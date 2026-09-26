use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::{
        activation::{gelu, sigmoid, softmax},
        backend::Backend,
        Tensor,
    },
};

use crate::model::jev::{ChoiceVerdict, JevError, NoulVerdict, ScoreVerdict, MAX_CHOICE_CANDIDATES};

// =====================================================================
// Heads: k-NN Metric Projection + Jev Decision Primitives
// =====================================================================

#[derive(Config, Debug)]
pub struct UnifiedHeadsConfig {
    pub d_model: usize,
    #[config(default = "256")]
    pub knn_dim: usize,
    #[config(default = "5")]
    pub num_rubric_bins: usize,
    /// Applied between each head's hidden layer and its output (training only).
    #[config(default = "0.1")]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct UnifiedHeads<B: Backend> {
    pub knn_fc1: Linear<B>,
    pub knn_fc2: Linear<B>,
    pub choice_fc1: Linear<B>,
    pub choice_fc2: Linear<B>,
    pub noul_fc1: Linear<B>,
    pub noul_fc2: Linear<B>,
    pub score_fc1: Linear<B>,
    pub score_fc2: Linear<B>,
    /// Post-hoc calibration temperatures, one per decision head. Frozen at 1.0 during
    /// training and fit afterwards on validation NLL (see `training::fit_temperatures`).
    pub choice_temperature: Param<Tensor<B, 1>>,
    pub noul_temperature: Param<Tensor<B, 1>>,
    pub score_temperature: Param<Tensor<B, 1>>,
    pub dropout: Dropout,
    pub num_rubric_bins: usize,
}

impl UnifiedHeadsConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UnifiedHeads<B> {
        let mid_dim = self.d_model / 2;

        let knn_fc1 = LinearConfig::new(self.d_model, mid_dim).init(device);
        let knn_fc2 = LinearConfig::new(mid_dim, self.knn_dim).init(device);

        let choice_fc1 = LinearConfig::new(self.d_model, mid_dim).init(device);
        let choice_fc2 = LinearConfig::new(mid_dim, 1).init(device);

        let noul_fc1 = LinearConfig::new(self.d_model, mid_dim).init(device);
        let noul_fc2 = LinearConfig::new(mid_dim, 1).init(device);

        let score_fc1 = LinearConfig::new(self.d_model, mid_dim).init(device);
        let score_fc2 = LinearConfig::new(mid_dim, self.num_rubric_bins - 1).init(device);
        let temperature = || Param::from_tensor(Tensor::<B, 1>::ones([1], device)).set_require_grad(false);

        UnifiedHeads {
            knn_fc1,
            knn_fc2,
            choice_fc1,
            choice_fc2,
            noul_fc1,
            noul_fc2,
            score_fc1,
            score_fc2,
            choice_temperature: temperature(),
            noul_temperature: temperature(),
            score_temperature: temperature(),
            dropout: DropoutConfig::new(self.dropout).init(),
            num_rubric_bins: self.num_rubric_bins,
        }
    }
}

/// A calibrated decision head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Head {
    Choice,
    Noul,
    Score,
}

impl Head {
    pub const ALL: [Head; 3] = [Head::Choice, Head::Noul, Head::Score];
}

impl<B: Backend> UnifiedHeads<B> {
    fn temperature_param(&self, head: Head) -> &Param<Tensor<B, 1>> {
        match head {
            Head::Choice => &self.choice_temperature,
            Head::Noul => &self.noul_temperature,
            Head::Score => &self.score_temperature,
        }
    }

    /// Returns `head`'s calibration temperature tensor clamped to [0.01, 10.0]
    pub fn temperature(&self, head: Head) -> Tensor<B, 1> {
        self.temperature_param(head).val().clamp(0.01, 10.0)
    }

    /// Returns the scalar float value of `head`'s clamped calibration temperature
    pub fn temperature_value(&self, head: Head) -> f32 {
        self.temperature(head).into_data().convert::<f32>().to_vec::<f32>().unwrap()[0]
    }

    /// Replaces the calibration temperatures (kept frozen: they are fit post hoc, not trained).
    pub fn set_temperatures(&mut self, choice: f32, noul: f32, score: f32) {
        let device = self.choice_temperature.device();
        let param = |t: f32| Param::from_tensor(Tensor::<B, 1>::from_floats([t], &device)).set_require_grad(false);
        self.choice_temperature = param(choice);
        self.noul_temperature = param(noul);
        self.score_temperature = param(score);
    }

    /// Extracts an L2-normalized vector for metric/HNSW search
    pub fn extract_knn_embedding(&self, pooled_state: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.dropout.forward(gelu(self.knn_fc1.forward(pooled_state)));
        let projected = self.knn_fc2.forward(h);

        // L2 Normalization: v / (||v||_2 + 1e-8)
        let norm = projected.clone().powf_scalar(2.0).sum_dim(1).sqrt() + 1e-8;
        projected / norm
    }

    /// Computes raw (unscaled) Noul boolean assertion logits [batch_size]
    pub fn forward_noul_raw_logits(&self, token_vector: Tensor<B, 2>) -> Tensor<B, 1> {
        let h = self.dropout.forward(gelu(self.noul_fc1.forward(token_vector)));
        self.noul_fc2.forward(h).squeeze_dim(1)
    }

    /// Computes temperature-calibrated Noul logits [batch_size]
    pub fn forward_noul_logits(&self, token_vector: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_noul_raw_logits(token_vector) / self.temperature(Head::Noul)
    }

    /// Evaluates calibrated True/False assertion probability tensor
    pub fn evaluate_noul(&self, token_vector: Tensor<B, 2>) -> Tensor<B, 1> {
        sigmoid(self.forward_noul_logits(token_vector))
    }

    /// Evaluates calibrated True/False assertion returning a typed NoulVerdict
    pub fn evaluate_noul_verdict(
        &self,
        token_vector: Tensor<B, 2>,
        threshold: f32,
    ) -> Result<NoulVerdict, JevError> {
        let prob_tensor = self.evaluate_noul(token_vector);
        let prob = prob_tensor.into_data().as_slice::<f32>().unwrap()[0];
        let temp = self.temperature_value(Head::Noul);
        NoulVerdict::new(prob, threshold, temp)
    }

    /// Helper evaluating calibrated True/False assertion with standard 0.5 threshold
    pub fn evaluate_noul_default(
        &self,
        token_vector: Tensor<B, 2>,
    ) -> Result<NoulVerdict, JevError> {
        self.evaluate_noul_verdict(token_vector, 0.5)
    }

    /// Evaluates calibrated True/False assertions returning typed NoulVerdicts for multiple questions [Q, d_model]
    pub fn evaluate_noul_verdicts(
        &self,
        token_vectors: Tensor<B, 2>,
        threshold: f32,
    ) -> Result<Vec<NoulVerdict>, JevError> {
        let prob_tensor = self.evaluate_noul(token_vectors);
        let probs: Vec<f32> = prob_tensor.into_data().as_slice::<f32>().unwrap().to_vec();
        let temp = self.temperature_value(Head::Noul);
        probs
            .into_iter()
            .map(|p| NoulVerdict::new(p, threshold, temp))
            .collect()
    }

    /// Helper evaluating calibrated True/False assertions with default 0.5 threshold for multiple questions
    pub fn evaluate_noul_verdicts_default(
        &self,
        token_vectors: Tensor<B, 2>,
    ) -> Result<Vec<NoulVerdict>, JevError> {
        self.evaluate_noul_verdicts(token_vectors, 0.5)
    }

    /// Computes raw cumulative ordinal rubric logits [batch_size, num_rubric_bins - 1]
    pub fn forward_score_raw_logits(&self, token_vector: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.dropout.forward(gelu(self.score_fc1.forward(token_vector)));
        self.score_fc2.forward(h)
    }

    /// Computes temperature-calibrated cumulative ordinal rubric logits [batch_size, num_rubric_bins - 1]
    pub fn forward_score_logits(&self, token_vector: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_score_raw_logits(token_vector) / self.temperature(Head::Score).reshape([1, 1])
    }

    /// Evaluates continuous ordinal rubric score tensor in [1.0, num_rubric_bins]
    pub fn evaluate_score(&self, token_vector: Tensor<B, 2>) -> Tensor<B, 1> {
        let cumulative_logits = self.forward_score_logits(token_vector);
        let cum_probs = sigmoid(cumulative_logits);
        cum_probs.sum_dim(1).squeeze_dim(1) + 1.0
    }

    /// Evaluates continuous ordinal rubric score returning a typed ScoreVerdict
    pub fn evaluate_score_verdict(
        &self,
        token_vector: Tensor<B, 2>,
    ) -> Result<ScoreVerdict, JevError> {
        let cumulative_logits = self.forward_score_logits(token_vector);
        let cum_probs_tensor = sigmoid(cumulative_logits);
        let cum_probs = cum_probs_tensor
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        ScoreVerdict::new(cum_probs, self.num_rubric_bins)
    }

    /// Evaluates continuous ordinal rubric scores returning typed ScoreVerdicts for multiple questions [Q, d_model]
    pub fn evaluate_score_verdicts(
        &self,
        token_vectors: Tensor<B, 2>,
    ) -> Result<Vec<ScoreVerdict>, JevError> {
        let cumulative_logits = self.forward_score_logits(token_vectors);
        let cum_probs_tensor = sigmoid(cumulative_logits);
        let [num_queries, num_thresholds] = cum_probs_tensor.dims();
        let flat_probs: Vec<f32> = cum_probs_tensor
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let mut verdicts = Vec::with_capacity(num_queries);
        for chunk in flat_probs.chunks_exact(num_thresholds) {
            verdicts.push(ScoreVerdict::new(chunk.to_vec(), self.num_rubric_bins)?);
        }
        Ok(verdicts)
    }

    /// Computes raw choice logits for candidate vectors [k]
    pub fn forward_choice_raw_logits(&self, cand_vectors: Tensor<B, 2>) -> Tensor<B, 1> {
        let h = self.dropout.forward(gelu(self.choice_fc1.forward(cand_vectors)));
        self.choice_fc2.forward(h).squeeze_dim(1)
    }

    /// Computes temperature-calibrated choice logits for candidate vectors [k]
    pub fn forward_choice_logits(&self, cand_vectors: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_choice_raw_logits(cand_vectors) / self.temperature(Head::Choice)
    }

    /// Computes temperature-calibrated choice logits for batched candidates [batch_size, max_k]
    pub fn forward_batched_choice_logits(&self, batched_cands: Tensor<B, 3>) -> Tensor<B, 2> {
        let h = self.dropout.forward(gelu(self.choice_fc1.forward(batched_cands)));
        let raw = self.choice_fc2.forward(h).squeeze_dim(2);
        raw / self.temperature(Head::Choice).reshape([1, 1])
    }

    /// Slices candidate tokens and computes normalized Choice distribution tensor
    pub fn evaluate_choice(&self, cand_vectors: Tensor<B, 2>) -> Tensor<B, 1> {
        let logits = self.forward_choice_logits(cand_vectors);
        softmax(logits, 0)
    }

    /// Evaluates candidate tokens returning a typed ChoiceVerdict with argmax selection
    pub fn evaluate_choice_verdict(
        &self,
        cand_vectors: Tensor<B, 2>,
    ) -> Result<ChoiceVerdict, JevError> {
        let k = cand_vectors.dims()[0];
        if k == 0 {
            return Err(JevError::EmptyCandidates);
        }
        if k > MAX_CHOICE_CANDIDATES {
            return Err(JevError::TooManyCandidates {
                count: k,
                max: MAX_CHOICE_CANDIDATES,
            });
        }

        let logits_tensor = self.forward_choice_logits(cand_vectors);
        let probs_tensor = softmax(logits_tensor.clone(), 0);

        let logits = logits_tensor.into_data().as_slice::<f32>().unwrap().to_vec();
        let probs = probs_tensor.into_data().as_slice::<f32>().unwrap().to_vec();

        ChoiceVerdict::new(probs, logits)
    }
}
