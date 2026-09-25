use burn::tensor::{backend::Backend, Int, Tensor};
use serde::{Deserialize, Serialize};

use crate::delimiters::{
    extract_batched_candidates, extract_cls_state, extract_noul_states, extract_score_states,
    DelimiterError, SequenceCoordinates,
};
use crate::model::BiMamba2Backbone;
use crate::training::calibration::brier_calibration_loss;
use crate::training::metric::benign_adversarial_metric_loss;
use crate::training::tasks::{choice_cross_entropy_loss, noul_bce_loss, ordinal_score_loss};

// =====================================================================
// Joint Loss Configuration & Batched Containers
// =====================================================================

/// Scalar weights controlling the multi-objective loss balance.
#[derive(Clone, Debug)]
pub struct JointLossConfig {
    pub lambda_metric: f32,
    pub lambda_noul: f32,
    pub lambda_choice: f32,
    pub lambda_score: f32,
    pub lambda_calibration: f32,
    pub metric_temperature: f32,
}

impl Default for JointLossConfig {
    fn default() -> Self {
        Self {
            lambda_metric: 0.1,
            lambda_noul: 1.0,
            lambda_choice: 1.0,
            lambda_score: 0.5,
            lambda_calibration: 0.2,
            metric_temperature: 0.07,
        }
    }
}

impl JointLossConfig {
    /// Weights used by `reflex-train` for head-only training on a frozen backbone
    /// (choice/noul 1.5, score 1.0, metric 0.1, calibration 0.2).
    pub fn frozen_backbone() -> Self {
        Self {
            lambda_metric: 0.1,
            lambda_noul: 1.5,
            lambda_choice: 1.5,
            lambda_score: 1.0,
            lambda_calibration: 0.2,
            metric_temperature: 0.07,
        }
    }

    pub fn with_lambda_metric(mut self, val: f32) -> Self {
        self.lambda_metric = val;
        self
    }

    pub fn with_lambda_noul(mut self, val: f32) -> Self {
        self.lambda_noul = val;
        self
    }

    pub fn with_lambda_choice(mut self, val: f32) -> Self {
        self.lambda_choice = val;
        self
    }

    pub fn with_lambda_score(mut self, val: f32) -> Self {
        self.lambda_score = val;
        self
    }

    pub fn with_lambda_calibration(mut self, val: f32) -> Self {
        self.lambda_calibration = val;
        self
    }

    pub fn with_metric_temperature(mut self, val: f32) -> Self {
        self.metric_temperature = val;
        self
    }
}

/// Batched container holding the padded input tokens and extracted target tensors.
pub struct TrainingBatch<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub coords: Vec<SequenceCoordinates>,
    pub is_benign: Vec<bool>,
    pub choice_targets: Option<Tensor<B, 1, Int>>,
    pub noul_targets: Option<Tensor<B, 1>>,
    pub score_targets: Option<Tensor<B, 1>>,
}

/// Multi-question targets for a single sequence or scenario.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MultiQuestionTargets {
    pub choice_targets: Vec<usize>,
    pub noul_targets: Vec<f32>,
    pub score_targets: Vec<f32>,
}

impl MultiQuestionTargets {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_choice_target(mut self, target: usize) -> Self {
        self.choice_targets.push(target);
        self
    }

    pub fn with_noul_target(mut self, target: f32) -> Self {
        self.noul_targets.push(target);
        self
    }

    pub fn with_score_target(mut self, target: f32) -> Self {
        self.score_targets.push(target);
        self
    }
}

/// Loss breakdown returned by `JointTier1Loss::forward`.
#[derive(Clone, Debug, Default)]
pub struct JointLossBreakdown {
    pub total_loss: f32,
    pub metric_loss: f32,
    pub choice_loss: f32,
    pub noul_loss: f32,
    pub score_loss: f32,
    pub calibration_loss: f32,
}

/// Result of evaluating the joint loss function on a batch.
pub struct JointLossOutput<B: Backend> {
    pub total_loss: Tensor<B, 1>,
    pub breakdown: JointLossBreakdown,
}

// =====================================================================
// Joint Tier 1 Reflex Loss Evaluator
// =====================================================================

/// Evaluates the combined multi-task objective across all active Tier 1 heads.
pub struct JointTier1Loss {
    config: JointLossConfig,
}

impl JointTier1Loss {
    pub fn new(config: JointLossConfig) -> Self {
        Self { config }
    }

    /// Evaluates the joint multi-task loss on a forward-passed model and batch.
    pub fn forward<B: Backend>(
        &self,
        model: &BiMamba2Backbone<B>,
        batch: TrainingBatch<B>,
        device: &B::Device,
    ) -> Result<JointLossOutput<B>, DelimiterError> {
        let batch_size = batch.coords.len();
        let hidden = model.forward_backbone(batch.input_ids);

        // 1. Metric Loss (Anomaly Projection Head)
        let cls_states = extract_cls_state(&hidden, 0, &batch.coords[0]);
        let mut cls_list = Vec::with_capacity(batch_size);
        cls_list.push(cls_states);

        for b in 1..batch_size {
            cls_list.push(extract_cls_state(&hidden, b, &batch.coords[b]));
        }

        let stacked_cls = Tensor::cat(cls_list, 0);
        let embeddings = model.heads.extract_knn_embedding(stacked_cls);

        let metric_loss = benign_adversarial_metric_loss(
            embeddings,
            &batch.is_benign,
            self.config.metric_temperature,
            device,
        );

        // 2. Choice Loss (Categorical Actions)
        let choice_loss = if let Some(ref targets) = batch.choice_targets {
            let (batched_cands, mask) = extract_batched_candidates(&hidden, &batch.coords, device)?;
            let choice_logits = model.heads.forward_batched_choice_logits(batched_cands);
            choice_cross_entropy_loss(choice_logits, targets.clone(), mask, None, device)
        } else {
            Tensor::<B, 1>::zeros([1], device)
        };

        // 3. Noul BCE & Calibration Loss (Boolean Assertions)
        let (noul_loss, cal_loss) = if let Some(ref targets) = batch.noul_targets {
            let mut noul_states_list = Vec::new();
            for (b, coord) in batch.coords.iter().enumerate() {
                if let Some(state) = extract_noul_states(&hidden, b, coord, device) {
                    noul_states_list.push(state);
                }
            }

            if !noul_states_list.is_empty() {
                let stacked_noul = Tensor::cat(noul_states_list, 0);
                let noul_logits = model.heads.forward_noul_logits(stacked_noul);
                let bce = noul_bce_loss(noul_logits.clone(), targets.clone(), None, device);
                let brier = brier_calibration_loss(noul_logits, targets.clone(), None, device);
                (bce, brier)
            } else {
                (Tensor::<B, 1>::zeros([1], device), Tensor::<B, 1>::zeros([1], device))
            }
        } else {
            (Tensor::<B, 1>::zeros([1], device), Tensor::<B, 1>::zeros([1], device))
        };

        // 4. Score Ordinal Loss (Metric Rubrics)
        let score_loss = if let Some(ref targets) = batch.score_targets {
            let mut score_states_list = Vec::new();
            for (b, coord) in batch.coords.iter().enumerate() {
                if let Some(state) = extract_score_states(&hidden, b, coord, device) {
                    score_states_list.push(state);
                }
            }

            if !score_states_list.is_empty() {
                let stacked_score = Tensor::cat(score_states_list, 0);
                let score_logits = model.heads.forward_score_logits(stacked_score);
                ordinal_score_loss(
                    score_logits,
                    targets.clone(),
                    model.heads.num_rubric_bins,
                    None,
                    device,
                )
            } else {
                Tensor::<B, 1>::zeros([1], device)
            }
        } else {
            Tensor::<B, 1>::zeros([1], device)
        };

        // 5. Multi-Objective Weighted Sum
        let total_loss = metric_loss.clone() * self.config.lambda_metric
            + choice_loss.clone() * self.config.lambda_choice
            + noul_loss.clone() * self.config.lambda_noul
            + score_loss.clone() * self.config.lambda_score
            + cal_loss.clone() * self.config.lambda_calibration;

        let breakdown = JointLossBreakdown {
            total_loss: total_loss.clone().into_data().as_slice::<f32>().unwrap()[0],
            metric_loss: metric_loss.into_data().as_slice::<f32>().unwrap()[0],
            choice_loss: choice_loss.into_data().as_slice::<f32>().unwrap()[0],
            noul_loss: noul_loss.into_data().as_slice::<f32>().unwrap()[0],
            score_loss: score_loss.into_data().as_slice::<f32>().unwrap()[0],
            calibration_loss: cal_loss.into_data().as_slice::<f32>().unwrap()[0],
        };

        Ok(JointLossOutput {
            total_loss,
            breakdown,
        })
    }
}
