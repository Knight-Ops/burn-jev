use std::fmt;

use burn::tensor::{backend::Backend, Int, Tensor};
use serde::Serialize;

use crate::anomaly::{AnomalyError, MetricEmbedding};
use crate::delimiters::{
    extract_candidate_states, extract_choice_question_candidates, extract_cls_state,
    extract_noul_states, extract_score_states, CoordinateResolver, DelimiterError,
    SequenceCoordinates,
};
use crate::model::{
    BiMamba2Backbone, ChoiceVerdict, JevError, NoulVerdict, ScoreVerdict,
};

/// Combined error type for the Tier 1 Reflex Engine.
#[derive(Clone, Debug, PartialEq)]
pub enum ReflexError {
    Delimiter(DelimiterError),
    Jev(JevError),
    Anomaly(AnomalyError),
}

impl fmt::Display for ReflexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Delimiter(e) => write!(f, "Delimiter coordinate error: {e:?}"),
            Self::Jev(e) => write!(f, "Jev decision error: {e:?}"),
            Self::Anomaly(e) => write!(f, "Anomaly / embedding error: {e}"),
        }
    }
}

impl std::error::Error for ReflexError {}

impl From<DelimiterError> for ReflexError {
    fn from(e: DelimiterError) -> Self {
        Self::Delimiter(e)
    }
}

impl From<JevError> for ReflexError {
    fn from(e: JevError) -> Self {
        Self::Jev(e)
    }
}

impl From<AnomalyError> for ReflexError {
    fn from(e: AnomalyError) -> Self {
        Self::Anomaly(e)
    }
}

/// Generic, typed reflex verdict produced by the System 1 engine.
///
/// Contains all parsed decisions and metric representations in a single pass:
/// - Categorical candidate selections (`choices`) for each delimited choice question.
/// - Calibrated boolean assertions (`nouls`) for each queried `<noul_q>`.
/// - Calibrated ordinal rubric scores (`scores`) for each queried `<score_q>`.
/// - Backward-compatible `choice`, `noul`, and `score` accessors for the primary/first question.
/// - L2-normalized 256-dim metric state embedding (`embedding`) from `[CLS]`.
///
/// Serializes to JSON without the backward-compatible `choice`/`noul`/`score` accessors,
/// which duplicate the first element of `choices`/`nouls`/`scores`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReflexVerdict {
    /// Categorical decisions for each in-context choice question.
    pub choices: Vec<ChoiceVerdict>,
    /// Calibrated boolean assertions for each noul question.
    pub nouls: Vec<NoulVerdict>,
    /// Calibrated continuous/ordinal scores for each rubric question.
    pub scores: Vec<ScoreVerdict>,

    /// Primary / first choice verdict (for backward compatibility).
    #[serde(skip)]
    pub choice: Option<ChoiceVerdict>,
    /// Primary / first noul assertion verdict (for backward compatibility).
    #[serde(skip)]
    pub noul: Option<NoulVerdict>,
    /// Primary / first rubric score verdict (for backward compatibility).
    #[serde(skip)]
    pub score: Option<ScoreVerdict>,

    /// L2-normalized 256-dim metric state embedding from [CLS].
    pub embedding: MetricEmbedding,
    /// Resolved token sequence coordinates.
    pub coordinates: SequenceCoordinates,
}

/// The general-purpose In-Context Reflex Engine.
///
/// Orchestrates dynamic sequence parsing, bidirectional SSD forward propagation,
/// and typed decision evaluation without autoregressive token generation.
pub struct ReflexEngine<B: Backend> {
    pub model: BiMamba2Backbone<B>,
    pub resolver: CoordinateResolver,
}

impl<B: Backend> ReflexEngine<B> {
    /// Construct a new ReflexEngine instance with a trained backbone and delimiter configuration.
    pub fn new(model: BiMamba2Backbone<B>, resolver: CoordinateResolver) -> Self {
        Self { model, resolver }
    }

    /// Evaluates a tokenized sequence in a single sub-15ms forward reflex pass.
    pub fn evaluate(
        &self,
        tokens: &[i64],
        device: &B::Device,
    ) -> Result<ReflexVerdict, ReflexError> {
        // 1. Resolve in-context delimiters dynamically (O(N) single-pass)
        let coords = self.resolver.resolve_coordinates(tokens)?;

        // 2. Prepare 2D input tensor [1, L]
        let input_ids = Tensor::<B, 1, Int>::from_data(tokens, device).unsqueeze_dim(0);

        // 3. Single parallel pass forward through bidirectional Mamba-2 layers
        let h = self.model.forward_backbone(input_ids);

        // 4. Extract CLS state and compute L2-normalized 256-dim metric state embedding
        let cls_token = extract_cls_state(&h, 0, &coords);
        let knn_tensor = self.model.heads.extract_knn_embedding(cls_token);
        let embedding = MetricEmbedding::from_burn_tensor_row(&knn_tensor, 0)?;

        // 5. Evaluate Jev Choice Head(s)
        let mut choices = Vec::with_capacity(coords.choice_questions.len());
        if !coords.choice_questions.is_empty() {
            for q in &coords.choice_questions {
                if !q.candidate_indices.is_empty() {
                    let cand_vectors = extract_choice_question_candidates(&h, 0, q, device)?;
                    choices.push(self.model.heads.evaluate_choice_verdict(cand_vectors)?);
                }
            }
        } else if coords.candidate_count() > 0 {
            let cand_vectors = extract_candidate_states(&h, 0, &coords, device)?;
            choices.push(self.model.heads.evaluate_choice_verdict(cand_vectors)?);
        }
        let choice = choices.first().cloned();

        // 6. Evaluate Jev Noul Head(s) across all boolean assertions
        let nouls = if coords.has_noul() {
            if let Some(noul_tokens) = extract_noul_states(&h, 0, &coords, device) {
                self.model.heads.evaluate_noul_verdicts_default(noul_tokens)?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let noul = nouls.first().cloned();

        // 7. Evaluate Jev Score Head(s) across all rubric queries
        let scores = if coords.has_score() {
            if let Some(score_tokens) = extract_score_states(&h, 0, &coords, device) {
                self.model.heads.evaluate_score_verdicts(score_tokens)?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let score = scores.first().cloned();

        Ok(ReflexVerdict {
            choices,
            nouls,
            scores,
            choice,
            noul,
            score,
            embedding,
            coordinates: coords,
        })
    }
}
