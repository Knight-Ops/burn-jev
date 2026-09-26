use std::fmt;

use burn::tensor::backend::Backend;
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::anomaly::{AnomalyError, MetricEmbedding};
use crate::dataset::ReflexRequest;
use crate::encoding::{EncodedScenario, EncodingConfig, EncodingError};
use crate::model::{ChoiceVerdict, DecisionModel, JevError, ModernBertEncoder, NoulVerdict, ScoreVerdict};

/// Combined error type for the Tier 1 Reflex Engine.
#[derive(Clone, Debug, PartialEq)]
pub enum ReflexError {
    Encoding(EncodingError),
    Jev(JevError),
    Anomaly(AnomalyError),
}

impl fmt::Display for ReflexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encoding(e) => write!(f, "Encoding error: {e}"),
            Self::Jev(e) => write!(f, "Jev decision error: {e:?}"),
            Self::Anomaly(e) => write!(f, "Anomaly / embedding error: {e}"),
        }
    }
}

impl std::error::Error for ReflexError {}

impl From<EncodingError> for ReflexError {
    fn from(e: EncodingError) -> Self {
        Self::Encoding(e)
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

/// Sequence facts reported alongside a verdict.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EncodingSummary {
    pub tokens: usize,
    pub context_tokens: usize,
    pub context_truncated: bool,
}

impl From<&EncodedScenario> for EncodingSummary {
    fn from(e: &EncodedScenario) -> Self {
        Self {
            tokens: e.len(),
            context_tokens: e.context.len(),
            context_truncated: e.context_truncated,
        }
    }
}

/// Generic, typed reflex verdict produced by the System 1 engine.
///
/// Contains all parsed decisions and metric representations in a single pass:
/// - Categorical candidate selections (`choices`) for each choice question.
/// - Calibrated boolean assertions (`nouls`) for each noul query.
/// - Calibrated ordinal rubric scores (`scores`) for each score rubric.
/// - Backward-compatible `choice`, `noul`, and `score` accessors for the primary/first question.
/// - L2-normalized 256-dim metric state embedding (`embedding`) of the pooled context.
///
/// Serializes to JSON without the backward-compatible `choice`/`noul`/`score` accessors,
/// which duplicate the first element of `choices`/`nouls`/`scores`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReflexVerdict {
    pub choices: Vec<ChoiceVerdict>,
    pub nouls: Vec<NoulVerdict>,
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

    /// L2-normalized 256-dim metric state embedding of the scenario context.
    pub embedding: MetricEmbedding,
    pub encoding: EncodingSummary,
}

/// The general-purpose In-Context Reflex Engine.
///
/// Encodes a request, runs one bidirectional pass through the frozen ModernBERT encoder,
/// and reads every decision item against the context with the trained decision model —
/// no autoregressive generation.
pub struct ReflexEngine<B: Backend> {
    pub encoder: ModernBertEncoder<B>,
    pub decision: DecisionModel<B>,
    pub encoding: EncodingConfig,
    pub tokenizer: Tokenizer,
}

impl<B: Backend> ReflexEngine<B> {
    pub fn new(
        encoder: ModernBertEncoder<B>,
        decision: DecisionModel<B>,
        encoding: EncodingConfig,
        tokenizer: Tokenizer,
    ) -> Self {
        Self {
            encoder,
            decision,
            encoding,
            tokenizer,
        }
    }

    pub fn evaluate(&self, request: &ReflexRequest, device: &B::Device) -> Result<ReflexVerdict, ReflexError> {
        let encoded = request.encode(&self.tokenizer, &self.encoding)?;
        self.evaluate_encoded(&encoded, device)
    }

    pub fn evaluate_encoded(&self, encoded: &EncodedScenario, device: &B::Device) -> Result<ReflexVerdict, ReflexError> {
        let features = self
            .encoder
            .scenario_features(&[encoded], self.encoding.pad_id, device)
            .pop()
            .expect("one scenario in, one out");
        let decisions = self.decision.decide(&features, device)?;
        let embedding = MetricEmbedding::from_burn_tensor_row(&decisions.embedding, 0)?;

        Ok(ReflexVerdict {
            choice: decisions.choices.first().cloned(),
            noul: decisions.nouls.first().cloned(),
            score: decisions.scores.first().cloned(),
            choices: decisions.choices,
            nouls: decisions.nouls,
            scores: decisions.scores,
            embedding,
            encoding: EncodingSummary::from(encoded),
        })
    }
}
