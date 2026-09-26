//! =====================================================================
//! JEV JSONL Dataset Format & Preprocessing Pipeline
//! =====================================================================
//!
//! Provides structured serialization, deserialization, and tokenization
//! for multi-question JEV training datasets in JSON Lines (.jsonl) format.
//!
//! Each line is a self-contained JSON scenario containing:
//! - Context / telemetry state
//! - Multiple choice questions with candidate actions and ground-truth index
//! - Multiple Noul boolean assertions with ground-truth float target (0.0 or 1.0)
//! - Multiple continuous/ordinal score rubrics with ground-truth float target (1.0 to 5.0)
//! - Optional benign/adversarial indicator for contrastive metric learning (absent = unknown,
//!   e.g. imported external data; such scenarios are left out of the metric loss)

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::encoding::{encode_scenario, EncodedScenario, EncodingConfig, EncodingError, ScenarioTexts};
use crate::training::MultiQuestionTargets;

// =====================================================================
// Schema Definitions
// =====================================================================

/// Representation of a single in-context choice question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChoiceQuestionRecord {
    pub prompt: String,
    pub candidates: Vec<String>,
    pub target: usize,
}

/// Representation of a single Noul boolean assertion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoulQueryRecord {
    pub assertion: String,
    pub target: f32,
}

/// Representation of a single continuous or ordinal rubric evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreRubricRecord {
    pub prompt: String,
    pub target: f32,
}

/// A complete grounded scenario record in JSONL format.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JevScenarioRecord {
    pub id: String,
    #[serde(default)]
    pub domain: Option<String>,
    pub context: String,
    /// `None` = unknown (external data); excluded from the contrastive metric loss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_benign: Option<bool>,
    #[serde(default)]
    pub choice_questions: Vec<ChoiceQuestionRecord>,
    #[serde(default)]
    pub noul_queries: Vec<NoulQueryRecord>,
    #[serde(default)]
    pub score_rubrics: Vec<ScoreRubricRecord>,
}

// =====================================================================
// Dataset Errors
// =====================================================================

#[derive(Debug)]
pub enum JevDatasetError {
    Io(std::io::Error),
    Json { line: usize, source: serde_json::Error },
    Tokenizer(String),
    Encoding(EncodingError),
    Validation(String),
}

impl std::fmt::Display for JevDatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "Dataset I/O error: {e}"),
            Self::Json { line, source } => write!(f, "JSON parse error at line {line}: {source}"),
            Self::Tokenizer(e) => write!(f, "Tokenizer error: {e}"),
            Self::Encoding(e) => write!(f, "Encoding error: {e}"),
            Self::Validation(e) => write!(f, "Dataset validation error: {e}"),
        }
    }
}

impl std::error::Error for JevDatasetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<EncodingError> for JevDatasetError {
    fn from(e: EncodingError) -> Self {
        Self::Encoding(e)
    }
}

impl From<std::io::Error> for JevDatasetError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// =====================================================================
// Tokenized Scenario Output
// =====================================================================

/// Preprocessed and tokenized representation ready for model forward pass.
#[derive(Clone, Debug)]
pub struct TokenizedScenario {
    pub id: String,
    pub encoded: EncodedScenario,
    pub targets: MultiQuestionTargets,
    pub is_benign: Option<bool>,
}

impl JevScenarioRecord {
    /// Validates the record fields for internal consistency.
    pub fn validate(&self) -> Result<(), JevDatasetError> {
        if self.id.trim().is_empty() {
            return Err(JevDatasetError::Validation("Scenario id cannot be empty".to_string()));
        }
        if self.context.trim().is_empty() {
            return Err(JevDatasetError::Validation(format!(
                "Scenario '{}' context cannot be empty",
                self.id
            )));
        }
        for (q_idx, q) in self.choice_questions.iter().enumerate() {
            if q.candidates.len() < 2 {
                return Err(JevDatasetError::Validation(format!(
                    "Scenario '{}' choice question {} must have at least 2 candidates",
                    self.id, q_idx + 1
                )));
            }
            if q.target >= q.candidates.len() {
                return Err(JevDatasetError::Validation(format!(
                    "Scenario '{}' choice question {} target ({}) exceeds candidate count ({})",
                    self.id, q_idx + 1, q.target, q.candidates.len()
                )));
            }
        }
        for (n_idx, n) in self.noul_queries.iter().enumerate() {
            if !(0.0..=1.0).contains(&n.target) {
                return Err(JevDatasetError::Validation(format!(
                    "Scenario '{}' noul query {} target ({}) must be in [0.0, 1.0]",
                    self.id, n_idx + 1, n.target
                )));
            }
        }
        for (s_idx, s) in self.score_rubrics.iter().enumerate() {
            if s.target < 1.0 {
                return Err(JevDatasetError::Validation(format!(
                    "Scenario '{}' score rubric {} target ({}) must be >= 1.0",
                    self.id, s_idx + 1, s.target
                )));
            }
        }
        Ok(())
    }

    /// Encodes this scenario and extracts multi-question training targets.
    pub fn encode(
        &self,
        tokenizer: &Tokenizer,
        cfg: &EncodingConfig,
    ) -> Result<TokenizedScenario, JevDatasetError> {
        self.validate()?;

        let encoded = self.to_request().encode(tokenizer, cfg)?;

        let mut targets = MultiQuestionTargets::new();
        for q_def in &self.choice_questions {
            targets = targets.with_choice_target(q_def.target);
        }
        for noul_def in &self.noul_queries {
            targets = targets.with_noul_target(noul_def.target);
        }
        for score_def in &self.score_rubrics {
            targets = targets.with_score_target(score_def.target);
        }

        Ok(TokenizedScenario {
            id: self.id.clone(),
            encoded,
            targets,
            is_benign: self.is_benign,
        })
    }

    /// Strips ground-truth labels, leaving the prompt that inference sees.
    pub fn to_request(&self) -> ReflexRequest {
        ReflexRequest {
            id: Some(self.id.clone()),
            context: self.context.clone(),
            choice_questions: self
                .choice_questions
                .iter()
                .map(|q| ChoiceQuestionRequest {
                    prompt: q.prompt.clone(),
                    candidates: q.candidates.clone(),
                })
                .collect(),
            noul_queries: self
                .noul_queries
                .iter()
                .map(|n| NoulQueryRequest {
                    assertion: n.assertion.clone(),
                })
                .collect(),
            score_rubrics: self
                .score_rubrics
                .iter()
                .map(|s| ScoreRubricRequest {
                    prompt: s.prompt.clone(),
                })
                .collect(),
        }
    }
}

// =====================================================================
// Unlabeled Inference Requests
// =====================================================================

/// A choice question without its ground-truth target.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChoiceQuestionRequest {
    pub prompt: String,
    pub candidates: Vec<String>,
}

/// A Noul assertion without its ground-truth target.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoulQueryRequest {
    pub assertion: String,
}

/// A score rubric without its ground-truth target.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreRubricRequest {
    pub prompt: String,
}

/// An unlabeled scenario submitted for inference.
///
/// Shares the JSONL shape of [`JevScenarioRecord`] minus the labels; unknown fields
/// (`target`, `is_benign`, `domain`) are ignored, so dataset files can be fed directly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReflexRequest {
    #[serde(default)]
    pub id: Option<String>,
    pub context: String,
    #[serde(default)]
    pub choice_questions: Vec<ChoiceQuestionRequest>,
    #[serde(default)]
    pub noul_queries: Vec<NoulQueryRequest>,
    #[serde(default)]
    pub score_rubrics: Vec<ScoreRubricRequest>,
}

impl ReflexRequest {
    /// Encodes the request into the span layout of [`crate::encoding`]; the single definition
    /// of the sequence layout, shared by training and inference.
    pub fn encode(&self, tokenizer: &Tokenizer, cfg: &EncodingConfig) -> Result<EncodedScenario, EncodingError> {
        let texts = ScenarioTexts {
            context: &self.context,
            choice_questions: self
                .choice_questions
                .iter()
                .map(|q| (q.prompt.as_str(), q.candidates.iter().map(String::as_str).collect()))
                .collect(),
            noul_queries: self.noul_queries.iter().map(|n| n.assertion.as_str()).collect(),
            score_rubrics: self.score_rubrics.iter().map(|s| s.prompt.as_str()).collect(),
        };
        encode_scenario(&texts, tokenizer, cfg)
    }
}

// =====================================================================
// JevDataset Container
// =====================================================================

/// In-memory collection of JEV scenario records with JSONL reading and writing.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JevDataset {
    pub records: Vec<JevScenarioRecord>,
}

impl JevDataset {
    pub fn new() -> Self {
        Self { records: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn add(&mut self, record: JevScenarioRecord) {
        self.records.push(record);
    }

    /// Loads a dataset from a JSON Lines string.
    pub fn from_jsonl_str(content: &str) -> Result<Self, JevDatasetError> {
        let mut records = Vec::new();
        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                continue;
            }
            let record: JevScenarioRecord = serde_json::from_str(trimmed).map_err(|source| {
                JevDatasetError::Json {
                    line: idx + 1,
                    source,
                }
            })?;
            record.validate()?;
            records.push(record);
        }
        Ok(Self { records })
    }

    /// Loads a dataset from a JSON Lines (.jsonl) file on disk.
    pub fn from_jsonl_file<P: AsRef<Path>>(path: P) -> Result<Self, JevDatasetError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();

        for (idx, line_res) in reader.lines().enumerate() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                continue;
            }
            let record: JevScenarioRecord = serde_json::from_str(trimmed).map_err(|source| {
                JevDatasetError::Json {
                    line: idx + 1,
                    source,
                }
            })?;
            record.validate()?;
            records.push(record);
        }

        Ok(Self { records })
    }

    /// Saves the dataset to a JSON Lines (.jsonl) file.
    pub fn save_jsonl_file<P: AsRef<Path>>(&self, path: P) -> Result<(), JevDatasetError> {
        let mut file = File::create(path)?;
        for record in &self.records {
            let line = serde_json::to_string(record)
                .map_err(|e| JevDatasetError::Validation(format!("Serialization error: {e}")))?;
            writeln!(file, "{line}")?;
        }
        Ok(())
    }

    /// Splits the dataset into train and validation partitions.
    pub fn split(self, val_fraction: f32) -> (Self, Self) {
        let n = self.records.len();
        let n_val = ((n as f32) * val_fraction).round() as usize;
        let n_train = n.saturating_sub(n_val);

        let mut train_records = Vec::with_capacity(n_train);
        let mut val_records = Vec::with_capacity(n_val);

        for (i, record) in self.records.into_iter().enumerate() {
            if i < n_train {
                train_records.push(record);
            } else {
                val_records.push(record);
            }
        }

        (Self { records: train_records }, Self { records: val_records })
    }
}
