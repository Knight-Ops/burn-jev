use std::fmt;
use burn::tensor::{backend::Backend, Bool, Int, Tensor};
use serde::{Deserialize, Serialize};

// =====================================================================
// Delimiter Token Registry & Configuration
// =====================================================================

/// Slicing coordinate target relative to the delimiter marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CoordinateTarget {
    /// Slice at the exact position of the delimiter marker token.
    #[default]
    MarkerToken,
    /// Slice at the token immediately following the marker (e.g., candidate payload token).
    NextToken,
}

/// Configuration defining special token IDs and parsing constraints.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelimiterConfig {
    pub cls_id: i64,
    pub sep_id: i64,
    pub eos_id: i64,
    pub cand_marker_id: i64,
    pub noul_query_marker_id: i64,
    pub score_query_marker_id: i64,
    pub choice_query_marker_id: i64,
    pub max_candidates: usize,
    pub max_choice_questions: usize,
    pub max_noul_questions: usize,
    pub max_score_questions: usize,
    pub target_mode: CoordinateTarget,
    pub require_cls: bool,
}

impl Default for DelimiterConfig {
    fn default() -> Self {
        Self {
            cls_id: 1,
            sep_id: 2,
            eos_id: 3,
            cand_marker_id: 4,
            noul_query_marker_id: 5,
            score_query_marker_id: 6,
            choice_query_marker_id: 7,
            max_candidates: 255,
            max_choice_questions: 32,
            max_noul_questions: 64,
            max_score_questions: 64,
            target_mode: CoordinateTarget::MarkerToken,
            require_cls: true,
        }
    }
}

impl DelimiterConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Dedicated delimiter IDs (50280..=50286) for Mamba-2 checkpoints using the GPT-NeoX
    /// tokenizer. They sit in the padding rows above the 50277-entry BPE vocabulary, so they
    /// never collide with real tokens (the `Default` IDs 1..=7 are ordinary punctuation there).
    pub fn mamba2_reserved() -> Self {
        Self::default()
            .with_cls_id(50280)
            .with_sep_id(50281)
            .with_eos_id(50282)
            .with_cand_marker_id(50283)
            .with_noul_query_marker_id(50284)
            .with_score_query_marker_id(50285)
            .with_choice_query_marker_id(50286)
    }

    /// Largest special token ID; must be below the backbone's embedding vocabulary size.
    pub fn max_token_id(&self) -> i64 {
        [
            self.cls_id,
            self.sep_id,
            self.eos_id,
            self.cand_marker_id,
            self.noul_query_marker_id,
            self.score_query_marker_id,
            self.choice_query_marker_id,
        ]
        .into_iter()
        .max()
        .unwrap()
    }

    pub fn with_cls_id(mut self, id: i64) -> Self {
        self.cls_id = id;
        self
    }

    pub fn with_sep_id(mut self, id: i64) -> Self {
        self.sep_id = id;
        self
    }

    pub fn with_eos_id(mut self, id: i64) -> Self {
        self.eos_id = id;
        self
    }

    pub fn with_cand_marker_id(mut self, id: i64) -> Self {
        self.cand_marker_id = id;
        self
    }

    pub fn with_noul_query_marker_id(mut self, id: i64) -> Self {
        self.noul_query_marker_id = id;
        self
    }

    pub fn with_score_query_marker_id(mut self, id: i64) -> Self {
        self.score_query_marker_id = id;
        self
    }

    pub fn with_choice_query_marker_id(mut self, id: i64) -> Self {
        self.choice_query_marker_id = id;
        self
    }

    pub fn with_max_candidates(mut self, max: usize) -> Self {
        self.max_candidates = max;
        self
    }

    pub fn with_max_choice_questions(mut self, max: usize) -> Self {
        self.max_choice_questions = max;
        self
    }

    pub fn with_max_noul_questions(mut self, max: usize) -> Self {
        self.max_noul_questions = max;
        self
    }

    pub fn with_max_score_questions(mut self, max: usize) -> Self {
        self.max_score_questions = max;
        self
    }

    pub fn with_target_mode(mut self, mode: CoordinateTarget) -> Self {
        self.target_mode = mode;
        self
    }

    pub fn with_require_cls(mut self, require: bool) -> Self {
        self.require_cls = require;
        self
    }
}

// =====================================================================
// Sequence Coordinates & Error Primitives
// =====================================================================

/// Resolved coordinate targets for a single in-context choice question.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ChoiceQuestionCoordinates {
    /// Optional index for the `<choice_q>` marker token initiating this choice question.
    pub query_index: Option<usize>,
    /// Resolved coordinates for candidates in this question.
    pub candidate_indices: Vec<usize>,
}

impl ChoiceQuestionCoordinates {
    pub fn new(query_index: Option<usize>, candidate_indices: Vec<usize>) -> Self {
        Self {
            query_index,
            candidate_indices,
        }
    }

    pub fn candidate_count(&self) -> usize {
        self.candidate_indices.len()
    }
}

/// Resolved token coordinate positions within a tokenized sequence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SequenceCoordinates {
    /// Index for the [CLS] token (typically 0) used for k-NN anomaly representation.
    pub cls_index: usize,
    /// Indices for in-context choice candidates (<cand>) across questions (or primary question).
    pub candidate_indices: Vec<usize>,
    /// Optional index for primary boolean query assertion (<noul_q>).
    pub noul_query_index: Option<usize>,
    /// Optional index for primary continuous/ordinal rubric evaluation (<score_q>).
    pub score_query_index: Option<usize>,
    /// Optional index for sequence separation [SEP].
    pub sep_index: Option<usize>,
    /// Optional index for sequence termination [EOS].
    pub eos_index: Option<usize>,
    /// Total sequence length.
    pub seq_len: usize,

    /// Multi-question choice sets (<choice_q> ... <cand> ...).
    pub choice_questions: Vec<ChoiceQuestionCoordinates>,
    /// All boolean query assertion markers (<noul_q>).
    pub noul_query_indices: Vec<usize>,
    /// All continuous/ordinal rubric query markers (<score_q>).
    pub score_query_indices: Vec<usize>,
}

impl SequenceCoordinates {
    pub fn candidate_count(&self) -> usize {
        self.candidate_indices.len()
    }

    pub fn choice_question_count(&self) -> usize {
        self.choice_questions.len()
    }

    pub fn noul_question_count(&self) -> usize {
        self.noul_query_indices.len()
    }

    pub fn score_question_count(&self) -> usize {
        self.score_query_indices.len()
    }

    pub fn has_noul(&self) -> bool {
        !self.noul_query_indices.is_empty()
    }

    pub fn has_score(&self) -> bool {
        !self.score_query_indices.is_empty()
    }

    pub fn has_choice(&self) -> bool {
        !self.choice_questions.is_empty()
            && self
                .choice_questions
                .iter()
                .any(|q| !q.candidate_indices.is_empty())
    }
}

/// Errors occurring during delimiter scanning and coordinate extraction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DelimiterError {
    EmptySequence,
    MissingClsToken,
    TooManyCandidates { count: usize, max: usize },
    TooManyChoiceQuestions { count: usize, max: usize },
    TooManyNoulQuestions { count: usize, max: usize },
    TooManyScoreQuestions { count: usize, max: usize },
    NoCandidatesFound,
    DuplicateNoulQuery { first: usize, second: usize },
    DuplicateScoreQuery { first: usize, second: usize },
    IndexOutOfBounds { index: usize, seq_len: usize },
    BatchMismatch { expected: usize, found: usize },
}

impl fmt::Display for DelimiterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySequence => write!(f, "Input token sequence is empty"),
            Self::MissingClsToken => write!(f, "Missing required CLS token"),
            Self::TooManyCandidates { count, max } => {
                write!(f, "Candidate count ({count}) exceeds maximum allowed ({max})")
            }
            Self::TooManyChoiceQuestions { count, max } => {
                write!(
                    f,
                    "Choice question count ({count}) exceeds maximum allowed ({max})"
                )
            }
            Self::TooManyNoulQuestions { count, max } => {
                write!(
                    f,
                    "Noul question count ({count}) exceeds maximum allowed ({max})"
                )
            }
            Self::TooManyScoreQuestions { count, max } => {
                write!(
                    f,
                    "Score question count ({count}) exceeds maximum allowed ({max})"
                )
            }
            Self::NoCandidatesFound => write!(f, "No candidate delimiters found in sequence"),
            Self::DuplicateNoulQuery { first, second } => write!(
                f,
                "Duplicate noul query marker found at index {second} (first at {first})"
            ),
            Self::DuplicateScoreQuery { first, second } => write!(
                f,
                "Duplicate score query marker found at index {second} (first at {first})"
            ),
            Self::IndexOutOfBounds { index, seq_len } => write!(
                f,
                "Coordinate index {index} out of bounds for sequence length {seq_len}"
            ),
            Self::BatchMismatch { expected, found } => write!(
                f,
                "Batch size mismatch: expected {expected} items, found {found}"
            ),
        }
    }
}

impl std::error::Error for DelimiterError {}

// =====================================================================
// Zero-Copy Single-Pass Coordinate Resolver
// =====================================================================

#[derive(Clone, Debug)]
pub struct CoordinateResolver {
    config: DelimiterConfig,
}

impl CoordinateResolver {
    pub fn new(config: DelimiterConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &DelimiterConfig {
        &self.config
    }

    /// Performs a single-pass O(N) scan over tokens to resolve delimiter coordinates.
    pub fn resolve_coordinates(&self, tokens: &[i64]) -> Result<SequenceCoordinates, DelimiterError> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(DelimiterError::EmptySequence);
        }

        let mut cls_index = None;
        let mut sep_index = None;
        let mut eos_index = None;
        let mut candidate_indices = Vec::new();
        let mut choice_questions: Vec<ChoiceQuestionCoordinates> = Vec::new();
        let mut noul_query_indices = Vec::new();
        let mut score_query_indices = Vec::new();

        for (idx, &token) in tokens.iter().enumerate() {
            if token == self.config.cls_id && cls_index.is_none() {
                cls_index = Some(idx);
            } else if token == self.config.sep_id {
                sep_index = Some(idx);
            } else if token == self.config.eos_id {
                eos_index = Some(idx);
            } else if token == self.config.choice_query_marker_id {
                let target_idx = match self.config.target_mode {
                    CoordinateTarget::MarkerToken => idx,
                    CoordinateTarget::NextToken => {
                        let next = idx + 1;
                        if next >= seq_len {
                            return Err(DelimiterError::IndexOutOfBounds {
                                index: next,
                                seq_len,
                            });
                        }
                        next
                    }
                };
                choice_questions.push(ChoiceQuestionCoordinates::new(Some(target_idx), Vec::new()));
                if choice_questions.len() > self.config.max_choice_questions {
                    return Err(DelimiterError::TooManyChoiceQuestions {
                        count: choice_questions.len(),
                        max: self.config.max_choice_questions,
                    });
                }
            } else if token == self.config.cand_marker_id {
                let target_idx = match self.config.target_mode {
                    CoordinateTarget::MarkerToken => idx,
                    CoordinateTarget::NextToken => {
                        let next = idx + 1;
                        if next >= seq_len {
                            return Err(DelimiterError::IndexOutOfBounds {
                                index: next,
                                seq_len,
                            });
                        }
                        next
                    }
                };
                candidate_indices.push(target_idx);
                if candidate_indices.len() > self.config.max_candidates {
                    return Err(DelimiterError::TooManyCandidates {
                        count: candidate_indices.len(),
                        max: self.config.max_candidates,
                    });
                }
                if choice_questions.is_empty() {
                    choice_questions.push(ChoiceQuestionCoordinates::new(None, vec![target_idx]));
                } else {
                    choice_questions
                        .last_mut()
                        .unwrap()
                        .candidate_indices
                        .push(target_idx);
                }
            } else if token == self.config.noul_query_marker_id {
                let target_idx = match self.config.target_mode {
                    CoordinateTarget::MarkerToken => idx,
                    CoordinateTarget::NextToken => {
                        let next = idx + 1;
                        if next >= seq_len {
                            return Err(DelimiterError::IndexOutOfBounds {
                                index: next,
                                seq_len,
                            });
                        }
                        next
                    }
                };
                noul_query_indices.push(target_idx);
                if noul_query_indices.len() > self.config.max_noul_questions {
                    return Err(DelimiterError::TooManyNoulQuestions {
                        count: noul_query_indices.len(),
                        max: self.config.max_noul_questions,
                    });
                }
            } else if token == self.config.score_query_marker_id {
                let target_idx = match self.config.target_mode {
                    CoordinateTarget::MarkerToken => idx,
                    CoordinateTarget::NextToken => {
                        let next = idx + 1;
                        if next >= seq_len {
                            return Err(DelimiterError::IndexOutOfBounds {
                                index: next,
                                seq_len,
                            });
                        }
                        next
                    }
                };
                score_query_indices.push(target_idx);
                if score_query_indices.len() > self.config.max_score_questions {
                    return Err(DelimiterError::TooManyScoreQuestions {
                        count: score_query_indices.len(),
                        max: self.config.max_score_questions,
                    });
                }
            }
        }

        let cls_index = match cls_index {
            Some(idx) => idx,
            None => {
                if self.config.require_cls {
                    return Err(DelimiterError::MissingClsToken);
                } else {
                    0
                }
            }
        };

        let noul_query_index = noul_query_indices.first().copied();
        let score_query_index = score_query_indices.first().copied();

        Ok(SequenceCoordinates {
            cls_index,
            candidate_indices,
            noul_query_index,
            score_query_index,
            sep_index,
            eos_index,
            seq_len,
            choice_questions,
            noul_query_indices,
            score_query_indices,
        })
    }

    /// Resolves coordinates for a batch of sequences.
    pub fn resolve_coordinates_batched(
        &self,
        batch_tokens: &[Vec<i64>],
    ) -> Result<Vec<SequenceCoordinates>, DelimiterError> {
        let mut results = Vec::with_capacity(batch_tokens.len());
        for tokens in batch_tokens {
            results.push(self.resolve_coordinates(tokens)?);
        }
        Ok(results)
    }
}

// =====================================================================
// Dynamic Tensor Coordinate Slicing
// =====================================================================

/// Extracts the [CLS] representation for k-NN anomaly metric indexing.
///
/// Returns a 2D tensor of shape `[1, d_model]`.
pub fn extract_cls_state<B: Backend>(
    hidden: &Tensor<B, 3>,
    batch_idx: usize,
    coords: &SequenceCoordinates,
) -> Tensor<B, 2> {
    let [_, _, d_model] = hidden.dims();
    let cls_idx = coords.cls_index;
    hidden
        .clone()
        .slice([batch_idx..batch_idx + 1, cls_idx..cls_idx + 1, 0..d_model])
        .squeeze_dim(1)
}

/// Extracts candidate representations for dynamic in-context Choice evaluation.
///
/// Returns a 2D tensor of shape `[K, d_model]` where K is the number of candidates.
pub fn extract_candidate_states<B: Backend>(
    hidden: &Tensor<B, 3>,
    batch_idx: usize,
    coords: &SequenceCoordinates,
    device: &B::Device,
) -> Result<Tensor<B, 2>, DelimiterError> {
    if coords.candidate_indices.is_empty() {
        return Err(DelimiterError::NoCandidatesFound);
    }

    let [_, seq_len, d_model] = hidden.dims();
    let indices_i64: Vec<i64> = coords.candidate_indices.iter().map(|&i| i as i64).collect();
    let indices_tensor = Tensor::<B, 1, Int>::from_data(indices_i64.as_slice(), device);

    let item_hidden = hidden
        .clone()
        .slice([batch_idx..batch_idx + 1, 0..seq_len, 0..d_model]);

    let selected = item_hidden.select(1, indices_tensor).squeeze_dim(0); // [K, d_model]

    Ok(selected)
}

/// Extracts candidate representations for a specific in-context choice question.
///
/// Returns a 2D tensor of shape `[K_q, d_model]` where K_q is the number of candidates in the question.
pub fn extract_choice_question_candidates<B: Backend>(
    hidden: &Tensor<B, 3>,
    batch_idx: usize,
    question: &ChoiceQuestionCoordinates,
    device: &B::Device,
) -> Result<Tensor<B, 2>, DelimiterError> {
    if question.candidate_indices.is_empty() {
        return Err(DelimiterError::NoCandidatesFound);
    }

    let [_, seq_len, d_model] = hidden.dims();
    let indices_i64: Vec<i64> = question
        .candidate_indices
        .iter()
        .map(|&i| i as i64)
        .collect();
    let indices_tensor = Tensor::<B, 1, Int>::from_data(indices_i64.as_slice(), device);

    let item_hidden = hidden
        .clone()
        .slice([batch_idx..batch_idx + 1, 0..seq_len, 0..d_model])
        .squeeze_dim(0);

    let selected = item_hidden.select(0, indices_tensor); // [K_q, d_model]

    Ok(selected)
}

/// Extracts the hidden state at the primary `<noul_q>` marker for Boolean assertion evaluation.
///
/// Returns `Some([1, d_model])` if present, or `None`.
pub fn extract_noul_state<B: Backend>(
    hidden: &Tensor<B, 3>,
    batch_idx: usize,
    coords: &SequenceCoordinates,
) -> Option<Tensor<B, 2>> {
    let [_, _, d_model] = hidden.dims();
    coords.noul_query_index.map(|idx| {
        hidden
            .clone()
            .slice([batch_idx..batch_idx + 1, idx..idx + 1, 0..d_model])
            .squeeze_dim(1)
    })
}

/// Extracts the hidden states at all `<noul_q>` markers for multi-question Boolean assertion evaluation.
///
/// Returns `Some([Q_noul, d_model])` if one or more noul queries are present, or `None`.
pub fn extract_noul_states<B: Backend>(
    hidden: &Tensor<B, 3>,
    batch_idx: usize,
    coords: &SequenceCoordinates,
    device: &B::Device,
) -> Option<Tensor<B, 2>> {
    if coords.noul_query_indices.is_empty() {
        return None;
    }
    let [_, seq_len, d_model] = hidden.dims();
    if coords.noul_query_indices.len() == 1 {
        let idx = coords.noul_query_indices[0];
        Some(
            hidden
                .clone()
                .slice([batch_idx..batch_idx + 1, idx..idx + 1, 0..d_model])
                .squeeze_dim(1),
        )
    } else {
        let indices_i64: Vec<i64> = coords
            .noul_query_indices
            .iter()
            .map(|&i| i as i64)
            .collect();
        let indices_tensor = Tensor::<B, 1, Int>::from_data(indices_i64.as_slice(), device);
        let item_hidden = hidden
            .clone()
            .slice([batch_idx..batch_idx + 1, 0..seq_len, 0..d_model])
            .squeeze_dim(0);
        Some(item_hidden.select(0, indices_tensor))
    }
}

/// Extracts the hidden state at the primary `<score_q>` marker for continuous/ordinal rubric evaluation.
///
/// Returns `Some([1, d_model])` if present, or `None`.
pub fn extract_score_state<B: Backend>(
    hidden: &Tensor<B, 3>,
    batch_idx: usize,
    coords: &SequenceCoordinates,
) -> Option<Tensor<B, 2>> {
    let [_, _, d_model] = hidden.dims();
    coords.score_query_index.map(|idx| {
        hidden
            .clone()
            .slice([batch_idx..batch_idx + 1, idx..idx + 1, 0..d_model])
            .squeeze_dim(1)
    })
}

/// Extracts the hidden states at all `<score_q>` markers for multi-question continuous/ordinal rubric evaluation.
///
/// Returns `Some([Q_score, d_model])` if one or more score queries are present, or `None`.
pub fn extract_score_states<B: Backend>(
    hidden: &Tensor<B, 3>,
    batch_idx: usize,
    coords: &SequenceCoordinates,
    device: &B::Device,
) -> Option<Tensor<B, 2>> {
    if coords.score_query_indices.is_empty() {
        return None;
    }
    let [_, seq_len, d_model] = hidden.dims();
    if coords.score_query_indices.len() == 1 {
        let idx = coords.score_query_indices[0];
        Some(
            hidden
                .clone()
                .slice([batch_idx..batch_idx + 1, idx..idx + 1, 0..d_model])
                .squeeze_dim(1),
        )
    } else {
        let indices_i64: Vec<i64> = coords
            .score_query_indices
            .iter()
            .map(|&i| i as i64)
            .collect();
        let indices_tensor = Tensor::<B, 1, Int>::from_data(indices_i64.as_slice(), device);
        let item_hidden = hidden
            .clone()
            .slice([batch_idx..batch_idx + 1, 0..seq_len, 0..d_model])
            .squeeze_dim(0);
        Some(item_hidden.select(0, indices_tensor))
    }
}

/// Batched extraction of candidate states with variable K per item.
///
/// Returns:
/// - Padded candidates tensor of shape `[B, max_K, d_model]`.
/// - Boolean mask tensor of shape `[B, max_K]` where `true` indicates a valid candidate.
pub fn extract_batched_candidates<B: Backend>(
    hidden: &Tensor<B, 3>,
    coords: &[SequenceCoordinates],
    device: &B::Device,
) -> Result<(Tensor<B, 3>, Tensor<B, 2, Bool>), DelimiterError> {
    let [batch_size, _, d_model] = hidden.dims();
    if coords.len() != batch_size {
        return Err(DelimiterError::BatchMismatch {
            expected: batch_size,
            found: coords.len(),
        });
    }

    let max_k = coords
        .iter()
        .map(|c| c.candidate_indices.len())
        .max()
        .unwrap_or(0);

    if max_k == 0 {
        return Err(DelimiterError::NoCandidatesFound);
    }

    let mut candidate_tensors = Vec::with_capacity(batch_size);
    let mut mask_data = Vec::with_capacity(batch_size * max_k);

    for (b, coord) in coords.iter().enumerate() {
        let k = coord.candidate_indices.len();
        if k > 0 {
            let item_candidates = extract_candidate_states(hidden, b, coord, device)?;
            if k < max_k {
                let pad_k = max_k - k;
                let padding = Tensor::<B, 2>::zeros([pad_k, d_model], device);
                let padded = Tensor::cat(vec![item_candidates, padding], 0); // [max_K, d_model]
                candidate_tensors.push(padded.unsqueeze_dim(0));
            } else {
                candidate_tensors.push(item_candidates.unsqueeze_dim(0));
            }
        } else {
            let zeros = Tensor::<B, 2>::zeros([max_k, d_model], device);
            candidate_tensors.push(zeros.unsqueeze_dim(0));
        }

        for i in 0..max_k {
            mask_data.push(i < k);
        }
    }

    let batched_candidates = Tensor::cat(candidate_tensors, 0); // [B, max_K, d_model]
    let mask_tensor = Tensor::<B, 1, Bool>::from_data(mask_data.as_slice(), device)
        .reshape([batch_size, max_k]);

    Ok((batched_candidates, mask_tensor))
}
