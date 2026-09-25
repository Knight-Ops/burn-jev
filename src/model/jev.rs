use std::fmt;

use serde::Serialize;

// =====================================================================
// Jev Typed Decision Primitives
// =====================================================================

/// Error types for the Jev typed decision layer.
#[derive(Clone, Debug, PartialEq)]
pub enum JevError {
    /// No candidates were provided for categorical choice evaluation.
    EmptyCandidates,
    /// Number of candidates exceeds architectural maximum (K <= 255).
    TooManyCandidates { count: usize, max: usize },
    /// Decision threshold is outside the valid range [0.0, 1.0].
    InvalidThreshold { threshold: f32 },
    /// Probability vector is empty or malformed.
    EmptyProbabilities,
    /// Dimension mismatch during tensor extraction or head evaluation.
    InvalidDimension { expected: usize, found: usize },
}

impl fmt::Display for JevError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidates => write!(f, "No candidates provided (K must be >= 1)"),
            Self::TooManyCandidates { count, max } => {
                write!(f, "Candidate count {} exceeds maximum allowed {}", count, max)
            }
            Self::InvalidThreshold { threshold } => {
                write!(
                    f,
                    "Invalid threshold {}: must be in range [0.0, 1.0]",
                    threshold
                )
            }
            Self::EmptyProbabilities => write!(f, "Probabilities list cannot be empty"),
            Self::InvalidDimension { expected, found } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, found {}",
                    expected, found
                )
            }
        }
    }
}

impl std::error::Error for JevError {}

/// Maximum number of candidates supported by the In-Context Choice head.
pub const MAX_CHOICE_CANDIDATES: usize = 255;

/// Typed verdict for a calibrated boolean assertion evaluated at a `<noul_q>` marker.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NoulVerdict {
    /// Discrete binary outcome: true if `probability >= threshold`.
    pub is_true: bool,
    /// Platt-calibrated probability in [0.0, 1.0].
    pub probability: f32,
    /// Calibrated temperature T parameter used during scaling.
    pub calibrated_temperature: f32,
}

impl NoulVerdict {
    /// Constructs and validates a NoulVerdict given probability, decision threshold, and temperature.
    pub fn new(probability: f32, threshold: f32, calibrated_temperature: f32) -> Result<Self, JevError> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(JevError::InvalidThreshold { threshold });
        }
        // Clamp numerical precision edge cases (e.g. 1.0000001 or -0.0000001) to [0.0, 1.0]
        let clamped_prob = probability.clamp(0.0, 1.0);
        let is_true = clamped_prob >= threshold;

        Ok(Self {
            is_true,
            probability: clamped_prob,
            calibrated_temperature,
        })
    }
}

/// Typed verdict for continuous bounded metric & monotonic ordinal rubric evaluated at `<score_q>`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScoreVerdict {
    /// Continuous expected score in range [1.0, M].
    pub expected_score: f32,
    /// Monotonic discrete rubric bin in range 1..=M.
    pub discrete_bin: usize,
    /// Vector of cumulative threshold probabilities [P(Score > 1), ..., P(Score > M-1)].
    pub cumulative_probs: Vec<f32>,
}

impl ScoreVerdict {
    /// Constructs a ScoreVerdict from cumulative threshold probabilities and total rubric bins M.
    pub fn new(cumulative_probs: Vec<f32>, num_rubric_bins: usize) -> Result<Self, JevError> {
        if num_rubric_bins < 2 {
            return Err(JevError::InvalidDimension {
                expected: 2,
                found: num_rubric_bins,
            });
        }
        let expected_thresholds = num_rubric_bins - 1;
        if cumulative_probs.len() != expected_thresholds {
            return Err(JevError::InvalidDimension {
                expected: expected_thresholds,
                found: cumulative_probs.len(),
            });
        }

        // Clamp probabilities to [0.0, 1.0]
        let clamped_probs: Vec<f32> = cumulative_probs
            .iter()
            .map(|&p| p.clamp(0.0, 1.0))
            .collect();

        // Expected score: 1.0 + sum(P(Score > m)) for m in 1..M-1
        let expected_score = 1.0 + clamped_probs.iter().sum::<f32>();

        // Discrete bin: 1 + number of cumulative thresholds >= 0.5
        let discrete_bin = 1 + clamped_probs.iter().filter(|&&p| p >= 0.5).count();

        Ok(Self {
            expected_score,
            discrete_bin,
            cumulative_probs: clamped_probs,
        })
    }
}

/// Typed verdict for in-context dynamic candidate selection evaluated across `<cand>` markers.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChoiceVerdict {
    /// Selected candidate index in 0..K-1 corresponding to argmax probability / logit.
    pub selected_candidate: usize,
    /// Calibrated categorical probability distribution over K candidates (sums to 1.0).
    pub probabilities: Vec<f32>,
    /// Scaled pre-softmax logits [z_0 / T, ..., z_{K-1} / T].
    pub logits: Vec<f32>,
}

impl ChoiceVerdict {
    /// Constructs and validates a ChoiceVerdict from categorical probabilities and logits.
    pub fn new(probabilities: Vec<f32>, logits: Vec<f32>) -> Result<Self, JevError> {
        let k = probabilities.len();
        if k == 0 {
            return Err(JevError::EmptyCandidates);
        }
        if k > MAX_CHOICE_CANDIDATES {
            return Err(JevError::TooManyCandidates {
                count: k,
                max: MAX_CHOICE_CANDIDATES,
            });
        }
        if logits.len() != k {
            return Err(JevError::InvalidDimension {
                expected: k,
                found: logits.len(),
            });
        }

        // Find argmax candidate index (favoring highest probability / logit)
        let mut best_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (idx, &prob) in probabilities.iter().enumerate() {
            if prob > max_val {
                max_val = prob;
                best_idx = idx;
            }
        }

        Ok(Self {
            selected_candidate: best_idx,
            probabilities,
            logits,
        })
    }
}
