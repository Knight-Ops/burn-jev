use burn::tensor::{backend::Backend, Tensor};
use serde::Serialize;

use crate::anomaly::traits::AnomalyError;

/// Dimension of the L2-normalized metric projection vector (z in R^256)
pub const KNN_EMBEDDING_DIM: usize = 256;

/// Tolerance for validating unit L2-norm (||z||_2 = 1.0)
const L2_NORM_TOLERANCE: f32 = 1e-3;

/// Small epsilon to avoid division by zero during normalization
const NORM_EPSILON: f32 = 1e-8;

/// An L2-normalized vector embedding in R^256 used for metric search,
/// state clustering, semantic caching, or anomaly gating.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct MetricEmbedding {
    values: Vec<f32>,
}

/// Backwards-compatible alias for metric embeddings used in anomaly gating.
pub type AnomalyEmbedding = MetricEmbedding;

impl MetricEmbedding {
    /// Construct a new MetricEmbedding from an existing L2-normalized vector.
    ///
    /// Validates that:
    /// 1. `values.len() == KNN_EMBEDDING_DIM` (256)
    /// 2. `||values||_2 \approx 1.0` within `1e-3` tolerance.
    pub fn new(values: Vec<f32>) -> Result<Self, AnomalyError> {
        if values.len() != KNN_EMBEDDING_DIM {
            return Err(AnomalyError::InvalidDimension {
                expected: KNN_EMBEDDING_DIM,
                found: values.len(),
            });
        }

        let norm_sq: f32 = values.iter().map(|&v| v * v).sum();
        let norm = norm_sq.sqrt();

        if (norm - 1.0).abs() > L2_NORM_TOLERANCE {
            return Err(AnomalyError::NormalizationError { norm });
        }

        Ok(Self { values })
    }

    /// Construct a MetricEmbedding by normalizing an arbitrary non-zero vector in R^256.
    pub fn from_raw_unnormalized(values: Vec<f32>) -> Result<Self, AnomalyError> {
        if values.len() != KNN_EMBEDDING_DIM {
            return Err(AnomalyError::InvalidDimension {
                expected: KNN_EMBEDDING_DIM,
                found: values.len(),
            });
        }

        let norm_sq: f32 = values.iter().map(|&v| v * v).sum();
        let norm = (norm_sq + NORM_EPSILON).sqrt();

        let normalized: Vec<f32> = values.iter().map(|&v| v / norm).collect();
        Ok(Self { values: normalized })
    }

    /// Extract a MetricEmbedding from a 1D Burn Tensor of shape `[256]`.
    pub fn from_burn_tensor<B: Backend>(tensor: Tensor<B, 1>) -> Result<Self, AnomalyError> {
        let dims: [usize; 1] = tensor.shape().dims();
        if dims[0] != KNN_EMBEDDING_DIM {
            return Err(AnomalyError::InvalidDimension {
                expected: KNN_EMBEDDING_DIM,
                found: dims[0],
            });
        }

        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().map_err(|_| AnomalyError::BackendError {
            message: "Failed to cast tensor data into f32 slice".to_string(),
        })?;

        Self::new(slice.to_vec())
    }

    /// Extract a MetricEmbedding from a row of a 2D Burn Tensor of shape `[B, 256]`.
    pub fn from_burn_tensor_row<B: Backend>(
        tensor: &Tensor<B, 2>,
        row: usize,
    ) -> Result<Self, AnomalyError> {
        let dims: [usize; 2] = tensor.shape().dims();
        if dims[1] != KNN_EMBEDDING_DIM {
            return Err(AnomalyError::InvalidDimension {
                expected: KNN_EMBEDDING_DIM,
                found: dims[1],
            });
        }
        if row >= dims[0] {
            return Err(AnomalyError::BackendError {
                message: format!(
                    "Row index {row} out of bounds for tensor with batch size {}",
                    dims[0]
                ),
            });
        }

        let row_tensor = tensor.clone().slice([row..(row + 1), 0..KNN_EMBEDDING_DIM]).squeeze_dim(0);
        Self::from_burn_tensor(row_tensor)
    }

    /// Read-only slice view of the 256 float components.
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Consume the wrapper to retrieve the raw `Vec<f32>`.
    pub fn into_vec(self) -> Vec<f32> {
        self.values
    }

    /// Compute the Euclidean distance ||self - other||_2.
    #[inline]
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        let mut sum_sq = 0.0f32;
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            let diff = a - b;
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }

    /// Compute the Cosine distance: 1.0 - (self . other).
    #[inline]
    pub fn cosine_distance(&self, other: &Self) -> f32 {
        let dot_product: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();
        (1.0 - dot_product).max(0.0)
    }
}
