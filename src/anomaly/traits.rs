use std::fmt;

use crate::anomaly::embedding::AnomalyEmbedding;

/// Error conditions arising during vector indexing, embedding extraction, and anomaly gating.
#[derive(Clone, Debug, PartialEq)]
pub enum AnomalyError {
    /// Vector length does not match expected dimension (e.g. 256)
    InvalidDimension { expected: usize, found: usize },
    /// Vector norm deviates from unit length ||z||_2 = 1.0 beyond tolerance
    NormalizationError { norm: f32 },
    /// Tenant or namespace is not registered in the system
    TenantNotFound { tenant_id: String },
    /// Insufficient samples available to perform k-NN query or calibration
    InsufficientSamples { needed: usize, found: usize },
    /// Anomaly threshold is non-positive or invalid
    InvalidThreshold { threshold: f32 },
    /// False Positive Rate must be in range (0.0, 1.0)
    InvalidFpr { fpr: f32 },
    /// Low-level backend failure (e.g., remote vector DB gRPC error or lock failure)
    BackendError { message: String },
}

impl fmt::Display for AnomalyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { expected, found } => {
                write!(f, "Invalid vector dimension: expected {expected}, found {found}")
            }
            Self::NormalizationError { norm } => {
                write!(f, "Vector is not L2-normalized: norm is {norm}, expected ~1.0")
            }
            Self::TenantNotFound { tenant_id } => {
                write!(f, "Tenant namespace not found: '{tenant_id}'")
            }
            Self::InsufficientSamples { needed, found } => {
                write!(f, "Insufficient samples in namespace: needed {needed}, found {found}")
            }
            Self::InvalidThreshold { threshold } => {
                write!(f, "Invalid anomaly threshold {threshold}: must be positive and finite")
            }
            Self::InvalidFpr { fpr } => {
                write!(f, "Invalid FPR {fpr}: must be strictly between 0.0 and 1.0")
            }
            Self::BackendError { message } => {
                write!(f, "Vector index backend error: {message}")
            }
        }
    }
}

impl std::error::Error for AnomalyError {}

/// Pluggable abstraction for nearest-neighbor vector stores.
///
/// This trait abstracts the physical vector index implementation, decoupling
/// the Tier 1 anomaly gating logic from the underlying storage engine.
/// Implementations may include:
/// - In-process SIMD FIFO Ring Buffer (`InMemoryRingIndex`, default)
/// - Remote vector databases (e.g., Qdrant, Milvus, pgvector via gRPC/HTTP)
/// - Embedded HNSW engines (e.g. USearch, Instant-Distance)
pub trait AnomalyIndex: Send + Sync {
    /// Compute the k-NN distance of the query vector to the nearest samples in the namespace.
    ///
    /// Returns a vector of distances sorted in ascending order (closest first).
    /// If fewer than `k` samples exist, returns distances to all available samples.
    fn search_knn(
        &self,
        namespace: &str,
        query: &AnomalyEmbedding,
        k: usize,
    ) -> Result<Vec<f32>, AnomalyError>;

    /// Insert a verified benign query vector into a tenant's historical namespace.
    fn insert_benign(
        &self,
        namespace: &str,
        vector: AnomalyEmbedding,
    ) -> Result<(), AnomalyError>;

    /// Return the count of vectors currently stored in this namespace.
    fn sample_count(&self, namespace: &str) -> usize;

    /// Retrieve all stored embeddings for a namespace (useful for offline threshold calibration).
    fn get_all(&self, namespace: &str) -> Result<Vec<AnomalyEmbedding>, AnomalyError>;

    /// Clear all vectors from a namespace.
    fn clear(&self, namespace: &str) -> Result<(), AnomalyError>;
}
