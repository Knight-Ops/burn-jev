pub mod embedding;
pub mod engine;
pub mod in_memory;
pub mod traits;

pub use embedding::{AnomalyEmbedding, MetricEmbedding, KNN_EMBEDDING_DIM};
pub use engine::{
    calibrate_threshold_from_distances, AnomalyVerdict, TenantConfig, TenantRegistry,
    DEFAULT_ANOMALY_THRESHOLD, DEFAULT_COLD_START_HORIZON, DEFAULT_KNN_K, GLOBAL_ANCHOR_NAMESPACE,
};
pub use in_memory::{FifoRingBuffer, InMemoryRingIndex, DEFAULT_RING_CAPACITY};
pub use traits::{AnomalyError, AnomalyIndex};
