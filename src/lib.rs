pub mod anomaly;
pub mod backend;
pub mod cache;
pub mod dataset;
pub mod encoding;
pub mod model;
pub mod pipeline;
pub mod training;

pub use anomaly::{
    calibrate_threshold_from_distances, AnomalyEmbedding, AnomalyError, AnomalyIndex,
    AnomalyVerdict, FifoRingBuffer, InMemoryRingIndex, MetricEmbedding, TenantConfig,
    TenantRegistry, DEFAULT_ANOMALY_THRESHOLD, DEFAULT_COLD_START_HORIZON, DEFAULT_KNN_K,
    DEFAULT_RING_CAPACITY, GLOBAL_ANCHOR_NAMESPACE, KNN_EMBEDDING_DIM,
};
pub use backend::{BackendKind, CpuBackend};
pub use cache::{CachedScenario, FeatureCache, FeatureCacheMetadata, ScenarioDiskMeta};
pub use dataset::{
    ChoiceQuestionRecord, ChoiceQuestionRequest, JevDataset, JevDatasetError, JevScenarioRecord,
    NoulQueryRecord, NoulQueryRequest, ReflexRequest, ScoreRubricRecord, ScoreRubricRequest,
    TokenizedScenario,
};
pub use encoding::{
    encode_scenario, EncodedScenario, EncodingConfig, EncodingError, ItemKind, ItemSpan,
    ScenarioTexts, ENCODING_VERSION, NUM_ITEM_TYPES,
};
pub use model::{
    load_artifact, read_artifact_metadata, save_artifact, sha256_file, ArtifactMetadata,
    ChoiceVerdict, DecisionModel, DecisionModelConfig, FeatureBatch, ItemReader,
    ItemReaderConfig, JevError, LoadError, LoadedEncoder, ModernBertConfig, ModernBertEncoder,
    Head, ModernBertLoader, NoulVerdict, ScenarioFeatures, ScoreVerdict, UnifiedHeads,
    UnifiedHeadsConfig, ARTIFACT_FORMAT_VERSION, MAX_CHOICE_CANDIDATES,
};
pub use pipeline::{
    EncodingSummary, EscalationReason, ReflexEngine, ReflexError, ReflexSecurityRouter,
    ReflexVerdict, Tier1Routing, DEFAULT_THREAT_THRESHOLD,
};
pub use training::{
    benign_adversarial_metric_loss, brier_calibration_loss, choice_cross_entropy_loss,
    compute_features, evaluate_dataset, get_or_compute_features, noul_bce_loss, ordinal_score_loss,
    fit_temperatures, train_decision_model, Baselines, CacheOutcome, CacheSettings, DecisionBatcher, EvalReport,
    FeatureContext, JointLossBreakdown, JointLossConfig, MetricLossConfig, MultiQuestionTargets,
    TrainConfig, TrainOutput,
};
