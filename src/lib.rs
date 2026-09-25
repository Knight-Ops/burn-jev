pub mod anomaly;
pub mod backend;
pub mod cache;
pub mod dataset;
pub mod delimiters;
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
pub use cache::{
    CachedScenario, FeatureCache, FeatureCacheMetadata, ScenarioDiskMeta,
};
pub use dataset::{
    ChoiceQuestionRecord, ChoiceQuestionRequest, JevDataset, JevDatasetError, JevScenarioRecord,
    NoulQueryRecord, NoulQueryRequest, ReflexRequest, ScoreRubricRecord, ScoreRubricRequest,
    TokenizedScenario,
};
pub use delimiters::{
    extract_batched_candidates, extract_candidate_states, extract_choice_question_candidates,
    extract_cls_state, extract_noul_state, extract_noul_states, extract_score_state,
    extract_score_states, ChoiceQuestionCoordinates, CoordinateResolver, CoordinateTarget,
    DelimiterConfig, DelimiterError, SequenceCoordinates,
};
pub use model::{
    BiMamba2Backbone, BiMamba2Config, BiMamba2JevKNN, BiMamba2JevKNNConfig, ChoiceVerdict,
    HeadsMetadata, JevError, LoadError, LoadReport, LoadedBackbone, LoaderOptions,
    Mamba2CheckpointLoader, Mamba2SSDBlock, Mamba2SSDConfig, NoulVerdict, ScoreVerdict,
    UnifiedHeads, UnifiedHeadsConfig, HEADS_FORMAT_VERSION, MAX_CHOICE_CANDIDATES,
};
pub use pipeline::{
    EscalationReason, ReflexEngine, ReflexError, ReflexSecurityRouter, ReflexVerdict,
    Tier1Routing, DEFAULT_THREAT_THRESHOLD,
};
pub use training::{
    benign_adversarial_metric_loss, brier_calibration_loss, choice_cross_entropy_loss,
    compute_features_from_backbone, evaluate_dataset, get_or_compute_features, noul_bce_loss,
    ordinal_score_loss, train_heads, CacheOutcome, CacheSettings, EvalReport, FeatureContext,
    JointLossBreakdown, JointLossConfig, JointLossOutput, JointTier1Loss, MetricLossConfig,
    MultiQuestionTargets, TrainConfig, TrainOutput, TrainingBatch,
};
