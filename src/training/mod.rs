pub mod calibration;
pub mod eval;
pub mod features;
pub mod joint;
pub mod metric;
pub mod tasks;
pub mod trainer;

pub use calibration::brier_calibration_loss;
pub use eval::{evaluate_dataset, ChoiceOutcome, EvalReport, NoulOutcome, ScenarioEval, ScoreOutcome};
pub use features::{
    compute_features_from_backbone, get_or_compute_features, CacheOutcome, CacheSettings,
    FeatureContext,
};
pub use joint::{
    JointLossBreakdown, JointLossConfig, JointLossOutput, JointTier1Loss, MultiQuestionTargets,
    TrainingBatch,
};
pub use metric::{benign_adversarial_metric_loss, MetricLossConfig};
pub use tasks::{choice_cross_entropy_loss, noul_bce_loss, ordinal_score_loss};
pub use trainer::{train_heads, TrainConfig, TrainOutput};
