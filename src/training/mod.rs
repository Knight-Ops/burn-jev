pub mod calibration;
pub mod eval;
pub mod features;
pub mod joint;
pub mod metric;
pub mod tasks;
pub mod temperature;
pub mod trainer;

pub use calibration::brier_calibration_loss;
pub use eval::{
    evaluate_dataset, Baselines, ChoiceOutcome, EvalReport, NoulOutcome, ScenarioEval, ScoreOutcome,
};
pub use features::{compute_features, get_or_compute_features, CacheOutcome, CacheSettings, FeatureContext};
pub use joint::{JointLossBreakdown, JointLossConfig, MultiQuestionTargets};
pub use metric::{benign_adversarial_metric_loss, MetricLossConfig};
pub use tasks::{choice_cross_entropy_loss, noul_bce_loss, ordinal_score_loss};
pub use temperature::{fit_temperature, fit_temperatures, HeadLogits};
pub use trainer::{
    train_decision_model, DecisionBatch, DecisionBatcher, DecisionMetric, DecisionStats, DecisionStepOutput,
    MetricKind, TrainConfig, TrainOutput,
};
