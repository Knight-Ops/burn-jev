//! Multi-task loss weights, per-scenario targets and the loss breakdown reported per epoch.

use serde::{Deserialize, Serialize};

// =====================================================================
// Joint Loss Configuration & Targets
// =====================================================================

/// Scalar weights controlling the multi-objective loss balance.
#[derive(Clone, Debug)]
pub struct JointLossConfig {
    pub lambda_metric: f32,
    pub lambda_noul: f32,
    pub lambda_choice: f32,
    pub lambda_score: f32,
    pub lambda_calibration: f32,
    pub metric_temperature: f32,
}

impl Default for JointLossConfig {
    fn default() -> Self {
        Self {
            lambda_metric: 0.1,
            lambda_noul: 1.0,
            lambda_choice: 1.0,
            lambda_score: 0.5,
            lambda_calibration: 0.2,
            metric_temperature: 0.07,
        }
    }
}

impl JointLossConfig {
    /// Weights used by `reflex-train` when training the decision model on a frozen encoder
    /// (choice/noul 1.5, score 1.0, metric 0.1, calibration 0.2).
    pub fn frozen_backbone() -> Self {
        Self {
            lambda_metric: 0.1,
            lambda_noul: 1.5,
            lambda_choice: 1.5,
            lambda_score: 1.0,
            lambda_calibration: 0.2,
            metric_temperature: 0.07,
        }
    }

    pub fn with_lambda_metric(mut self, val: f32) -> Self {
        self.lambda_metric = val;
        self
    }

    pub fn with_lambda_noul(mut self, val: f32) -> Self {
        self.lambda_noul = val;
        self
    }

    pub fn with_lambda_choice(mut self, val: f32) -> Self {
        self.lambda_choice = val;
        self
    }

    pub fn with_lambda_score(mut self, val: f32) -> Self {
        self.lambda_score = val;
        self
    }

    pub fn with_lambda_calibration(mut self, val: f32) -> Self {
        self.lambda_calibration = val;
        self
    }

    pub fn with_metric_temperature(mut self, val: f32) -> Self {
        self.metric_temperature = val;
        self
    }
}

/// Multi-question targets for a single sequence or scenario.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MultiQuestionTargets {
    pub choice_targets: Vec<usize>,
    pub noul_targets: Vec<f32>,
    pub score_targets: Vec<f32>,
}

impl MultiQuestionTargets {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_choice_target(mut self, target: usize) -> Self {
        self.choice_targets.push(target);
        self
    }

    pub fn with_noul_target(mut self, target: f32) -> Self {
        self.noul_targets.push(target);
        self
    }

    pub fn with_score_target(mut self, target: f32) -> Self {
        self.score_targets.push(target);
        self
    }
}

/// Mean per-task losses (unweighted) and the weighted total.
#[derive(Clone, Debug, Default)]
pub struct JointLossBreakdown {
    pub total_loss: f32,
    pub metric_loss: f32,
    pub choice_loss: f32,
    pub noul_loss: f32,
    pub score_loss: f32,
    pub calibration_loss: f32,
}
