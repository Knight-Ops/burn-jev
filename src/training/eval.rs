//! Evaluation of a [`ReflexEngine`] against a labeled dataset.

use burn::tensor::backend::Backend;
use serde::Serialize;

use crate::dataset::JevDataset;
use crate::pipeline::{ReflexEngine, ReflexVerdict};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChoiceOutcome {
    pub target: usize,
    pub predicted: usize,
    /// Probability assigned to the predicted candidate.
    pub confidence: f32,
}

impl ChoiceOutcome {
    pub fn correct(&self) -> bool {
        self.target == self.predicted
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NoulOutcome {
    pub target: bool,
    pub predicted: bool,
    pub probability: f32,
}

impl NoulOutcome {
    pub fn correct(&self) -> bool {
        self.target == self.predicted
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScoreOutcome {
    pub target: f32,
    pub predicted: f32,
}

impl ScoreOutcome {
    pub fn abs_error(&self) -> f32 {
        (self.predicted - self.target).abs()
    }
}

/// Per-scenario predictions paired with their labels.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScenarioEval {
    pub id: String,
    pub domain: Option<String>,
    pub choices: Vec<ChoiceOutcome>,
    pub nouls: Vec<NoulOutcome>,
    pub scores: Vec<ScoreOutcome>,
    #[serde(skip)]
    pub verdict: ReflexVerdict,
}

/// Aggregate metrics over a dataset, plus the per-scenario breakdown.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EvalReport {
    pub scenarios: Vec<ScenarioEval>,
}

impl EvalReport {
    pub fn choice_counts(&self) -> (usize, usize) {
        count(self.scenarios.iter().flat_map(|s| &s.choices).map(ChoiceOutcome::correct))
    }

    pub fn noul_counts(&self) -> (usize, usize) {
        count(self.scenarios.iter().flat_map(|s| &s.nouls).map(NoulOutcome::correct))
    }

    pub fn score_count(&self) -> usize {
        self.scenarios.iter().map(|s| s.scores.len()).sum()
    }

    /// Fraction of choice questions answered correctly; `None` if there were none.
    pub fn choice_accuracy(&self) -> Option<f64> {
        ratio(self.choice_counts())
    }

    /// Fraction of noul assertions answered correctly; `None` if there were none.
    pub fn noul_accuracy(&self) -> Option<f64> {
        ratio(self.noul_counts())
    }

    /// Root-mean-square error of expected rubric scores; `None` if there were none.
    pub fn score_rmse(&self) -> Option<f32> {
        let n = self.score_count();
        if n == 0 {
            return None;
        }
        let sq: f32 = self
            .scenarios
            .iter()
            .flat_map(|s| &s.scores)
            .map(|o| o.abs_error() * o.abs_error())
            .sum();
        Some((sq / n as f32).sqrt())
    }

    /// Expected calibration error of choice top-1 confidence; `None` if there were no choices.
    pub fn choice_ece(&self) -> Option<f64> {
        expected_calibration_error(
            self.scenarios.iter().flat_map(|s| &s.choices).map(|o| (o.confidence, o.correct())),
            ECE_BINS,
        )
    }

    /// Expected calibration error of the noul verdict's confidence (`max(p, 1 - p)`);
    /// `None` if there were no nouls.
    pub fn noul_ece(&self) -> Option<f64> {
        expected_calibration_error(
            self.scenarios
                .iter()
                .flat_map(|s| &s.nouls)
                .map(|o| (o.probability.max(1.0 - o.probability), o.correct())),
            ECE_BINS,
        )
    }
}

const ECE_BINS: usize = 10;

/// Equal-width-bin ECE over `(confidence, correct)` pairs: the item-weighted mean of
/// |accuracy − mean confidence| per bin.
pub fn expected_calibration_error(outcomes: impl Iterator<Item = (f32, bool)>, bins: usize) -> Option<f64> {
    let mut acc = vec![(0usize, 0.0f64, 0usize); bins];
    let mut total = 0usize;
    for (conf, correct) in outcomes {
        let conf = conf.clamp(0.0, 1.0) as f64;
        let bin = ((conf * bins as f64) as usize).min(bins - 1);
        acc[bin].0 += 1;
        acc[bin].1 += conf;
        acc[bin].2 += correct as usize;
        total += 1;
    }
    (total > 0).then(|| {
        acc.iter()
            .filter(|(n, _, _)| *n > 0)
            .map(|&(n, conf, correct)| (correct as f64 / n as f64 - conf / n as f64).abs() * n as f64)
            .sum::<f64>()
            / total as f64
    })
}

fn count(outcomes: impl Iterator<Item = bool>) -> (usize, usize) {
    outcomes.fold((0, 0), |(c, t), ok| (c + ok as usize, t + 1))
}

fn ratio((correct, total): (usize, usize)) -> Option<f64> {
    (total > 0).then(|| correct as f64 / total as f64)
}

/// Runs every record through the full engine (encode → encoder → decision model) and
/// compares the verdicts with the labels.
pub fn evaluate_dataset<B: Backend>(
    engine: &ReflexEngine<B>,
    dataset: &JevDataset,
    device: &B::Device,
) -> Result<EvalReport, Box<dyn std::error::Error>> {
    let mut scenarios = Vec::with_capacity(dataset.len());

    for record in &dataset.records {
        record.validate()?;
        let verdict = engine.evaluate(&record.to_request(), device)?;

        let choices = verdict
            .choices
            .iter()
            .zip(&record.choice_questions)
            .map(|(v, q)| ChoiceOutcome {
                target: q.target,
                predicted: v.selected_candidate,
                confidence: v.probabilities[v.selected_candidate],
            })
            .collect();
        let nouls = verdict
            .nouls
            .iter()
            .zip(&record.noul_queries)
            .map(|(v, n)| NoulOutcome {
                target: n.target >= 0.5,
                predicted: v.is_true,
                probability: v.probability,
            })
            .collect();
        let scores = verdict
            .scores
            .iter()
            .zip(&record.score_rubrics)
            .map(|(v, s)| ScoreOutcome {
                target: s.target,
                predicted: v.expected_score,
            })
            .collect();

        scenarios.push(ScenarioEval {
            id: record.id.clone(),
            domain: record.domain.clone(),
            choices,
            nouls,
            scores,
            verdict,
        });
    }

    Ok(EvalReport { scenarios })
}

/// Scores of label-free predictors on an evaluation set; a model that does not beat these
/// has learned nothing usable.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Baselines {
    /// Expected accuracy of picking a candidate uniformly at random.
    pub choice_uniform: Option<f64>,
    /// Always picking the longest candidate (by characters): catches a length shortcut.
    pub choice_longest: Option<f64>,
    /// The reference set's majority noul label.
    pub noul_majority_label: bool,
    /// Accuracy of always answering `noul_majority_label`.
    pub noul_majority: Option<f64>,
    /// Mean score target of the reference set.
    pub score_reference_mean: f32,
    /// RMSE of always predicting `score_reference_mean`.
    pub score_mean_rmse: Option<f32>,
}

impl Baselines {
    /// Label statistics come from `reference` (normally the training set) and are scored on
    /// `eval`, so the baselines see no evaluation labels.
    pub fn compute(reference: &JevDataset, eval: &JevDataset) -> Self {
        let ref_nouls: Vec<bool> = reference
            .records
            .iter()
            .flat_map(|r| r.noul_queries.iter().map(|n| n.target >= 0.5))
            .collect();
        let noul_majority_label = ref_nouls.iter().filter(|&&t| t).count() * 2 >= ref_nouls.len();
        let ref_scores: Vec<f32> = reference
            .records
            .iter()
            .flat_map(|r| r.score_rubrics.iter().map(|s| s.target))
            .collect();
        let score_reference_mean = if ref_scores.is_empty() {
            0.0
        } else {
            ref_scores.iter().sum::<f32>() / ref_scores.len() as f32
        };

        let questions: Vec<_> = eval.records.iter().flat_map(|r| &r.choice_questions).collect();
        let mean = |values: Vec<f64>| (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64);
        let choice_uniform = mean(questions.iter().map(|q| 1.0 / q.candidates.len() as f64).collect());
        let choice_longest = mean(
            questions
                .iter()
                .map(|q| {
                    let longest = (0..q.candidates.len())
                        .max_by_key(|&i| q.candidates[i].chars().count())
                        .unwrap_or(0);
                    (longest == q.target) as u8 as f64
                })
                .collect(),
        );
        let noul_majority = mean(
            eval.records
                .iter()
                .flat_map(|r| &r.noul_queries)
                .map(|n| ((n.target >= 0.5) == noul_majority_label) as u8 as f64)
                .collect(),
        );
        let score_mean_rmse = mean(
            eval.records
                .iter()
                .flat_map(|r| &r.score_rubrics)
                .map(|s| ((s.target - score_reference_mean) as f64).powi(2))
                .collect(),
        )
        .map(|mse| mse.sqrt() as f32);

        Self {
            choice_uniform,
            choice_longest,
            noul_majority_label,
            noul_majority,
            score_reference_mean,
            score_mean_rmse,
        }
    }
}
