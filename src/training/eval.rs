//! Evaluation of a [`ReflexEngine`] against a labeled dataset.

use burn::tensor::backend::Backend;
use serde::Serialize;
use tokenizers::Tokenizer;

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
}

fn count(outcomes: impl Iterator<Item = bool>) -> (usize, usize) {
    outcomes.fold((0, 0), |(c, t), ok| (c + ok as usize, t + 1))
}

fn ratio((correct, total): (usize, usize)) -> Option<f64> {
    (total > 0).then(|| correct as f64 / total as f64)
}

/// Runs every record through the engine and compares the verdicts with the labels.
///
/// Records are tokenized with the engine's own delimiter configuration.
pub fn evaluate_dataset<B: Backend>(
    engine: &ReflexEngine<B>,
    dataset: &JevDataset,
    tokenizer: &Tokenizer,
    device: &B::Device,
) -> Result<EvalReport, Box<dyn std::error::Error>> {
    let delimiters = engine.resolver.config();
    let mut scenarios = Vec::with_capacity(dataset.len());

    for record in &dataset.records {
        let tokenized = record.encode(tokenizer, delimiters)?;
        let verdict = engine.evaluate(&tokenized.token_ids, device)?;

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
