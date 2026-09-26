//! Decision-model (reader + heads) training on Burn's native supervised trainer.
//!
//! Cached encoder features are served by a Burn [`DataLoader`]; [`DecisionModel`] implements
//! [`TrainStep`]/[`InferenceStep`], and [`SupervisedTraining`] drives the loop: TUI (or CLI
//! when stdout is not a terminal), AdamW with linear warmup × cosine decay, gradient-norm
//! clipping, per-epoch validation, early stopping and best-epoch checkpointing, all keyed on
//! the validation selection loss (weighted choice + noul + score).

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use burn::data::dataloader::{batcher::Batcher, DataLoaderBuilder};
use burn::data::dataset::InMemDataset;
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::composed::ComposedLrSchedulerConfig;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::module::Module;
use burn::optim::AdamWConfig;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::{
    backend::{AutodiffBackend, Backend},
    Int, Tensor, TensorData, Transaction,
};
use burn::train::checkpoint::{CheckpointingAction, CheckpointingStrategy};
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::store::{Aggregate, EventStoreClient, Split};
use burn::train::metric::{
    Adaptor, ItemLazy, LearningRateMetric, Metric, MetricAttributes, MetricMetadata, MetricName,
    Numeric, NumericAttributes, NumericEntry, SerializedEntry,
};
use burn::train::{
    EarlyStoppingStrategy, InferenceStep, Learner, SupervisedTraining, TrainOutput as StepOutput, TrainStep,
};

use crate::cache::CachedScenario;
use crate::model::{DecisionModel, DecisionModelConfig, FeatureBatch, ScenarioFeatures};
use crate::training::calibration::brier_calibration_loss;
use crate::training::joint::JointLossConfig;
use crate::training::metric::benign_adversarial_metric_loss;
use crate::training::tasks::{choice_cross_entropy_loss, noul_bce_loss, ordinal_score_loss};
use crate::training::temperature::fit_temperatures;

/// Hyperparameters for [`train_decision_model`].
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub epochs: usize,
    /// Peak learning rate, reached after warmup.
    pub lr: f64,
    pub weight_decay: f32,
    /// Fraction of all iterations spent in linear warmup.
    pub warmup_frac: f64,
    /// Cosine decay floor as a fraction of `lr`.
    pub min_lr_frac: f64,
    pub grad_clip_norm: Option<f32>,
    /// Scenarios per optimizer step. The contrastive metric loss is only non-zero for
    /// batches with at least two benign scenarios and one adversarial one.
    pub batch_size: usize,
    /// Stop after this many epochs without validation improvement.
    pub patience: usize,
    pub loss_weights: JointLossConfig,
    /// Seeds the backend RNG (init, dropout) and the data-loader shuffle.
    pub seed: u64,
    /// Where the trainer writes logs, metrics and checkpoints for this run.
    pub run_dir: PathBuf,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 60,
            lr: 3e-4,
            weight_decay: 1e-2,
            warmup_frac: 0.05,
            min_lr_frac: 0.1,
            grad_clip_norm: Some(1.0),
            batch_size: 16,
            patience: 8,
            loss_weights: JointLossConfig::frozen_backbone(),
            seed: 42,
            run_dir: PathBuf::from("data/.runs/latest"),
        }
    }
}

pub struct TrainOutput<B: AutodiffBackend> {
    /// Weights restored from `best_epoch`, with calibration temperatures fit on validation.
    pub model: DecisionModel<B::InnerBackend>,
    pub best_epoch: usize,
    /// `[choice, noul, score]` temperatures set on `model`.
    pub temperatures: [f32; 3],
}

/// Trains a freshly initialized decision model; validation drives early stopping, the
/// choice of which epoch's weights are returned, and the post-hoc temperature fit.
pub fn train_decision_model<B: AutodiffBackend>(
    train: Vec<CachedScenario>,
    val: Vec<CachedScenario>,
    model_config: &DecisionModelConfig,
    config: &TrainConfig,
    device: &B::Device,
) -> Result<TrainOutput<B>, Box<dyn std::error::Error>> {
    if train.is_empty() || val.is_empty() {
        return Err("training needs non-empty train and validation sets".into());
    }
    let run_dir = &config.run_dir;
    let checkpoint_dir = run_dir.join("checkpoint");
    if checkpoint_dir.exists() && std::fs::read_dir(&checkpoint_dir)?.next().is_some() {
        return Err(format!("{} already holds checkpoints; use a fresh run directory", checkpoint_dir.display()).into());
    }

    B::seed(device, config.seed);
    let batch_size = config.batch_size.max(1);
    let steps_per_epoch = train.len().div_ceil(batch_size);
    let total_iters = (steps_per_epoch * config.epochs).max(1);
    let warmup_iters = ((total_iters as f64 * config.warmup_frac).ceil() as usize).max(1);

    let train_loader = DataLoaderBuilder::new(DecisionBatcher)
        .batch_size(batch_size)
        .shuffle(config.seed)
        .set_device(device.clone())
        .build(InMemDataset::new(train));
    // Shared so the temperature fit can reuse the validation features without a copy.
    let val: Vec<Arc<CachedScenario>> = val.into_iter().map(Arc::new).collect();
    let valid_loader = DataLoaderBuilder::<B::InnerBackend, _, _>::new(DecisionBatcher)
        .batch_size(batch_size)
        .set_device(device.clone())
        .build(InMemDataset::new(val.clone()));

    let model = model_config.init::<B>(device).with_loss_weights(config.loss_weights.clone());
    let optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .with_grad_clipping(config.grad_clip_norm.map(GradientClippingConfig::Norm))
        .init();
    // Prod reduction: (warmup ramp 1/w → 1) × (cosine lr → lr * min_lr_frac).
    let lr_scheduler = ComposedLrSchedulerConfig::new()
        .linear(LinearLrSchedulerConfig::new(1.0 / warmup_iters as f64, 1.0, warmup_iters))
        .cosine(CosineAnnealingLrSchedulerConfig::new(config.lr, total_iters).with_min_lr(config.lr * config.min_lr_frac))
        .init()?;

    // Burn's metric strategies query the event store right after validation, but metrics
    // reach it through an async processor, so the current epoch is often not there yet
    // (MetricCheckpointingStrategy then never saves an improving epoch). The strategies
    // below tolerate that lag, and the final pick uses a tracker that is complete once
    // training has returned.
    let tracker = EpochTracker::default();
    let selection_name = MetricKind::SelectionLoss.name();
    let mut training = SupervisedTraining::new(run_dir, train_loader, valid_loader)
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(recorder())
        .with_checkpointing_strategy(LagTolerantCheckpointing::new(selection_name))
        .early_stopping(LagTolerantEarlyStopping::new(selection_name, config.patience))
        .num_epochs(config.epochs)
        .summary();
    for kind in MetricKind::ALL {
        let valid = DecisionMetric::new(kind);
        let valid = if kind == MetricKind::SelectionLoss { valid.with_tracker(tracker.clone()) } else { valid };
        training = training.metric_train_numeric(DecisionMetric::new(kind)).metric_valid_numeric(valid);
    }

    let result = training.launch(Learner::new(model, optim, lr_scheduler));
    // Drop the renderer so the TUI releases the terminal (and prints the summary).
    drop(result.renderer);

    let best_epoch = best_checkpoint_epoch(&checkpoint_dir, &tracker)?;
    let record = recorder().load(checkpoint_dir.join(format!("model-{best_epoch}")), device)?;
    let mut model = model_config
        .init::<B::InnerBackend>(device)
        .load_record(record)
        .with_loss_weights(config.loss_weights.clone());

    // Validation already picked the epoch; reusing it for calibration is acceptable because
    // the fit is only three scalars over the whole validation set.
    let temperatures = fit_temperatures(&model, &val, batch_size, device);
    let [tc, tn, ts] = temperatures;
    model.heads.set_temperatures(tc, tn, ts);
    eprintln!("[calibrate] T choice={tc:.3} noul={tn:.3} score={ts:.3}");
    Ok(TrainOutput {
        model,
        best_epoch,
        temperatures,
    })
}

fn recorder() -> NamedMpkFileRecorder<FullPrecisionSettings> {
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
}

/// The saved checkpoint with the lowest validation selection loss.
fn best_checkpoint_epoch(dir: &Path, tracker: &EpochTracker) -> Result<usize, Box<dyn std::error::Error>> {
    let saved: Vec<usize> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            name.strip_prefix("model-")?.strip_suffix(".mpk")?.parse().ok()
        })
        .collect();
    let means = tracker.means();
    let global_best = argmin(means.iter().map(|(&e, &v)| (e, v)));
    let best = argmin(saved.iter().filter_map(|e| means.get(e).map(|&v| (*e, v))))
        .ok_or_else(|| format!("no model checkpoint with validation metrics in {}", dir.display()))?;
    if global_best != Some(best) {
        eprintln!("[train] warning: best epoch {global_best:?} has no checkpoint; using epoch {best}");
    }
    Ok(best)
}

fn argmin(values: impl Iterator<Item = (usize, f64)>) -> Option<usize> {
    values.fold(None, |best: Option<(usize, f64)>, (e, v)| match best {
        Some((_, bv)) if bv <= v => best,
        _ => Some((e, v)),
    })
    .map(|(e, _)| e)
}

/// Per-epoch validation selection loss, filled by the valid [`DecisionMetric`] and shared
/// with the trainer (metrics are cloned into the processor, the map is not).
#[derive(Clone, Default)]
pub struct EpochTracker(Arc<Mutex<BTreeMap<usize, (f64, f64)>>>);

impl EpochTracker {
    fn record(&self, epoch: usize, value: f64, weight: usize) {
        let mut map = self.0.lock().unwrap();
        let entry = map.entry(epoch).or_insert((0.0, 0.0));
        entry.0 += value * weight as f64;
        entry.1 += weight as f64;
    }

    /// Weighted mean per epoch.
    pub fn means(&self) -> BTreeMap<usize, f64> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, (_, w))| *w > 0.0)
            .map(|(&e, &(s, w))| (e, s / w))
            .collect()
    }
}

/// Validation values of `metric` for epochs whose aggregate is final.
///
/// The event store caches the first aggregate it computes per (metric, epoch) and never
/// refreshes it, so querying an epoch whose validation batches are still being processed
/// would pin a partial mean. Validation events are processed in order, so epoch `e` is
/// complete once epoch `e + 1` has any entry; that is probed through a different metric
/// ([`PROBE_METRIC`]) so the decision metric is only ever read for complete epochs.
fn final_epochs(store: &EventStoreClient, metric: &str, upto: usize) -> Vec<(usize, f64)> {
    (1..upto)
        .take_while(|&e| store.find_metric(PROBE_METRIC, e + 1, Aggregate::Mean, &Split::Valid).is_some())
        .filter_map(|e| store.find_metric(metric, e, Aggregate::Mean, &Split::Valid).map(|v| (e, v)))
        .collect()
}

/// Metric used only to detect that an epoch has started reaching the store.
const PROBE_METRIC: &str = "Loss";

/// Saves every epoch, then deletes saved epochs whose metric is final and not the best;
/// epochs whose metrics are missing or may still be partial are kept.
struct LagTolerantCheckpointing {
    metric: String,
    saved: BTreeSet<usize>,
}

impl LagTolerantCheckpointing {
    fn new(metric: &str) -> Self {
        Self {
            metric: metric.to_string(),
            saved: BTreeSet::new(),
        }
    }
}

impl CheckpointingStrategy for LagTolerantCheckpointing {
    fn checkpointing(&mut self, epoch: usize, store: &EventStoreClient) -> Vec<CheckpointingAction> {
        let known = final_epochs(store, &self.metric, epoch);
        let best = argmin(known.iter().copied());
        let known: BTreeSet<usize> = known.into_iter().map(|(e, _)| e).collect();
        let stale: Vec<usize> = self
            .saved
            .iter()
            .copied()
            .filter(|e| known.contains(e) && Some(*e) != best)
            .collect();
        let mut actions: Vec<_> = stale.iter().map(|&e| CheckpointingAction::Delete(e)).collect();
        for e in stale {
            self.saved.remove(&e);
        }
        actions.push(CheckpointingAction::Save);
        self.saved.insert(epoch);
        actions
    }
}

/// Stops once the newest final epoch is `patience` epochs past the best final one (at most
/// a couple of epochs later than an exact rule would, never earlier).
#[derive(Clone)]
struct LagTolerantEarlyStopping {
    metric: String,
    patience: usize,
}

impl LagTolerantEarlyStopping {
    fn new(metric: &str, patience: usize) -> Self {
        Self {
            metric: metric.to_string(),
            patience,
        }
    }
}

impl EarlyStoppingStrategy for LagTolerantEarlyStopping {
    fn should_stop(&mut self, epoch: usize, store: &EventStoreClient) -> bool {
        let known = final_epochs(store, &self.metric, epoch);
        match (argmin(known.iter().copied()), known.last()) {
            (Some(best), Some(&(latest, _))) => latest - best >= self.patience,
            _ => false,
        }
    }
}

// =====================================================================
// Data: cached scenarios → padded batches
// =====================================================================

#[derive(Clone)]
pub struct DecisionBatch<B: Backend> {
    pub features: Arc<FeatureBatch<B>>,
    pub is_benign: Vec<bool>,
    pub choice_targets: Vec<i64>,
    pub noul_targets: Vec<f32>,
    pub score_targets: Vec<f32>,
}

impl<B: Backend> std::fmt::Debug for DecisionBatch<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionBatch")
            .field("scenarios", &self.is_benign.len())
            .field("choice_questions", &self.choice_targets.len())
            .field("noul_queries", &self.noul_targets.len())
            .field("score_rubrics", &self.score_targets.len())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DecisionBatcher;

impl<B: Backend, I: Borrow<CachedScenario>> Batcher<B, I, DecisionBatch<B>> for DecisionBatcher {
    fn batch(&self, items: Vec<I>, device: &B::Device) -> DecisionBatch<B> {
        let items: Vec<&CachedScenario> = items.iter().map(Borrow::borrow).collect();
        let features: Vec<&ScenarioFeatures> = items.iter().map(|s| &s.features).collect();
        DecisionBatch {
            features: Arc::new(FeatureBatch::new(&features, device)),
            is_benign: items.iter().map(|s| s.is_benign).collect(),
            choice_targets: items
                .iter()
                .flat_map(|s| s.targets.choice_targets.iter().map(|&t| t as i64))
                .collect(),
            noul_targets: items.iter().flat_map(|s| s.targets.noul_targets.iter().copied()).collect(),
            score_targets: items.iter().flat_map(|s| s.targets.score_targets.iter().copied()).collect(),
        }
    }
}

// =====================================================================
// Train / valid steps
// =====================================================================

/// Losses and logits of one step, synced lazily into [`DecisionStats`] for the metrics.
pub struct DecisionStepOutput<B: Backend> {
    total: Tensor<B, 1>,
    /// `[choice, noul, score, metric, calibration]`, each `[1]`.
    parts: [Tensor<B, 1>; 5],
    choice_logits: Option<Tensor<B, 2>>,
    choice_counts: Vec<usize>,
    choice_targets: Vec<i64>,
    noul_logits: Option<Tensor<B, 1>>,
    noul_targets: Vec<f32>,
    score_logits: Option<Tensor<B, 2>>,
    score_targets: Vec<f32>,
    weights: [f32; 3],
}

fn step_loss<B: Backend>(model: &DecisionModel<B>, batch: DecisionBatch<B>) -> DecisionStepOutput<B> {
    let device = batch.features.ctx.device();
    let w = &model.loss_weights;
    let fb = &batch.features;
    let out = model.forward(fb);
    let zero = || Tensor::<B, 1>::zeros([1], &device);

    let metric_loss = benign_adversarial_metric_loss(out.embedding, &batch.is_benign, w.metric_temperature, &device);

    let choice_loss = match (out.choice_logits.clone(), fb.choice.as_ref()) {
        (Some(logits), Some(rows)) => {
            assert_eq!(batch.choice_targets.len(), rows.questions, "choice targets do not match questions");
            let targets = Tensor::<B, 1, Int>::from_data(
                TensorData::new(batch.choice_targets.clone(), [rows.questions]),
                &device,
            );
            choice_cross_entropy_loss(logits, targets, rows.mask.clone(), None, &device)
        }
        _ => zero(),
    };

    let (noul_loss, calibration_loss) = match out.noul_logits.clone() {
        Some(logits) => {
            let n = batch.noul_targets.len();
            assert_eq!(n, logits.dims()[0], "noul targets do not match queries");
            let targets = Tensor::<B, 1>::from_data(TensorData::new(batch.noul_targets.clone(), [n]), &device);
            (
                noul_bce_loss(logits.clone(), targets.clone(), None, &device),
                brier_calibration_loss(logits, targets, None, &device),
            )
        }
        None => (zero(), zero()),
    };

    let score_loss = match out.score_logits.clone() {
        Some(logits) => {
            let n = batch.score_targets.len();
            assert_eq!(n, logits.dims()[0], "score targets do not match rubrics");
            let targets = Tensor::<B, 1>::from_data(TensorData::new(batch.score_targets.clone(), [n]), &device);
            ordinal_score_loss(logits, targets, model.heads.num_rubric_bins, None, &device)
        }
        None => zero(),
    };

    let total = metric_loss.clone() * w.lambda_metric
        + choice_loss.clone() * w.lambda_choice
        + noul_loss.clone() * w.lambda_noul
        + score_loss.clone() * w.lambda_score
        + calibration_loss.clone() * w.lambda_calibration;

    DecisionStepOutput {
        total,
        parts: [choice_loss, noul_loss, score_loss, metric_loss, calibration_loss],
        choice_logits: out.choice_logits,
        choice_counts: fb.choice.as_ref().map(|c| c.counts.clone()).unwrap_or_default(),
        choice_targets: batch.choice_targets,
        noul_logits: out.noul_logits,
        noul_targets: batch.noul_targets,
        score_logits: out.score_logits,
        score_targets: batch.score_targets,
        weights: [w.lambda_choice, w.lambda_noul, w.lambda_score],
    }
}

impl<B: AutodiffBackend> TrainStep for DecisionModel<B> {
    type Input = DecisionBatch<B>;
    type Output = DecisionStepOutput<B>;

    fn step(&self, batch: DecisionBatch<B>) -> StepOutput<DecisionStepOutput<B>> {
        let out = step_loss(self, batch);
        let grads = out.total.backward();
        StepOutput::new(self, grads, out)
    }
}

impl<B: Backend> InferenceStep for DecisionModel<B> {
    type Input = DecisionBatch<B>;
    type Output = DecisionStepOutput<B>;

    fn step(&self, batch: DecisionBatch<B>) -> DecisionStepOutput<B> {
        step_loss(self, batch)
    }
}

// =====================================================================
// Metrics
// =====================================================================

/// Host-side summary of one step: losses plus the counts behind accuracy metrics.
#[derive(Clone, Debug, Default)]
pub struct DecisionStats {
    pub total_loss: f64,
    pub choice_loss: f64,
    pub noul_loss: f64,
    pub score_loss: f64,
    pub metric_loss: f64,
    pub calibration_loss: f64,
    /// Weighted choice + noul + score loss: the early-stopping/checkpoint criterion. The
    /// metric and calibration terms are left out because they depend on batch composition.
    pub selection_loss: f64,
    pub choice_correct: usize,
    pub choice_total: usize,
    pub noul_correct: usize,
    pub noul_total: usize,
    pub score_sq_err: f64,
    pub score_total: usize,
}

impl<B: Backend> ItemLazy for DecisionStepOutput<B> {
    type ItemSync = DecisionStats;

    fn sync(self) -> DecisionStats {
        let mut tx = Transaction::default().register(self.total);
        for part in self.parts {
            tx = tx.register(part);
        }
        let (has_c, has_n, has_s) = (self.choice_logits.is_some(), self.noul_logits.is_some(), self.score_logits.is_some());
        let score_bins = self.score_logits.as_ref().map_or(0, |l| l.dims()[1]);
        let c_max = self.choice_logits.as_ref().map_or(0, |l| l.dims()[1]);
        if let Some(t) = self.choice_logits {
            tx = tx.register(t);
        }
        if let Some(t) = self.noul_logits {
            tx = tx.register(t);
        }
        if let Some(t) = self.score_logits {
            tx = tx.register(t);
        }
        let mut data = tx
            .execute()
            .into_iter()
            .map(|d| d.convert::<f32>().to_vec::<f32>().expect("f32 step output"));
        let mut next = || data.next().expect("registered tensor");

        let scalar = |v: Vec<f32>| v[0] as f64;
        let mut stats = DecisionStats {
            total_loss: scalar(next()),
            choice_loss: scalar(next()),
            noul_loss: scalar(next()),
            score_loss: scalar(next()),
            metric_loss: scalar(next()),
            calibration_loss: scalar(next()),
            ..DecisionStats::default()
        };
        let [wc, wn, ws] = self.weights.map(f64::from);
        stats.selection_loss = wc * stats.choice_loss + wn * stats.noul_loss + ws * stats.score_loss;

        if has_c {
            let logits = next();
            for (q, (&count, &target)) in self.choice_counts.iter().zip(&self.choice_targets).enumerate() {
                let row = &logits[q * c_max..q * c_max + count];
                stats.choice_correct += (argmax(row) == target as usize) as usize;
            }
            stats.choice_total = self.choice_targets.len();
        }
        if has_n {
            for (&z, &t) in next().iter().zip(&self.noul_targets) {
                stats.noul_correct += ((z >= 0.0) == (t >= 0.5)) as usize;
            }
            stats.noul_total = self.noul_targets.len();
        }
        if has_s {
            for (row, &t) in next().chunks_exact(score_bins).zip(&self.score_targets) {
                let expected = 1.0 + row.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).sum::<f32>();
                stats.score_sq_err += ((expected - t) as f64).powi(2);
            }
            stats.score_total = self.score_targets.len();
        }
        stats
    }
}

impl Adaptor<DecisionStats> for DecisionStats {
    fn adapt(&self) -> DecisionStats {
        self.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricKind {
    Loss,
    SelectionLoss,
    ChoiceLoss,
    NoulLoss,
    ScoreLoss,
    ChoiceAccuracy,
    NoulAccuracy,
    ScoreMse,
}

impl MetricKind {
    pub const ALL: [MetricKind; 8] = [
        MetricKind::Loss,
        MetricKind::SelectionLoss,
        MetricKind::ChoiceLoss,
        MetricKind::NoulLoss,
        MetricKind::ScoreLoss,
        MetricKind::ChoiceAccuracy,
        MetricKind::NoulAccuracy,
        MetricKind::ScoreMse,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MetricKind::Loss => "Loss",
            MetricKind::SelectionLoss => "Selection Loss",
            MetricKind::ChoiceLoss => "Choice Loss",
            MetricKind::NoulLoss => "Noul Loss",
            MetricKind::ScoreLoss => "Score Loss",
            MetricKind::ChoiceAccuracy => "Choice Accuracy",
            MetricKind::NoulAccuracy => "Noul Accuracy",
            MetricKind::ScoreMse => "Score MSE",
        }
    }

    /// `(value, weight)` for one step; the epoch value is the weighted mean.
    fn read(self, s: &DecisionStats) -> (f64, usize) {
        let pct = |c: usize, t: usize| if t == 0 { 0.0 } else { 100.0 * c as f64 / t as f64 };
        match self {
            MetricKind::Loss => (s.total_loss, 1),
            MetricKind::SelectionLoss => (s.selection_loss, 1),
            MetricKind::ChoiceLoss => (s.choice_loss, (s.choice_total > 0) as usize),
            MetricKind::NoulLoss => (s.noul_loss, (s.noul_total > 0) as usize),
            MetricKind::ScoreLoss => (s.score_loss, (s.score_total > 0) as usize),
            MetricKind::ChoiceAccuracy => (pct(s.choice_correct, s.choice_total), s.choice_total),
            MetricKind::NoulAccuracy => (pct(s.noul_correct, s.noul_total), s.noul_total),
            MetricKind::ScoreMse => (
                if s.score_total == 0 { 0.0 } else { s.score_sq_err / s.score_total as f64 },
                s.score_total,
            ),
        }
    }

    fn higher_is_better(self) -> bool {
        matches!(self, MetricKind::ChoiceAccuracy | MetricKind::NoulAccuracy)
    }
}

/// One numeric metric over [`DecisionStats`], weighted by item count so epoch values are
/// exact (accuracy over all questions, MSE over all rubrics).
#[derive(Clone)]
pub struct DecisionMetric {
    kind: MetricKind,
    name: MetricName,
    state: NumericMetricState,
    tracker: Option<EpochTracker>,
}

impl DecisionMetric {
    pub fn new(kind: MetricKind) -> Self {
        Self {
            kind,
            name: Arc::new(kind.name().to_string()),
            state: NumericMetricState::default(),
            tracker: None,
        }
    }

    /// Also records per-epoch values into `tracker`.
    pub fn with_tracker(mut self, tracker: EpochTracker) -> Self {
        self.tracker = Some(tracker);
        self
    }
}

impl Metric for DecisionMetric {
    type Input = DecisionStats;

    fn update(&mut self, stats: &DecisionStats, metadata: &MetricMetadata) -> SerializedEntry {
        let (value, weight) = self.kind.read(stats);
        if let Some(ref tracker) = self.tracker {
            tracker.record(metadata.global_progress.items_processed, value, weight);
        }
        let mut format = FormatOptions::new(self.name()).precision(if self.kind.higher_is_better() { 1 } else { 4 });
        if self.kind.higher_is_better() {
            format = format.unit("%");
        }
        self.state.update(value, weight, format)
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: self.kind.higher_is_better().then(|| "%".to_string()),
            higher_is_better: self.kind.higher_is_better(),
        }
        .into()
    }
}

impl Numeric for DecisionMetric {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
        .0
}
