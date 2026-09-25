//! Head-only training over precomputed backbone features.

use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{backend::AutodiffBackend, Bool, Int, Tensor, TensorData};

use crate::cache::CachedScenario;
use crate::model::{UnifiedHeads, UnifiedHeadsConfig};
use crate::training::calibration::brier_calibration_loss;
use crate::training::joint::{JointLossBreakdown, JointLossConfig};
use crate::training::metric::benign_adversarial_metric_loss;
use crate::training::tasks::{choice_cross_entropy_loss, noul_bce_loss, ordinal_score_loss};

/// Hyperparameters for [`train_heads`].
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub epochs: usize,
    pub lr: f64,
    /// Scenarios per optimizer step; `0` is treated as `1`.
    ///
    /// The contrastive metric loss is only non-zero for batches with at least two benign
    /// scenarios and one adversarial one, so `1` leaves the k-NN embedding head untrained.
    pub batch_size: usize,
    pub loss_weights: JointLossConfig,
    /// Seeds the backend RNG before head initialization and the per-epoch shuffle, making
    /// runs reproducible.
    pub seed: Option<u64>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            lr: 0.02,
            batch_size: 4,
            loss_weights: JointLossConfig::frozen_backbone(),
            seed: None,
        }
    }
}

/// Trained heads plus the per-epoch mean loss breakdown.
pub struct TrainOutput<B: AutodiffBackend> {
    pub heads: UnifiedHeads<B::InnerBackend>,
    pub history: Vec<JointLossBreakdown>,
}

/// Trains freshly initialized heads with Adam, one optimizer step per mini-batch of
/// `config.batch_size` scenarios.
///
/// Scenario order is reshuffled every epoch (seeded from `config.seed`, or `0`) so the
/// contrastive pairs change between epochs. With `batch_size == 1` the order is left fixed,
/// since shuffling would not change batch membership.
///
/// `on_epoch(epoch, breakdown)` is called after each epoch (1-based) with the epoch's mean
/// losses over its batches, so callers can report progress; the same values are returned in
/// `history`.
pub fn train_heads<B: AutodiffBackend>(
    scenarios: &[CachedScenario<B>],
    heads_config: &UnifiedHeadsConfig,
    config: &TrainConfig,
    device: &B::Device,
    mut on_epoch: impl FnMut(usize, &JointLossBreakdown),
) -> TrainOutput<B> {
    if let Some(seed) = config.seed {
        B::seed(device, seed);
    }
    let mut heads: UnifiedHeads<B> = heads_config.init(device);
    let mut optim = AdamConfig::new().init();
    let weights = &config.loss_weights;
    let batch_size = config.batch_size.max(1);
    let mut rng = SplitMix64(config.seed.unwrap_or(0));
    let mut order: Vec<usize> = (0..scenarios.len()).collect();
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 1..=config.epochs {
        if batch_size > 1 {
            rng.shuffle(&mut order);
        }
        let mut sum = JointLossBreakdown::default();
        let mut steps = 0usize;

        for chunk in order.chunks(batch_size) {
            let batch: Vec<&CachedScenario<B>> = chunk.iter().map(|&i| &scenarios[i]).collect();
            let (total, breakdown) = batch_loss(&heads, &batch, weights, device);
            accumulate(&mut sum, &breakdown);
            steps += 1;

            let grads = GradientsParams::from_grads(total.backward(), &heads);
            heads = optim.step(config.lr, heads, grads);
        }

        let mean = mean_of(&sum, steps.max(1) as f32);
        on_epoch(epoch, &mean);
        history.push(mean);
    }

    TrainOutput {
        heads: heads.valid(),
        history,
    }
}

/// Unweighted per-task losses for one scenario; `None` when the scenario has no such task.
struct TaskLosses<B: AutodiffBackend> {
    choice: Option<Tensor<B, 1>>,
    noul: Option<Tensor<B, 1>>,
    calibration: Option<Tensor<B, 1>>,
    score: Option<Tensor<B, 1>>,
}

/// Weighted multi-task loss for a mini-batch.
///
/// The metric loss is computed once over the whole batch; each task loss is averaged over
/// the scenarios that have that task, so scenarios without it do not dilute it with zeros.
fn batch_loss<B: AutodiffBackend>(
    heads: &UnifiedHeads<B>,
    batch: &[&CachedScenario<B>],
    w: &JointLossConfig,
    device: &B::Device,
) -> (Tensor<B, 1>, JointLossBreakdown) {
    let zero = || Tensor::<B, 1>::zeros([1], device);

    let cls = Tensor::cat(batch.iter().map(|item| item.cls_state.clone()).collect(), 0);
    let is_benign: Vec<bool> = batch.iter().map(|item| item.is_benign).collect();
    let embedding = heads.extract_knn_embedding(cls);
    let metric_loss =
        benign_adversarial_metric_loss(embedding, &is_benign, w.metric_temperature, device);

    let (mut choice, mut noul, mut calibration, mut score) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for item in batch {
        let t = task_losses(heads, item, device);
        choice.extend(t.choice);
        noul.extend(t.noul);
        calibration.extend(t.calibration);
        score.extend(t.score);
    }
    let mean = |losses: Vec<Tensor<B, 1>>| {
        let n = losses.len();
        losses.into_iter().reduce(|a, b| a + b).map_or_else(zero, |sum| sum / n as f32)
    };
    let choice_loss = mean(choice);
    let noul_loss = mean(noul);
    let calibration_loss = mean(calibration);
    let score_loss = mean(score);

    let total = metric_loss.clone() * w.lambda_metric
        + choice_loss.clone() * w.lambda_choice
        + noul_loss.clone() * w.lambda_noul
        + score_loss.clone() * w.lambda_score
        + calibration_loss.clone() * w.lambda_calibration;

    let breakdown = JointLossBreakdown {
        total_loss: scalar(&total),
        metric_loss: scalar(&metric_loss),
        choice_loss: scalar(&choice_loss),
        noul_loss: scalar(&noul_loss),
        score_loss: scalar(&score_loss),
        calibration_loss: scalar(&calibration_loss),
    };
    (total, breakdown)
}

/// Choice, noul, calibration and score losses for a single scenario.
fn task_losses<B: AutodiffBackend>(
    heads: &UnifiedHeads<B>,
    item: &CachedScenario<B>,
    device: &B::Device,
) -> TaskLosses<B> {
    let choice = (!item.choice_questions.is_empty()).then(|| {
        let mut choice_loss = Tensor::<B, 1>::zeros([1], device);
        for (cands, &target) in item.choice_questions.iter().zip(&item.targets.choice_targets) {
            let k = cands.dims()[0];
            let logits = heads.forward_choice_logits(cands.clone()).reshape([1, k]);
            let targets = Tensor::<B, 1, Int>::from_data([target as i64], device);
            let mask = Tensor::<B, 2, Bool>::from_data(TensorData::new(vec![true; k], [1, k]), device);
            choice_loss = choice_loss + choice_cross_entropy_loss(logits, targets, mask, None, device);
        }
        choice_loss / item.choice_questions.len() as f32
    });

    let (noul, calibration) = match item.noul_states {
        Some(ref states) => {
            let n = item.targets.noul_targets.len().min(states.dims()[0]);
            let logits = heads.forward_noul_logits(states.clone().slice([0..n]));
            let targets = Tensor::<B, 1>::from_data(
                TensorData::new(item.targets.noul_targets[..n].to_vec(), [n]),
                device,
            );
            (
                Some(noul_bce_loss(logits.clone(), targets.clone(), None, device)),
                Some(brier_calibration_loss(logits, targets, None, device)),
            )
        }
        None => (None, None),
    };

    let score = item.score_states.as_ref().map(|states| {
        let n = item.targets.score_targets.len().min(states.dims()[0]);
        let logits = heads.forward_score_logits(states.clone().slice([0..n]));
        let targets = Tensor::<B, 1>::from_data(
            TensorData::new(item.targets.score_targets[..n].to_vec(), [n]),
            device,
        );
        ordinal_score_loss(logits, targets, heads.num_rubric_bins, None, device)
    });

    TaskLosses {
        choice,
        noul,
        calibration,
        score,
    }
}

/// Minimal SplitMix64 PRNG for reproducible epoch shuffles without an RNG dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Fisher–Yates shuffle.
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

fn scalar<B: AutodiffBackend>(t: &Tensor<B, 1>) -> f32 {
    t.clone().into_data().convert::<f32>().as_slice::<f32>().unwrap()[0]
}

fn accumulate(sum: &mut JointLossBreakdown, b: &JointLossBreakdown) {
    sum.total_loss += b.total_loss;
    sum.metric_loss += b.metric_loss;
    sum.choice_loss += b.choice_loss;
    sum.noul_loss += b.noul_loss;
    sum.score_loss += b.score_loss;
    sum.calibration_loss += b.calibration_loss;
}

fn mean_of(sum: &JointLossBreakdown, n: f32) -> JointLossBreakdown {
    JointLossBreakdown {
        total_loss: sum.total_loss / n,
        metric_loss: sum.metric_loss / n,
        choice_loss: sum.choice_loss / n,
        noul_loss: sum.noul_loss / n,
        score_loss: sum.score_loss / n,
        calibration_loss: sum.calibration_loss / n,
    }
}
