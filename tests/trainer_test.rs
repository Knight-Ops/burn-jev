//! `train_heads` mini-batching. Kept in its own test binary because the backend RNG is
//! process-global: seeded runs must not interleave with unrelated tests that draw from it.

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
use burn_flex::{Flex, FlexDevice};
use burn_mamba::{
    train_heads, CachedScenario, JointLossBreakdown, MultiQuestionTargets, TrainConfig,
    UnifiedHeadsConfig,
};

type DiffBackend = burn::backend::Autodiff<Flex<f32, i32>>;

const TRAIN_D_MODEL: usize = 16;

/// The backend RNG is global, so seeded runs must not interleave with other tests.
static SEEDED_RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// 8 synthetic scenarios (5 benign, 3 adversarial), each with random CLS state and one
/// 3-candidate choice question.
fn synthetic_scenarios(device: &FlexDevice) -> Vec<CachedScenario<DiffBackend>> {
    DiffBackend::seed(device, 7);
    (0..8)
        .map(|i| CachedScenario {
            id: format!("s{i}"),
            cls_state: Tensor::random([1, TRAIN_D_MODEL], Distribution::Normal(0.0, 1.0), device),
            choice_questions: vec![Tensor::random([3, TRAIN_D_MODEL], Distribution::Normal(0.0, 1.0), device)],
            noul_states: None,
            score_states: None,
            targets: MultiQuestionTargets {
                choice_targets: vec![i % 3],
                noul_targets: vec![],
                score_targets: vec![],
            },
            is_benign: i % 3 != 2,
        })
        .collect()
}

fn run_train_heads(batch_size: usize, seed: u64) -> Vec<JointLossBreakdown> {
    let _guard = SEEDED_RNG.lock().unwrap_or_else(|e| e.into_inner());
    let device = FlexDevice;
    let scenarios = synthetic_scenarios(&device);
    let heads_config = UnifiedHeadsConfig::new(TRAIN_D_MODEL).with_knn_dim(8);
    let config = TrainConfig {
        epochs: 30,
        batch_size,
        seed: Some(seed),
        ..TrainConfig::default()
    };
    train_heads(&scenarios, &heads_config, &config, &device, |_, _| {}).history
}

#[test]
fn test_train_heads_minibatch_trains_metric_head() {
    let history = run_train_heads(4, 0);
    let first = history.first().unwrap().metric_loss;
    let last = history.last().unwrap().metric_loss;
    assert!(first > 0.0, "metric loss must be non-zero with mixed batches of 4, got {first}");
    assert!(last < first, "metric loss should fall during training: first {first}, last {last}");
}

#[test]
fn test_train_heads_batch_size_one_has_zero_metric_loss() {
    let history = run_train_heads(1, 0);
    assert!(
        history.iter().all(|b| b.metric_loss == 0.0),
        "single-scenario batches have no contrastive peers, so metric loss must be 0"
    );
}

#[test]
fn test_train_heads_is_deterministic_for_a_seed() {
    let a = run_train_heads(4, 3);
    let b = run_train_heads(4, 3);
    let key = |h: &[JointLossBreakdown]| {
        h.iter().map(|b| (b.total_loss, b.metric_loss, b.choice_loss)).collect::<Vec<_>>()
    };
    assert_eq!(key(&a), key(&b));
}
