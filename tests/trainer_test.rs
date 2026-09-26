//! The native-trainer path end to end on a synthetic task that needs cross-attention: each
//! context hides a "signal" token; true assertions and correct candidates are noisy copies
//! of it, distractors are random. Items can only be answered by consulting the context.

use burn::backend::Autodiff;
use burn_flex::{Flex, FlexDevice};
use burn_mamba::{
    train_decision_model, CachedScenario, DecisionModelConfig, ItemKind, ItemReaderConfig, MultiQuestionTargets,
    ScenarioFeatures, TrainConfig, UnifiedHeadsConfig,
};

type B = Flex<f32, i32>;
const D: usize = 16;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, scale: f32) -> Vec<f32> {
        (0..D).map(|_| self.next() * scale).collect()
    }
}

fn scenario(rng: &mut Lcg, i: usize) -> CachedScenario {
    let signal = rng.vec(1.5);
    let near = |rng: &mut Lcg| signal.iter().zip(rng.vec(0.3)).map(|(s, n)| s + n).collect::<Vec<_>>();
    let ctx_len = 6;
    let hidden = (rng.next().abs() * ctx_len as f32) as usize % ctx_len;
    let mut ctx = Vec::new();
    for t in 0..ctx_len {
        ctx.extend(if t == hidden { signal.clone() } else { rng.vec(1.5) });
    }

    let target = (rng.next().abs() * 3.0) as usize % 3;
    let mut items = Vec::new();
    let mut kinds = Vec::new();
    for c in 0..3 {
        items.extend(if c == target { near(rng) } else { rng.vec(1.5) });
        kinds.push(ItemKind::Choice { question: 0, candidate: c });
    }
    let truth = i % 2 == 0;
    items.extend(if truth { near(rng) } else { rng.vec(1.5) });
    kinds.push(ItemKind::Noul { index: 0 });
    items.extend(if truth { rng.vec(1.5) } else { near(rng) });
    kinds.push(ItemKind::Noul { index: 1 });
    items.extend(rng.vec(1.5));
    kinds.push(ItemKind::Score { index: 0 });

    CachedScenario {
        id: format!("syn_{i}"),
        is_benign: i % 3 != 0,
        targets: MultiQuestionTargets::new()
            .with_choice_target(target)
            .with_noul_target(truth as u8 as f32)
            .with_noul_target(!truth as u8 as f32)
            .with_score_target(3.0),
        features: ScenarioFeatures { d_model: D, ctx, ctx_len, items, kinds },
    }
}

#[test]
fn native_trainer_learns_a_cross_attention_task_and_restores_the_best_epoch() {
    let device = FlexDevice;
    let mut rng = Lcg(7);
    let train: Vec<_> = (0..1536).map(|i| scenario(&mut rng, i)).collect();
    let val: Vec<_> = (0..128).map(|i| scenario(&mut rng, 1000 + i)).collect();

    let reader = ItemReaderConfig::new(D).with_d_reader(32).with_n_heads(4).with_dropout(0.0);
    let model_config = DecisionModelConfig::with_reader(reader, UnifiedHeadsConfig::new(32).with_dropout(0.0));
    let run_dir = std::env::temp_dir().join(format!("burn_mamba_trainer_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&run_dir);
    let config = TrainConfig {
        epochs: 30,
        lr: 3e-3,
        batch_size: 32,
        patience: 6,
        seed: 1,
        run_dir: run_dir.clone(),
        ..TrainConfig::default()
    };

    let out = train_decision_model::<Autodiff<B>>(train, val.clone(), &model_config, &config, &device).unwrap();
    assert!((1..=30).contains(&out.best_epoch));
    assert!(run_dir.join("checkpoint").join(format!("model-{}.mpk", out.best_epoch)).exists());

    let (mut choice_ok, mut noul_ok, mut noul_n) = (0, 0, 0);
    for s in &val {
        let d = out.model.decide(&s.features, &device).unwrap();
        choice_ok += (d.choices[0].selected_candidate == s.targets.choice_targets[0]) as usize;
        for (v, &t) in d.nouls.iter().zip(&s.targets.noul_targets) {
            noul_ok += (v.is_true == (t >= 0.5)) as usize;
            noul_n += 1;
        }
    }
    let choice_acc = choice_ok as f64 / val.len() as f64;
    let noul_acc = noul_ok as f64 / noul_n as f64;
    eprintln!("best epoch {}: val choice {choice_acc:.3}, noul {noul_acc:.3}", out.best_epoch);
    assert!(choice_acc > 0.8, "choice accuracy {choice_acc} (chance 0.33)");
    assert!(noul_acc > 0.8, "noul accuracy {noul_acc} (chance 0.5)");
    let _ = std::fs::remove_dir_all(&run_dir);
}
