mod common;

use burn::backend::Autodiff;
use burn::tensor::{Bool, Int, Tensor, TensorData};
use burn_flex::FlexDevice;
use burn_jev::{DecisionModelConfig, FeatureBatch, ItemKind, ScenarioFeatures};
use common::{tiny_encoder_config, TestBackend, PAD};

const D: usize = 16;

fn features(ctx_len: usize, seed: f32, kinds: Vec<ItemKind>) -> ScenarioFeatures {
    let n = kinds.len();
    ScenarioFeatures {
        d_model: D,
        ctx: (0..ctx_len * D).map(|i| (i as f32 * 0.7 + seed).sin()).collect(),
        ctx_len,
        items: (0..n * D).map(|i| (i as f32 * 0.3 + seed).cos()).collect(),
        kinds,
    }
}

fn kinds() -> Vec<ItemKind> {
    vec![
        ItemKind::Choice { question: 0, candidate: 0 },
        ItemKind::Choice { question: 0, candidate: 1 },
        ItemKind::Choice { question: 0, candidate: 2 },
        ItemKind::Choice { question: 1, candidate: 0 },
        ItemKind::Choice { question: 1, candidate: 1 },
        ItemKind::Noul { index: 0 },
        ItemKind::Noul { index: 1 },
        ItemKind::Score { index: 0 },
    ]
}

fn model() -> burn_jev::DecisionModel<TestBackend> {
    let reader = burn_jev::ItemReaderConfig::new(D).with_d_reader(8).with_n_heads(2);
    DecisionModelConfig::with_reader(reader, burn_jev::UnifiedHeadsConfig::new(8)).init(&FlexDevice)
}

fn to_vec<const N: usize>(t: Tensor<TestBackend, N>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn batch_shapes_and_grouping() {
    let m = model();
    let a = features(5, 0.0, kinds());
    let b = features(3, 1.0, vec![ItemKind::Noul { index: 0 }]);
    let batch = FeatureBatch::<TestBackend>::new(&[&a, &b], &FlexDevice);
    let out = m.forward(&batch);
    assert_eq!(out.embedding.dims(), [2, 256]);
    // 2 questions from `a`; padded to 3 candidates.
    assert_eq!(out.choice_logits.unwrap().dims(), [2, 3]);
    assert_eq!(batch.choice.as_ref().unwrap().counts, vec![3, 2]);
    assert_eq!(out.noul_logits.unwrap().dims(), [3]);
    assert_eq!(out.score_logits.unwrap().dims(), [1, 4]);
}

#[test]
fn padding_in_a_batch_does_not_change_outputs() {
    let m = model();
    let a = features(3, 0.0, kinds());
    let long = features(9, 2.0, vec![ItemKind::Score { index: 0 }]);
    let alone = m.forward(&FeatureBatch::new(&[&a], &FlexDevice));
    let padded = m.forward(&FeatureBatch::new(&[&a, &long], &FlexDevice));

    let close = |x: Vec<f32>, y: Vec<f32>| x.iter().zip(&y).all(|(p, q)| (p - q).abs() < 1e-5);
    assert!(close(to_vec(alone.noul_logits.unwrap()), to_vec(padded.noul_logits.unwrap())));
    assert!(close(to_vec(alone.choice_logits.unwrap()), to_vec(padded.choice_logits.unwrap())));
    let emb_alone = to_vec(alone.embedding);
    assert!(close(emb_alone.clone(), to_vec(padded.embedding)[..emb_alone.len()].to_vec()));
}

#[test]
fn decide_returns_normalized_verdicts_per_question() {
    let m = model();
    let d = m.decide(&features(4, 0.5, kinds()), &FlexDevice).unwrap();
    assert_eq!(d.choices.len(), 2);
    assert_eq!(d.choices[0].probabilities.len(), 3);
    assert_eq!(d.choices[1].probabilities.len(), 2);
    for c in &d.choices {
        assert!((c.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
    assert_eq!(d.nouls.len(), 2);
    assert_eq!(d.scores.len(), 1);
    let norm: f32 = to_vec(d.embedding).iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4);
}

#[test]
fn gradients_reach_the_cross_attention() {
    type AD = Autodiff<TestBackend>;
    let reader = burn_jev::ItemReaderConfig::new(D).with_d_reader(8).with_n_heads(2);
    let m = DecisionModelConfig::with_reader(reader, burn_jev::UnifiedHeadsConfig::new(8)).init::<AD>(&FlexDevice);
    let a = features(4, 0.0, kinds());
    let out = m.forward(&FeatureBatch::new(&[&a], &FlexDevice));
    let grads = out.noul_logits.unwrap().sum().backward();
    let g = m.reader.blocks[0].attn.value.weight.grad(&grads).expect("attention value projection gets a gradient");
    assert!(g.abs().sum().into_scalar() > 0.0);
    assert!(m.reader.proj.weight.grad(&grads).is_some());
}

#[test]
fn encoder_padding_does_not_change_real_tokens() {
    let cfg = tiny_encoder_config();
    let enc = cfg.init::<TestBackend>(&FlexDevice);
    let (l_short, l_long) = (7, 12);
    let ids: Vec<i64> = (0..l_long as i64).map(|i| (i * 5) % 32).collect();

    let one = enc.forward(
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(ids[..l_short].to_vec(), [1, l_short]), &FlexDevice),
        Tensor::<TestBackend, 2, Bool>::from_data(TensorData::new(vec![true; l_short], [1, l_short]), &FlexDevice),
    );
    let mut padded_ids = ids[..l_short].to_vec();
    padded_ids.resize(l_long, PAD);
    let mut mask = vec![true; l_short];
    mask.resize(l_long, false);
    let batch_ids: Vec<i64> = padded_ids.iter().chain(&ids).copied().collect();
    let batch_mask: Vec<bool> = mask.iter().copied().chain(std::iter::repeat_n(true, l_long)).collect();
    let both = enc.forward(
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(batch_ids, [2, l_long]), &FlexDevice),
        Tensor::<TestBackend, 2, Bool>::from_data(TensorData::new(batch_mask, [2, l_long]), &FlexDevice),
    );
    let a = to_vec(one);
    let b = to_vec(both.slice([0..1, 0..l_short]));
    let max_err = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(max_err < 1e-4, "padding leaked into real tokens: {max_err}");
}
