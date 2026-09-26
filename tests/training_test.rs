use burn::tensor::{Int, Tensor};
use burn_flex::{Flex, FlexDevice};
use burn_jev::{
    benign_adversarial_metric_loss, brier_calibration_loss, choice_cross_entropy_loss,
    noul_bce_loss, ordinal_score_loss,
};

type TestBackend = Flex<f32, i32>;

#[test]
fn test_benign_adversarial_metric_loss_dynamics() {
    let device = FlexDevice;

    // 4 samples: 2 benign, 2 adversarial
    let is_benign = vec![true, true, false, false];

    // Case 1: Benign samples are close (cosine sim ~ 1.0), adversarial are orthogonal/distant
    let good_data: [f32; 8] = [
        1.0, 0.0, // Benign 0
        0.99, 0.14, // Benign 1 (normalized ~ 1.0)
        0.0, 1.0, // Adversarial 2
        -1.0, 0.0, // Adversarial 3
    ];
    let good_embeddings = Tensor::<TestBackend, 1>::from_data(good_data.as_slice(), &device)
        .reshape([4, 2]);

    let low_loss = benign_adversarial_metric_loss(good_embeddings, &is_benign, 0.1, &device);
    let low_val = low_loss.into_data().as_slice::<f32>().unwrap()[0];

    // Case 2: Benign samples are far apart (Sample 0: [1, 0], Sample 1: [-1, 0])
    let bad_data: [f32; 8] = [
        1.0, 0.0, // Benign 0
        -1.0, 0.0, // Benign 1
        0.0, 1.0, // Adversarial 2
        0.0, -1.0, // Adversarial 3
    ];
    let bad_embeddings = Tensor::<TestBackend, 1>::from_data(bad_data.as_slice(), &device)
        .reshape([4, 2]);

    let high_loss = benign_adversarial_metric_loss(bad_embeddings, &is_benign, 0.1, &device);
    let high_val = high_loss.into_data().as_slice::<f32>().unwrap()[0];

    assert!(
        low_val < high_val,
        "Clustered benign embeddings must have lower contrastive metric loss ({low_val}) than distant benign embeddings ({high_val})"
    );
}

#[test]
fn test_metric_loss_handles_zero_benign_pairs_gracefully() {
    let device = FlexDevice;

    // Batch with only 1 benign item (no positive peers possible)
    let is_benign = vec![true, false, false];
    let emb_data: [f32; 6] = [
        1.0, 0.0,
        0.0, 1.0,
        -1.0, 0.0,
    ];
    let embeddings = Tensor::<TestBackend, 1>::from_data(emb_data.as_slice(), &device)
        .reshape([3, 2]);

    let loss = benign_adversarial_metric_loss(embeddings, &is_benign, 0.07, &device);
    let val = loss.into_data().as_slice::<f32>().unwrap()[0];
    assert_eq!(val, 0.0, "Zero valid positive pairs should yield 0.0 loss without NaN or panic");
}

#[test]
fn test_noul_bce_loss_correctness() {
    let device = FlexDevice;

    // Logits: confident positive (+5.0), confident negative (-5.0)
    let logits_data: [f32; 2] = [5.0, -5.0];
    let targets_data: [f32; 2] = [1.0, 0.0];
    let logits = Tensor::<TestBackend, 1>::from_data(logits_data.as_slice(), &device);
    let targets = Tensor::<TestBackend, 1>::from_data(targets_data.as_slice(), &device);

    let loss = noul_bce_loss(logits, targets, None, &device);
    let val = loss.into_data().as_slice::<f32>().unwrap()[0];

    // Highly confident correct predictions should produce loss near zero
    assert!(val < 0.05, "Confident correct BCE loss should be near zero, got {val}");

    // Inverted predictions (+5.0 for y=0, -5.0 for y=1)
    let bad_data: [f32; 2] = [-5.0, 5.0];
    let bad_logits = Tensor::<TestBackend, 1>::from_data(bad_data.as_slice(), &device);
    let high_loss = noul_bce_loss(bad_logits, Tensor::<TestBackend, 1>::from_data(targets_data.as_slice(), &device), None, &device);
    let high_val = high_loss.into_data().as_slice::<f32>().unwrap()[0];
    assert!(high_val > 4.5, "Inverted predictions should yield high BCE loss, got {high_val}");
}

#[test]
fn test_choice_cross_entropy_with_padding_mask() {
    let device = FlexDevice;

    // Batch size 2, max candidates 3
    // Item 0 has 3 candidates, target is index 1
    // Item 1 has 2 candidates (third is padding), target is index 0
    let logits_data: [f32; 6] = [
        -2.0, 5.0, -1.0, // Item 0: target 1 is dominant
        4.0, -2.0, 999.0, // Item 1: target 0 is dominant, index 2 has large value but IS PADDED
    ];
    let logits = Tensor::<TestBackend, 1>::from_data(logits_data.as_slice(), &device)
        .reshape([2, 3]);

    let targets_data: [i32; 2] = [1, 0];
    let targets = Tensor::<TestBackend, 1, Int>::from_data(targets_data.as_slice(), &device);

    // Candidate mask: item 1 has false at index 2
    let mask_data: [bool; 6] = [
        true, true, true,
        true, true, false, // index 2 is padding
    ];
    let mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_data(mask_data.as_slice(), &device)
        .reshape([2, 3]);

    let loss = choice_cross_entropy_loss(logits, targets, mask, None, &device);
    let val = loss.into_data().as_slice::<f32>().unwrap()[0];

    // Despite index 2 having a huge 999.0 logit in item 1, the padding mask must suppress it
    assert!(val < 0.05, "Padded candidate with large logit must be masked out; loss was {val}");
}

#[test]
fn test_ordinal_score_loss_monotonicity() {
    let device = FlexDevice;
    let num_bins = 5; // cutoffs at 1.0, 2.0, 3.0, 4.0

    // Item with true score 3.0
    let targets_data: [f32; 1] = [3.0];
    let targets = Tensor::<TestBackend, 1>::from_data(targets_data.as_slice(), &device);

    // Accurate model: logits [+5, +5, -5, -5]
    let accurate_data: [f32; 4] = [5.0, 5.0, -5.0, -5.0];
    let accurate_logits = Tensor::<TestBackend, 1>::from_data(accurate_data.as_slice(), &device).reshape([1, 4]);
    let good_loss = ordinal_score_loss(accurate_logits, targets.clone(), num_bins, None, &device);
    let good_val = good_loss.into_data().as_slice::<f32>().unwrap()[0];

    // Inaccurate model: logits [-5, -5, +5, +5]
    let inaccurate_data: [f32; 4] = [-5.0, -5.0, 5.0, 5.0];
    let inaccurate_logits = Tensor::<TestBackend, 1>::from_data(inaccurate_data.as_slice(), &device).reshape([1, 4]);
    let bad_loss = ordinal_score_loss(inaccurate_logits, targets, num_bins, None, &device);
    let bad_val = bad_loss.into_data().as_slice::<f32>().unwrap()[0];

    assert!(good_val < 0.05, "Accurate ordinal logits should have low loss, got {good_val}");
    assert!(bad_val > 4.5, "Inaccurate ordinal logits should have high loss, got {bad_val}");
}

#[test]
fn test_brier_calibration_loss_proper_scoring() {
    let device = FlexDevice;

    // Logits: +4.0 (prob ~0.982), -4.0 (prob ~0.018)
    let logits_data: [f32; 2] = [4.0, -4.0];
    let targets_data: [f32; 2] = [1.0, 0.0];
    let logits = Tensor::<TestBackend, 1>::from_data(logits_data.as_slice(), &device);
    let targets = Tensor::<TestBackend, 1>::from_data(targets_data.as_slice(), &device);

    let loss = brier_calibration_loss(logits, targets, None, &device);
    let val = loss.into_data().as_slice::<f32>().unwrap()[0];

    // Brier score: (0.982 - 1)^2 + (0.018 - 0)^2 ~ 0.0003 + 0.0003 ~ 0.0006
    assert!(val < 0.01, "Well-calibrated probabilities must yield near-zero Brier score, got {val}");
}
