use burn::backend::Autodiff;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor};
use burn_flex::{Flex, FlexDevice};
use burn_mamba::{
    benign_adversarial_metric_loss, brier_calibration_loss, choice_cross_entropy_loss,
    noul_bce_loss, ordinal_score_loss, BiMamba2Config, CoordinateResolver, DelimiterConfig,
    JointLossConfig, JointTier1Loss, TrainingBatch,
};

type TestBackend = Flex<f32, i32>;
type DiffBackend = Autodiff<TestBackend>;

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

#[test]
fn test_end_to_end_joint_loss_gradient_flow() {
    let device = FlexDevice;
    let d_model = 32;
    let vocab_size = 100;

    let config = BiMamba2Config::new(vocab_size)
        .with_d_model(d_model)
        .with_d_state(16)
        .with_headdim(16)
        .with_expand(1)
        .with_n_layers(2)
        .with_knn_dim(16)
        .with_num_rubric_bins(5);

    let model = config.init::<DiffBackend>(&device);

    let delims = DelimiterConfig::default();
    let resolver = CoordinateResolver::new(delims.clone());

    // Construct a synthetic batch of 2 sequences:
    let seq0: Vec<i64> = vec![
        delims.cls_id, 10,
        delims.cand_marker_id, 11,
        delims.cand_marker_id, 12,
        delims.noul_query_marker_id, 13,
        delims.score_query_marker_id, 14,
    ];
    let seq1: Vec<i64> = vec![
        delims.cls_id, 20,
        delims.cand_marker_id, 21,
        delims.cand_marker_id, 22,
        delims.noul_query_marker_id, 23,
        delims.score_query_marker_id, 24,
    ];
    let seq_len = seq0.len();

    let coords0 = resolver.resolve_coordinates(&seq0).unwrap();
    let coords1 = resolver.resolve_coordinates(&seq1).unwrap();

    let mut input_data: Vec<i32> = Vec::new();
    input_data.extend(seq0.iter().map(|&x| x as i32));
    input_data.extend(seq1.iter().map(|&x| x as i32));

    let input_ids = Tensor::<DiffBackend, 1, Int>::from_data(input_data.as_slice(), &device)
        .reshape([2, seq_len]);

    let choice_targets_data: [i32; 2] = [0, 1];
    let noul_targets_data: [f32; 2] = [1.0, 0.0];
    let score_targets_data: [f32; 2] = [4.0, 2.0];

    let batch = TrainingBatch {
        input_ids,
        coords: vec![coords0, coords1],
        is_benign: vec![true, true],
        choice_targets: Some(Tensor::<DiffBackend, 1, Int>::from_data(choice_targets_data.as_slice(), &device)),
        noul_targets: Some(Tensor::<DiffBackend, 1>::from_data(noul_targets_data.as_slice(), &device)),
        score_targets: Some(Tensor::<DiffBackend, 1>::from_data(score_targets_data.as_slice(), &device)),
    };

    let joint_engine = JointTier1Loss::new(
        JointLossConfig::default()
            .with_lambda_metric(0.5)
            .with_lambda_noul(1.0)
            .with_lambda_choice(1.0)
            .with_lambda_score(1.0)
            .with_lambda_calibration(0.5),
    );

    let output = joint_engine.forward(&model, batch, &device).unwrap();

    // Verify loss breakdown values are populated and finite
    assert!(!output.breakdown.total_loss.is_nan());
    assert!(!output.breakdown.metric_loss.is_nan());
    assert!(!output.breakdown.noul_loss.is_nan());
    assert!(!output.breakdown.choice_loss.is_nan());
    assert!(!output.breakdown.score_loss.is_nan());
    assert!(!output.breakdown.calibration_loss.is_nan());
    assert!(output.breakdown.total_loss > 0.0);

    // Backward pass: Compute autograd gradients across entire pipeline
    let grads = output.total_loss.backward();

    // Check gradients on Backbone layers
    let d_skip_grad = model.layers[0].d_skip.grad(&grads);
    assert!(d_skip_grad.is_some(), "Gradient for layers[0].d_skip must exist");
    for &g in d_skip_grad.unwrap().into_data().as_slice::<f32>().unwrap() {
        assert!(!g.is_nan() && !g.is_infinite(), "d_skip grad must be finite");
    }

    let emb_grad = model.embedding.weight.grad(&grads);
    assert!(emb_grad.is_some(), "Gradient for embedding.weight must exist");

    // Check gradients on Head layers
    let knn_grad = model.heads.knn_fc1.weight.grad(&grads);
    assert!(knn_grad.is_some(), "Gradient for knn_fc1.weight must exist");

    let choice_grad = model.heads.choice_fc1.weight.grad(&grads);
    assert!(choice_grad.is_some(), "Gradient for choice_fc1.weight must exist");

    let noul_grad = model.heads.noul_fc1.weight.grad(&grads);
    assert!(noul_grad.is_some(), "Gradient for noul_fc1.weight must exist");

    let score_grad1 = model.heads.score_fc1.weight.grad(&grads);
    assert!(score_grad1.is_some(), "Gradient for score_fc1.weight must exist");
    let score_grad2 = model.heads.score_fc2.weight.grad(&grads);
    assert!(score_grad2.is_some(), "Gradient for score_fc2.weight must exist");

    let temp_grad = model.heads.temperature.grad(&grads);
    assert!(temp_grad.is_some(), "Gradient for temperature parameter must exist");
}

#[test]
fn test_optimizer_multi_step_joint_loss_reduction() {
    let device = FlexDevice;
    let d_model = 32;
    let vocab_size = 100;

    let config = BiMamba2Config::new(vocab_size)
        .with_d_model(d_model)
        .with_d_state(16)
        .with_headdim(16)
        .with_expand(1)
        .with_n_layers(2)
        .with_knn_dim(16)
        .with_num_rubric_bins(5);

    let mut model = config.init::<DiffBackend>(&device);
    let mut optim = AdamConfig::new().init();

    let delims = DelimiterConfig::default();
    let resolver = CoordinateResolver::new(delims.clone());

    let seq0: Vec<i64> = vec![
        delims.cls_id, 10,
        delims.cand_marker_id, 11,
        delims.cand_marker_id, 12,
        delims.noul_query_marker_id, 13,
        delims.score_query_marker_id, 14,
    ];
    let seq_len = seq0.len();
    let coords0 = resolver.resolve_coordinates(&seq0).unwrap();

    let joint_engine = JointTier1Loss::new(JointLossConfig::default());

    let mut losses = Vec::new();

    let choice_data: [i32; 1] = [0];
    let noul_data: [f32; 1] = [1.0];
    let score_data: [f32; 1] = [4.0];

    for _step in 0..3 {
        let input_ids = Tensor::<DiffBackend, 1, Int>::from_data(
            seq0.iter().map(|&x| x as i32).collect::<Vec<_>>().as_slice(),
            &device,
        )
        .reshape([1, seq_len]);

        let batch = TrainingBatch {
            input_ids,
            coords: vec![coords0.clone()],
            is_benign: vec![true],
            choice_targets: Some(Tensor::<DiffBackend, 1, Int>::from_data(choice_data.as_slice(), &device)),
            noul_targets: Some(Tensor::<DiffBackend, 1>::from_data(noul_data.as_slice(), &device)),
            score_targets: Some(Tensor::<DiffBackend, 1>::from_data(score_data.as_slice(), &device)),
        };

        let output = joint_engine.forward(&model, batch, &device).unwrap();
        losses.push(output.breakdown.total_loss);

        let grads = output.total_loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(0.05, model, grads);
    }

    assert_eq!(losses.len(), 3);
    assert!(
        losses[2] < losses[0],
        "Loss after 2 Adam steps ({}) must be lower than initial loss ({})",
        losses[2],
        losses[0]
    );
}
