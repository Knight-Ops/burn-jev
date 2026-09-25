use burn::{module::Param, tensor::Tensor};
use burn_flex::{Flex, FlexDevice};
use burn_mamba::{
    ChoiceVerdict, JevError, NoulVerdict, ScoreVerdict, UnifiedHeads, UnifiedHeadsConfig,
    MAX_CHOICE_CANDIDATES,
};

type TestBackend = Flex<f32, i32>;

#[test]
fn test_noul_verdict_thresholding() {
    let device = FlexDevice;
    let d_model = 64;
    let config = UnifiedHeadsConfig::new(d_model);
    let heads: UnifiedHeads<TestBackend> = config.init(&device);

    let token_vector = Tensor::<TestBackend, 1>::zeros([d_model], &device).unsqueeze_dim(0); // [1, D]

    // Default threshold 0.5
    let verdict = heads
        .evaluate_noul_default(token_vector.clone())
        .expect("Noul default evaluation should succeed");

    assert!(
        verdict.probability >= 0.0 && verdict.probability <= 1.0,
        "Probability must be in [0.0, 1.0], got {}",
        verdict.probability
    );
    assert_eq!(
        verdict.is_true,
        verdict.probability >= 0.5,
        "is_true must reflect probability >= 0.5"
    );
    assert_eq!(
        verdict.calibrated_temperature,
        heads.temperature_value(),
        "Temperature must match heads temperature"
    );

    // Custom strict threshold
    let strict_verdict = heads
        .evaluate_noul_verdict(token_vector.clone(), 0.99)
        .expect("Strict threshold should succeed");
    assert_eq!(strict_verdict.is_true, strict_verdict.probability >= 0.99);

    // Custom lenient threshold
    let lenient_verdict = heads
        .evaluate_noul_verdict(token_vector, 0.01)
        .expect("Lenient threshold should succeed");
    assert_eq!(lenient_verdict.is_true, lenient_verdict.probability >= 0.01);

    // Direct constructor unit check
    let direct_noul = NoulVerdict::new(0.75, 0.5, 1.0).unwrap();
    assert!(direct_noul.is_true);
    assert_eq!(direct_noul.probability, 0.75);
}

#[test]
fn test_noul_invalid_threshold() {
    let device = FlexDevice;
    let d_model = 32;
    let config = UnifiedHeadsConfig::new(d_model);
    let heads: UnifiedHeads<TestBackend> = config.init(&device);

    let token_vector = Tensor::<TestBackend, 1>::zeros([d_model], &device).unsqueeze_dim(0);

    let err_neg = heads.evaluate_noul_verdict(token_vector.clone(), -0.1);
    assert!(matches!(err_neg, Err(JevError::InvalidThreshold { .. })));

    let err_high = heads.evaluate_noul_verdict(token_vector, 1.05);
    assert!(matches!(err_high, Err(JevError::InvalidThreshold { .. })));
}

#[test]
fn test_score_verdict_monotonicity_and_bounds() {
    let device = FlexDevice;
    let d_model = 64;
    let num_rubric_bins = 5;
    let config = UnifiedHeadsConfig::new(d_model).with_num_rubric_bins(num_rubric_bins);
    let heads: UnifiedHeads<TestBackend> = config.init(&device);

    let token_vector = Tensor::<TestBackend, 1>::zeros([d_model], &device).unsqueeze_dim(0);

    let verdict = heads
        .evaluate_score_verdict(token_vector)
        .expect("Score evaluation should succeed");

    assert_eq!(
        verdict.cumulative_probs.len(),
        num_rubric_bins - 1,
        "Should have M-1 cumulative threshold probabilities"
    );

    for (idx, &prob) in verdict.cumulative_probs.iter().enumerate() {
        assert!(
            prob >= 0.0 && prob <= 1.0,
            "Cumulative prob at index {} must be in [0, 1], got {}",
            idx,
            prob
        );
    }

    assert!(
        verdict.expected_score >= 1.0 && verdict.expected_score <= num_rubric_bins as f32,
        "Expected score {} must be in [1.0, {}]",
        verdict.expected_score,
        num_rubric_bins
    );

    assert!(
        verdict.discrete_bin >= 1 && verdict.discrete_bin <= num_rubric_bins,
        "Discrete bin {} must be in 1..={}",
        verdict.discrete_bin,
        num_rubric_bins
    );

    // Verify synthetic boundary conditions
    let min_score = ScoreVerdict::new(vec![0.0, 0.0, 0.0, 0.0], 5).unwrap();
    assert_eq!(min_score.expected_score, 1.0);
    assert_eq!(min_score.discrete_bin, 1);

    let max_score = ScoreVerdict::new(vec![1.0, 1.0, 1.0, 1.0], 5).unwrap();
    assert_eq!(max_score.expected_score, 5.0);
    assert_eq!(max_score.discrete_bin, 5);

    let mid_score = ScoreVerdict::new(vec![0.9, 0.7, 0.2, 0.1], 5).unwrap();
    assert!((mid_score.expected_score - 2.9).abs() < 1e-4);
    assert_eq!(mid_score.discrete_bin, 3); // 1 + 2 thresholds >= 0.5
}

#[test]
fn test_choice_verdict_distribution_and_argmax() {
    let device = FlexDevice;
    let d_model = 64;
    let config = UnifiedHeadsConfig::new(d_model);
    let heads: UnifiedHeads<TestBackend> = config.init(&device);

    // 3 candidates
    let k = 3;
    let cand_vectors = Tensor::<TestBackend, 1>::zeros([k * d_model], &device).reshape([k, d_model]);

    let verdict = heads
        .evaluate_choice_verdict(cand_vectors)
        .expect("Choice evaluation should succeed");

    assert_eq!(verdict.probabilities.len(), k);
    assert_eq!(verdict.logits.len(), k);
    assert!(verdict.selected_candidate < k);

    let sum_probs: f32 = verdict.probabilities.iter().sum();
    assert!(
        (sum_probs - 1.0).abs() < 1e-5,
        "Probabilities must sum to 1.0, got {}",
        sum_probs
    );

    for &p in &verdict.probabilities {
        assert!(p >= 0.0 && p <= 1.0, "Probability {} out of [0, 1]", p);
    }

    // Verify argmax consistency
    let max_prob = verdict
        .probabilities
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    assert_eq!(
        verdict.probabilities[verdict.selected_candidate], max_prob,
        "Selected candidate must correspond to maximum probability"
    );

    // K = 1 single candidate test
    let single_cand = Tensor::<TestBackend, 1>::zeros([d_model], &device).reshape([1, d_model]);
    let single_verdict = heads
        .evaluate_choice_verdict(single_cand)
        .expect("Single candidate choice should succeed");
    assert_eq!(single_verdict.selected_candidate, 0);
    assert_eq!(single_verdict.probabilities.len(), 1);
    assert!((single_verdict.probabilities[0] - 1.0).abs() < 1e-5);

    // Direct constructor unit check
    let direct_choice = ChoiceVerdict::new(vec![0.1, 0.7, 0.2], vec![0.5, 2.5, 1.2]).unwrap();
    assert_eq!(direct_choice.selected_candidate, 1);
}

#[test]
fn test_choice_verdict_candidate_limits() {
    let device = FlexDevice;
    let d_model = 16;
    let config = UnifiedHeadsConfig::new(d_model);
    let heads: UnifiedHeads<TestBackend> = config.init(&device);

    // K = 0 (Empty)
    let empty_cand = Tensor::<TestBackend, 1>::zeros([0], &device).reshape([0, d_model]);
    let err_empty = heads.evaluate_choice_verdict(empty_cand);
    assert_eq!(err_empty, Err(JevError::EmptyCandidates));

    // K = 256 (> MAX_CHOICE_CANDIDATES = 255)
    let overflow_k = MAX_CHOICE_CANDIDATES + 1;
    let overflow_cand =
        Tensor::<TestBackend, 1>::zeros([overflow_k * d_model], &device).reshape([overflow_k, d_model]);
    let err_overflow = heads.evaluate_choice_verdict(overflow_cand);
    assert_eq!(
        err_overflow,
        Err(JevError::TooManyCandidates {
            count: 256,
            max: 255
        })
    );
}

#[test]
fn test_platt_temperature_clamping() {
    let device = FlexDevice;
    let d_model = 32;
    let config = UnifiedHeadsConfig::new(d_model);
    let mut heads: UnifiedHeads<TestBackend> = config.init(&device);

    // Default initialized temperature is 1.0
    assert!((heads.temperature_value() - 1.0).abs() < 1e-5);

    // Manually set extreme low temperature (< 0.01)
    heads.temperature = Param::from_tensor(Tensor::<TestBackend, 1>::from_data([0.0001f32], &device));
    assert!(
        (heads.temperature_value() - 0.01).abs() < 1e-5,
        "Temperature must clamp to 0.01 min, got {}",
        heads.temperature_value()
    );

    // Manually set extreme high temperature (> 10.0)
    heads.temperature = Param::from_tensor(Tensor::<TestBackend, 1>::from_data([100.0f32], &device));
    assert!(
        (heads.temperature_value() - 10.0).abs() < 1e-5,
        "Temperature must clamp to 10.0 max, got {}",
        heads.temperature_value()
    );

    // Perform forward pass with clamped temperature: verify numerical stability
    let token = Tensor::<TestBackend, 1>::zeros([d_model], &device).unsqueeze_dim(0);
    let noul = heads.evaluate_noul_default(token).unwrap();
    assert!(!noul.probability.is_nan() && !noul.probability.is_infinite());
}
