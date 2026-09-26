use burn::tensor::Tensor;
use burn_flex::{Flex, FlexDevice};
use burn_mamba::{
    ChoiceVerdict, Head, JevError, NoulVerdict, ScoreVerdict, UnifiedHeads, UnifiedHeadsConfig,
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
        heads.temperature_value(Head::Noul),
        "Temperature must match the noul head temperature"
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
fn test_per_head_temperatures() {
    let device = FlexDevice;
    let d_model = 32;
    let config = UnifiedHeadsConfig::new(d_model);
    let mut heads: UnifiedHeads<TestBackend> = config.init(&device);

    // Every head starts at 1.0 and is frozen (fit post hoc, never trained).
    for head in Head::ALL {
        assert!((heads.temperature_value(head) - 1.0).abs() < 1e-5);
    }
    assert!(!heads.choice_temperature.is_require_grad());

    // Each head reads its own temperature.
    heads.set_temperatures(2.0, 0.5, 4.0);
    assert!((heads.temperature_value(Head::Choice) - 2.0).abs() < 1e-5);
    assert!((heads.temperature_value(Head::Noul) - 0.5).abs() < 1e-5);
    assert!((heads.temperature_value(Head::Score) - 4.0).abs() < 1e-5);

    let token = Tensor::<TestBackend, 1>::from_floats([0.3; 32], &device).unsqueeze_dim(0);
    let raw = heads.forward_noul_raw_logits(token.clone()).into_scalar();
    let scaled = heads.forward_noul_logits(token.clone()).into_scalar();
    assert!((scaled - raw / 0.5).abs() < 1e-4, "noul logits must use the noul temperature");
    let raw = heads.forward_score_raw_logits(token.clone()).into_data().to_vec::<f32>().unwrap();
    let scaled = heads.forward_score_logits(token.clone()).into_data().to_vec::<f32>().unwrap();
    for (r, s) in raw.iter().zip(&scaled) {
        assert!((s - r / 4.0).abs() < 1e-4, "score logits must use the score temperature");
    }
    let raw = heads.forward_choice_raw_logits(token.clone()).into_scalar();
    let scaled = heads.forward_choice_logits(token.clone()).into_scalar();
    assert!((scaled - raw / 2.0).abs() < 1e-4, "choice logits must use the choice temperature");
    assert_eq!(heads.evaluate_noul_default(token.clone()).unwrap().calibrated_temperature, 0.5);

    // Extreme values are clamped to [0.01, 10.0] and stay numerically stable.
    heads.set_temperatures(0.0001, 100.0, 1.0);
    assert!((heads.temperature_value(Head::Choice) - 0.01).abs() < 1e-5);
    assert!((heads.temperature_value(Head::Noul) - 10.0).abs() < 1e-5);
    let noul = heads.evaluate_noul_default(token).unwrap();
    assert!(!noul.probability.is_nan() && !noul.probability.is_infinite());
}
