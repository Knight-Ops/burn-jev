use std::sync::Arc;
use burn::tensor::Tensor;
use burn_flex::{Flex, FlexDevice};

use burn_jev::anomaly::{
    calibrate_threshold_from_distances, AnomalyEmbedding, AnomalyError, FifoRingBuffer,
    InMemoryRingIndex, TenantConfig, TenantRegistry, KNN_EMBEDDING_DIM,
};

type TestBackend = Flex<f32, i32>;

/// Helper to generate a deterministic unit vector with primary weight at a given coordinate.
fn make_unit_vector(active_idx: usize) -> AnomalyEmbedding {
    let mut raw = vec![0.0f32; KNN_EMBEDDING_DIM];
    raw[active_idx % KNN_EMBEDDING_DIM] = 1.0;
    AnomalyEmbedding::new(raw).expect("Must create valid unit vector")
}

/// Helper to generate a perturbed vector around a base vector.
fn make_perturbed_vector(active_idx: usize, noise_level: f32) -> AnomalyEmbedding {
    let mut raw = vec![0.0f32; KNN_EMBEDDING_DIM];
    raw[active_idx % KNN_EMBEDDING_DIM] = 1.0;
    for i in 0..KNN_EMBEDDING_DIM {
        raw[i] += ((i as f32 * 0.17).sin()) * noise_level;
    }
    AnomalyEmbedding::from_raw_unnormalized(raw).expect("Must normalize")
}

#[test]
fn test_anomaly_embedding_validation_and_distances() {
    // Valid unit vector
    let v1 = make_unit_vector(0);
    assert_eq!(v1.as_slice().len(), KNN_EMBEDDING_DIM);
    assert_eq!(v1.euclidean_distance(&v1), 0.0);
    assert_eq!(v1.cosine_distance(&v1), 0.0);

    // Orthogonal unit vector
    let v2 = make_unit_vector(1);
    let dist = v1.euclidean_distance(&v2);
    // For orthogonal unit vectors, ||u - v||_2 = sqrt(1^2 + (-1)^2) = sqrt(2) ≈ 1.4142
    assert!((dist - std::f32::consts::SQRT_2).abs() < 1e-4);
    assert!((v1.cosine_distance(&v2) - 1.0).abs() < 1e-4);

    // Dimension mismatch error
    let invalid_dim = vec![1.0; 128];
    assert!(matches!(
        AnomalyEmbedding::new(invalid_dim),
        Err(AnomalyError::InvalidDimension { expected: 256, found: 128 })
    ));

    // Normalization error
    let unnormalized = vec![1.0; KNN_EMBEDDING_DIM];
    assert!(matches!(
        AnomalyEmbedding::new(unnormalized),
        Err(AnomalyError::NormalizationError { .. })
    ));

    // from_raw_unnormalized succeeds and normalizes
    let raw = vec![2.0; KNN_EMBEDDING_DIM];
    let normalized = AnomalyEmbedding::from_raw_unnormalized(raw).unwrap();
    let norm: f32 = normalized.as_slice().iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4);
}

#[test]
fn test_burn_tensor_embedding_extraction() {
    let device = FlexDevice;

    // 1D Tensor extraction
    let mut raw_data = vec![0.0f32; KNN_EMBEDDING_DIM];
    raw_data[42] = 1.0;
    let tensor_1d = Tensor::<TestBackend, 1>::from_floats(raw_data.as_slice(), &device);
    let emb_1d = AnomalyEmbedding::from_burn_tensor(tensor_1d).expect("Extraction must succeed");
    assert_eq!(emb_1d.as_slice()[42], 1.0);

    // 2D Tensor row extraction (Batch = 2)
    let mut raw_batch = vec![0.0f32; 2 * KNN_EMBEDDING_DIM];
    raw_batch[10] = 1.0; // Row 0
    raw_batch[KNN_EMBEDDING_DIM + 20] = 1.0; // Row 1
    let tensor_2d = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(raw_batch, [2, KNN_EMBEDDING_DIM]),
        &device,
    );

    let row_0 = AnomalyEmbedding::from_burn_tensor_row(&tensor_2d, 0).unwrap();
    let row_1 = AnomalyEmbedding::from_burn_tensor_row(&tensor_2d, 1).unwrap();
    assert_eq!(row_0.as_slice()[10], 1.0);
    assert_eq!(row_1.as_slice()[20], 1.0);

    // Out of bounds row index
    assert!(AnomalyEmbedding::from_burn_tensor_row(&tensor_2d, 2).is_err());
}

#[test]
fn test_fifo_ring_buffer_eviction_and_k_nearest() {
    let capacity = 5;
    let mut ring = FifoRingBuffer::new(capacity);
    assert_eq!(ring.len(), 0);
    assert!(ring.is_empty());

    // Insert 3 items
    for i in 0..3 {
        ring.insert(make_unit_vector(i));
    }
    assert_eq!(ring.len(), 3);

    // Query 2 nearest to unit_vector(0)
    let q = make_unit_vector(0);
    let distances = ring.k_nearest_distances(&q, 2);
    assert_eq!(distances.len(), 2);
    assert_eq!(distances[0], 0.0); // Exact match with item 0
    assert!((distances[1] - std::f32::consts::SQRT_2).abs() < 1e-4);

    // Fill to capacity
    ring.insert(make_unit_vector(3));
    ring.insert(make_unit_vector(4));
    assert_eq!(ring.len(), 5);

    // Insert 2 more items to trigger FIFO overwrite of index 0 and 1
    ring.insert(make_unit_vector(5)); // Overwrites 0
    ring.insert(make_unit_vector(6)); // Overwrites 1
    assert_eq!(ring.len(), 5);

    // Now query for unit_vector(0) which was evicted: closest should now be ~sqrt(2)
    let post_evict_d = ring.k_nearest_distances(&q, 1);
    assert_eq!(post_evict_d.len(), 1);
    assert!(post_evict_d[0] > 1.0); // 0 was evicted, no exact 0.0 distance
}

#[test]
fn test_threshold_calibration_percentile() {
    // 100 sample distances from 0.01 to 1.00
    let distances: Vec<f32> = (1..=100).map(|i| i as f32 / 100.0).collect();

    // 5% FPR -> 95th percentile
    let tau_95 = calibrate_threshold_from_distances(&distances, 0.05).unwrap();
    assert!((tau_95 - 0.9505).abs() < 0.02);

    // 1% FPR -> 99th percentile
    let tau_99 = calibrate_threshold_from_distances(&distances, 0.01).unwrap();
    assert!((tau_99 - 0.99).abs() < 0.02);

    // Invalid FPR
    assert!(matches!(
        calibrate_threshold_from_distances(&distances, 0.0),
        Err(AnomalyError::InvalidFpr { .. })
    ));
    assert!(matches!(
        calibrate_threshold_from_distances(&distances, 1.0),
        Err(AnomalyError::InvalidFpr { .. })
    ));

    // Empty slice
    assert!(matches!(
        calibrate_threshold_from_distances(&[], 0.05),
        Err(AnomalyError::InsufficientSamples { .. })
    ));
}

#[test]
fn test_cold_start_interpolation_lifecycle() {
    let index = Arc::new(InMemoryRingIndex::new());
    let registry = TenantRegistry::new(index.clone());

    // 1. Setup Global Anchor baseline: 10 safe anchors clustered around index 0
    let global_anchors: Vec<AnomalyEmbedding> = (0..10)
        .map(|i| make_perturbed_vector(0, i as f32 * 0.01))
        .collect();
    registry.set_global_anchors(&global_anchors).unwrap();
    assert_eq!(registry.global_sample_count(), 10);

    let tenant_id = "tenant_fintech_corp";
    let tenant_config = TenantConfig::new(tenant_id, 0.50, 5)
        .unwrap()
        .with_cold_start_horizon(100); // 100 samples for test horizon
    registry.register_tenant(tenant_config).unwrap();

    // Query 1: Close to global anchor baseline
    let query_benign = make_perturbed_vector(0, 0.01);

    // State A: N = 0 (Pure Cold Start, alpha = 0.0)
    let verdict_cold = registry.evaluate(tenant_id, &query_benign).unwrap();
    assert_eq!(verdict_cold.cold_start_alpha, 0.0);
    assert_eq!(verdict_cold.tenant_distance, None);
    assert_eq!(verdict_cold.effective_distance, verdict_cold.global_distance);
    assert!(!verdict_cold.is_anomaly); // Distance to global anchor is small

    // State B: Add 50 benign samples clustered around coordinate 10 (different domain than global)
    for _ in 0..50 {
        let sample = make_perturbed_vector(10, 0.02);
        registry.record_benign(tenant_id, sample).unwrap();
    }
    assert_eq!(registry.tenant_sample_count(tenant_id), 50);

    // Evaluate query clustered at coordinate 10:
    let query_tenant_domain = make_perturbed_vector(10, 0.02);
    let verdict_halfway = registry.evaluate(tenant_id, &query_tenant_domain).unwrap();
    assert!((verdict_halfway.cold_start_alpha - 0.50).abs() < 1e-4);
    assert!(verdict_halfway.tenant_distance.is_some());
    let tenant_d = verdict_halfway.tenant_distance.unwrap();
    let global_d = verdict_halfway.global_distance;
    let expected_s = 0.50 * tenant_d + 0.50 * global_d;
    assert!((verdict_halfway.effective_distance - expected_s).abs() < 1e-4);

    // State C: Add 50 more samples (Total N = 100 >= cold_start_horizon, alpha = 1.0)
    for _ in 0..50 {
        let sample = make_perturbed_vector(10, 0.02);
        registry.record_benign(tenant_id, sample).unwrap();
    }
    assert_eq!(registry.tenant_sample_count(tenant_id), 100);

    let verdict_mature = registry.evaluate(tenant_id, &query_tenant_domain).unwrap();
    assert_eq!(verdict_mature.cold_start_alpha, 1.0);
    assert_eq!(
        verdict_mature.effective_distance,
        verdict_mature.tenant_distance.unwrap()
    );
    assert!(!verdict_mature.is_anomaly); // Benign query within mature tenant domain

    // Outlier query far from both global anchors and tenant cluster (coordinate 200)
    let query_attack = make_unit_vector(200);
    let verdict_attack = registry.evaluate(tenant_id, &query_attack).unwrap();
    assert!(verdict_attack.is_anomaly); // effective_distance > 0.50 threshold
    assert!(verdict_attack.effective_distance > verdict_attack.threshold);
}

#[test]
fn test_multi_tenant_namespace_isolation() {
    let index = Arc::new(InMemoryRingIndex::new());
    let registry = TenantRegistry::new(index.clone());

    // Setup global anchors
    let global_anchors = vec![make_unit_vector(0), make_unit_vector(1)];
    registry.set_global_anchors(&global_anchors).unwrap();

    let tenant_a = "tenant_a";
    let tenant_b = "tenant_b";

    // Insert 5 vectors for tenant A (at index 50)
    for _ in 0..5 {
        registry.record_benign(tenant_a, make_unit_vector(50)).unwrap();
    }

    // Insert 10 vectors for tenant B (at index 100)
    for _ in 0..10 {
        registry.record_benign(tenant_b, make_unit_vector(100)).unwrap();
    }

    assert_eq!(registry.tenant_sample_count(tenant_a), 5);
    assert_eq!(registry.tenant_sample_count(tenant_b), 10);

    // Query close to Tenant A's cluster
    let q_a = make_unit_vector(50);
    let v_a = registry.evaluate(tenant_a, &q_a).unwrap();
    let v_b = registry.evaluate(tenant_b, &q_a).unwrap();

    // Tenant A sees small distance (0.0), Tenant B sees orthogonal distance (~1.414)
    assert_eq!(v_a.tenant_distance.unwrap(), 0.0);
    assert!((v_b.tenant_distance.unwrap() - std::f32::consts::SQRT_2).abs() < 1e-3);
}
