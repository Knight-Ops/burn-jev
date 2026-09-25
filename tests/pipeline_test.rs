use std::sync::Arc;
use burn_flex::{Flex, FlexDevice};

use burn_mamba::{
    AnomalyEmbedding, BiMamba2Config, CoordinateResolver, DelimiterConfig, EscalationReason,
    InMemoryRingIndex, ReflexEngine, ReflexSecurityRouter, TenantConfig, TenantRegistry,
    Tier1Routing, KNN_EMBEDDING_DIM,
};

type TestBackend = Flex<f32, i32>;

fn init_test_engine() -> (ReflexEngine<TestBackend>, FlexDevice, DelimiterConfig) {
    let device = FlexDevice;
    let delimiter_config = DelimiterConfig::default();
    let resolver = CoordinateResolver::new(delimiter_config.clone());

    let config = BiMamba2Config::new(500)
        .with_d_model(64)
        .with_n_layers(2)
        .with_d_state(32)
        .with_headdim(32)
        .with_knn_dim(KNN_EMBEDDING_DIM);

    let model = config.init(&device);
    let engine = ReflexEngine::new(model, resolver);

    (engine, device, delimiter_config)
}

#[test]
fn test_general_purpose_reflex_engine_agent_tool_selection() {
    let (engine, device, delimiters) = init_test_engine();

    // General-purpose Agent action selection without any security/anomaly assumptions:
    // Prompt: [CLS] ... user prompt ... [SEP] <cand> Bash Tool <cand> SQL Tool <cand> Python Tool [EOS]
    let tokens = vec![
        delimiters.cls_id,              // [CLS]
        50, 60, 70, 80,                 // User request
        delimiters.sep_id,              // [SEP]
        delimiters.cand_marker_id, 101, // Tool 1 (Bash)
        delimiters.cand_marker_id, 102, // Tool 2 (SQL)
        delimiters.cand_marker_id, 103, // Tool 3 (Python)
        delimiters.eos_id,              // [EOS]
    ];

    let verdict = engine.evaluate(&tokens, &device).expect("Evaluation must succeed");

    // Choice was evaluated dynamically
    assert!(verdict.choice.is_some());
    let choice = verdict.choice.unwrap();
    assert_eq!(choice.probabilities.len(), 3);
    assert!((choice.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    assert!(choice.selected_candidate < 3);

    // No Noul or Score markers were present
    assert!(verdict.noul.is_none());
    assert!(verdict.score.is_none());
    assert_eq!(verdict.choices.len(), 1);
    assert!(verdict.nouls.is_empty());
    assert!(verdict.scores.is_empty());

    // Metric embedding is valid and normalized
    assert_eq!(verdict.embedding.as_slice().len(), KNN_EMBEDDING_DIM);
    let norm: f32 = verdict.embedding.as_slice().iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3);
}

#[test]
fn test_general_purpose_reflex_engine_multi_questions() {
    let (engine, device, delimiters) = init_test_engine();

    // Prompt layout with:
    // - 2 distinct Choice questions:
    //   Q1: 2 candidates (Action options)
    //   Q2: 3 candidates (Routing options)
    // - 2 Noul assertion questions (Boolean checks)
    // - 2 Rubric score questions (Ordinal scales 1-5)
    let tokens = vec![
        delimiters.cls_id,              // [CLS]
        10, 20, 30,                     // Context
        delimiters.sep_id,              // [SEP]
        delimiters.choice_query_marker_id, 100, // Choice Q1
        delimiters.cand_marker_id, 101, // Opt 1
        delimiters.cand_marker_id, 102, // Opt 2
        delimiters.choice_query_marker_id, 200, // Choice Q2
        delimiters.cand_marker_id, 201, // Opt A
        delimiters.cand_marker_id, 202, // Opt B
        delimiters.cand_marker_id, 203, // Opt C
        delimiters.noul_query_marker_id, 301,  // Noul Q1
        delimiters.noul_query_marker_id, 302,  // Noul Q2
        delimiters.score_query_marker_id, 401, // Score Q1
        delimiters.score_query_marker_id, 402, // Score Q2
        delimiters.eos_id,              // [EOS]
    ];

    let verdict = engine.evaluate(&tokens, &device).expect("Multi-question evaluation must succeed");

    // 1. Multiple Choices Verification
    assert_eq!(verdict.choices.len(), 2);
    assert_eq!(verdict.choices[0].probabilities.len(), 2);
    assert_eq!(verdict.choices[1].probabilities.len(), 3);
    assert!((verdict.choices[0].probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    assert!((verdict.choices[1].probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    // Legacy choice accessor matches first question
    assert_eq!(verdict.choice, Some(verdict.choices[0].clone()));

    // 2. Multiple Nouls Verification
    assert_eq!(verdict.nouls.len(), 2);
    for noul in &verdict.nouls {
        assert!(noul.probability >= 0.0 && noul.probability <= 1.0);
    }
    // Legacy noul accessor matches first question
    assert_eq!(verdict.noul, Some(verdict.nouls[0].clone()));

    // 3. Multiple Scores Verification
    assert_eq!(verdict.scores.len(), 2);
    for score in &verdict.scores {
        assert!(score.expected_score >= 1.0 && score.expected_score <= 5.0);
        assert_eq!(score.cumulative_probs.len(), 4);
    }
    // Legacy score accessor matches first question
    assert_eq!(verdict.score, Some(verdict.scores[0].clone()));

    // 4. Metric Embedding is unchanged
    assert_eq!(verdict.embedding.as_slice().len(), KNN_EMBEDDING_DIM);
}

#[test]
fn test_security_router_fast_path_and_auto_record() {
    let (engine, device, delimiters) = init_test_engine();
    let index = Arc::new(InMemoryRingIndex::new());
    let registry = Arc::new(TenantRegistry::new(index));

    let tenant_id = "tenant_agent_production";
    let tenant_cfg = TenantConfig::new(tenant_id, 0.95, 3).unwrap();
    registry.register_tenant(tenant_cfg).unwrap();

    let tokens = vec![
        delimiters.cls_id,
        10, 20, 30,
        delimiters.sep_id,
        delimiters.cand_marker_id, 101,
        delimiters.cand_marker_id, 102,
        delimiters.noul_query_marker_id,
        delimiters.eos_id,
    ];

    // Seed global anchor with a vector extracted from the model on this sequence
    let baseline_verdict = engine.evaluate(&tokens, &device).unwrap();
    registry
        .set_global_anchors(&[baseline_verdict.embedding.clone()])
        .unwrap();

    let router = ReflexSecurityRouter::new(engine, registry.clone())
        .with_threat_threshold(0.90)
        .with_auto_record_benign(true);

    assert_eq!(registry.tenant_sample_count(tenant_id), 0);

    // Route benign request
    let routing = router.route(tenant_id, &tokens, &device).unwrap();
    assert!(!routing.is_escalated());

    match routing {
        Tier1Routing::FastPathPass { verdict, anomaly_verdict } => {
            assert!(verdict.choice.is_some());
            assert!(verdict.noul.is_some());
            assert!(anomaly_verdict.is_some());
            assert!(!anomaly_verdict.unwrap().is_anomaly);
        }
        Tier1Routing::EscalateToTier2 { .. } => panic!("Expected FastPathPass"),
    }

    // Auto-record should have added 1 sample into tenant history
    assert_eq!(registry.tenant_sample_count(tenant_id), 1);
}

#[test]
fn test_security_router_anomaly_escalation() {
    let (engine, device, delimiters) = init_test_engine();
    let index = Arc::new(InMemoryRingIndex::new());
    let registry = Arc::new(TenantRegistry::new(index));

    let tenant_id = "tenant_locked_down";
    // Set a very strict outlier threshold (0.01) to force anomaly trigger
    let tenant_cfg = TenantConfig::new(tenant_id, 0.01, 1).unwrap();
    registry.register_tenant(tenant_cfg).unwrap();

    // Populate global anchor with an orthogonal dummy vector
    let mut orthogonal_raw = vec![0.0f32; KNN_EMBEDDING_DIM];
    orthogonal_raw[150] = 1.0;
    registry
        .set_global_anchors(&[AnomalyEmbedding::new(orthogonal_raw).unwrap()])
        .unwrap();

    let router = ReflexSecurityRouter::new(engine, registry)
        .with_threat_threshold(0.99); // high threat threshold so only anomaly fires

    let tokens = vec![
        delimiters.cls_id,
        15, 25, 35,
        delimiters.eos_id,
    ];

    let routing = router.route(tenant_id, &tokens, &device).unwrap();
    assert!(routing.is_escalated());

    match routing {
        Tier1Routing::EscalateToTier2 { reason, raw_prompt_tokens, .. } => {
            assert_eq!(raw_prompt_tokens, tokens);
            assert!(matches!(reason, EscalationReason::OutlierAnomaly { .. }));
        }
        Tier1Routing::FastPathPass { .. } => panic!("Expected EscalateToTier2"),
    }
}
