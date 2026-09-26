mod common;

use std::sync::Arc;

use burn_flex::FlexDevice;
use burn_mamba::{
    AnomalyEmbedding, ChoiceQuestionRequest, DecisionModelConfig, EscalationReason, InMemoryRingIndex,
    NoulQueryRequest, ReflexEngine, ReflexRequest, ReflexSecurityRouter, ScoreRubricRequest, TenantConfig,
    TenantRegistry, Tier1Routing, KNN_EMBEDDING_DIM,
};
use common::{tiny_encoder_config, tiny_encoding, tiny_tokenizer, TestBackend};

fn engine() -> ReflexEngine<TestBackend> {
    let device = FlexDevice;
    let cfg = tiny_encoder_config();
    let encoder = cfg.init(&device);
    let decision = DecisionModelConfig::for_encoder(cfg.hidden_size).init(&device);
    ReflexEngine::new(encoder, decision, tiny_encoding(), tiny_tokenizer())
}

fn tool_request() -> ReflexRequest {
    ReflexRequest {
        id: Some("tools".into()),
        context: "the agent requests a tool".into(),
        choice_questions: vec![ChoiceQuestionRequest {
            prompt: "select tool".into(),
            candidates: vec!["bash".into(), "sql".into(), "python".into()],
        }],
        noul_queries: vec![],
        score_rubrics: vec![],
    }
}

#[test]
fn engine_selects_among_candidates() {
    let verdict = engine().evaluate(&tool_request(), &FlexDevice).unwrap();
    let choice = verdict.choice.as_ref().unwrap();
    assert_eq!(choice.probabilities.len(), 3);
    assert!((choice.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    assert!(verdict.nouls.is_empty() && verdict.scores.is_empty());
    assert_eq!(verdict.embedding.as_slice().len(), KNN_EMBEDDING_DIM);
    let norm: f32 = verdict.embedding.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3);
    assert_eq!(verdict.encoding.context_tokens, 5);
}

#[test]
fn engine_answers_every_question() {
    let mut req = tool_request();
    req.choice_questions.push(ChoiceQuestionRequest {
        prompt: "select action".into(),
        candidates: vec!["allow".into(), "block".into()],
    });
    req.noul_queries = vec![
        NoulQueryRequest { assertion: "is this malicious".into() },
        NoulQueryRequest { assertion: "is this benign".into() },
    ];
    req.score_rubrics = vec![ScoreRubricRequest { prompt: "rate risk from 1 to 5".into() }];

    let verdict = engine().evaluate(&req, &FlexDevice).unwrap();
    assert_eq!(verdict.choices.len(), 2);
    assert_eq!(verdict.choices[1].probabilities.len(), 2);
    assert_eq!(verdict.nouls.len(), 2);
    assert_eq!(verdict.scores.len(), 1);
    let s = verdict.scores[0].expected_score;
    assert!((1.0..=5.0).contains(&s));
}

#[test]
fn router_fast_path_records_benign_traffic() {
    let engine = engine();
    let registry = Arc::new(TenantRegistry::new(Arc::new(InMemoryRingIndex::new())));
    let tenant = "tenant_agent_production";
    registry.register_tenant(TenantConfig::new(tenant, 0.95, 3).unwrap()).unwrap();

    let mut req = tool_request();
    req.noul_queries = vec![NoulQueryRequest { assertion: "is this malicious".into() }];
    let baseline = engine.evaluate(&req, &FlexDevice).unwrap();
    registry.set_global_anchors(&[baseline.embedding.clone()]).unwrap();

    let router = ReflexSecurityRouter::new(engine, registry.clone())
        .with_threat_threshold(0.99)
        .with_auto_record_benign(true);
    let routing = router.route(tenant, &req, &FlexDevice).unwrap();
    match routing {
        Tier1Routing::FastPathPass { verdict, anomaly_verdict } => {
            assert!(verdict.noul.is_some());
            assert!(!anomaly_verdict.unwrap().is_anomaly);
        }
        Tier1Routing::EscalateToTier2 { .. } => panic!("expected FastPathPass"),
    }
    assert_eq!(registry.tenant_sample_count(tenant), 1);
}

#[test]
fn router_escalates_outliers_with_the_request() {
    let registry = Arc::new(TenantRegistry::new(Arc::new(InMemoryRingIndex::new())));
    let tenant = "tenant_locked_down";
    registry.register_tenant(TenantConfig::new(tenant, 0.01, 1).unwrap()).unwrap();
    let mut orthogonal = vec![0.0f32; KNN_EMBEDDING_DIM];
    orthogonal[150] = 1.0;
    registry.set_global_anchors(&[AnomalyEmbedding::new(orthogonal).unwrap()]).unwrap();

    let router = ReflexSecurityRouter::new(engine(), registry).with_threat_threshold(0.99);
    let req = tool_request();
    match router.route(tenant, &req, &FlexDevice).unwrap() {
        Tier1Routing::EscalateToTier2 { reason, request, .. } => {
            assert_eq!(request, req);
            assert!(matches!(reason, EscalationReason::OutlierAnomaly { .. }));
        }
        Tier1Routing::FastPathPass { .. } => panic!("expected EscalateToTier2"),
    }
}
