use burn_jev::{
    ChoiceQuestionRecord, EncodingConfig, JevDataset, JevScenarioRecord, NoulQueryRecord,
    ScoreRubricRecord,
};
use tokenizers::Tokenizer;

#[test]
fn test_jsonl_parse_and_validate() {
    let raw_jsonl = r#"
    {"id":"scen_01","domain":"devops","context":"Server memory 99%","is_benign":true,"choice_questions":[{"prompt":"Action:","candidates":["Drain","Kill"],"target":0}],"noul_queries":[{"assertion":"Is root?","target":1.0}],"score_rubrics":[{"prompt":"Severity","target":4.5}]}
    {"id":"scen_02","domain":"sec","context":"SQL inject attack","is_benign":false,"choice_questions":[{"prompt":"Action:","candidates":["Block","Pass"],"target":0}],"noul_queries":[{"assertion":"Is attack?","target":1.0}],"score_rubrics":[{"prompt":"Threat","target":5.0}]}
    "#;

    let dataset = JevDataset::from_jsonl_str(raw_jsonl).expect("Valid JSONL parse");
    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.records[0].id, "scen_01");
    assert_eq!(dataset.records[1].id, "scen_02");
    assert_eq!(dataset.records[0].choice_questions[0].target, 0);
    assert_eq!(dataset.records[0].noul_queries[0].target, 1.0);
    assert_eq!(dataset.records[0].score_rubrics[0].target, 4.5);
}

#[test]
fn test_jsonl_validation_errors() {
    // Missing id
    let bad_id = r#"{"id":"","context":"telemetry","choice_questions":[{"prompt":"Q","candidates":["A","B"],"target":0}]}"#;
    assert!(JevDataset::from_jsonl_str(bad_id).is_err());

    // Too few candidates (<2)
    let bad_cands = r#"{"id":"1","context":"telemetry","choice_questions":[{"prompt":"Q","candidates":["A"],"target":0}]}"#;
    assert!(JevDataset::from_jsonl_str(bad_cands).is_err());

    // Target out of range
    let bad_target = r#"{"id":"1","context":"telemetry","choice_questions":[{"prompt":"Q","candidates":["A","B"],"target":3}]}"#;
    assert!(JevDataset::from_jsonl_str(bad_target).is_err());

    // Noul target out of [0, 1]
    let bad_noul = r#"{"id":"1","context":"telemetry","noul_queries":[{"assertion":"A","target":1.5}]}"#;
    assert!(JevDataset::from_jsonl_str(bad_noul).is_err());

    // Score target < 1.0
    let bad_score = r#"{"id":"1","context":"telemetry","score_rubrics":[{"prompt":"S","target":0.5}]}"#;
    assert!(JevDataset::from_jsonl_str(bad_score).is_err());
}

#[test]
fn test_dataset_split() {
    let mut dataset = JevDataset::new();
    for i in 0..10 {
        dataset.add(JevScenarioRecord {
            id: format!("scen_{i}"),
            domain: Some("test".into()),
            context: format!("Context {i}"),
            is_benign: true,
            choice_questions: vec![ChoiceQuestionRecord {
                prompt: "P".into(),
                candidates: vec!["A".into(), "B".into()],
                target: 0,
            }],
            noul_queries: vec![NoulQueryRecord {
                assertion: "N".into(),
                target: 1.0,
            }],
            score_rubrics: vec![ScoreRubricRecord {
                prompt: "S".into(),
                target: 3.0,
            }],
        });
    }

    let (train, val) = dataset.split(0.2); // 20% val
    assert_eq!(train.len(), 8);
    assert_eq!(val.len(), 2);
    assert_eq!(train.records[0].id, "scen_0");
    assert_eq!(val.records[0].id, "scen_8");
}

#[test]
fn test_dataset_encode_with_tokenizer() {
    let tokenizer_path = std::path::Path::new("models/modernbert-base/tokenizer.json");
    if !tokenizer_path.exists() {
        return;
    }
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let cfg = EncodingConfig::default();

    let record = JevScenarioRecord {
        id: "test_encode".into(),
        domain: Some("sysadmin".into()),
        context: "Cluster telemetry shows CPU load average at 12.4 on 8-core host.".into(),
        is_benign: true,
        choice_questions: vec![ChoiceQuestionRecord {
            prompt: "Select mitigation:".into(),
            candidates: vec![
                "Throttle batch worker background jobs".into(),
                "Reboot node immediately".into(),
            ],
            target: 0,
        }],
        noul_queries: vec![NoulQueryRecord {
            assertion: "Is host CPU overloaded?".into(),
            target: 1.0,
        }],
        score_rubrics: vec![ScoreRubricRecord {
            prompt: "Rate incident severity:".into(),
            target: 3.8,
        }],
    };

    let encoded = record.encode(&tokenizer, &cfg).unwrap();
    assert_eq!(encoded.id, "test_encode");
    assert_eq!(encoded.encoded.input_ids[0], cfg.cls_id);
    assert_eq!(*encoded.encoded.input_ids.last().unwrap(), cfg.sep_id);
    assert!(!encoded.encoded.context.is_empty());
    // One item per candidate, noul and rubric of the record.
    let record_items = record.choice_questions.iter().map(|q| q.candidates.len()).sum::<usize>()
        + record.noul_queries.len()
        + record.score_rubrics.len();
    assert_eq!(encoded.encoded.items.len(), record_items);
    assert_eq!(encoded.targets.choice_targets, vec![0]);
    assert_eq!(encoded.targets.noul_targets, vec![1.0]);
    assert_eq!(encoded.targets.score_targets, vec![3.8]);
}
