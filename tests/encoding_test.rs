mod common;

use burn_mamba::{ChoiceQuestionRequest, EncodingError, ItemKind, NoulQueryRequest, ReflexRequest, ScoreRubricRequest};
use common::{tiny_encoding, tiny_tokenizer, CLS, SEP};

fn request() -> ReflexRequest {
    ReflexRequest {
        id: Some("r1".into()),
        context: "the agent requests a tool".into(),
        choice_questions: vec![ChoiceQuestionRequest {
            prompt: "select action".into(),
            candidates: vec!["run bash".into(), "query sql".into(), "block".into()],
        }],
        noul_queries: vec![NoulQueryRequest { assertion: "is this malicious".into() }],
        score_rubrics: vec![ScoreRubricRequest { prompt: "rate risk from 1 to 5".into() }],
    }
}

#[test]
fn layout_uses_only_cls_and_sep_and_spans_cover_their_text() {
    let tok = tiny_tokenizer();
    let enc = request().encode(&tok, &tiny_encoding()).unwrap();
    let ids = &enc.input_ids;
    let word = |w: &str| common::WORDS.iter().position(|x| *x == w).unwrap() as i64;

    assert_eq!(ids[0], CLS);
    assert_eq!(*ids.last().unwrap(), SEP);
    assert_eq!(enc.context, 1..6);
    assert_eq!(ids[6], SEP);
    assert!(!enc.context_truncated);
    // No marker tokens: everything except [CLS]/[SEP] is text.
    assert!(ids.iter().all(|&t| t == CLS || t == SEP || t < common::WORDS.len() as i64));

    // Items: 3 candidates (question-major), then noul, then score.
    let kinds: Vec<ItemKind> = enc.items.iter().map(|i| i.kind).collect();
    assert_eq!(
        kinds,
        vec![
            ItemKind::Choice { question: 0, candidate: 0 },
            ItemKind::Choice { question: 0, candidate: 1 },
            ItemKind::Choice { question: 0, candidate: 2 },
            ItemKind::Noul { index: 0 },
            ItemKind::Score { index: 0 },
        ]
    );
    let text = |r: &std::ops::Range<usize>| ids[r.clone()].to_vec();
    assert_eq!(text(&enc.items[0].range), vec![word("run"), word("bash")]);
    assert_eq!(text(&enc.items[2].range), vec![word("block")]);
    assert_eq!(text(enc.items[1].prompt_range.as_ref().unwrap()), vec![word("select"), word("action")]);
    assert_eq!(enc.items[3].prompt_range, None);
    // Each item span is followed by [SEP].
    for item in &enc.items {
        assert_eq!(ids[item.range.end], SEP);
    }
    assert_eq!(enc.choice_candidate_counts(), vec![3]);
}

#[test]
fn only_the_context_is_truncated() {
    let tok = tiny_tokenizer();
    let mut req = request();
    req.context = vec!["user"; 200].join(" ");
    let full = request().encode(&tok, &tiny_encoding()).unwrap();
    let items_len = full.len() - full.context.len() - 2;

    let enc = req.encode(&tok, &tiny_encoding().with_max_seq_len(40)).unwrap();
    assert!(enc.context_truncated);
    assert_eq!(enc.len(), 40);
    assert_eq!(enc.context.len(), 40 - 2 - items_len);
    assert_eq!(enc.items.len(), full.items.len());
}

#[test]
fn rejects_sequences_whose_items_alone_do_not_fit() {
    let tok = tiny_tokenizer();
    let err = request().encode(&tok, &tiny_encoding().with_max_seq_len(10)).unwrap_err();
    assert!(matches!(err, EncodingError::TooLong { .. }), "{err:?}");
}

#[test]
fn empty_item_text_falls_back_to_its_separator() {
    let tok = tiny_tokenizer();
    let mut req = request();
    req.noul_queries[0].assertion = String::new();
    let enc = req.encode(&tok, &tiny_encoding()).unwrap();
    let noul = &enc.items[3];
    assert_eq!(noul.range.len(), 1);
    assert_eq!(enc.input_ids[noul.range.start], SEP);
}

#[test]
fn enforces_limits() {
    let tok = tiny_tokenizer();
    let mut req = request();
    req.choice_questions[0].candidates.clear();
    assert!(matches!(req.encode(&tok, &tiny_encoding()), Err(EncodingError::EmptyCandidates { question: 0 })));

    let mut cfg = tiny_encoding();
    cfg.max_noul_queries = 0;
    assert!(matches!(request().encode(&tok, &cfg), Err(EncodingError::TooMany { .. })));
}
