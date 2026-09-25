use burn::tensor::Tensor;
use burn_flex::Flex;

use burn_mamba::delimiters::{
    extract_batched_candidates, extract_candidate_states, extract_choice_question_candidates,
    extract_cls_state, extract_noul_state, extract_noul_states, extract_score_state,
    extract_score_states, CoordinateResolver, CoordinateTarget, DelimiterConfig, DelimiterError,
};

type TestBackend = Flex;

#[test]
fn test_delimiter_resolver_nominal() {
    let config = DelimiterConfig::default();
    let resolver = CoordinateResolver::new(config);

    // Format: [CLS] ... <cand> 10 <cand> 20 <cand> 30 [SEP] <noul_q> <score_q> [EOS]
    let tokens = vec![1, 100, 101, 4, 10, 4, 20, 4, 30, 2, 5, 6, 3];
    let coords = resolver.resolve_coordinates(&tokens).unwrap();

    assert_eq!(coords.cls_index, 0);
    assert_eq!(coords.candidate_indices, vec![3, 5, 7]);
    assert_eq!(coords.noul_query_index, Some(10));
    assert_eq!(coords.score_query_index, Some(11));
    assert_eq!(coords.sep_index, Some(9));
    assert_eq!(coords.eos_index, Some(12));
    assert_eq!(coords.seq_len, 13);
    assert_eq!(coords.candidate_count(), 3);
    assert_eq!(coords.choice_question_count(), 1);
    assert_eq!(coords.noul_question_count(), 1);
    assert_eq!(coords.score_query_index, Some(11));
    assert_eq!(coords.score_question_count(), 1);
    assert!(coords.has_noul());
    assert!(coords.has_score());
    assert!(coords.has_choice());
}

#[test]
fn test_delimiter_resolver_optional_queries() {
    let config = DelimiterConfig::default();
    let resolver = CoordinateResolver::new(config);

    // Only candidates, no noul or score queries
    let tokens = vec![1, 100, 4, 10, 4, 20, 3];
    let coords = resolver.resolve_coordinates(&tokens).unwrap();

    assert_eq!(coords.cls_index, 0);
    assert_eq!(coords.candidate_indices, vec![2, 4]);
    assert_eq!(coords.noul_query_index, None);
    assert_eq!(coords.score_query_index, None);
    assert_eq!(coords.choice_questions.len(), 1);
    assert_eq!(coords.noul_query_indices.len(), 0);
    assert_eq!(coords.score_query_indices.len(), 0);
    assert!(!coords.has_noul());
    assert!(!coords.has_score());
    assert!(coords.has_choice());
}

#[test]
fn test_delimiter_resolver_target_mode_next_token() {
    let config = DelimiterConfig::default().with_target_mode(CoordinateTarget::NextToken);
    let resolver = CoordinateResolver::new(config);

    // Format: [CLS] ... <cand> 10 <cand> 20 [SEP] <noul_q> 50 [EOS]
    let tokens = vec![1, 100, 4, 10, 4, 20, 2, 5, 50, 3];
    let coords = resolver.resolve_coordinates(&tokens).unwrap();

    // Slicing should target the token *following* the delimiter
    assert_eq!(coords.candidate_indices, vec![3, 5]); // indices of 10 and 20
    assert_eq!(coords.noul_query_index, Some(8)); // index of 50
}

#[test]
fn test_delimiter_resolver_max_candidates_limit() {
    let config = DelimiterConfig::default().with_max_candidates(2);
    let resolver = CoordinateResolver::new(config);

    let tokens = vec![1, 4, 10, 4, 20, 4, 30, 3];
    let result = resolver.resolve_coordinates(&tokens);

    assert_eq!(
        result.unwrap_err(),
        DelimiterError::TooManyCandidates { count: 3, max: 2 }
    );
}

#[test]
fn test_delimiter_resolver_error_variants() {
    let config = DelimiterConfig::default();
    let resolver = CoordinateResolver::new(config);

    // 1. Empty sequence
    assert_eq!(
        resolver.resolve_coordinates(&[]).unwrap_err(),
        DelimiterError::EmptySequence
    );

    // 2. Missing CLS token
    assert_eq!(
        resolver.resolve_coordinates(&[4, 10, 4, 20]).unwrap_err(),
        DelimiterError::MissingClsToken
    );

    // 3. Exceeding max noul queries
    let noul_lim_resolver = CoordinateResolver::new(
        DelimiterConfig::default().with_max_noul_questions(1),
    );
    assert_eq!(
        noul_lim_resolver
            .resolve_coordinates(&[1, 5, 20, 5, 30])
            .unwrap_err(),
        DelimiterError::TooManyNoulQuestions {
            count: 2,
            max: 1
        }
    );

    // 4. Exceeding max score queries
    let score_lim_resolver = CoordinateResolver::new(
        DelimiterConfig::default().with_max_score_questions(1),
    );
    assert_eq!(
        score_lim_resolver
            .resolve_coordinates(&[1, 6, 20, 6, 30])
            .unwrap_err(),
        DelimiterError::TooManyScoreQuestions {
            count: 2,
            max: 1
        }
    );

    // 5. Exceeding max choice questions
    let choice_lim_resolver = CoordinateResolver::new(
        DelimiterConfig::default().with_max_choice_questions(1),
    );
    assert_eq!(
        choice_lim_resolver
            .resolve_coordinates(&[1, 7, 4, 10, 7, 4, 20])
            .unwrap_err(),
        DelimiterError::TooManyChoiceQuestions {
            count: 2,
            max: 1
        }
    );

    // 6. Index out of bounds with NextToken mode
    let next_resolver = CoordinateResolver::new(
        DelimiterConfig::default().with_target_mode(CoordinateTarget::NextToken),
    );
    // Trailing CAND marker has no next token
    assert_eq!(
        next_resolver.resolve_coordinates(&[1, 10, 4]).unwrap_err(),
        DelimiterError::IndexOutOfBounds {
            index: 3,
            seq_len: 3
        }
    );
}

#[test]
fn test_delimiter_resolver_multi_questions() {
    let config = DelimiterConfig::default();
    let resolver = CoordinateResolver::new(config);

    // Tokens layout:
    // [CLS] 100
    // <choice_q> 110 <cand> 111 <cand> 112
    // <choice_q> 120 <cand> 121 <cand> 122 <cand> 123
    // <noul_q> 130 <noul_q> 131 <noul_q> 132
    // <score_q> 140 <score_q> 141
    // [EOS]
    let tokens = vec![
        1, 100, // [CLS]
        7, 110, 4, 111, 4, 112, // Choice Q1 with 2 candidates (indices 4, 6)
        7, 120, 4, 121, 4, 122, 4, 123, // Choice Q2 with 3 candidates (indices 10, 12, 14)
        5, 130, 5, 131, 5, 132, // 3 Noul queries (indices 16, 18, 20)
        6, 140, 6, 141, // 2 Score queries (indices 22, 24)
        3, // [EOS]
    ];

    let coords = resolver.resolve_coordinates(&tokens).unwrap();
    assert_eq!(coords.cls_index, 0);
    assert_eq!(coords.choice_question_count(), 2);
    assert_eq!(coords.choice_questions[0].candidate_indices, vec![4, 6]);
    assert_eq!(coords.choice_questions[1].candidate_indices, vec![10, 12, 14]);
    assert_eq!(coords.candidate_indices, vec![4, 6, 10, 12, 14]);

    assert_eq!(coords.noul_question_count(), 3);
    assert_eq!(coords.noul_query_indices, vec![16, 18, 20]);
    assert_eq!(coords.noul_query_index, Some(16));

    assert_eq!(coords.score_question_count(), 2);
    assert_eq!(coords.score_query_indices, vec![22, 24]);
    assert_eq!(coords.score_query_index, Some(22));

    // Dynamic Slicing Verification
    let device = Default::default();
    let d_model = 16;
    let seq_len = tokens.len();
    let mut hidden_data = Vec::with_capacity(seq_len * d_model);
    for t in 0..seq_len {
        for d in 0..d_model {
            hidden_data.push((t * 100 + d) as f32);
        }
    }
    let hidden =
        Tensor::<TestBackend, 1>::from_data(hidden_data.as_slice(), &device).reshape([1, seq_len, d_model]);

    // Multi-Noul extraction -> [3, 16]
    let noul_states = extract_noul_states(&hidden, 0, &coords, &device).unwrap();
    assert_eq!(noul_states.dims(), [3, d_model]);

    // Multi-Score extraction -> [2, 16]
    let score_states = extract_score_states(&hidden, 0, &coords, &device).unwrap();
    assert_eq!(score_states.dims(), [2, d_model]);

    // Choice Q1 extraction -> [2, 16]
    let cands_q1 =
        extract_choice_question_candidates(&hidden, 0, &coords.choice_questions[0], &device).unwrap();
    assert_eq!(cands_q1.dims(), [2, d_model]);

    // Choice Q2 extraction -> [3, 16]
    let cands_q2 =
        extract_choice_question_candidates(&hidden, 0, &coords.choice_questions[1], &device).unwrap();
    assert_eq!(cands_q2.dims(), [3, d_model]);
}

#[test]
fn test_delimiter_resolver_batched() {
    let config = DelimiterConfig::default();
    let resolver = CoordinateResolver::new(config);

    let batch = vec![
        vec![1, 10, 4, 20, 4, 30, 3],
        vec![1, 50, 4, 60, 5, 70, 3],
    ];

    let results = resolver.resolve_coordinates_batched(&batch).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].candidate_indices, vec![2, 4]);
    assert_eq!(results[1].candidate_indices, vec![2]);
    assert_eq!(results[1].noul_query_index, Some(4));
}

#[test]
fn test_tensor_slicing_integration() {
    let device = Default::default();
    let config = DelimiterConfig::default();
    let resolver = CoordinateResolver::new(config);

    let tokens = vec![1, 100, 4, 10, 4, 20, 2, 5, 6, 3];
    let coords = resolver.resolve_coordinates(&tokens).unwrap();

    let d_model = 8;
    let seq_len = tokens.len();
    let mut data = Vec::with_capacity(seq_len * d_model);
    for t in 0..seq_len {
        for d in 0..d_model {
            data.push((t * 10 + d) as f32);
        }
    }
    let hidden =
        Tensor::<TestBackend, 1>::from_data(data.as_slice(), &device).reshape([1, seq_len, d_model]);

    // 1. CLS extraction
    let cls = extract_cls_state(&hidden, 0, &coords);
    assert_eq!(cls.dims(), [1, d_model]);

    // 2. Candidate extraction
    let cands = extract_candidate_states(&hidden, 0, &coords, &device).unwrap();
    assert_eq!(cands.dims(), [2, d_model]);

    // 3. Noul extraction
    let noul = extract_noul_state(&hidden, 0, &coords).unwrap();
    assert_eq!(noul.dims(), [1, d_model]);

    // 4. Score extraction
    let score = extract_score_state(&hidden, 0, &coords).unwrap();
    assert_eq!(score.dims(), [1, d_model]);
}

#[test]
fn test_batched_candidates_with_padding() {
    let device = Default::default();
    let config = DelimiterConfig::default();
    let resolver = CoordinateResolver::new(config);

    // Sequence 0: 2 candidates
    let tokens0 = vec![1, 10, 4, 20, 4, 30, 3];
    // Sequence 1: 3 candidates
    let tokens1 = vec![1, 10, 4, 20, 4, 30, 4, 40, 3];

    let coords0 = resolver.resolve_coordinates(&tokens0).unwrap();
    let coords1 = resolver.resolve_coordinates(&tokens1).unwrap();
    let coords_batch = vec![coords0, coords1];

    let d_model = 4;
    let max_len = 8;
    let hidden = Tensor::<TestBackend, 3>::zeros([2, max_len, d_model], &device);

    let (batched_cands, mask) =
        extract_batched_candidates(&hidden, &coords_batch, &device).unwrap();

    // Batch size 2, max_K 3, d_model 4
    assert_eq!(batched_cands.dims(), [2, 3, d_model]);
    assert_eq!(mask.dims(), [2, 3]);

    let mask_data = mask.into_data();
    let slice = mask_data.as_slice::<bool>().unwrap();
    // Seq 0: true, true, false
    assert_eq!(slice[0..3], [true, true, false]);
    // Seq 1: true, true, true
    assert_eq!(slice[3..6], [true, true, true]);
}
