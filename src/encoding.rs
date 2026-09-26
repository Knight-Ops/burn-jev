//! =====================================================================
//! Span-Based Sequence Encoding for the Tier 1 Reader
//! =====================================================================
//!
//! Lays a scenario out as one bidirectional-encoder input and records the token span of the
//! context and of every decision item:
//!
//! ```text
//! [CLS] context [SEP] (prompt [SEP] (cand [SEP])*)* (noul [SEP])* (score [SEP])*
//! ```
//!
//! Only special tokens the encoder was pretrained with (`[CLS]`, `[SEP]`, `[PAD]`) appear;
//! items are identified by position, never by marker tokens. Features are mean-pooled over an
//! item's own span, so they always cover the text they describe.

use std::ops::Range;

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::model::ModernBertConfig;

/// Bumped whenever the layout below changes; part of feature-cache and artifact identity.
pub const ENCODING_VERSION: u32 = 1;

/// Number of distinct [`ItemKind`]s (size of the reader's type embedding).
pub const NUM_ITEM_TYPES: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodingConfig {
    pub cls_id: i64,
    pub sep_id: i64,
    pub pad_id: i64,
    /// Upper bound on the whole sequence. Only the context is truncated to fit.
    pub max_seq_len: usize,
    pub max_candidates: usize,
    pub max_choice_questions: usize,
    pub max_noul_queries: usize,
    pub max_score_rubrics: usize,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            cls_id: 50281,
            sep_id: 50282,
            pad_id: 50283,
            max_seq_len: 2048,
            max_candidates: crate::model::MAX_CHOICE_CANDIDATES,
            max_choice_questions: 32,
            max_noul_queries: 64,
            max_score_rubrics: 64,
        }
    }
}

impl EncodingConfig {
    /// Special tokens taken from the encoder's own config.
    pub fn for_encoder(config: &ModernBertConfig) -> Self {
        Self {
            cls_id: config.cls_token_id,
            sep_id: config.sep_token_id,
            pad_id: config.pad_token_id,
            max_seq_len: Self::default().max_seq_len.min(config.max_position_embeddings),
            ..Self::default()
        }
    }

    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }
}

/// Which decision an item span feeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ItemKind {
    Choice { question: usize, candidate: usize },
    Noul { index: usize },
    Score { index: usize },
}

impl ItemKind {
    /// Index into the reader's type embedding.
    pub fn type_id(self) -> usize {
        match self {
            ItemKind::Choice { .. } => 0,
            ItemKind::Noul { .. } => 1,
            ItemKind::Score { .. } => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItemSpan {
    pub kind: ItemKind,
    /// Tokens of the item text; the following `[SEP]` if the text tokenized to nothing.
    pub range: Range<usize>,
    /// For candidates: the span of their question prompt.
    pub prompt_range: Option<Range<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodedScenario {
    pub input_ids: Vec<i64>,
    /// Context tokens (excluding `[CLS]`/`[SEP]`); the reader's keys.
    pub context: Range<usize>,
    /// Choice candidates (question-major), then noul queries, then score rubrics.
    pub items: Vec<ItemSpan>,
    pub context_truncated: bool,
}

impl EncodedScenario {
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }

    /// Candidate count of each choice question, in question order.
    pub fn choice_candidate_counts(&self) -> Vec<usize> {
        candidate_counts(self.items.iter().map(|i| i.kind))
    }
}

/// Candidate count per choice question from a question-major item list.
pub fn candidate_counts(kinds: impl IntoIterator<Item = ItemKind>) -> Vec<usize> {
    let mut counts: Vec<usize> = Vec::new();
    for kind in kinds {
        if let ItemKind::Choice { question, .. } = kind {
            if counts.len() <= question {
                counts.resize(question + 1, 0);
            }
            counts[question] += 1;
        }
    }
    counts
}

#[derive(Clone, Debug, PartialEq)]
pub enum EncodingError {
    Tokenizer(String),
    EmptyContext,
    EmptyCandidates { question: usize },
    TooMany { what: &'static str, count: usize, max: usize },
    /// The items alone (plus one context token) exceed `max_seq_len`.
    TooLong { required: usize, max: usize },
}

impl std::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tokenizer(e) => write!(f, "tokenizer error: {e}"),
            Self::EmptyContext => write!(f, "context tokenized to nothing"),
            Self::EmptyCandidates { question } => write!(f, "choice question {question} has no candidates"),
            Self::TooMany { what, count, max } => write!(f, "{count} {what} exceeds the limit of {max}"),
            Self::TooLong { required, max } => write!(
                f,
                "questions need {required} tokens, over max_seq_len {max} even with the context truncated to one token"
            ),
        }
    }
}

impl std::error::Error for EncodingError {}

/// The texts of one scenario, in encoding order.
pub struct ScenarioTexts<'a> {
    pub context: &'a str,
    /// `(prompt, candidates)` per choice question.
    pub choice_questions: Vec<(&'a str, Vec<&'a str>)>,
    pub noul_queries: Vec<&'a str>,
    pub score_rubrics: Vec<&'a str>,
}

pub fn encode_scenario(
    texts: &ScenarioTexts<'_>,
    tokenizer: &Tokenizer,
    cfg: &EncodingConfig,
) -> Result<EncodedScenario, EncodingError> {
    check_limit("choice questions", texts.choice_questions.len(), cfg.max_choice_questions)?;
    check_limit("noul queries", texts.noul_queries.len(), cfg.max_noul_queries)?;
    check_limit("score rubrics", texts.score_rubrics.len(), cfg.max_score_rubrics)?;
    for (q, (_, cands)) in texts.choice_questions.iter().enumerate() {
        if cands.is_empty() {
            return Err(EncodingError::EmptyCandidates { question: q });
        }
        check_limit("candidates", cands.len(), cfg.max_candidates)?;
    }

    let tok = |text: &str| -> Result<Vec<i64>, EncodingError> {
        let enc = tokenizer
            .encode(text, false)
            .map_err(|e| EncodingError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().iter().map(|&id| id as i64).collect())
    };

    let mut context_ids = tok(texts.context)?;
    if context_ids.is_empty() {
        return Err(EncodingError::EmptyContext);
    }
    let choice: Vec<(Vec<i64>, Vec<Vec<i64>>)> = texts
        .choice_questions
        .iter()
        .map(|(prompt, cands)| Ok((tok(prompt)?, cands.iter().map(|c| tok(c)).collect::<Result<_, _>>()?)))
        .collect::<Result<_, EncodingError>>()?;
    let nouls: Vec<Vec<i64>> = texts.noul_queries.iter().map(|t| tok(t)).collect::<Result<_, _>>()?;
    let scores: Vec<Vec<i64>> = texts.score_rubrics.iter().map(|t| tok(t)).collect::<Result<_, _>>()?;

    // Every segment is followed by one [SEP].
    let seg = |ids: &Vec<i64>| ids.len() + 1;
    let item_tokens: usize = choice
        .iter()
        .map(|(p, cands)| seg(p) + cands.iter().map(seg).sum::<usize>())
        .sum::<usize>()
        + nouls.iter().map(seg).sum::<usize>()
        + scores.iter().map(seg).sum::<usize>();
    let fixed = 2; // [CLS] + [SEP] after the context
    if fixed + item_tokens + 1 > cfg.max_seq_len {
        return Err(EncodingError::TooLong {
            required: fixed + item_tokens + 1,
            max: cfg.max_seq_len,
        });
    }
    let budget = cfg.max_seq_len - fixed - item_tokens;
    let context_truncated = context_ids.len() > budget;
    context_ids.truncate(budget);

    let mut ids = Vec::with_capacity(fixed + context_ids.len() + item_tokens);
    ids.push(cfg.cls_id);
    let context = push_segment(&mut ids, &context_ids, cfg.sep_id);
    let mut items = Vec::new();

    for (q, (prompt, cands)) in choice.iter().enumerate() {
        let prompt_range = push_segment(&mut ids, prompt, cfg.sep_id);
        for (c, cand) in cands.iter().enumerate() {
            items.push(ItemSpan {
                kind: ItemKind::Choice { question: q, candidate: c },
                range: push_segment(&mut ids, cand, cfg.sep_id),
                prompt_range: Some(prompt_range.clone()),
            });
        }
    }
    for (index, noul) in nouls.iter().enumerate() {
        items.push(ItemSpan {
            kind: ItemKind::Noul { index },
            range: push_segment(&mut ids, noul, cfg.sep_id),
            prompt_range: None,
        });
    }
    for (index, score) in scores.iter().enumerate() {
        items.push(ItemSpan {
            kind: ItemKind::Score { index },
            range: push_segment(&mut ids, score, cfg.sep_id),
            prompt_range: None,
        });
    }

    Ok(EncodedScenario {
        input_ids: ids,
        context,
        items,
        context_truncated,
    })
}

/// Appends `segment` then `[SEP]`; returns the segment's span, or the `[SEP]`'s if empty.
fn push_segment(ids: &mut Vec<i64>, segment: &[i64], sep_id: i64) -> Range<usize> {
    let start = ids.len();
    ids.extend_from_slice(segment);
    ids.push(sep_id);
    if segment.is_empty() {
        start..start + 1
    } else {
        start..start + segment.len()
    }
}

fn check_limit(what: &'static str, count: usize, max: usize) -> Result<(), EncodingError> {
    if count > max {
        return Err(EncodingError::TooMany { what, count, max });
    }
    Ok(())
}
