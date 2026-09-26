//! Shared fixtures: a hermetic word-level tokenizer and a tiny random ModernBERT, so tests
//! need no downloaded weights.
#![allow(dead_code)]

use std::collections::HashMap;

use burn_flex::Flex;
use burn_jev::{EncodingConfig, ModernBertConfig};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

pub type TestBackend = Flex<f32, i32>;

pub const WORDS: &[&str] = &[
    "[UNK]", "the", "agent", "requests", "a", "tool", "bash", "sql", "python", "run", "query",
    "is", "this", "malicious", "benign", "rate", "risk", "from", "1", "to", "5", "login",
    "failed", "user", "admin", "deploy", "build", "select", "action", "block", "allow", "alert",
];
pub const CLS: i64 = 60;
pub const SEP: i64 = 61;
pub const PAD: i64 = 62;

pub fn tiny_tokenizer() -> Tokenizer {
    let vocab: HashMap<String, u32> = WORDS.iter().enumerate().map(|(i, w)| (w.to_string(), i as u32)).collect();
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer
}

/// 3 layers so both global (layer 0) and local (layers 1, 2) attention run; a 4-token
/// window so the local mask actually excludes keys in short test sequences.
pub fn tiny_encoder_config() -> ModernBertConfig {
    ModernBertConfig::new(64, 32, 3, 2, 48)
        .with_local_attention(4)
        .with_max_position_embeddings(256)
        .with_cls_token_id(CLS)
        .with_sep_token_id(SEP)
        .with_pad_token_id(PAD)
}

pub fn tiny_encoding() -> EncodingConfig {
    EncodingConfig::for_encoder(&tiny_encoder_config()).with_max_seq_len(128)
}
