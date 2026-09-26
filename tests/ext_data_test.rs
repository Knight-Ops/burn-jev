//! Imported auxiliary sets (`data/ext/*.jsonl`, from `scripts/import_typed_decisions.py`) must load,
//! validate and encode without context truncation at the default `max_seq_len`.

use burn_jev::{EncodingConfig, JevDataset};
use tokenizers::Tokenizer;

#[test]
fn test_ext_datasets_load_and_encode_untruncated() {
    let tokenizer_path = std::path::Path::new("models/modernbert-base/tokenizer.json");
    let Ok(dir) = std::fs::read_dir("data/ext") else { return };
    if !tokenizer_path.exists() {
        return;
    }
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let cfg = EncodingConfig::default();

    for entry in dir {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let ds = JevDataset::from_jsonl_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(!ds.is_empty(), "{} is empty", path.display());
        for record in &ds.records {
            assert!(record.id.starts_with("ext_"), "{}: id {} lacks the ext_ prefix", path.display(), record.id);
            let enc = record.encode(&tokenizer, &cfg).unwrap_or_else(|e| panic!("{}: {e}", record.id));
            assert!(!enc.encoded.context_truncated, "{}: context truncated at {} tokens", record.id, cfg.max_seq_len);
        }
    }
}
