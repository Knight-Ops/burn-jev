mod common;

use std::path::PathBuf;

use burn_flex::FlexDevice;
use burn_mamba::{
    load_artifact, read_artifact_metadata, save_artifact, ArtifactMetadata, DecisionModelConfig, Head, ItemKind,
    ItemReaderConfig, LoadError, ScenarioFeatures, UnifiedHeadsConfig,
};
use common::{tiny_encoding, TestBackend};

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("burn_mamba_artifact_{}_{name}.safetensors", std::process::id()))
}

fn config() -> DecisionModelConfig {
    DecisionModelConfig::with_reader(ItemReaderConfig::new(16).with_d_reader(8).with_n_heads(2), UnifiedHeadsConfig::new(8))
}

fn probe() -> ScenarioFeatures {
    ScenarioFeatures {
        d_model: 16,
        ctx: (0..4 * 16).map(|i| (i as f32).sin()).collect(),
        ctx_len: 4,
        items: (0..3 * 16).map(|i| (i as f32 * 0.5).cos()).collect(),
        kinds: vec![
            ItemKind::Choice { question: 0, candidate: 0 },
            ItemKind::Choice { question: 0, candidate: 1 },
            ItemKind::Noul { index: 0 },
        ],
    }
}

#[test]
fn roundtrip_preserves_decisions_and_metadata() {
    let device = FlexDevice;
    let mut model = config().init::<TestBackend>(&device);
    model.heads.set_temperatures(1.5, 0.7, 2.5);
    // Params are lazily initialized; a forward pass materializes them, as training does
    // (otherwise the clone serialized below would draw its own random values).
    let before = model.decide(&probe(), &device).unwrap();
    let path = temp_path("roundtrip");
    let mut meta = ArtifactMetadata::new(config(), tiny_encoding(), "enc-sha".into(), "tok-sha".into());
    meta.best_epoch = Some(7);
    save_artifact(&model, &meta, &path).unwrap();

    let read = read_artifact_metadata(&path).unwrap();
    assert_eq!(read.best_epoch, Some(7));
    assert_eq!(read.encoding, tiny_encoding());
    assert_eq!(read.model.reader, config().reader);

    let (loaded, _) = load_artifact::<TestBackend, _>(&path, "enc-sha", "tok-sha", &device).unwrap();
    let temps = Head::ALL.map(|h| loaded.heads.temperature_value(h));
    assert_eq!(temps, [1.5, 0.7, 2.5]);
    let after = loaded.decide(&probe(), &device).unwrap();
    assert_eq!(before.choices, after.choices);
    assert_eq!(before.nouls, after.nouls);
    let _ = std::fs::remove_file(path);
}

#[test]
fn refuses_other_encoder_or_tokenizer() {
    let device = FlexDevice;
    let path = temp_path("mismatch");
    let meta = ArtifactMetadata::new(config(), tiny_encoding(), "enc-sha".into(), "tok-sha".into());
    save_artifact(&config().init::<TestBackend>(&device), &meta, &path).unwrap();

    let err = load_artifact::<TestBackend, _>(&path, "other-enc", "tok-sha", &device).unwrap_err();
    assert!(matches!(err, LoadError::EncoderMismatch { what: "encoder", .. }), "{err}");
    let err = load_artifact::<TestBackend, _>(&path, "enc-sha", "other-tok", &device).unwrap_err();
    assert!(matches!(err, LoadError::EncoderMismatch { what: "tokenizer", .. }), "{err}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn rejects_files_that_are_not_artifacts() {
    let path = temp_path("not_artifact");
    let data = vec![0u8; 16];
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4], &data).unwrap();
    std::fs::write(&path, safetensors::serialize([("w", view)], None).unwrap()).unwrap();
    let err = read_artifact_metadata(&path).unwrap_err();
    assert!(matches!(err, LoadError::InvalidConfiguration(_)), "{err}");
    let _ = std::fs::remove_file(path);
}
