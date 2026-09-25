use std::path::PathBuf;

use burn::tensor::Tensor;
use burn_flex::{Flex, FlexDevice};
use burn_mamba::{
    BiMamba2Backbone, BiMamba2Config, DelimiterConfig, FeatureCache, HeadsMetadata, LoadError,
    Mamba2CheckpointLoader, UnifiedHeads,
};

type TestBackend = Flex<f32, i32>;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("burn_mamba_{}_{name}", std::process::id()))
}

fn tiny_config() -> BiMamba2Config {
    BiMamba2Config::new(64)
        .with_d_model(32)
        .with_n_layers(1)
        .with_d_state(16)
        .with_headdim(16)
        .with_expand(2)
        .with_knn_dim(16)
        .with_num_rubric_bins(4)
}

/// Writes a small random backbone checkpoint and returns (path, sha256).
fn write_backbone(name: &str) -> (PathBuf, String) {
    let model: BiMamba2Backbone<TestBackend> = tiny_config().init(&FlexDevice);
    let path = temp_path(name);
    model.save_safetensors_file(&path).unwrap();
    let sha = FeatureCache::compute_file_hash(&path).unwrap();
    (path, sha)
}

fn outputs(heads: &UnifiedHeads<TestBackend>, x: Tensor<TestBackend, 2>) -> Vec<Vec<f32>> {
    let to_vec = |t: Tensor<TestBackend, 1>| t.into_data().to_vec::<f32>().unwrap();
    vec![
        heads.extract_knn_embedding(x.clone()).into_data().to_vec::<f32>().unwrap(),
        to_vec(heads.forward_choice_logits(x.clone())),
        to_vec(heads.forward_noul_logits(x.clone())),
        heads.forward_score_logits(x).into_data().to_vec::<f32>().unwrap(),
    ]
}

#[test]
fn test_heads_artifact_roundtrip() {
    let device = FlexDevice;
    let (backbone_path, sha) = write_backbone("roundtrip_backbone.safetensors");
    let loaded = Mamba2CheckpointLoader::load_backbone_file::<TestBackend, _>(&backbone_path, &device).unwrap();
    assert_eq!(loaded.sha256, sha);
    assert!(!loaded.report.heads_loaded, "backbone loading must ignore heads.* tensors");

    let heads_config = tiny_config().heads_config();
    let heads: UnifiedHeads<TestBackend> = heads_config.init(&device);
    let delimiters = DelimiterConfig::mamba2_reserved();
    let metadata = HeadsMetadata::new(&heads_config, delimiters.clone(), sha.clone());

    let heads_path = temp_path("roundtrip_heads.safetensors");
    Mamba2CheckpointLoader::save_heads_file(&heads, &metadata, &heads_path).unwrap();
    let (restored, restored_meta) =
        Mamba2CheckpointLoader::load_heads_file::<TestBackend, _>(&heads_path, &sha, &device).unwrap();
    let read_meta = Mamba2CheckpointLoader::read_heads_metadata(&heads_path).unwrap();
    let artifact_size = heads_path.metadata().unwrap().len();
    let backbone_size = backbone_path.metadata().unwrap().len();
    let _ = std::fs::remove_file(&heads_path);
    let _ = std::fs::remove_file(&backbone_path);

    assert_eq!(restored_meta, metadata);
    assert_eq!(read_meta, metadata);
    assert_eq!(restored_meta.delimiters, delimiters);
    assert_eq!(restored.num_rubric_bins, 4);
    assert!(artifact_size < backbone_size, "heads artifact must not embed the backbone");

    let x = Tensor::<TestBackend, 2>::random([3, 32], burn::tensor::Distribution::Default, &device);
    assert_eq!(outputs(&heads, x.clone()), outputs(&restored, x));
    assert_eq!(heads.temperature_value(), restored.temperature_value());
}

#[test]
fn test_heads_artifact_rejects_mismatched_backbone() {
    let device = FlexDevice;
    let heads_config = tiny_config().heads_config();
    let heads: UnifiedHeads<TestBackend> = heads_config.init(&device);
    let metadata = HeadsMetadata::new(&heads_config, DelimiterConfig::mamba2_reserved(), "a".repeat(64));

    let heads_path = temp_path("mismatch_heads.safetensors");
    Mamba2CheckpointLoader::save_heads_file(&heads, &metadata, &heads_path).unwrap();
    let result =
        Mamba2CheckpointLoader::load_heads_file::<TestBackend, _>(&heads_path, &"b".repeat(64), &device);
    let _ = std::fs::remove_file(&heads_path);

    match result {
        Err(LoadError::BackboneMismatch { expected, found }) => {
            assert_eq!(expected, "a".repeat(64));
            assert_eq!(found, "b".repeat(64));
        }
        other => panic!("expected BackboneMismatch, got {:?}", other.map(|(_, m)| m)),
    }
}

#[test]
fn test_heads_loader_rejects_full_checkpoint() {
    let device = FlexDevice;
    let (backbone_path, sha) = write_backbone("full_as_heads.safetensors");
    let result = Mamba2CheckpointLoader::load_heads_file::<TestBackend, _>(&backbone_path, &sha, &device);
    let _ = std::fs::remove_file(&backbone_path);
    assert!(matches!(result, Err(LoadError::InvalidConfiguration(_))));
}
