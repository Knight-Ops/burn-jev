use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_mamba::{CachedScenario, FeatureCache, MultiQuestionTargets};
use std::fs;
use std::path::PathBuf;

type Backend = Flex<f32, i32>;

#[test]
fn test_feature_cache_roundtrip() {
    let device = FlexDevice;
    let temp_dir = PathBuf::from("target/test_cache_roundtrip");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let d_model = 64;
    let cls_tensor = Tensor::<Backend, 2>::from_data(
        TensorData::new(vec![0.5f32; d_model], vec![1, d_model]),
        &device,
    );
    let choice_1 = Tensor::<Backend, 2>::from_data(
        TensorData::new(vec![0.1f32; 3 * d_model], vec![3, d_model]),
        &device,
    );
    let choice_2 = Tensor::<Backend, 2>::from_data(
        TensorData::new(vec![0.2f32; 2 * d_model], vec![2, d_model]),
        &device,
    );
    let noul_tensor = Tensor::<Backend, 2>::from_data(
        TensorData::new(vec![0.8f32; 2 * d_model], vec![2, d_model]),
        &device,
    );
    let score_tensor = Tensor::<Backend, 2>::from_data(
        TensorData::new(vec![0.9f32; 2 * d_model], vec![2, d_model]),
        &device,
    );

    let scenario = CachedScenario {
        id: "scen_test_01".to_string(),
        cls_state: cls_tensor,
        choice_questions: vec![choice_1, choice_2],
        noul_states: Some(noul_tensor),
        score_states: Some(score_tensor),
        targets: MultiQuestionTargets::new()
            .with_choice_target(0)
            .with_choice_target(1)
            .with_noul_target(1.0)
            .with_noul_target(0.0)
            .with_score_target(4.5)
            .with_score_target(3.2),
        is_benign: true,
    };

    let safetensors_path = temp_dir.join("test.safetensors");
    let meta_path = temp_dir.join("test.meta.json");
    let dataset_hash = "fake_hash_123".to_string();
    let model_id = "test_model_v1".to_string();

    // 1. Save to disk
    FeatureCache::save_to_disk(
        &[scenario],
        &safetensors_path,
        &meta_path,
        dataset_hash.clone(),
        model_id.clone(),
        d_model,
    )
    .expect("Save to disk must succeed");

    assert!(safetensors_path.exists());
    assert!(meta_path.exists());

    // 2. Validate cache validation checks
    assert!(FeatureCache::is_cache_valid(
        &safetensors_path,
        &meta_path,
        &dataset_hash,
        &model_id,
        d_model
    ));
    assert!(!FeatureCache::is_cache_valid(
        &safetensors_path,
        &meta_path,
        "wrong_hash",
        &model_id,
        d_model
    ));

    // 3. Load from disk
    let loaded: Vec<CachedScenario<Backend>> =
        FeatureCache::load_from_disk(&safetensors_path, &meta_path, &device)
            .expect("Load from disk must succeed");

    assert_eq!(loaded.len(), 1);
    let l = &loaded[0];
    assert_eq!(l.id, "scen_test_01");
    assert_eq!(l.cls_state.dims(), [1, d_model]);
    assert_eq!(l.choice_questions.len(), 2);
    assert_eq!(l.choice_questions[0].dims(), [3, d_model]);
    assert_eq!(l.choice_questions[1].dims(), [2, d_model]);
    assert_eq!(l.noul_states.as_ref().unwrap().dims(), [2, d_model]);
    assert_eq!(l.score_states.as_ref().unwrap().dims(), [2, d_model]);
    assert_eq!(l.targets.choice_targets, vec![0, 1]);
    assert_eq!(l.targets.noul_targets, vec![1.0, 0.0]);
    assert_eq!(l.targets.score_targets, vec![4.5, 3.2]);
    assert!(l.is_benign);

    let _ = fs::remove_dir_all(&temp_dir);
}
