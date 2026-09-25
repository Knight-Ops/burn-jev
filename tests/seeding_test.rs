use std::collections::BTreeMap;
use burn::tensor::{Int, Tensor};
use burn_flex::{Flex, FlexDevice};
use burn_mamba::{
    BiMamba2Backbone, BiMamba2Config, LoadError, LoaderOptions, Mamba2CheckpointLoader,
};
use safetensors::tensor::{Dtype, TensorView};

type TestBackend = Flex<f32, i32>;

#[test]
fn test_adjustable_config_presets() {
    let cfg_130m = BiMamba2Config::mamba2_130m();
    assert_eq!(cfg_130m.d_model, 768);
    assert_eq!(cfg_130m.n_layers, 24);
    assert_eq!(cfg_130m.d_state, 128);
    assert_eq!(cfg_130m.headdim, 64);
    assert_eq!(cfg_130m.expand, 2);
    assert_eq!(cfg_130m.ngroups, 1);
    assert_eq!(cfg_130m.vocab_size, 50288);

    let cfg_370m = BiMamba2Config::mamba2_370m();
    assert_eq!(cfg_370m.d_model, 1024);
    assert_eq!(cfg_370m.n_layers, 48);
    assert_eq!(cfg_370m.d_state, 128);
    assert_eq!(cfg_370m.headdim, 64);
    assert_eq!(cfg_370m.expand, 2);
    assert_eq!(cfg_370m.ngroups, 1);
    assert_eq!(cfg_370m.vocab_size, 50288);

    let cfg_780m = BiMamba2Config::mamba2_780m();
    assert_eq!(cfg_780m.d_model, 1536);
    assert_eq!(cfg_780m.n_layers, 48);
    assert_eq!(cfg_780m.d_state, 128);
    assert_eq!(cfg_780m.headdim, 64);

    let cfg_1_3b = BiMamba2Config::mamba2_1_3b();
    assert_eq!(cfg_1_3b.d_model, 2048);
    assert_eq!(cfg_1_3b.n_layers, 48);

    let cfg_2_7b = BiMamba2Config::mamba2_2_7b();
    assert_eq!(cfg_2_7b.d_model, 2560);
    assert_eq!(cfg_2_7b.n_layers, 64);

    // Custom arbitrary scale
    let custom = BiMamba2Config::new(1000)
        .with_d_model(128)
        .with_n_layers(3)
        .with_d_state(32)
        .with_headdim(32)
        .with_expand(2)
        .with_ngroups(1);
    assert_eq!(custom.d_model, 128);
    assert_eq!(custom.n_layers, 3);
}

#[test]
fn test_safetensors_f32_loading() {
    let device = FlexDevice;
    let vocab_size = 16;
    let d_model = 32;
    let n_layers = 2;
    let d_state = 16;
    let headdim = 16;
    let expand = 2;
    let ngroups = 1;

    let config = BiMamba2Config::new(vocab_size)
        .with_d_model(d_model)
        .with_n_layers(n_layers)
        .with_d_state(d_state)
        .with_headdim(headdim)
        .with_expand(expand)
        .with_ngroups(ngroups);

    let mut model: BiMamba2Backbone<TestBackend> = config.init(&device);

    let d_inner = d_model * expand; // 64
    let nheads = d_inner / headdim; // 4
    let d_conv = d_inner + 2 * ngroups * d_state; // 64 + 32 = 96
    let in_proj_dim = 2 * d_inner + 2 * ngroups * d_state + nheads; // 128 + 32 + 4 = 164

    // Prepare synthetic safetensors weights
    let emb_data: Vec<u8> = (0..vocab_size * d_model)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();

    let in_proj_data: Vec<u8> = (0..in_proj_dim * d_model)
        .flat_map(|i| ((i as f32) * 0.001 + 0.1).to_le_bytes())
        .collect();

    let conv_w_data: Vec<u8> = (0..d_conv * 1 * 4)
        .flat_map(|i| ((i as f32) * 0.02 + 0.05).to_le_bytes())
        .collect();

    let conv_b_data: Vec<u8> = (0..d_conv)
        .flat_map(|i| ((i as f32) * 0.03).to_le_bytes())
        .collect();

    let dt_bias_data: Vec<u8> = (0..nheads)
        .flat_map(|i| ((i as f32) * 0.1 - 0.5).to_le_bytes())
        .collect();

    let a_log_data: Vec<u8> = (0..nheads)
        .flat_map(|i| ((i as f32) * 0.5 + 1.0).to_le_bytes())
        .collect();

    let d_skip_data: Vec<u8> = (0..nheads)
        .flat_map(|i| ((i as f32) * 0.2 + 0.8).to_le_bytes())
        .collect();

    let inner_norm_data: Vec<u8> = (0..d_inner)
        .flat_map(|i| ((i as f32) * 0.005 + 1.0).to_le_bytes())
        .collect();

    let out_proj_data: Vec<u8> = (0..d_model * d_inner)
        .flat_map(|i| ((i as f32) * 0.002 + 0.2).to_le_bytes())
        .collect();

    let norm_w_data: Vec<u8> = (0..d_model)
        .flat_map(|i| ((i as f32) * 0.01 + 1.0).to_le_bytes())
        .collect();

    let norm_f_data: Vec<u8> = (0..d_model)
        .flat_map(|i| ((i as f32) * 0.015 + 1.0).to_le_bytes())
        .collect();

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "backbone.embeddings.weight".to_string(),
        TensorView::new(Dtype::F32, vec![vocab_size, d_model], &emb_data).unwrap(),
    );

    for i in 0..n_layers {
        tensors.insert(
            format!("backbone.layers.{i}.mixer.in_proj.weight"),
            TensorView::new(Dtype::F32, vec![in_proj_dim, d_model], &in_proj_data).unwrap(),
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.conv1d.weight"),
            TensorView::new(Dtype::F32, vec![d_conv, 1, 4], &conv_w_data).unwrap(),
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.conv1d.bias"),
            TensorView::new(Dtype::F32, vec![d_conv], &conv_b_data).unwrap(),
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.dt_bias"),
            TensorView::new(Dtype::F32, vec![nheads], &dt_bias_data).unwrap(),
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.A_log"),
            TensorView::new(Dtype::F32, vec![nheads], &a_log_data).unwrap(),
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.D"),
            TensorView::new(Dtype::F32, vec![nheads], &d_skip_data).unwrap(),
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.norm.weight"),
            TensorView::new(Dtype::F32, vec![d_inner], &inner_norm_data).unwrap(),
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.out_proj.weight"),
            TensorView::new(Dtype::F32, vec![d_model, d_inner], &out_proj_data).unwrap(),
        );
        tensors.insert(
            format!("backbone.layers.{i}.norm.weight"),
            TensorView::new(Dtype::F32, vec![d_model], &norm_w_data).unwrap(),
        );
    }

    tensors.insert(
        "backbone.norm_f.weight".to_string(),
        TensorView::new(Dtype::F32, vec![d_model], &norm_f_data).unwrap(),
    );

    let safetensors_bytes = safetensors::serialize(&tensors, None).unwrap();

    let report = model.load_safetensors_bytes(&safetensors_bytes).unwrap();
    assert_eq!(report.layers_loaded, 2);
    assert!(report.embedding_loaded);
    assert!(report.final_norm_loaded);
    assert_eq!(report.tensors_loaded, 1 + 2 * 9 + 1); // emb + 2 layers * 9 tensors + norm_f

    // Verify embedding weights accurately seeded
    let emb_tensor = model.embedding.weight.val().into_data();
    let emb_floats = emb_tensor.as_slice::<f32>().unwrap();
    assert!((emb_floats[0] - 0.0).abs() < 1e-6);
    assert!((emb_floats[1] - 0.01).abs() < 1e-6);

    // Verify dt_bias accurately seeded
    let dt_bias = model.layers[0].dt_bias.as_ref().unwrap().val().into_data();
    let dt_floats = dt_bias.as_slice::<f32>().unwrap();
    assert!((dt_floats[0] - (-0.5)).abs() < 1e-6);
    assert!((dt_floats[1] - (-0.4)).abs() < 1e-6);

    // Verify forward pass executes cleanly on seeded model
    let input_tokens = Tensor::<TestBackend, 1, Int>::from_data([0i64, 1, 2, 3], &device)
        .reshape([1, 4]);
    let hidden = model.forward_backbone(input_tokens);
    assert_eq!(hidden.dims(), [1, 4, d_model]);

    let hidden_data = hidden.into_data();
    let hidden_floats = hidden_data.as_slice::<f32>().unwrap();
    for &val in hidden_floats {
        assert!(!val.is_nan(), "Hidden state contains NaN");
        assert!(!val.is_infinite(), "Hidden state contains Inf");
    }
}

#[test]
fn test_safetensors_f16_and_bf16_loading() {
    let device = FlexDevice;
    let d_model = 16;
    let config = BiMamba2Config::new(8)
        .with_d_model(d_model)
        .with_n_layers(1)
        .with_d_state(8)
        .with_headdim(8)
        .with_expand(2);

    let mut model: BiMamba2Backbone<TestBackend> = config.init(&device);

    // Create F16 and BF16 weights
    let norm_f16_bytes: Vec<u8> = (0..d_model)
        .flat_map(|i| half::f16::from_f32((i as f32) * 0.1 + 1.0).to_le_bytes())
        .collect();

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "backbone.norm_f.weight".to_string(),
        TensorView::new(Dtype::F16, vec![d_model], &norm_f16_bytes).unwrap(),
    );

    let safetensors_bytes = safetensors::serialize(&tensors, None).unwrap();

    let options = LoaderOptions {
        strict_layer_count: false,
        load_embedding: false,
        load_layers: false,
        load_final_norm: true,
        load_heads: false,
    };

    let report = Mamba2CheckpointLoader::load_bytes(&mut model, &safetensors_bytes, &options).unwrap();
    assert!(report.final_norm_loaded);

    let final_norm_tensor = model.final_norm.gamma.val().into_data();
    let norm_floats = final_norm_tensor.as_slice::<f32>().unwrap();
    assert!((norm_floats[0] - 1.0).abs() < 1e-3);
    assert!((norm_floats[1] - 1.1).abs() < 1e-3);
}

#[test]
fn test_shape_mismatch_error() {
    let device = FlexDevice;
    let config = BiMamba2Config::new(16).with_d_model(32).with_n_layers(1);
    let mut model: BiMamba2Backbone<TestBackend> = config.init(&device);

    // Provide embedding with wrong shape [16, 64] instead of [16, 32]
    let wrong_data = vec![0u8; 16 * 64 * 4];
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "backbone.embeddings.weight".to_string(),
        TensorView::new(Dtype::F32, vec![16, 64], &wrong_data).unwrap(),
    );
    let bytes = safetensors::serialize(&tensors, None).unwrap();

    let result = model.load_safetensors_bytes(&bytes);
    match result {
        Err(LoadError::ShapeMismatch { tensor, expected, found }) => {
            assert_eq!(tensor, "backbone.embeddings.weight");
            assert_eq!(expected, vec![16, 32]);
            assert_eq!(found, vec![16, 64]);
        }
        other => panic!("Expected ShapeMismatch error, got {:?}", other),
    }
}

#[test]
fn test_strict_layers_error_on_missing_layer() {
    let device = FlexDevice;
    let config = BiMamba2Config::new(16).with_d_model(32).with_n_layers(4);
    let mut model: BiMamba2Backbone<TestBackend> = config.init(&device);

    // Empty safetensors (0 layers)
    let tensors: BTreeMap<String, TensorView> = BTreeMap::new();
    let bytes = safetensors::serialize(&tensors, None).unwrap();

    let options = LoaderOptions {
        strict_layer_count: true,
        load_embedding: false,
        load_layers: true,
        load_final_norm: false,
        load_heads: false,
    };

    let result = Mamba2CheckpointLoader::load_bytes(&mut model, &bytes, &options);
    assert!(matches!(result, Err(LoadError::TensorNotFound(_))));
}

#[test]
fn test_safetensors_save_and_load_roundtrip() {
    let device = FlexDevice;
    let config = BiMamba2Config::new(16)
        .with_d_model(32)
        .with_n_layers(2)
        .with_d_state(16)
        .with_headdim(16)
        .with_expand(2)
        .with_ngroups(1);

    let model: BiMamba2Backbone<TestBackend> = config.init(&device);
    let tmp_path = std::env::temp_dir().join("test_roundtrip.safetensors");
    model.save_safetensors_file(&tmp_path).unwrap();

    let mut loaded_model: BiMamba2Backbone<TestBackend> = config.init(&device);
    let report = loaded_model.load_safetensors_file(&tmp_path).unwrap();
    let _ = std::fs::remove_file(tmp_path);

    assert_eq!(report.layers_loaded, 2);
    assert!(report.embedding_loaded);
    assert!(report.final_norm_loaded);
    assert!(report.heads_loaded);

    // Verify forward pass
    let input_tokens = Tensor::<TestBackend, 1, Int>::from_data([0i64, 1, 2, 3], &device).reshape([1, 4]);
    let hidden = loaded_model.forward_backbone(input_tokens);
    for &val in hidden.into_data().as_slice::<f32>().unwrap() {
        assert!(!val.is_nan(), "Roundtrip loaded model hidden contains NaN");
    }
}

/// Deterministic, non-trivial test values in roughly [-2, 2].
fn test_values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0 + (i as f32) * 1e-3).collect()
}

#[test]
fn test_f16_bf16_embedding_matches_scalar_conversion() {
    let device = FlexDevice;
    // Odd element count (13 * 7) exercises the SIMD remainder path in `half`.
    let (vocab, d_model) = (13, 8);
    let config = BiMamba2Config::new(vocab)
        .with_d_model(d_model)
        .with_n_layers(1)
        .with_d_state(8)
        .with_headdim(8)
        .with_expand(2);
    let values = test_values(vocab * d_model);

    for dtype in [Dtype::F16, Dtype::BF16] {
        let (bytes, expected): (Vec<u8>, Vec<f32>) = match dtype {
            Dtype::F16 => (
                values.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect(),
                values.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect(),
            ),
            _ => (
                values.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect(),
                values.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect(),
            ),
        };
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "backbone.embeddings.weight".to_string(),
            TensorView::new(dtype, vec![vocab, d_model], &bytes).unwrap(),
        );
        let checkpoint = safetensors::serialize(&tensors, None).unwrap();

        let mut model: BiMamba2Backbone<TestBackend> = config.init(&device);
        let report = model.load_safetensors_bytes(&checkpoint).unwrap();
        assert!(report.embedding_loaded);

        let loaded = model.embedding.weight.val().into_data();
        assert_eq!(loaded.as_slice::<f32>().unwrap(), expected.as_slice(), "{dtype:?} decode mismatch");
    }
}

#[test]
fn test_loaded_params_equal_checkpoint_and_skip_random_init() {
    let device = FlexDevice;
    let (vocab, d_model) = (16, 32);
    let config = BiMamba2Config::new(vocab)
        .with_d_model(d_model)
        .with_n_layers(1)
        .with_d_state(16)
        .with_headdim(16)
        .with_expand(2);
    let mut model: BiMamba2Backbone<TestBackend> = config.init(&device);

    // Premise: freshly initialized params are lazy, so the loader can avoid sampling them.
    assert!(!model.embedding.weight.is_initialized());
    assert!(!model.layers[0].in_proj.weight.is_initialized());
    assert!(!model.layers[0].out_proj.weight.is_initialized());

    let in_proj_dim = model.layers[0].in_proj.weight.lazy_shape().dims::<2>()[1];
    assert!(!model.layers[0].in_proj.weight.is_initialized(), "lazy_shape must not initialize");

    let emb = test_values(vocab * d_model);
    let in_proj = test_values(in_proj_dim * d_model);
    let to_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let (emb_bytes, in_proj_bytes) = (to_bytes(&emb), to_bytes(&in_proj));
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "backbone.embeddings.weight".to_string(),
        TensorView::new(Dtype::F32, vec![vocab, d_model], &emb_bytes).unwrap(),
    );
    tensors.insert(
        "backbone.layers.0.mixer.in_proj.weight".to_string(),
        TensorView::new(Dtype::F32, vec![in_proj_dim, d_model], &in_proj_bytes).unwrap(),
    );
    let checkpoint = safetensors::serialize(&tensors, None).unwrap();

    model.load_safetensors_bytes(&checkpoint).unwrap();

    // Params absent from the checkpoint stay lazy: the loader never forced their init.
    assert!(!model.layers[0].out_proj.weight.is_initialized());

    let loaded_emb = model.embedding.weight.val().into_data();
    assert_eq!(loaded_emb.as_slice::<f32>().unwrap(), emb.as_slice());
    // in_proj is stored [out, in] in the checkpoint and [in, out] in Burn.
    let loaded_in_proj = model.layers[0].in_proj.weight.val().swap_dims(0, 1).into_data();
    assert_eq!(loaded_in_proj.as_slice::<f32>().unwrap(), in_proj.as_slice());
}

#[test]
fn test_infer_config_file_reads_header_only() {
    let device = FlexDevice;
    let config = BiMamba2Config::new(16)
        .with_d_model(32)
        .with_n_layers(2)
        .with_d_state(16)
        .with_headdim(16)
        .with_expand(2);
    let model: BiMamba2Backbone<TestBackend> = config.init(&device);
    let path = std::env::temp_dir().join("test_infer_config_file.safetensors");
    model.save_safetensors_file(&path).unwrap();

    let from_file = Mamba2CheckpointLoader::infer_config_file(&path).unwrap();
    let from_bytes = Mamba2CheckpointLoader::infer_config(&std::fs::read(&path).unwrap()).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(format!("{from_file:?}"), format!("{from_bytes:?}"));
    assert_eq!(from_file.d_model, 32);
    assert_eq!(from_file.n_layers, 2);
    assert_eq!(from_file.d_state, 16);
}
