use burn::backend::Autodiff;
use burn::module::Param;
use burn::tensor::{backend::Backend, Distribution, ElementConversion, Int, Tensor};
use burn_flex::{Flex, FlexDevice};
use burn_mamba::{BiMamba2Config, Mamba2SSDBlock, Mamba2SSDConfig};

type TestBackend = Flex<f32, i32>;
type DiffBackend = Autodiff<TestBackend>;

#[test]
fn test_ssd_block_forward_dimensions() {
    let device = FlexDevice;
    let config = Mamba2SSDConfig::new(64)
        .with_d_state(32)
        .with_headdim(32)
        .with_expand(2);

    let block: Mamba2SSDBlock<TestBackend> = config.init(&device);

    let batch: usize = 2;
    let seq_len: usize = 16;
    let d_model: usize = 64;

    let input_data: Vec<f32> = (0..batch * seq_len * d_model)
        .map(|i| ((i % 100) as f32) / 100.0)
        .collect();

    let input = Tensor::<TestBackend, 1>::from_data(input_data.as_slice(), &device)
        .reshape([batch, seq_len, d_model]);

    let output = block.forward(input);
    assert_eq!(output.dims(), [batch, seq_len, d_model]);

    // Check for NaN or Inf
    let output_data = output.into_data();
    let values = output_data.as_slice::<f32>().unwrap();
    for &val in values {
        assert!(!val.is_nan(), "Output contains NaN");
        assert!(!val.is_infinite(), "Output contains Inf");
    }
}

#[test]
fn test_direct_feedthrough_d_skip_parameter() {
    let device = FlexDevice;
    let d_model: usize = 64;
    let expand: usize = 2;
    let headdim: usize = 32;
    let expected_nheads = (d_model * expand) / headdim; // 4

    let config = Mamba2SSDConfig::new(d_model)
        .with_d_state(32)
        .with_headdim(headdim)
        .with_expand(expand);

    let block: Mamba2SSDBlock<TestBackend> = config.init(&device);

    // Verify d_skip parameter dimension and initialization
    assert_eq!(block.d_skip.val().dims(), [expected_nheads]);
    let d_skip_vals = block.d_skip.val().into_data();
    for &val in d_skip_vals.as_slice::<f32>().unwrap() {
        assert_eq!(val, 1.0, "d_skip must be initialized to 1.0");
    }
}

#[test]
fn test_native_sequence_reversal() {
    let device = FlexDevice;
    let config = Mamba2SSDConfig::new(32)
        .with_d_state(16)
        .with_headdim(16)
        .with_expand(1);
    let block: Mamba2SSDBlock<TestBackend> = config.init(&device);

    // Create tensor [1, 4, 1] with values [10.0, 20.0, 30.0, 40.0]
    let values = [10.0f32, 20.0, 30.0, 40.0];
    let input = Tensor::<TestBackend, 1>::from_data(values.as_slice(), &device)
        .reshape([1, 4, 1]);

    let reversed = block.reverse_sequence(input);
    let reversed_vals = reversed.into_data();
    let rev_slice = reversed_vals.as_slice::<f32>().unwrap();
    assert_eq!(rev_slice, &[40.0, 30.0, 20.0, 10.0]);
}

#[test]
fn test_decay_initialization_log_spaced() {
    let device = FlexDevice;
    let d_model: usize = 64;
    let expand: usize = 2;
    let headdim: usize = 32;
    let nheads = (d_model * expand) / headdim; // 4

    let config = Mamba2SSDConfig::new(d_model)
        .with_d_state(32)
        .with_headdim(headdim)
        .with_expand(expand);
    let block: Mamba2SSDBlock<TestBackend> = config.init(&device);

    let a_log_vals = block.a_log.val().into_data();
    let slice = a_log_vals.as_slice::<f32>().unwrap();
    assert_eq!(slice.len(), nheads);

    // Should be monotonic non-decreasing spanning ln(1.0) = 0 to ln(16.0) ~ 2.7725887
    assert!((slice[0] - 0.0).abs() < 1e-5);
    assert!((slice[nheads - 1] - 16.0f32.ln()).abs() < 1e-4);
    for i in 1..nheads {
        assert!(slice[i] >= slice[i - 1]);
    }
}

#[test]
fn test_d_skip_gradient_propagation() {
    let device = FlexDevice;
    let d_model: usize = 32;
    let expand: usize = 1;
    let headdim: usize = 16;
    let nheads = (d_model * expand) / headdim; // 2

    let config = Mamba2SSDConfig::new(d_model)
        .with_d_state(16)
        .with_headdim(headdim)
        .with_expand(expand);

    let block: Mamba2SSDBlock<DiffBackend> = config.init(&device);

    let batch = 1;
    let seq_len = 4;
    let input_vals = vec![0.5f32; batch * seq_len * d_model];
    let input = Tensor::<DiffBackend, 1>::from_data(input_vals.as_slice(), &device)
        .reshape([batch, seq_len, d_model]);

    let output = block.forward(input);
    let loss = output.sum();
    let grads = loss.backward();

    let d_skip_grad = block.d_skip.grad(&grads);
    assert!(d_skip_grad.is_some(), "Gradient for d_skip must exist");
    let grad_tensor = d_skip_grad.unwrap();
    assert_eq!(grad_tensor.dims(), [nheads]);
    for &g in grad_tensor.into_data().as_slice::<f32>().unwrap() {
        assert!(!g.is_nan(), "d_skip gradient contains NaN");
        assert!(!g.is_infinite(), "d_skip gradient contains Inf");
        assert!(g != 0.0, "d_skip gradient must be non-zero");
    }
}

#[test]
fn test_backbone_and_heads_contract() {
    let device = FlexDevice;
    let vocab_size: usize = 100;
    let d_model: usize = 64;
    let knn_dim: usize = 32;
    let num_rubric_bins: usize = 5;

    let config = BiMamba2Config::new(vocab_size)
        .with_d_model(d_model)
        .with_n_layers(2)
        .with_d_state(32)
        .with_headdim(32)
        .with_knn_dim(knn_dim)
        .with_num_rubric_bins(num_rubric_bins);

    let model = config.init::<TestBackend>(&device);

    let batch: usize = 2;
    let seq_len: usize = 16;
    let tokens: Vec<i64> = (0..(batch * seq_len) as i64)
        .map(|i| (i * 3) % vocab_size as i64)
        .collect();
    let input_ids = Tensor::<TestBackend, 1, Int>::from_data(tokens.as_slice(), &device)
        .reshape([batch, seq_len]);

    // 1. Backbone forward pass
    let hidden = model.forward_backbone(input_ids);
    assert_eq!(hidden.dims(), [batch, seq_len, d_model]);

    // 2. k-NN L2 Normalized Embedding
    let cls_tokens = hidden.clone().slice([0..batch, 0..1, 0..d_model]).squeeze_dim(1);
    let knn_vecs = model.heads.extract_knn_embedding(cls_tokens);
    assert_eq!(knn_vecs.dims(), [batch, knn_dim]);

    // Verify L2 norm == 1.0 (approx)
    let norms = knn_vecs.powf_scalar(2.0).sum_dim(1).sqrt().into_data();
    for &norm in norms.as_slice::<f32>().unwrap() {
        assert!((norm - 1.0).abs() < 1e-3, "k-NN embedding must have unit L2 norm, got {}", norm);
    }

    // 3. Noul boolean probability
    let noul_token = hidden.clone().slice([0..1, 10..11, 0..d_model]).squeeze_dim(1);
    let noul_prob = model.heads.evaluate_noul(noul_token).into_data();
    let noul_val = noul_prob.as_slice::<f32>().unwrap()[0];
    assert!(noul_val >= 0.0 && noul_val <= 1.0, "Noul prob out of [0, 1]: {}", noul_val);

    // 4. Score ordinal metric
    let score_token = hidden.clone().slice([0..1, 12..13, 0..d_model]).squeeze_dim(1);
    let score = model.heads.evaluate_score(score_token).into_data();
    let score_val = score.as_slice::<f32>().unwrap()[0];
    assert!(
        score_val >= 1.0 && score_val <= num_rubric_bins as f32,
        "Score {} out of [1.0, {}]", score_val, num_rubric_bins
    );

    // 5. Choice categorical distribution
    let cand_indices = [5, 7, 9];
    let cand_tensor = Tensor::<TestBackend, 1, Int>::from_data(cand_indices.as_slice(), &device);
    let cand_vectors = hidden.select(1, cand_tensor).slice([0..1, 0..3, 0..d_model]).squeeze_dim(0);
    let choice_dist = model.heads.evaluate_choice(cand_vectors).into_data();
    let probs = choice_dist.as_slice::<f32>().unwrap();
    assert_eq!(probs.len(), 3);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "Choice probabilities must sum to 1.0, got {}", sum);
}

// =====================================================================
// Chunked SSD scan vs. token-by-token recurrent reference
// =====================================================================

/// Block with non-trivial A_log / D so the decay and feedthrough paths are exercised.
fn scan_block<B: Backend>(ngroups: usize, device: &B::Device) -> Mamba2SSDBlock<B> {
    let mut block: Mamba2SSDBlock<B> = Mamba2SSDConfig::new(32)
        .with_d_state(16)
        .with_headdim(16)
        .with_expand(2)
        .with_ngroups(ngroups)
        .init(device);
    let nheads = block.nheads;
    block.a_log = Param::from_tensor(Tensor::random([nheads], Distribution::Uniform(-1.0, 2.0), device));
    block.d_skip = Param::from_tensor(Tensor::random([nheads], Distribution::Uniform(0.5, 1.5), device));
    block
}

/// Random scan inputs: (x, dt, B, C) with dt = softplus(raw + dt_bias) as in `forward_pass`.
fn scan_inputs<B: Backend>(
    block: &Mamba2SSDBlock<B>,
    batch: usize,
    seq_len: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
    let (h, p, g, n) = (block.nheads, block.headdim, block.ngroups, block.d_state);
    let x = Tensor::random([batch, seq_len, h, p], Distribution::Uniform(-1.0, 1.0), device);
    let dt_bias = Tensor::<B, 1>::random([h], Distribution::Uniform(-2.0, 1.0), device);
    let dt_raw = Tensor::<B, 3>::random([batch, seq_len, h], Distribution::Uniform(-1.0, 1.0), device);
    let dt = burn::tensor::activation::softplus(dt_raw + dt_bias.reshape([1, 1, h]), 1.0);
    let b = Tensor::random([batch, seq_len, g, n], Distribution::Uniform(-1.0, 1.0), device);
    let c = Tensor::random([batch, seq_len, g, n], Distribution::Uniform(-1.0, 1.0), device);
    (x, dt, b, c)
}

fn max_abs_diff<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>) -> f32 {
    (a - b).abs().max().into_scalar().elem::<f32>()
}

#[test]
fn test_chunked_scan_matches_recurrent() {
    let device = FlexDevice;
    for ngroups in [1, 2] {
        let block = scan_block::<TestBackend>(ngroups, &device);
        for seq_len in [1, 7, 64, 65, 150] {
            let (x, dt, b, c) = scan_inputs(&block, 2, seq_len, &device);
            let chunked = block.ssd_scan(x.clone(), dt.clone(), b.clone(), c.clone());
            let reference = block.ssd_scan_recurrent(x, dt, b, c);
            assert_eq!(chunked.dims(), reference.dims());
            let diff = max_abs_diff(chunked, reference);
            assert!(
                diff < 1e-4,
                "chunked vs recurrent scan differ by {diff} (L={seq_len}, ngroups={ngroups})"
            );
        }
    }
}

#[test]
fn test_chunked_scan_gradients() {
    let device = FlexDevice;
    let block = scan_block::<DiffBackend>(2, &device);
    let (x, dt, b, c) = scan_inputs(&block, 2, 150, &device);
    let x = x.require_grad();

    let grads = block.ssd_scan(x.clone(), dt.clone(), b.clone(), c.clone()).sum().backward();
    let ref_grads = block.ssd_scan_recurrent(x.clone(), dt, b, c).sum().backward();

    let check = |name: &str, g: Tensor<TestBackend, 1>, r: Tensor<TestBackend, 1>| {
        for &v in g.clone().into_data().as_slice::<f32>().unwrap() {
            assert!(v.is_finite(), "{name} gradient is not finite: {v}");
        }
        let scale = r.clone().abs().max().into_scalar().max(1.0);
        let diff = max_abs_diff(g, r);
        assert!(diff / scale < 1e-4, "{name} gradient differs from recurrent by {diff} (scale {scale})");
    };
    check("a_log", block.a_log.grad(&grads).unwrap(), block.a_log.grad(&ref_grads).unwrap());
    check("d_skip", block.d_skip.grad(&grads).unwrap(), block.d_skip.grad(&ref_grads).unwrap());
    check(
        "x",
        x.grad(&grads).unwrap().flatten(0, 3),
        x.grad(&ref_grads).unwrap().flatten(0, 3),
    );
}
