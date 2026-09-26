//! ModernBERT building blocks, matching HF `modeling_modernbert` (eager attention path).

use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{
        activation::{gelu, softmax},
        backend::Backend,
        Tensor, TensorData,
    },
};

use super::config::ModernBertConfig;

/// Additive bias for masked attention scores. Large enough that `exp` underflows to 0,
/// small enough that adding two of them stays finite in f32.
pub(crate) const MASK_BIAS: f32 = -1.0e9;

// =====================================================================
// Rotary position embedding (rotate_half / NeoX convention)
// =====================================================================

/// Precomputed `cos`/`sin` tables `[1, 1, L, head_dim]` for one RoPE base.
///
/// `burn::nn::RotaryEncoding` rotates interleaved pairs `(x0, x1), (x2, x3)…`; HF ModernBERT
/// rotates the two halves `(x_i, x_{i + d/2})`, so we need our own.
#[derive(Clone, Debug)]
pub struct RopeTables<B: Backend> {
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
}

impl<B: Backend> RopeTables<B> {
    pub fn new(seq_len: usize, head_dim: usize, theta: f32, device: &B::Device) -> Self {
        let half = head_dim / 2;
        // Same op order as HF: inv_freq in f32, then position * inv_freq in f32.
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / (theta as f64).powf((2 * i) as f64 / head_dim as f64)) as f32)
            .collect();
        let mut cos = Vec::with_capacity(seq_len * head_dim);
        let mut sin = Vec::with_capacity(seq_len * head_dim);
        for p in 0..seq_len {
            for _ in 0..2 {
                for &f in &inv_freq {
                    let angle = p as f32 * f;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
        }
        let shape = [1, 1, seq_len, head_dim];
        Self {
            cos: Tensor::from_data(TensorData::new(cos, shape), device),
            sin: Tensor::from_data(TensorData::new(sin, shape), device),
        }
    }

    /// Applies the rotation to `x: [B, H, L, head_dim]`.
    pub fn apply(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, l, d] = x.dims();
        let half = d / 2;
        let x1 = x.clone().slice([0..b, 0..h, 0..l, 0..half]);
        let x2 = x.clone().slice([0..b, 0..h, 0..l, half..d]);
        let rotated = Tensor::cat(vec![x2.neg(), x1], 3);
        x * self.cos.clone() + rotated * self.sin.clone()
    }
}

// =====================================================================
// Attention
// =====================================================================

#[derive(Module, Debug)]
pub struct ModernBertAttention<B: Backend> {
    /// Fused `[d, 3d]` projection; output columns are `q | k | v`, each head-major.
    pub wqkv: Linear<B>,
    pub wo: Linear<B>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> ModernBertAttention<B> {
    /// `x: [B, L, d]`, `bias: [B, 1, L, L]` additive mask (0 or [`MASK_BIAS`]).
    pub fn forward(&self, x: Tensor<B, 3>, rope: &RopeTables<B>, bias: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, l, d] = x.dims();
        let (h, hd) = (self.n_heads, self.head_dim);

        let qkv = self.wqkv.forward(x).reshape([b, l, 3, h, hd]);
        let split = |i: usize| {
            qkv.clone()
                .slice([0..b, 0..l, i..i + 1, 0..h, 0..hd])
                .reshape([b, l, h, hd])
                .swap_dims(1, 2) // [B, H, L, hd]
        };
        let q = rope.apply(split(0));
        let k = rope.apply(split(1));
        let v = split(2);

        let scale = (hd as f32).powf(-0.5);
        let scores = q.matmul(k.swap_dims(2, 3)) * scale + bias;
        let weights = softmax(scores, 3);
        let context = weights.matmul(v).swap_dims(1, 2).reshape([b, l, d]);
        self.wo.forward(context)
    }
}

// =====================================================================
// GeGLU MLP
// =====================================================================

#[derive(Module, Debug)]
pub struct ModernBertMlp<B: Backend> {
    /// `[d, 2 * inter]`; the first half is the activation input, the second the gate.
    pub wi: Linear<B>,
    pub wo: Linear<B>,
    pub intermediate: usize,
}

impl<B: Backend> ModernBertMlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, l, _] = x.dims();
        let inter = self.intermediate;
        let projected = self.wi.forward(x);
        let input = projected.clone().slice([0..b, 0..l, 0..inter]);
        let gate = projected.slice([0..b, 0..l, inter..2 * inter]);
        self.wo.forward(gelu(input) * gate)
    }
}

// =====================================================================
// Encoder layer
// =====================================================================

#[derive(Module, Debug)]
pub struct ModernBertLayer<B: Backend> {
    /// `None` for layer 0, where HF uses `nn.Identity` (the embedding norm precedes it).
    pub attn_norm: Option<LayerNorm<B>>,
    pub attn: ModernBertAttention<B>,
    pub mlp_norm: LayerNorm<B>,
    pub mlp: ModernBertMlp<B>,
    /// Global (full) attention; otherwise sliding-window local attention.
    pub global: bool,
}

impl<B: Backend> ModernBertLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &RopeTables<B>,
        bias: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let normed = match self.attn_norm {
            Some(ref norm) => norm.forward(x.clone()),
            None => x.clone(),
        };
        let x = x + self.attn.forward(normed, rope, bias);
        let mlp_out = self.mlp.forward(self.mlp_norm.forward(x.clone()));
        x + mlp_out
    }
}

pub(crate) fn layer_norm<B: Backend>(config: &ModernBertConfig, device: &B::Device) -> LayerNorm<B> {
    LayerNormConfig::new(config.hidden_size)
        .with_epsilon(config.norm_eps)
        .with_bias(false)
        .init(device)
}

pub(crate) fn init_layer<B: Backend>(
    config: &ModernBertConfig,
    index: usize,
    device: &B::Device,
) -> ModernBertLayer<B> {
    let d = config.hidden_size;
    let inter = config.intermediate_size;
    let linear = |i, o| LinearConfig::new(i, o).with_bias(false).init(device);
    ModernBertLayer {
        attn_norm: (index > 0).then(|| layer_norm(config, device)),
        attn: ModernBertAttention {
            wqkv: linear(d, 3 * d),
            wo: linear(d, d),
            n_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
        },
        mlp_norm: layer_norm(config, device),
        mlp: ModernBertMlp {
            wi: linear(d, 2 * inter),
            wo: linear(inter, d),
            intermediate: inter,
        },
        global: config.is_global_layer(index),
    }
}
