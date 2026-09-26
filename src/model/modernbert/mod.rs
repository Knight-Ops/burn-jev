//! Frozen ModernBERT encoder (answerdotai/ModernBERT-base / -large) ported to Burn.
//!
//! Bidirectional, pre-norm transformer with alternating global and sliding-window local
//! attention, NeoX-style RoPE (separate bases for global and local layers), GeGLU MLPs and
//! bias-free LayerNorms.

pub mod config;
pub mod layers;
pub mod loader;

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm},
    tensor::{backend::Backend, Bool, Int, Tensor, TensorData},
};

pub use config::ModernBertConfig;
pub use layers::{ModernBertAttention, ModernBertLayer, ModernBertMlp, RopeTables};
pub use loader::{EncoderLoadReport, LoadedEncoder, ModernBertLoader};

use layers::{init_layer, layer_norm, MASK_BIAS};

#[derive(Module, Debug)]
pub struct ModernBertEncoder<B: Backend> {
    pub tok_embeddings: Embedding<B>,
    pub embeddings_norm: LayerNorm<B>,
    pub layers: Vec<ModernBertLayer<B>>,
    pub final_norm: LayerNorm<B>,
    #[module(skip)]
    pub config: ModernBertConfig,
}

impl ModernBertConfig {
    /// Randomly initialized encoder (tests); use [`ModernBertLoader`] for pretrained weights.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ModernBertEncoder<B> {
        ModernBertEncoder {
            tok_embeddings: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            embeddings_norm: layer_norm(self, device),
            layers: (0..self.num_hidden_layers).map(|i| init_layer(self, i, device)).collect(),
            final_norm: layer_norm(self, device),
            config: self.clone(),
        }
    }
}

/// Per-forward attention inputs shared by all layers.
struct AttentionContext<B: Backend> {
    global_rope: RopeTables<B>,
    local_rope: RopeTables<B>,
    global_bias: Tensor<B, 4>,
    local_bias: Tensor<B, 4>,
}

impl<B: Backend> ModernBertEncoder<B> {
    pub fn d_model(&self) -> usize {
        self.config.hidden_size
    }

    /// `input_ids: [B, L]`, `attention_mask: [B, L]` (true = real token) → `[B, L, d]`.
    ///
    /// Padded positions produce unspecified states; callers must only read real tokens.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>, attention_mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let (mut x, ctx) = self.embed(input_ids, attention_mask);
        for layer in &self.layers {
            x = self.run_layer(layer, x, &ctx);
        }
        self.final_norm.forward(x)
    }

    /// Like HF `output_hidden_states=True`: the embedding output followed by every layer's
    /// output (pre final norm), plus the final normed state as the last element.
    pub fn forward_hidden_states(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2, Bool>,
    ) -> Vec<Tensor<B, 3>> {
        let (mut x, ctx) = self.embed(input_ids, attention_mask);
        let mut states = Vec::with_capacity(self.layers.len() + 2);
        states.push(x.clone());
        for layer in &self.layers {
            x = self.run_layer(layer, x, &ctx);
            states.push(x.clone());
        }
        states.push(self.final_norm.forward(x));
        states
    }

    fn run_layer(&self, layer: &ModernBertLayer<B>, x: Tensor<B, 3>, ctx: &AttentionContext<B>) -> Tensor<B, 3> {
        if layer.global {
            layer.forward(x, &ctx.global_rope, ctx.global_bias.clone())
        } else {
            layer.forward(x, &ctx.local_rope, ctx.local_bias.clone())
        }
    }

    fn embed(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2, Bool>,
    ) -> (Tensor<B, 3>, AttentionContext<B>) {
        let device = input_ids.device();
        let [b, l] = input_ids.dims();
        let cfg = &self.config;
        let head_dim = cfg.head_dim();

        // Key padding: [B, 1, 1, L] → broadcast over heads and query rows.
        let pad_bias = attention_mask
            .bool_not()
            .float()
            .mul_scalar(MASK_BIAS)
            .reshape([b, 1, 1, l])
            .expand([b, 1, l, l]);

        // Local layers additionally only see keys with |i - j| <= local_attention / 2.
        let half_window = (cfg.local_attention / 2) as i64;
        let window: Vec<f32> = (0..l as i64)
            .flat_map(|i| {
                (0..l as i64).map(move |j| if (i - j).abs() <= half_window { 0.0 } else { MASK_BIAS })
            })
            .collect();
        let window = Tensor::<B, 4>::from_data(TensorData::new(window, [1, 1, l, l]), &device)
            .expand([b, 1, l, l]);

        let ctx = AttentionContext {
            global_rope: RopeTables::new(l, head_dim, cfg.global_rope_theta, &device),
            local_rope: RopeTables::new(l, head_dim, cfg.local_rope_theta, &device),
            local_bias: pad_bias.clone() + window,
            global_bias: pad_bias,
        };
        let x = self.embeddings_norm.forward(self.tok_embeddings.forward(input_ids));
        (x, ctx)
    }
}
