//! Item→token cross-attention reader.
//!
//! Each decision item (choice candidate, noul assertion, score rubric) becomes one query: its
//! mean-pooled encoder span plus a learned item-type embedding. Queries cross-attend to the
//! scenario's context tokens through a few pre-LN blocks; queries never attend to each other,
//! so every item is read against the evidence independently. `n_blocks = 0` degenerates to a
//! mean-pool probe (projection + type embedding + norm), used as an ablation.

use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    tensor::{activation::gelu, backend::Backend, Bool, Distribution, Int, Tensor},
};

use crate::encoding::NUM_ITEM_TYPES;

#[derive(Config, Debug, PartialEq)]
pub struct ItemReaderConfig {
    /// Encoder hidden size.
    pub d_in: usize,
    #[config(default = "256")]
    pub d_reader: usize,
    #[config(default = "2")]
    pub n_blocks: usize,
    #[config(default = "4")]
    pub n_heads: usize,
    #[config(default = "2")]
    pub ffn_mult: usize,
    #[config(default = "0.1")]
    pub dropout: f64,
    /// Probability of hiding each real context token from the reader's keys (training only).
    #[config(default = "0.0")]
    pub ctx_token_drop: f64,
}

#[derive(Module, Debug)]
pub struct ReaderBlock<B: Backend> {
    pub q_norm: LayerNorm<B>,
    pub kv_norm: LayerNorm<B>,
    pub attn: MultiHeadAttention<B>,
    pub ffn_norm: LayerNorm<B>,
    pub ffn_in: Linear<B>,
    pub ffn_out: Linear<B>,
    pub dropout: Dropout,
}

impl<B: Backend> ReaderBlock<B> {
    fn forward(&self, q: Tensor<B, 3>, kv: Tensor<B, 3>, ctx_pad: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let kv = self.kv_norm.forward(kv);
        let attended = self
            .attn
            .forward(MhaInput::new(self.q_norm.forward(q.clone()), kv.clone(), kv).mask_pad(ctx_pad))
            .context;
        let q = q + self.dropout.forward(attended);
        let ffn = self.ffn_out.forward(gelu(self.ffn_in.forward(self.ffn_norm.forward(q.clone()))));
        q + self.dropout.forward(ffn)
    }
}

#[derive(Module, Debug)]
pub struct ItemReader<B: Backend> {
    /// Shared projection of encoder states (item queries and context keys live in the same
    /// space) into the reader width.
    pub proj: Linear<B>,
    pub type_emb: Embedding<B>,
    pub blocks: Vec<ReaderBlock<B>>,
    pub out_norm: LayerNorm<B>,
    pub dropout: Dropout,
    #[module(skip)]
    pub ctx_token_drop: f64,
}

impl ItemReaderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ItemReader<B> {
        let r = self.d_reader;
        let norm = || LayerNormConfig::new(r).init(device);
        let blocks = (0..self.n_blocks)
            .map(|_| ReaderBlock {
                q_norm: norm(),
                kv_norm: norm(),
                attn: MultiHeadAttentionConfig::new(r, self.n_heads)
                    .with_dropout(self.dropout)
                    .init(device),
                ffn_norm: norm(),
                ffn_in: LinearConfig::new(r, r * self.ffn_mult).init(device),
                ffn_out: LinearConfig::new(r * self.ffn_mult, r).init(device),
                dropout: DropoutConfig::new(self.dropout).init(),
            })
            .collect();
        ItemReader {
            proj: LinearConfig::new(self.d_in, r).init(device),
            type_emb: EmbeddingConfig::new(NUM_ITEM_TYPES, r).init(device),
            blocks,
            out_norm: norm(),
            dropout: DropoutConfig::new(self.dropout).init(),
            ctx_token_drop: self.ctx_token_drop,
        }
    }
}

impl<B: Backend> ItemReader<B> {
    /// `queries: [B, K, d_in]`, `types: [B, K]`, `ctx: [B, L, d_in]`, `ctx_pad: [B, L]`
    /// (true = padding) → `[B, K, d_reader]`.
    pub fn forward(
        &self,
        queries: Tensor<B, 3>,
        types: Tensor<B, 2, Int>,
        ctx: Tensor<B, 3>,
        ctx_pad: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let mut q = self.dropout.forward(self.proj.forward(queries) + self.type_emb.forward(types));
        if !self.blocks.is_empty() {
            let ctx_pad = self.drop_context_tokens(ctx_pad);
            let kv = self.proj.forward(ctx);
            for block in &self.blocks {
                q = block.forward(q, kv.clone(), ctx_pad.clone());
            }
        }
        self.out_norm.forward(q)
    }

    /// Masks a random `ctx_token_drop` fraction of context tokens, only when autodiff is on
    /// (the same gate as [`Dropout`]), so validation and inference see every token. A row
    /// with every token masked is harmless: MHA's finite mask value yields uniform attention.
    fn drop_context_tokens(&self, ctx_pad: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
        let device = ctx_pad.device();
        if self.ctx_token_drop <= 0.0 || !B::ad_enabled(&device) {
            return ctx_pad;
        }
        let drop = Tensor::<B, 2>::random(ctx_pad.dims(), Distribution::Bernoulli(self.ctx_token_drop), &device)
            .greater_elem(0.5);
        ctx_pad.bool_or(drop)
    }

    /// Scenario-level embedding for the k-NN head: the masked mean of the context tokens,
    /// projected and normed like the item outputs. `pool: [B, L]` holds `1/len` on real
    /// context tokens and 0 on padding → `[B, d_reader]`.
    pub fn context_embedding(&self, ctx: Tensor<B, 3>, pool: Tensor<B, 2>) -> Tensor<B, 2> {
        let [b, _, d] = ctx.dims();
        let pooled = (ctx * pool.unsqueeze_dim(2)).sum_dim(1).reshape([b, d]);
        self.out_norm.forward(self.proj.forward(pooled))
    }
}
