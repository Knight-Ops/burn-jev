use std::path::Path;

use burn::config::Config;
use serde::Deserialize;

use crate::model::weights::LoadError;

/// Architecture of a ModernBERT encoder (base: d=768/22 layers, large: d=1024/28 layers).
#[derive(Config, Debug, PartialEq)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Width of one GeGLU half; `Wi` projects to `2 * intermediate_size`.
    pub intermediate_size: usize,
    /// Layer `i` uses global attention when `i % global_attn_every_n_layers == 0`.
    #[config(default = "3")]
    pub global_attn_every_n_layers: usize,
    /// Sliding-window width of local layers; a token sees keys within `local_attention / 2`.
    #[config(default = "128")]
    pub local_attention: usize,
    #[config(default = "160000.0")]
    pub global_rope_theta: f32,
    #[config(default = "10000.0")]
    pub local_rope_theta: f32,
    #[config(default = "1e-5")]
    pub norm_eps: f64,
    #[config(default = "8192")]
    pub max_position_embeddings: usize,
    #[config(default = "50281")]
    pub cls_token_id: i64,
    #[config(default = "50282")]
    pub sep_token_id: i64,
    #[config(default = "50283")]
    pub pad_token_id: i64,
}

/// The subset of HF `config.json` we read, plus the fields that must hold for our
/// implementation (no biases, exact-erf GELU) to be faithful.
#[derive(Deserialize)]
struct HfConfig {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    global_attn_every_n_layers: usize,
    local_attention: usize,
    global_rope_theta: f32,
    local_rope_theta: f32,
    #[serde(default)]
    norm_eps: Option<f64>,
    #[serde(default)]
    layer_norm_eps: Option<f64>,
    max_position_embeddings: usize,
    cls_token_id: i64,
    sep_token_id: i64,
    pad_token_id: i64,
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default)]
    norm_bias: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
}

impl ModernBertConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn is_global_layer(&self, layer: usize) -> bool {
        layer % self.global_attn_every_n_layers.max(1) == 0
    }

    /// Parses an HF ModernBERT `config.json`, rejecting variants this port does not implement.
    pub fn from_hf_json(json: &str) -> Result<Self, LoadError> {
        let hf: HfConfig = serde_json::from_str(json)
            .map_err(|e| LoadError::InvalidConfiguration(format!("bad ModernBERT config.json: {e}")))?;
        let unsupported = |what: &str| {
            Err(LoadError::InvalidConfiguration(format!("unsupported ModernBERT variant: {what}")))
        };
        if hf.model_type != "modernbert" {
            return unsupported(&format!("model_type = {}", hf.model_type));
        }
        if hf.norm_bias || hf.attention_bias || hf.mlp_bias {
            return unsupported("biased norms/attention/MLP");
        }
        if let Some(act) = hf.hidden_activation.as_deref() {
            if act != "gelu" {
                return unsupported(&format!("hidden_activation = {act}"));
            }
        }
        if hf.hidden_size % hf.num_attention_heads != 0 || (hf.hidden_size / hf.num_attention_heads) % 2 != 0 {
            return unsupported("head_dim must be an even divisor of hidden_size");
        }

        Ok(Self {
            vocab_size: hf.vocab_size,
            hidden_size: hf.hidden_size,
            num_hidden_layers: hf.num_hidden_layers,
            num_attention_heads: hf.num_attention_heads,
            intermediate_size: hf.intermediate_size,
            global_attn_every_n_layers: hf.global_attn_every_n_layers,
            local_attention: hf.local_attention,
            global_rope_theta: hf.global_rope_theta,
            local_rope_theta: hf.local_rope_theta,
            norm_eps: hf.norm_eps.or(hf.layer_norm_eps).unwrap_or(1e-5),
            max_position_embeddings: hf.max_position_embeddings,
            cls_token_id: hf.cls_token_id,
            sep_token_id: hf.sep_token_id,
            pad_token_id: hf.pad_token_id,
        })
    }

    pub fn from_hf_file<P: AsRef<Path>>(path: P) -> Result<Self, LoadError> {
        Self::from_hf_json(&std::fs::read_to_string(path)?)
    }
}
