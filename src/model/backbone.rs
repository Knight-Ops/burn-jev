use std::path::Path;
use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig},
    tensor::{backend::Backend, Int, Tensor},
};

use crate::model::heads::{UnifiedHeads, UnifiedHeadsConfig};
use crate::model::loader::{LoadError, LoadReport, LoaderOptions, Mamba2CheckpointLoader};
use crate::model::ssd::{Mamba2SSDBlock, Mamba2SSDConfig};

// =====================================================================
// Bi-Mamba-2 Architecture Configuration
// =====================================================================

#[derive(Config, Debug)]
pub struct BiMamba2Config {
    pub vocab_size: usize,
    #[config(default = "768")]
    pub d_model: usize,
    #[config(default = "24")]
    pub n_layers: usize,
    #[config(default = "128")]
    pub d_state: usize,
    #[config(default = "64")]
    pub headdim: usize,
    #[config(default = "2")]
    pub expand: usize,
    #[config(default = "1")]
    pub ngroups: usize,
    #[config(default = "true")]
    pub use_dt_bias: bool,
    #[config(default = "true")]
    pub use_inner_norm: bool,
    #[config(default = "256")]
    pub knn_dim: usize,
    #[config(default = "5")]
    pub num_rubric_bins: usize,
}

pub type BiMamba2JevKNN<B> = BiMamba2Backbone<B>;
pub type BiMamba2JevKNNConfig = BiMamba2Config;

impl BiMamba2Config {
    /// Presets for standard Mamba-2 open-source checkpoints
    /// Mamba-2 130M parameter preset
    pub fn mamba2_130m() -> Self {
        Self::new(50288)
            .with_d_model(768)
            .with_n_layers(24)
            .with_d_state(128)
            .with_headdim(64)
            .with_expand(2)
            .with_ngroups(1)
            .with_use_dt_bias(true)
            .with_use_inner_norm(true)
    }

    /// Mamba-2 370M parameter preset
    pub fn mamba2_370m() -> Self {
        Self::new(50288)
            .with_d_model(1024)
            .with_n_layers(48)
            .with_d_state(128)
            .with_headdim(64)
            .with_expand(2)
            .with_ngroups(1)
            .with_use_dt_bias(true)
            .with_use_inner_norm(true)
    }

    /// Mamba-2 780M parameter preset
    pub fn mamba2_780m() -> Self {
        Self::new(50288)
            .with_d_model(1536)
            .with_n_layers(48)
            .with_d_state(128)
            .with_headdim(64)
            .with_expand(2)
            .with_ngroups(1)
            .with_use_dt_bias(true)
            .with_use_inner_norm(true)
    }

    /// Mamba-2 1.3B parameter preset
    pub fn mamba2_1_3b() -> Self {
        Self::new(50288)
            .with_d_model(2048)
            .with_n_layers(48)
            .with_d_state(128)
            .with_headdim(64)
            .with_expand(2)
            .with_ngroups(1)
            .with_use_dt_bias(true)
            .with_use_inner_norm(true)
    }

    /// Mamba-2 2.7B parameter preset
    pub fn mamba2_2_7b() -> Self {
        Self::new(50288)
            .with_d_model(2560)
            .with_n_layers(64)
            .with_d_state(128)
            .with_headdim(64)
            .with_expand(2)
            .with_ngroups(1)
            .with_use_dt_bias(true)
            .with_use_inner_norm(true)
    }

    /// Configuration of the JEV heads that sit on top of this backbone.
    pub fn heads_config(&self) -> UnifiedHeadsConfig {
        UnifiedHeadsConfig::new(self.d_model)
            .with_knn_dim(self.knn_dim)
            .with_num_rubric_bins(self.num_rubric_bins)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BiMamba2Backbone<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);

        let ssd_config = Mamba2SSDConfig::new(self.d_model)
            .with_d_state(self.d_state)
            .with_headdim(self.headdim)
            .with_expand(self.expand)
            .with_ngroups(self.ngroups)
            .with_use_dt_bias(self.use_dt_bias)
            .with_use_inner_norm(self.use_inner_norm);

        let layers = (0..self.n_layers)
            .map(|_| ssd_config.init(device))
            .collect();

        let final_norm = RmsNormConfig::new(self.d_model).init(device);
        let heads = self.heads_config().init(device);

        BiMamba2Backbone {
            embedding,
            layers,
            final_norm,
            heads,
        }
    }
}

#[derive(Module, Debug)]
pub struct BiMamba2Backbone<B: Backend> {
    pub embedding: Embedding<B>,
    pub layers: Vec<Mamba2SSDBlock<B>>,
    pub final_norm: RmsNorm<B>,
    pub heads: UnifiedHeads<B>,
}

impl<B: Backend> BiMamba2Backbone<B> {
    /// Hidden size, read without forcing lazy parameter initialization.
    pub fn d_model(&self) -> usize {
        self.embedding.weight.lazy_shape().dims::<2>()[1]
    }

    /// Ingests tokenized state and runs bidirectional SSD passes through all layers
    pub fn forward_backbone(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embedding.forward(input_ids);
        for layer in &self.layers {
            x = layer.forward(x);
        }
        self.final_norm.forward(x)
    }

    /// Loads pretrained weights from a safetensors file with default options
    pub fn load_safetensors_file<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<LoadReport, LoadError> {
        Mamba2CheckpointLoader::load_file(self, path, &LoaderOptions::default())
    }

    /// Loads pretrained weights from in-memory safetensors bytes with default options
    pub fn load_safetensors_bytes(&mut self, bytes: &[u8]) -> Result<LoadReport, LoadError> {
        Mamba2CheckpointLoader::load_bytes(self, bytes, &LoaderOptions::default())
    }

    /// Loads pretrained weights with custom loader options
    pub fn load_safetensors_with_options<P: AsRef<Path>>(
        &mut self,
        path: P,
        options: &LoaderOptions,
    ) -> Result<LoadReport, LoadError> {
        Mamba2CheckpointLoader::load_file(self, path, options)
    }

    /// Saves all model weights (backbone and JEV heads) to a safetensors file
    pub fn save_safetensors_file<P: AsRef<Path>>(&self, path: P) -> Result<(), LoadError> {
        Mamba2CheckpointLoader::save_file(self, path)
    }
}
