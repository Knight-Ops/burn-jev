//! Loads HF ModernBERT checkpoints (`config.json` + `model.safetensors`) into Burn.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use burn::{
    module::Param,
    nn::{LayerNorm, Linear},
    tensor::{backend::Backend, Tensor, TensorData},
};
use safetensors::SafeTensors;

use super::{ModernBertConfig, ModernBertEncoder};
use crate::model::weights::{sha256_file, tensor_to_f32_vec, transpose_host, validate_shape, LoadError};

pub const CONFIG_FILE: &str = "config.json";
pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// Tensors that belong to the masked-LM head, not the encoder.
const MLM_PREFIXES: [&str; 2] = ["head.", "decoder."];

#[derive(Clone, Debug)]
pub struct EncoderLoadReport {
    pub tensors_loaded: usize,
    pub parameters_transferred: usize,
    /// Checkpoint tensors deliberately skipped (the MLM head).
    pub ignored: Vec<String>,
}

/// An encoder loaded by [`ModernBertLoader::load_dir`].
pub struct LoadedEncoder<B: Backend> {
    pub model: ModernBertEncoder<B>,
    pub config: ModernBertConfig,
    /// SHA-256 of `model.safetensors`; artifacts and feature caches are bound to it.
    pub sha256: String,
    pub report: EncoderLoadReport,
    /// `tokenizer.json` next to the weights, if present.
    pub tokenizer_path: Option<PathBuf>,
}

pub struct ModernBertLoader;

impl ModernBertLoader {
    /// Loads a HuggingFace ModernBERT directory (as written by `hf download`).
    pub fn load_dir<B: Backend, P: AsRef<Path>>(dir: P, device: &B::Device) -> Result<LoadedEncoder<B>, LoadError> {
        let dir = dir.as_ref();
        let config = ModernBertConfig::from_hf_file(dir.join(CONFIG_FILE))?;
        let weights_path = dir.join(WEIGHTS_FILE);
        let file = std::fs::File::open(&weights_path)?;
        // SAFETY: the checkpoint is treated as read-only for the duration of the load; every
        // tensor is copied out of the mapping before this function returns. Concurrent external
        // modification of the file is not supported.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let (model, report) = Self::load_bytes(&config, &mmap, device)?;
        let tokenizer_path = Some(dir.join(TOKENIZER_FILE)).filter(|p| p.exists());
        Ok(LoadedEncoder {
            model,
            config,
            sha256: sha256_file(&weights_path)?,
            report,
            tokenizer_path,
        })
    }

    /// Builds an encoder from in-memory safetensors bytes. Every encoder tensor must be
    /// present with the expected shape, and every checkpoint tensor must be consumed (apart
    /// from the MLM head), so a config/checkpoint mismatch fails loudly.
    pub fn load_bytes<B: Backend>(
        config: &ModernBertConfig,
        bytes: &[u8],
        device: &B::Device,
    ) -> Result<(ModernBertEncoder<B>, EncoderLoadReport), LoadError> {
        let st = SafeTensors::deserialize(bytes)?;
        // Params about to be replaced are lazy, so they are never randomly initialized.
        let mut model: ModernBertEncoder<B> = config.init(device);
        let mut w = WeightReader {
            st: &st,
            device,
            consumed: BTreeSet::new(),
            report: EncoderLoadReport {
                tensors_loaded: 0,
                parameters_transferred: 0,
                ignored: Vec::new(),
            },
        };

        let d = config.hidden_size;
        let emb = w.read("embeddings.tok_embeddings.weight", &[config.vocab_size, d])?;
        model.tok_embeddings.weight = Param::from_tensor(emb);
        w.norm(&mut model.embeddings_norm, "embeddings.norm.weight", d)?;

        for (i, layer) in model.layers.iter_mut().enumerate() {
            let p = format!("layers.{i}");
            if let Some(ref mut norm) = layer.attn_norm {
                w.norm(norm, &format!("{p}.attn_norm.weight"), d)?;
            }
            w.linear(&mut layer.attn.wqkv, &format!("{p}.attn.Wqkv.weight"), d, 3 * d)?;
            w.linear(&mut layer.attn.wo, &format!("{p}.attn.Wo.weight"), d, d)?;
            w.norm(&mut layer.mlp_norm, &format!("{p}.mlp_norm.weight"), d)?;
            let inter = config.intermediate_size;
            w.linear(&mut layer.mlp.wi, &format!("{p}.mlp.Wi.weight"), d, 2 * inter)?;
            w.linear(&mut layer.mlp.wo, &format!("{p}.mlp.Wo.weight"), inter, d)?;
        }
        w.norm(&mut model.final_norm, "final_norm.weight", d)?;

        let mut unexpected = Vec::new();
        for name in st.names() {
            if w.consumed.contains(name) {
                continue;
            }
            if MLM_PREFIXES.iter().any(|p| name.starts_with(p)) {
                w.report.ignored.push(name.to_string());
            } else {
                unexpected.push(name.to_string());
            }
        }
        if !unexpected.is_empty() {
            unexpected.sort();
            return Err(LoadError::InvalidConfiguration(format!(
                "checkpoint has tensors the config does not account for (layer count mismatch?): {}",
                unexpected.join(", ")
            )));
        }
        w.report.ignored.sort();
        Ok((model, w.report))
    }
}

struct WeightReader<'a, B: Backend> {
    st: &'a SafeTensors<'a>,
    device: &'a B::Device,
    consumed: BTreeSet<String>,
    report: EncoderLoadReport,
}

impl<B: Backend> WeightReader<'_, B> {
    /// Reads `name` (with or without the `model.` prefix) as an f32 tensor of `shape`.
    fn read<const D: usize>(&mut self, name: &str, shape: &[usize; D]) -> Result<Tensor<B, D>, LoadError> {
        let data = self.read_vec(name, shape)?;
        Ok(Tensor::from_data(TensorData::new(data, *shape), self.device))
    }

    fn read_vec(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, LoadError> {
        let prefixed = format!("model.{name}");
        let (key, view) = [prefixed.as_str(), name]
            .into_iter()
            .find_map(|k| self.st.tensor(k).ok().map(|v| (k.to_string(), v)))
            .ok_or_else(|| LoadError::TensorNotFound(prefixed.clone()))?;
        validate_shape(&key, view.shape(), shape)?;
        let data = tensor_to_f32_vec(&view)?;
        self.report.tensors_loaded += 1;
        self.report.parameters_transferred += data.len();
        self.consumed.insert(key);
        Ok(data)
    }

    fn norm(&mut self, norm: &mut LayerNorm<B>, name: &str, d: usize) -> Result<(), LoadError> {
        norm.gamma = Param::from_tensor(self.read(name, &[d])?);
        Ok(())
    }

    /// HF stores Linear weights `[out, in]`; Burn wants contiguous `[in, out]`.
    fn linear(&mut self, linear: &mut Linear<B>, name: &str, d_in: usize, d_out: usize) -> Result<(), LoadError> {
        let data = self.read_vec(name, &[d_out, d_in])?;
        let transposed = transpose_host(&data, d_out, d_in);
        linear.weight = Param::from_tensor(Tensor::from_data(TensorData::new(transposed, [d_in, d_out]), self.device));
        Ok(())
    }
}
