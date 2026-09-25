use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use burn::{
    module::Param,
    nn::Linear,
    tensor::{backend::Backend, Tensor, TensorData},
};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use half::slice::HalfFloatSliceExt;
use serde::{Deserialize, Serialize};

use crate::cache::FeatureCache;
use crate::delimiters::DelimiterConfig;

// Safetensors payloads are little-endian; decoding reinterprets bytes in native order.
#[cfg(target_endian = "big")]
compile_error!("burn-mamba's safetensors loader assumes a little-endian target");

/// Upper bound on the safetensors header JSON (same limit the safetensors crate enforces).
const MAX_HEADER_SIZE: u64 = 100_000_000;

use super::backbone::{BiMamba2Backbone, BiMamba2Config};
use super::heads::{UnifiedHeads, UnifiedHeadsConfig};

// =====================================================================
// Dynamic Mamba-2 Checkpoint Loader (130M, 370M, 780M, 1.3B, 2.7B+)
// =====================================================================

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Safetensors(safetensors::SafeTensorError),
    TensorNotFound(String),
    ShapeMismatch {
        tensor: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    UnsupportedDtype(String),
    InvalidConfiguration(String),
    /// A heads artifact was trained against a different backbone than the one supplied.
    BackboneMismatch { expected: String, found: String },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "I/O error loading checkpoint: {}", e),
            LoadError::Safetensors(e) => write!(f, "Safetensors parse error: {}", e),
            LoadError::TensorNotFound(name) => write!(f, "Tensor not found in checkpoint: {}", name),
            LoadError::ShapeMismatch { tensor, expected, found } => write!(
                f,
                "Shape mismatch for tensor '{}': expected {:?}, found {:?}",
                tensor, expected, found
            ),
            LoadError::UnsupportedDtype(dt) => write!(f, "Unsupported tensor data type: {}", dt),
            LoadError::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            LoadError::BackboneMismatch { expected, found } => write!(
                f,
                "Heads were trained on backbone sha256 {expected}, but the supplied backbone is {found}"
            ),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

impl From<safetensors::SafeTensorError> for LoadError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        LoadError::Safetensors(e)
    }
}

#[derive(Debug, Clone)]
pub struct LoaderOptions {
    pub strict_layer_count: bool,
    pub load_embedding: bool,
    pub load_layers: bool,
    pub load_final_norm: bool,
    pub load_heads: bool,
}

impl Default for LoaderOptions {
    fn default() -> Self {
        Self {
            strict_layer_count: false,
            load_embedding: true,
            load_layers: true,
            load_final_norm: true,
            load_heads: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadReport {
    pub layers_loaded: usize,
    pub total_model_layers: usize,
    pub tensors_loaded: usize,
    pub parameters_transferred: usize,
    pub embedding_loaded: bool,
    pub final_norm_loaded: bool,
    pub heads_loaded: bool,
}

pub struct Mamba2CheckpointLoader;

impl Mamba2CheckpointLoader {
    /// Inspects safetensors metadata from file and automatically infers BiMamba2Config dimensions
    /// Only the 8-byte length prefix and the header JSON are read, never the tensor payload.
    pub fn infer_config_file<P: AsRef<Path>>(path: P) -> Result<BiMamba2Config, LoadError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)?;
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)?;
        let header_len = u64::from_le_bytes(len_bytes);
        if header_len > MAX_HEADER_SIZE {
            return Err(LoadError::InvalidConfiguration(format!(
                "Safetensors header length {header_len} exceeds {MAX_HEADER_SIZE} bytes"
            )));
        }
        let mut json = vec![0u8; header_len as usize];
        file.read_exact(&mut json)?;
        Self::infer_config_from_json(&json)
    }

    /// Inspects safetensors metadata and automatically infers BiMamba2Config dimensions
    pub fn infer_config(bytes: &[u8]) -> Result<BiMamba2Config, LoadError> {
        if bytes.len() < 8 {
            return Err(LoadError::InvalidConfiguration("Buffer too small for safetensors header".to_string()));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        if bytes.len() < 8 + header_len {
            return Err(LoadError::InvalidConfiguration("Buffer incomplete for safetensors header JSON".to_string()));
        }
        Self::infer_config_from_json(&bytes[8..8 + header_len])
    }

    /// Helper to parse safetensors header JSON map without needing tensor binary payloads
    pub fn infer_config_from_json(json_bytes: &[u8]) -> Result<BiMamba2Config, LoadError> {
        let map: std::collections::HashMap<String, serde_json::Value> = serde_json::from_slice(json_bytes)
            .map_err(|e| LoadError::InvalidConfiguration(format!("Failed to parse safetensors header JSON: {}", e)))?;

        let get_shape = |key: &str| -> Option<Vec<usize>> {
            map.get(key)
                .and_then(|v| v.get("shape"))
                .and_then(|s| s.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
        };

        let find_shape = |keys: &[&str]| -> Option<(String, Vec<usize>)> {
            for &k in keys {
                if let Some(shape) = get_shape(k) {
                    return Some((k.to_string(), shape));
                }
            }
            None
        };

        // 1. Find embedding to determine vocab_size and d_model
        let emb_keys = [
            "backbone.embeddings.weight",
            "embeddings.weight",
            "backbone.embedding.weight",
            "embedding.weight",
        ];
        let (_, emb_shape) = find_shape(&emb_keys)
            .ok_or_else(|| LoadError::TensorNotFound("embedding weight not found to infer config".to_string()))?;
        let vocab_size = emb_shape[0];
        let d_model = emb_shape[1];

        // 2. Find layer 0 inner norm or out_proj to determine d_inner
        let inner_keys = ["backbone.layers.0.mixer.norm.weight", "layers.0.mixer.norm.weight"];
        let d_inner = if let Some((_, shape)) = find_shape(&inner_keys) {
            shape[0]
        } else {
            d_model * 2
        };
        let expand = if d_model > 0 { d_inner / d_model } else { 2 };

        // 3. Find dt_bias or A_log or D to determine nheads
        let dt_keys = [
            "backbone.layers.0.mixer.dt_bias",
            "layers.0.mixer.dt_bias",
            "backbone.layers.0.mixer.A_log",
            "layers.0.mixer.A_log",
        ];
        let nheads = if let Some((_, shape)) = find_shape(&dt_keys) {
            shape[0]
        } else {
            d_inner / 64
        };
        let headdim = if nheads > 0 { d_inner / nheads } else { 64 };

        // 4. Count layers
        let mut n_layers = 0;
        while get_shape(&format!("backbone.layers.{n_layers}.mixer.in_proj.weight")).is_some()
            || get_shape(&format!("layers.{n_layers}.mixer.in_proj.weight")).is_some()
        {
            n_layers += 1;
        }
        if n_layers == 0 {
            n_layers = 1;
        }

        // 5. Find conv1d to determine d_conv and d_state
        let conv_keys = ["backbone.layers.0.mixer.conv1d.weight", "layers.0.mixer.conv1d.weight"];
        let (d_state, ngroups) = if let Some((_, shape)) = find_shape(&conv_keys) {
            let d_conv = shape[0];
            let extra = d_conv.saturating_sub(d_inner);
            let ngroups = 1;
            let d_state = if extra > 0 { extra / 2 } else { 128 };
            (d_state, ngroups)
        } else {
            (128, 1)
        };

        Ok(BiMamba2Config::new(vocab_size)
            .with_d_model(d_model)
            .with_n_layers(n_layers)
            .with_d_state(d_state)
            .with_headdim(headdim)
            .with_expand(expand)
            .with_ngroups(ngroups)
            .with_use_dt_bias(true)
            .with_use_inner_norm(true))
    }


    /// Loads weights from a safetensors file into an existing BiMamba2Backbone.
    /// The file is memory-mapped, so the payload is never copied into an intermediate buffer.
    pub fn load_file<B: Backend, P: AsRef<Path>>(
        model: &mut BiMamba2Backbone<B>,
        path: P,
        options: &LoaderOptions,
    ) -> Result<LoadReport, LoadError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the checkpoint is treated as read-only for the duration of the load; every
        // tensor is copied out of the mapping before this function returns. Concurrent external
        // modification of the file is not supported.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::load_bytes(model, &mmap, options)
    }

    /// Loads weights from in-memory safetensors bytes into an existing BiMamba2Backbone
    pub fn load_bytes<B: Backend>(
        model: &mut BiMamba2Backbone<B>,
        bytes: &[u8],
        options: &LoaderOptions,
    ) -> Result<LoadReport, LoadError> {
        let safetensors = SafeTensors::deserialize(bytes)?;
        // Shape/device queries use the lazy accessors so params about to be replaced are never
        // randomly initialized.
        let device = model.embedding.weight.lazy_device();

        let mut report = LoadReport {
            layers_loaded: 0,
            total_model_layers: model.layers.len(),
            tensors_loaded: 0,
            parameters_transferred: 0,
            embedding_loaded: false,
            final_norm_loaded: false,
            heads_loaded: false,
        };

        // 1. Load Embedding
        if options.load_embedding {
            let embedding_keys = [
                "backbone.embeddings.weight",
                "embeddings.weight",
                "backbone.embedding.weight",
                "embedding.weight",
            ];
            if let Some((name, view)) = find_tensor(&safetensors, &embedding_keys) {
                let [vocab, d_model] = lazy_dims(&model.embedding.weight);
                validate_shape(name, view.shape(), &[vocab, d_model])?;
                let data = tensor_to_f32_vec(&view)?;
                let tensor = Tensor::<B, 2>::from_data(
                    TensorData::new(data, vec![vocab, d_model]),
                    &device,
                );
                model.embedding.weight = Param::from_tensor(tensor);
                report.embedding_loaded = true;
                report.tensors_loaded += 1;
                report.parameters_transferred += vocab * d_model;
            }
        }

        // 2. Load Layers
        if options.load_layers {
            for (i, layer) in model.layers.iter_mut().enumerate() {
                // Check if this layer exists in the checkpoint
                let in_proj_keys = [
                    format!("backbone.layers.{i}.mixer.in_proj.weight"),
                    format!("layers.{i}.mixer.in_proj.weight"),
                ];
                let in_proj_view = find_tensor_owned(&safetensors, &in_proj_keys);

                if in_proj_view.is_none() {
                    if options.strict_layer_count {
                        return Err(LoadError::TensorNotFound(format!(
                            "Missing layer {i} weights in checkpoint"
                        )));
                    }
                    // Stop if checkpoint has fewer layers than requested and strict is false
                    break;
                }

                // In-projection weight: checkpoint has [in_proj_dim, d_model], Burn Linear has [d_model, in_proj_dim]
                let (name, view) = in_proj_view.unwrap();
                let [d_model, in_proj_dim] = lazy_dims(&layer.in_proj.weight);
                validate_shape(&name, view.shape(), &[in_proj_dim, d_model])?;
                let in_proj_data = tensor_to_f32_vec(&view)?;
                // Transpose from [in_proj_dim, d_model] to [d_model, in_proj_dim]
                let in_proj_t = Tensor::<B, 2>::from_data(
                    TensorData::new(in_proj_data, vec![in_proj_dim, d_model]),
                    &device,
                ).swap_dims(0, 1);
                layer.in_proj.weight = Param::from_tensor(in_proj_t);
                report.tensors_loaded += 1;
                report.parameters_transferred += d_model * in_proj_dim;

                // Conv1d weight: [d_conv, 1, 4]
                let conv_w_keys = [
                    format!("backbone.layers.{i}.mixer.conv1d.weight"),
                    format!("layers.{i}.mixer.conv1d.weight"),
                ];
                if let Some((name, view)) = find_tensor_owned(&safetensors, &conv_w_keys) {
                    let [d_conv, in_ch, k] = lazy_dims(&layer.conv1d.weight);
                    validate_shape(&name, view.shape(), &[d_conv, in_ch, k])?;
                    let data = tensor_to_f32_vec(&view)?;
                    let tensor = Tensor::<B, 3>::from_data(
                        TensorData::new(data, vec![d_conv, in_ch, k]),
                        &device,
                    );
                    layer.conv1d.weight = Param::from_tensor(tensor);
                    report.tensors_loaded += 1;
                    report.parameters_transferred += d_conv * in_ch * k;
                }

                // Conv1d bias: [conv_channels]
                let conv_b_keys = [
                    format!("backbone.layers.{i}.mixer.conv1d.bias"),
                    format!("layers.{i}.mixer.conv1d.bias"),
                ];
                if let Some((name, view)) = find_tensor_owned(&safetensors, &conv_b_keys) {
                    let [conv_channels, _, _] = lazy_dims(&layer.conv1d.weight);
                    validate_shape(&name, view.shape(), &[conv_channels])?;
                    let data = tensor_to_f32_vec(&view)?;
                    let tensor = Tensor::<B, 1>::from_data(
                        TensorData::new(data, vec![conv_channels]),
                        &device,
                    );
                    layer.conv1d.bias = Some(Param::from_tensor(tensor));
                    report.tensors_loaded += 1;
                    report.parameters_transferred += conv_channels;
                }

                // dt_bias: [nheads]
                if let Some(dt_bias_param) = &mut layer.dt_bias {
                    let dt_bias_keys = [
                        format!("backbone.layers.{i}.mixer.dt_bias"),
                        format!("layers.{i}.mixer.dt_bias"),
                    ];
                    if let Some((name, view)) = find_tensor_owned(&safetensors, &dt_bias_keys) {
                        let nheads = layer.nheads;
                        validate_shape(&name, view.shape(), &[nheads])?;
                        let data = tensor_to_f32_vec(&view)?;
                        let tensor = Tensor::<B, 1>::from_data(
                            TensorData::new(data, vec![nheads]),
                            &device,
                        );
                        *dt_bias_param = Param::from_tensor(tensor);
                        report.tensors_loaded += 1;
                        report.parameters_transferred += nheads;
                    }
                }

                // A_log: [nheads]
                let a_log_keys = [
                    format!("backbone.layers.{i}.mixer.A_log"),
                    format!("layers.{i}.mixer.A_log"),
                ];
                if let Some((name, view)) = find_tensor_owned(&safetensors, &a_log_keys) {
                    let nheads = layer.nheads;
                    validate_shape(&name, view.shape(), &[nheads])?;
                    let data = tensor_to_f32_vec(&view)?;
                    let tensor = Tensor::<B, 1>::from_data(
                        TensorData::new(data, vec![nheads]),
                        &device,
                    );
                    layer.a_log = Param::from_tensor(tensor);
                    report.tensors_loaded += 1;
                    report.parameters_transferred += nheads;
                }

                // D (d_skip): [nheads]
                let d_keys = [
                    format!("backbone.layers.{i}.mixer.D"),
                    format!("layers.{i}.mixer.D"),
                ];
                if let Some((name, view)) = find_tensor_owned(&safetensors, &d_keys) {
                    let nheads = layer.nheads;
                    validate_shape(&name, view.shape(), &[nheads])?;
                    let data = tensor_to_f32_vec(&view)?;
                    let tensor = Tensor::<B, 1>::from_data(
                        TensorData::new(data, vec![nheads]),
                        &device,
                    );
                    layer.d_skip = Param::from_tensor(tensor);
                    report.tensors_loaded += 1;
                    report.parameters_transferred += nheads;
                }

                // inner_norm (mixer.norm.weight): [d_inner]
                if let Some(inner_norm) = &mut layer.inner_norm {
                    let inner_norm_keys = [
                        format!("backbone.layers.{i}.mixer.norm.weight"),
                        format!("layers.{i}.mixer.norm.weight"),
                    ];
                    if let Some((name, view)) = find_tensor_owned(&safetensors, &inner_norm_keys) {
                        let d_inner = layer.d_inner;
                        validate_shape(&name, view.shape(), &[d_inner])?;
                        let data = tensor_to_f32_vec(&view)?;
                        let tensor = Tensor::<B, 1>::from_data(
                            TensorData::new(data, vec![d_inner]),
                            &device,
                        );
                        inner_norm.gamma = Param::from_tensor(tensor);
                        report.tensors_loaded += 1;
                        report.parameters_transferred += d_inner;
                    }
                }

                // Out-projection weight: checkpoint has [d_model, d_inner], Burn Linear has [d_inner, d_model]
                let out_proj_keys = [
                    format!("backbone.layers.{i}.mixer.out_proj.weight"),
                    format!("layers.{i}.mixer.out_proj.weight"),
                ];
                if let Some((name, view)) = find_tensor_owned(&safetensors, &out_proj_keys) {
                    let [d_inner, d_model] = lazy_dims(&layer.out_proj.weight);
                    validate_shape(&name, view.shape(), &[d_model, d_inner])?;
                    let data = tensor_to_f32_vec(&view)?;
                    let out_proj_t = Tensor::<B, 2>::from_data(
                        TensorData::new(data, vec![d_model, d_inner]),
                        &device,
                    ).swap_dims(0, 1);
                    layer.out_proj.weight = Param::from_tensor(out_proj_t);
                    report.tensors_loaded += 1;
                    report.parameters_transferred += d_inner * d_model;
                }

                // Layer RMSNorm weight: [d_model]
                let norm_keys = [
                    format!("backbone.layers.{i}.norm.weight"),
                    format!("layers.{i}.norm.weight"),
                ];
                if let Some((name, view)) = find_tensor_owned(&safetensors, &norm_keys) {
                    let [d_model] = lazy_dims(&layer.norm.gamma);
                    validate_shape(&name, view.shape(), &[d_model])?;
                    let data = tensor_to_f32_vec(&view)?;
                    let tensor = Tensor::<B, 1>::from_data(
                        TensorData::new(data, vec![d_model]),
                        &device,
                    );
                    layer.norm.gamma = Param::from_tensor(tensor);
                    report.tensors_loaded += 1;
                    report.parameters_transferred += d_model;
                }

                report.layers_loaded += 1;
            }
        }

        // 3. Load Final RMSNorm
        if options.load_final_norm {
            let final_norm_keys = [
                "backbone.norm_f.weight",
                "norm_f.weight",
                "backbone.final_norm.weight",
                "final_norm.weight",
            ];
            if let Some((name, view)) = find_tensor(&safetensors, &final_norm_keys) {
                let [d_model] = lazy_dims(&model.final_norm.gamma);
                validate_shape(name, view.shape(), &[d_model])?;
                let data = tensor_to_f32_vec(&view)?;
                let tensor = Tensor::<B, 1>::from_data(
                    TensorData::new(data, vec![d_model]),
                    &device,
                );
                model.final_norm.gamma = Param::from_tensor(tensor);
                report.final_norm_loaded = true;
                report.tensors_loaded += 1;
                report.parameters_transferred += d_model;
            }
        }

        // 4. Load JEV Decision & Metric Heads
        if options.load_heads {
            let mut heads_found = 0;
            heads_found += load_linear_layer(&mut model.heads.choice_fc1, "heads.choice_fc1", &safetensors, &device, &mut report)?;
            heads_found += load_linear_layer(&mut model.heads.choice_fc2, "heads.choice_fc2", &safetensors, &device, &mut report)?;
            heads_found += load_linear_layer(&mut model.heads.noul_fc1, "heads.noul_fc1", &safetensors, &device, &mut report)?;
            heads_found += load_linear_layer(&mut model.heads.noul_fc2, "heads.noul_fc2", &safetensors, &device, &mut report)?;
            heads_found += load_linear_layer(&mut model.heads.score_fc1, "heads.score_fc1", &safetensors, &device, &mut report)?;
            heads_found += load_linear_layer(&mut model.heads.score_fc2, "heads.score_fc2", &safetensors, &device, &mut report)?;
            heads_found += load_linear_layer(&mut model.heads.knn_fc1, "heads.knn_fc1", &safetensors, &device, &mut report)?;
            heads_found += load_linear_layer(&mut model.heads.knn_fc2, "heads.knn_fc2", &safetensors, &device, &mut report)?;

            let temp_keys = [
                "heads.temperature".to_string(),
                "backbone.heads.temperature".to_string(),
            ];
            if let Some((_, view)) = find_tensor_owned(&safetensors, &temp_keys) {
                let data = tensor_to_f32_vec(&view)?;
                let tensor = Tensor::<B, 1>::from_data(TensorData::new(data, vec![1]), &device);
                model.heads.temperature = Param::from_tensor(tensor);
                report.tensors_loaded += 1;
                report.parameters_transferred += 1;
                heads_found += 1;
            }

            if heads_found > 0 {
                report.heads_loaded = true;
            }
        }

        Ok(report)
    }

    /// Serializes and saves all model weights (backbone and JEV heads) to a safetensors file
    pub fn save_file<B: Backend, P: AsRef<Path>>(
        model: &BiMamba2Backbone<B>,
        path: P,
    ) -> Result<(), LoadError> {
        let mut tensors_data: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

        // 1. Embedding
        let [vocab, d_model] = model.embedding.weight.dims();
        let emb_bytes = param_bytes(model.embedding.weight.val());
        tensors_data.push(("backbone.embeddings.weight".to_string(), vec![vocab, d_model], emb_bytes));

        // 2. Layers
        for (i, layer) in model.layers.iter().enumerate() {
            // in_proj weight: transpose from [d_model, in_proj_dim] to [in_proj_dim, d_model]
            let [d_model, in_proj_dim] = layer.in_proj.weight.dims();
            let in_proj_t = layer.in_proj.weight.val().swap_dims(0, 1);
            let in_proj_bytes = param_bytes(in_proj_t);
            tensors_data.push((
                format!("backbone.layers.{i}.mixer.in_proj.weight"),
                vec![in_proj_dim, d_model],
                in_proj_bytes,
            ));

            // conv1d weight
            let conv_bytes = param_bytes(layer.conv1d.weight.val());
            tensors_data.push((
                format!("backbone.layers.{i}.mixer.conv1d.weight"),
                layer.conv1d.weight.dims().to_vec(),
                conv_bytes,
            ));

            // conv1d bias
            if let Some(ref bias) = layer.conv1d.bias {
                let bias_bytes = param_bytes(bias.val());
                tensors_data.push((
                    format!("backbone.layers.{i}.mixer.conv1d.bias"),
                    bias.dims().to_vec(),
                    bias_bytes,
                ));
            }

            // dt_bias
            if let Some(ref dt) = layer.dt_bias {
                let dt_bytes = param_bytes(dt.val());
                tensors_data.push((
                    format!("backbone.layers.{i}.mixer.dt_bias"),
                    dt.dims().to_vec(),
                    dt_bytes,
                ));
            }

            // A_log
            let a_log_bytes = param_bytes(layer.a_log.val());
            tensors_data.push((
                format!("backbone.layers.{i}.mixer.A_log"),
                layer.a_log.dims().to_vec(),
                a_log_bytes,
            ));

            // D (d_skip)
            let d_skip_bytes = param_bytes(layer.d_skip.val());
            tensors_data.push((
                format!("backbone.layers.{i}.mixer.D"),
                layer.d_skip.dims().to_vec(),
                d_skip_bytes,
            ));

            // inner_norm
            if let Some(ref inner_norm) = layer.inner_norm {
                let norm_bytes = param_bytes(inner_norm.gamma.val());
                tensors_data.push((
                    format!("backbone.layers.{i}.mixer.norm.weight"),
                    inner_norm.gamma.dims().to_vec(),
                    norm_bytes,
                ));
            }

            // out_proj weight: transpose from [d_inner, d_model] to [d_model, d_inner]
            let [d_inner, d_model] = layer.out_proj.weight.dims();
            let out_proj_t = layer.out_proj.weight.val().swap_dims(0, 1);
            let out_proj_bytes = param_bytes(out_proj_t);
            tensors_data.push((
                format!("backbone.layers.{i}.mixer.out_proj.weight"),
                vec![d_model, d_inner],
                out_proj_bytes,
            ));

            // Layer RMSNorm
            let norm_bytes = param_bytes(layer.norm.gamma.val());
            tensors_data.push((
                format!("backbone.layers.{i}.norm.weight"),
                layer.norm.gamma.dims().to_vec(),
                norm_bytes,
            ));
        }

        // 3. Final RMSNorm
        let final_norm_bytes = param_bytes(model.final_norm.gamma.val());
        tensors_data.push((
            "backbone.norm_f.weight".to_string(),
            model.final_norm.gamma.dims().to_vec(),
            final_norm_bytes,
        ));

        // 4. JEV Heads
        push_heads_tensors(&model.heads, &mut tensors_data);

        write_safetensors(&tensors_data, None, path.as_ref())
    }

    /// Initializes a backbone from a checkpoint, inferring its architecture from the header.
    ///
    /// Any `heads.*` tensors in the file are ignored; heads come from a separate artifact
    /// (see [`Mamba2CheckpointLoader::load_heads_file`]). The returned `sha256` identifies the
    /// backbone file and is what heads artifacts are bound to.
    pub fn load_backbone_file<B: Backend, P: AsRef<Path>>(
        path: P,
        device: &B::Device,
    ) -> Result<LoadedBackbone<B>, LoadError> {
        let path = path.as_ref();
        let config = Self::infer_config_file(path)?;
        let mut model: BiMamba2Backbone<B> = config.init(device);
        let options = LoaderOptions {
            load_heads: false,
            ..LoaderOptions::default()
        };
        let report = Self::load_file(&mut model, path, &options)?;
        if report.layers_loaded == 0 || !report.embedding_loaded {
            return Err(LoadError::InvalidConfiguration(format!(
                "{} does not look like a Mamba-2 backbone (embedding loaded: {}, layers loaded: {})",
                path.display(),
                report.embedding_loaded,
                report.layers_loaded
            )));
        }
        let sha256 = FeatureCache::compute_file_hash(path)?;
        Ok(LoadedBackbone {
            model,
            config,
            sha256,
            report,
        })
    }

    /// Saves only the JEV head tensors plus the metadata needed to use them at inference time.
    pub fn save_heads_file<B: Backend, P: AsRef<Path>>(
        heads: &UnifiedHeads<B>,
        metadata: &HeadsMetadata,
        path: P,
    ) -> Result<(), LoadError> {
        let mut tensors_data = Vec::new();
        push_heads_tensors(heads, &mut tensors_data);

        let meta_json = serde_json::to_string(metadata).map_err(|e| {
            LoadError::InvalidConfiguration(format!("Failed to serialize heads metadata: {e}"))
        })?;
        let header = HashMap::from([
            (HEADS_FORMAT_KEY.to_string(), HEADS_FORMAT.to_string()),
            (HEADS_METADATA_KEY.to_string(), meta_json),
        ]);
        write_safetensors(&tensors_data, Some(header), path.as_ref())
    }

    /// Reads just the metadata of a heads artifact.
    pub fn read_heads_metadata<P: AsRef<Path>>(path: P) -> Result<HeadsMetadata, LoadError> {
        let bytes = std::fs::read(path)?;
        parse_heads_metadata(&bytes)
    }

    /// Loads a heads artifact, refusing it unless it was trained on the backbone whose file
    /// hash is `backbone_sha256`.
    pub fn load_heads_file<B: Backend, P: AsRef<Path>>(
        path: P,
        backbone_sha256: &str,
        device: &B::Device,
    ) -> Result<(UnifiedHeads<B>, HeadsMetadata), LoadError> {
        let bytes = std::fs::read(path)?;
        let metadata = parse_heads_metadata(&bytes)?;
        if metadata.backbone_sha256 != backbone_sha256 {
            return Err(LoadError::BackboneMismatch {
                expected: metadata.backbone_sha256,
                found: backbone_sha256.to_string(),
            });
        }

        let safetensors = SafeTensors::deserialize(&bytes)?;
        let mut heads: UnifiedHeads<B> = UnifiedHeadsConfig::new(metadata.d_model)
            .with_knn_dim(metadata.knn_dim)
            .with_num_rubric_bins(metadata.num_rubric_bins)
            .init(device);

        let mut report = LoadReport {
            layers_loaded: 0,
            total_model_layers: 0,
            tensors_loaded: 0,
            parameters_transferred: 0,
            embedding_loaded: false,
            final_norm_loaded: false,
            heads_loaded: false,
        };
        for (prefix, linear) in heads_linears_mut(&mut heads) {
            let found = load_linear_layer(linear, prefix, &safetensors, device, &mut report)?;
            if found == 0 {
                return Err(LoadError::TensorNotFound(format!("{prefix}.weight")));
            }
        }
        let (_, view) = find_tensor_owned(&safetensors, &["heads.temperature".to_string()])
            .ok_or_else(|| LoadError::TensorNotFound("heads.temperature".to_string()))?;
        validate_shape("heads.temperature", view.shape(), &[1])?;
        let data = tensor_to_f32_vec(&view)?;
        heads.temperature = Param::from_tensor(Tensor::<B, 1>::from_data(
            TensorData::new(data, vec![1]),
            device,
        ));

        Ok((heads, metadata))
    }
}

/// A backbone loaded by [`Mamba2CheckpointLoader::load_backbone_file`].
pub struct LoadedBackbone<B: Backend> {
    pub model: BiMamba2Backbone<B>,
    pub config: BiMamba2Config,
    /// SHA-256 of the checkpoint file.
    pub sha256: String,
    pub report: LoadReport,
}

// =====================================================================
// Heads-Only Artifact
// =====================================================================

const HEADS_FORMAT_KEY: &str = "format";
const HEADS_FORMAT: &str = "burn-mamba/reflex-heads";
const HEADS_METADATA_KEY: &str = "reflex_heads";
/// Bumped whenever the heads artifact layout or metadata schema changes incompatibly.
pub const HEADS_FORMAT_VERSION: u32 = 1;

/// Contract between a trained heads artifact and the inference app, stored in the
/// safetensors header.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeadsMetadata {
    pub format_version: u32,
    /// Delimiter token IDs the heads were trained with; inference must tokenize identically.
    pub delimiters: DelimiterConfig,
    pub d_model: usize,
    pub knn_dim: usize,
    pub num_rubric_bins: usize,
    /// SHA-256 of the backbone checkpoint whose hidden states the heads were fit to.
    pub backbone_sha256: String,
}

impl HeadsMetadata {
    pub fn new(
        heads_config: &UnifiedHeadsConfig,
        delimiters: DelimiterConfig,
        backbone_sha256: String,
    ) -> Self {
        Self {
            format_version: HEADS_FORMAT_VERSION,
            delimiters,
            d_model: heads_config.d_model,
            knn_dim: heads_config.knn_dim,
            num_rubric_bins: heads_config.num_rubric_bins,
            backbone_sha256,
        }
    }
}

fn parse_heads_metadata(bytes: &[u8]) -> Result<HeadsMetadata, LoadError> {
    let (_, header) = SafeTensors::read_metadata(bytes)?;
    let map = header.metadata().as_ref();
    let format = map.and_then(|m| m.get(HEADS_FORMAT_KEY));
    if format.map(String::as_str) != Some(HEADS_FORMAT) {
        return Err(LoadError::InvalidConfiguration(format!(
            "Not a reflex heads artifact (format = {format:?}); full-model checkpoints are not accepted here"
        )));
    }
    let json = map
        .and_then(|m| m.get(HEADS_METADATA_KEY))
        .ok_or_else(|| LoadError::InvalidConfiguration("Heads artifact is missing its metadata".to_string()))?;
    let metadata: HeadsMetadata = serde_json::from_str(json).map_err(|e| {
        LoadError::InvalidConfiguration(format!("Failed to parse heads metadata: {e}"))
    })?;
    if metadata.format_version != HEADS_FORMAT_VERSION {
        return Err(LoadError::InvalidConfiguration(format!(
            "Unsupported heads format version {} (expected {HEADS_FORMAT_VERSION})",
            metadata.format_version
        )));
    }
    Ok(metadata)
}

type NamedTensorBytes = Vec<(String, Vec<usize>, Vec<u8>)>;

fn heads_linears<B: Backend>(heads: &UnifiedHeads<B>) -> [(&'static str, &Linear<B>); 8] {
    [
        ("heads.choice_fc1", &heads.choice_fc1),
        ("heads.choice_fc2", &heads.choice_fc2),
        ("heads.noul_fc1", &heads.noul_fc1),
        ("heads.noul_fc2", &heads.noul_fc2),
        ("heads.score_fc1", &heads.score_fc1),
        ("heads.score_fc2", &heads.score_fc2),
        ("heads.knn_fc1", &heads.knn_fc1),
        ("heads.knn_fc2", &heads.knn_fc2),
    ]
}

fn heads_linears_mut<B: Backend>(heads: &mut UnifiedHeads<B>) -> [(&'static str, &mut Linear<B>); 8] {
    [
        ("heads.choice_fc1", &mut heads.choice_fc1),
        ("heads.choice_fc2", &mut heads.choice_fc2),
        ("heads.noul_fc1", &mut heads.noul_fc1),
        ("heads.noul_fc2", &mut heads.noul_fc2),
        ("heads.score_fc1", &mut heads.score_fc1),
        ("heads.score_fc2", &mut heads.score_fc2),
        ("heads.knn_fc1", &mut heads.knn_fc1),
        ("heads.knn_fc2", &mut heads.knn_fc2),
    ]
}

/// Appends `heads.*` tensors in checkpoint layout (Linear weights stored `[out, in]`).
fn push_heads_tensors<B: Backend>(heads: &UnifiedHeads<B>, list: &mut NamedTensorBytes) {
    for (prefix, linear) in heads_linears(heads) {
        let [d_in, d_out] = linear.weight.dims();
        let wt = linear.weight.val().swap_dims(0, 1);
        list.push((format!("{prefix}.weight"), vec![d_out, d_in], param_bytes(wt)));

        if let Some(ref bias) = linear.bias {
            list.push((format!("{prefix}.bias"), bias.dims().to_vec(), param_bytes(bias.val())));
        }
    }
    list.push((
        "heads.temperature".to_string(),
        vec![1],
        param_bytes(heads.temperature.val()),
    ));
}

fn write_safetensors(
    tensors_data: &NamedTensorBytes,
    metadata: Option<HashMap<String, String>>,
    path: &Path,
) -> Result<(), LoadError> {
    let mut views = BTreeMap::new();
    for (name, shape, bytes) in tensors_data {
        views.insert(
            name.clone(),
            TensorView::new(Dtype::F32, shape.clone(), bytes)?,
        );
    }

    let serialized = safetensors::serialize(&views, metadata)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serialized)?;
    Ok(())
}

fn load_linear_layer<B: Backend>(
    linear: &mut Linear<B>,
    prefix: &str,
    safetensors: &SafeTensors,
    device: &B::Device,
    report: &mut LoadReport,
) -> Result<usize, LoadError> {
    let mut found = 0;
    let [d_in, d_out] = lazy_dims(&linear.weight);
    let weight_keys = [
        format!("{prefix}.weight"),
        format!("backbone.{prefix}.weight"),
    ];
    if let Some((name, view)) = find_tensor_owned(safetensors, &weight_keys) {
        let shape = view.shape();
        let data = tensor_to_f32_vec(&view)?;
        let tensor = if shape == [d_out, d_in] {
            // Transpose on the host so the weight is contiguous `[d_in, d_out]`, exactly as a
            // freshly trained Linear is; a strided view changes matmul accumulation order.
            let mut transposed = vec![0.0f32; data.len()];
            for (o, row) in data.chunks_exact(d_in).enumerate() {
                for (i, &v) in row.iter().enumerate() {
                    transposed[i * d_out + o] = v;
                }
            }
            Tensor::<B, 2>::from_data(TensorData::new(transposed, vec![d_in, d_out]), device)
        } else if shape == [d_in, d_out] {
            Tensor::<B, 2>::from_data(TensorData::new(data, vec![d_in, d_out]), device)
        } else {
            return Err(LoadError::ShapeMismatch {
                tensor: name,
                expected: vec![d_out, d_in],
                found: shape.to_vec(),
            });
        };
        linear.weight = Param::from_tensor(tensor);
        report.tensors_loaded += 1;
        report.parameters_transferred += d_in * d_out;
        found += 1;
    }

    let bias_keys = [
        format!("{prefix}.bias"),
        format!("backbone.{prefix}.bias"),
    ];
    if let Some((_, view)) = find_tensor_owned(safetensors, &bias_keys) {
        let shape = view.shape();
        if shape == [d_out] {
            let data = tensor_to_f32_vec(&view)?;
            let tensor = Tensor::<B, 1>::from_data(TensorData::new(data, vec![d_out]), device);
            linear.bias = Some(Param::from_tensor(tensor));
            report.tensors_loaded += 1;
            report.parameters_transferred += d_out;
            found += 1;
        }
    }
    Ok(found)
}

fn find_tensor<'a>(
    safetensors: &'a SafeTensors<'a>,
    keys: &[&'a str],
) -> Option<(&'a str, safetensors::tensor::TensorView<'a>)> {
    for &key in keys {
        if let Ok(view) = safetensors.tensor(key) {
            return Some((key, view));
        }
    }
    None
}

fn find_tensor_owned<'a>(
    safetensors: &'a SafeTensors<'a>,
    keys: &[String],
) -> Option<(String, safetensors::tensor::TensorView<'a>)> {
    for key in keys {
        if let Ok(view) = safetensors.tensor(key) {
            return Some((key.clone(), view));
        }
    }
    None
}

fn validate_shape(name: &str, found: &[usize], expected: &[usize]) -> Result<(), LoadError> {
    if found != expected {
        return Err(LoadError::ShapeMismatch {
            tensor: name.to_string(),
            expected: expected.to_vec(),
            found: found.to_vec(),
        });
    }
    Ok(())
}

/// Decodes a safetensors view (F32/F16/BF16, little-endian) into an f32 vector.
/// `pod_collect_to_vec` copies into an aligned buffer, so unaligned mmap offsets are fine.
pub(crate) fn tensor_to_f32_vec(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>, LoadError> {
    let data = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => Ok(bytemuck::pod_collect_to_vec::<u8, f32>(data)),
        safetensors::Dtype::F16 => {
            let halfs = bytemuck::pod_collect_to_vec::<u8, half::f16>(data);
            let mut out = vec![0.0f32; halfs.len()];
            halfs.convert_to_f32_slice(&mut out);
            Ok(out)
        }
        safetensors::Dtype::BF16 => {
            let halfs = bytemuck::pod_collect_to_vec::<u8, half::bf16>(data);
            let mut out = vec![0.0f32; halfs.len()];
            halfs.convert_to_f32_slice(&mut out);
            Ok(out)
        }
        other => Err(LoadError::UnsupportedDtype(format!("{:?}", other))),
    }
}

/// Parameter dims without triggering lazy initialization.
fn lazy_dims<B: Backend, const D: usize>(param: &Param<Tensor<B, D>>) -> [usize; D] {
    param.lazy_shape().dims()
}

/// Raw little-endian f32 bytes of a tensor, laid out contiguously in logical order.
pub(crate) fn param_bytes<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<u8> {
    tensor.into_data().convert::<f32>().as_bytes().to_vec()
}
