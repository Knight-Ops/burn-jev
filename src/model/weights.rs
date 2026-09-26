//! Shared safetensors plumbing: error type, dtype decoding, and serialization helpers used by
//! the encoder loader, the decision artifact and the feature cache.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use half::slice::HalfFloatSliceExt;
use safetensors::tensor::{Dtype, TensorView};

// Safetensors payloads are little-endian; decoding reinterprets bytes in native order.
#[cfg(target_endian = "big")]
compile_error!("burn-jev's safetensors loader assumes a little-endian target");

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
    /// An artifact was trained against a different encoder (or tokenizer) than the one supplied.
    EncoderMismatch {
        what: &'static str,
        expected: String,
        found: String,
    },
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
            LoadError::EncoderMismatch { what, expected, found } => write!(
                f,
                "Artifact was trained on {what} sha256 {expected}, but the supplied one is {found}"
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

/// `(name, shape, dtype, little-endian bytes)` ready for [`write_safetensors`].
pub(crate) type NamedTensorBytes = Vec<(String, Vec<usize>, Dtype, Vec<u8>)>;

/// Decodes a safetensors view (F32/F16/BF16, little-endian) into an f32 vector.
/// `pod_collect_to_vec` copies into an aligned buffer, so unaligned mmap offsets are fine.
pub(crate) fn tensor_to_f32_vec(view: &TensorView) -> Result<Vec<f32>, LoadError> {
    let data = view.data();
    match view.dtype() {
        Dtype::F32 => Ok(bytemuck::pod_collect_to_vec::<u8, f32>(data)),
        Dtype::F16 => {
            let halfs = bytemuck::pod_collect_to_vec::<u8, half::f16>(data);
            let mut out = vec![0.0f32; halfs.len()];
            halfs.convert_to_f32_slice(&mut out);
            Ok(out)
        }
        Dtype::BF16 => {
            let halfs = bytemuck::pod_collect_to_vec::<u8, half::bf16>(data);
            let mut out = vec![0.0f32; halfs.len()];
            halfs.convert_to_f32_slice(&mut out);
            Ok(out)
        }
        other => Err(LoadError::UnsupportedDtype(format!("{:?}", other))),
    }
}

/// Little-endian f16 bytes of an f32 slice (used for the feature cache).
pub(crate) fn f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut halfs = vec![half::f16::ZERO; values.len()];
    halfs.convert_from_f32_slice(values);
    bytemuck::cast_slice(&halfs).to_vec()
}

pub(crate) fn write_safetensors(
    tensors_data: &NamedTensorBytes,
    metadata: Option<HashMap<String, String>>,
    path: &Path,
) -> Result<(), LoadError> {
    let mut views = BTreeMap::new();
    for (name, shape, dtype, bytes) in tensors_data {
        views.insert(name.clone(), TensorView::new(*dtype, shape.clone(), bytes)?);
    }

    let serialized = safetensors::serialize(&views, metadata)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serialized)?;
    Ok(())
}

pub(crate) fn validate_shape(name: &str, found: &[usize], expected: &[usize]) -> Result<(), LoadError> {
    if found != expected {
        return Err(LoadError::ShapeMismatch {
            tensor: name.to_string(),
            expected: expected.to_vec(),
            found: found.to_vec(),
        });
    }
    Ok(())
}

/// Transposes a row-major `[rows, cols]` buffer on the host so the result is contiguous
/// `[cols, rows]`; a strided device view would change matmul accumulation order.
pub(crate) fn transpose_host(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for (r, row) in data.chunks_exact(cols).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            out[c * rows + r] = v;
        }
    }
    out
}

/// SHA-256 of a file's contents as lowercase hex.
///
/// Stable across Rust releases and machines, so it doubles as the encoder/tokenizer identity
/// recorded in artifacts and feature-cache keys.
pub fn sha256_file<P: AsRef<Path>>(path: P) -> Result<String, std::io::Error> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
