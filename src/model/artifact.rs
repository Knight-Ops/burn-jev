//! The trained decision artifact: reader + heads plus the contract needed to use them.
//!
//! A single safetensors file. The header metadata carries [`ArtifactMetadata`] (which
//! encoder and tokenizer the model was fit to, the encoding layout and the model config);
//! the payload is one `U8` tensor holding the Burn record of the [`DecisionModel`]. The
//! encoder itself is never stored: it is frozen, and identified by its SHA-256.

use std::collections::HashMap;
use std::path::Path;

use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::backend::Backend,
};
use safetensors::{tensor::Dtype, SafeTensors};
use serde::{Deserialize, Serialize};

use crate::encoding::{EncodingConfig, ENCODING_VERSION};
use crate::model::decision::{DecisionModel, DecisionModelConfig};
use crate::model::weights::{write_safetensors, LoadError};

const FORMAT_KEY: &str = "format";
const FORMAT: &str = "burn-mamba/reflex-decision";
const METADATA_KEY: &str = "reflex_decision";
const RECORD_TENSOR: &str = "record";
/// Bumped whenever the artifact layout or metadata schema changes incompatibly.
pub const ARTIFACT_FORMAT_VERSION: u32 = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub format_version: u32,
    /// SHA-256 of the encoder's `model.safetensors`.
    pub encoder_sha256: String,
    /// SHA-256 of the `tokenizer.json` used to build the training features.
    pub tokenizer_sha256: String,
    pub encoding_version: u32,
    pub encoding: EncodingConfig,
    pub model: DecisionModelConfig,
    /// Epoch whose weights were kept (early stopping), if known.
    #[serde(default)]
    pub best_epoch: Option<usize>,
}

impl ArtifactMetadata {
    pub fn new(
        model: DecisionModelConfig,
        encoding: EncodingConfig,
        encoder_sha256: String,
        tokenizer_sha256: String,
    ) -> Self {
        Self {
            format_version: ARTIFACT_FORMAT_VERSION,
            encoder_sha256,
            tokenizer_sha256,
            encoding_version: ENCODING_VERSION,
            encoding,
            model,
            best_epoch: None,
        }
    }
}

fn recorder() -> NamedMpkBytesRecorder<FullPrecisionSettings> {
    NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
}

pub fn save_artifact<B: Backend, P: AsRef<Path>>(
    model: &DecisionModel<B>,
    metadata: &ArtifactMetadata,
    path: P,
) -> Result<(), LoadError> {
    let bytes = Recorder::<B>::record(&recorder(), model.clone().into_record(), ())
        .map_err(|e| LoadError::InvalidConfiguration(format!("failed to serialize decision model: {e}")))?;
    let meta_json = serde_json::to_string(metadata)
        .map_err(|e| LoadError::InvalidConfiguration(format!("failed to serialize artifact metadata: {e}")))?;
    let header = HashMap::from([
        (FORMAT_KEY.to_string(), FORMAT.to_string()),
        (METADATA_KEY.to_string(), meta_json),
    ]);
    let len = bytes.len();
    write_safetensors(&vec![(RECORD_TENSOR.to_string(), vec![len], Dtype::U8, bytes)], Some(header), path.as_ref())
}

/// Reads just the metadata of an artifact.
pub fn read_artifact_metadata<P: AsRef<Path>>(path: P) -> Result<ArtifactMetadata, LoadError> {
    parse_metadata(&std::fs::read(path)?)
}

/// Loads an artifact, refusing it unless it was trained on the encoder and tokenizer whose
/// file hashes are given.
pub fn load_artifact<B: Backend, P: AsRef<Path>>(
    path: P,
    encoder_sha256: &str,
    tokenizer_sha256: &str,
    device: &B::Device,
) -> Result<(DecisionModel<B>, ArtifactMetadata), LoadError> {
    let bytes = std::fs::read(path)?;
    let metadata = parse_metadata(&bytes)?;
    if metadata.encoder_sha256 != encoder_sha256 {
        return Err(LoadError::EncoderMismatch {
            what: "encoder",
            expected: metadata.encoder_sha256,
            found: encoder_sha256.to_string(),
        });
    }
    if metadata.tokenizer_sha256 != tokenizer_sha256 {
        return Err(LoadError::EncoderMismatch {
            what: "tokenizer",
            expected: metadata.tokenizer_sha256,
            found: tokenizer_sha256.to_string(),
        });
    }

    let st = SafeTensors::deserialize(&bytes)?;
    let view = st
        .tensor(RECORD_TENSOR)
        .map_err(|_| LoadError::TensorNotFound(RECORD_TENSOR.to_string()))?;
    let record = Recorder::<B>::load(&recorder(), view.data().to_vec(), device)
        .map_err(|e| LoadError::InvalidConfiguration(format!("failed to decode decision model: {e}")))?;
    let model = metadata.model.init::<B>(device).load_record(record);
    Ok((model, metadata))
}

fn parse_metadata(bytes: &[u8]) -> Result<ArtifactMetadata, LoadError> {
    let (_, header) = SafeTensors::read_metadata(bytes)?;
    let map = header.metadata().as_ref();
    let format = map.and_then(|m| m.get(FORMAT_KEY));
    if format.map(String::as_str) != Some(FORMAT) {
        return Err(LoadError::InvalidConfiguration(format!(
            "not a reflex decision artifact (format = {format:?}); encoder checkpoints and v1 heads files are not accepted"
        )));
    }
    let json = map
        .and_then(|m| m.get(METADATA_KEY))
        .ok_or_else(|| LoadError::InvalidConfiguration("artifact is missing its metadata".to_string()))?;
    let metadata: ArtifactMetadata = serde_json::from_str(json)
        .map_err(|e| LoadError::InvalidConfiguration(format!("failed to parse artifact metadata: {e}")))?;
    if metadata.format_version != ARTIFACT_FORMAT_VERSION {
        return Err(LoadError::InvalidConfiguration(format!(
            "unsupported artifact format version {} (expected {ARTIFACT_FORMAT_VERSION})",
            metadata.format_version
        )));
    }
    if metadata.encoding_version != ENCODING_VERSION {
        return Err(LoadError::InvalidConfiguration(format!(
            "artifact uses encoding v{} but this build encodes v{ENCODING_VERSION}",
            metadata.encoding_version
        )));
    }
    Ok(metadata)
}
