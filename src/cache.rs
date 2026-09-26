//! =====================================================================
//! Persistent Feature Cache for the Frozen Encoder
//! =====================================================================
//!
//! Stores each scenario's [`ScenarioFeatures`] (context token states and pooled item queries)
//! as f16 safetensors plus a metadata JSON, so the encoder runs once per dataset and training
//! runs only touch the small decision model.

use std::fs;
use std::path::{Path, PathBuf};

use safetensors::tensor::Dtype;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

use crate::encoding::ItemKind;
use crate::model::weights::{f16_bytes, tensor_to_f32_vec, write_safetensors, NamedTensorBytes};
use crate::model::ScenarioFeatures;
use crate::training::MultiQuestionTargets;

const CACHE_VERSION: u32 = 2;

/// A scenario's features plus its labels, ready for decision-model training.
#[derive(Clone, Debug, PartialEq)]
pub struct CachedScenario {
    pub id: String,
    pub is_benign: bool,
    pub targets: MultiQuestionTargets,
    pub features: ScenarioFeatures,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScenarioDiskMeta {
    pub id: String,
    pub targets: MultiQuestionTargets,
    pub is_benign: bool,
    pub ctx_len: usize,
    pub kinds: Vec<ItemKind>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureCacheMetadata {
    pub version: u32,
    pub dataset_hash: String,
    /// Encoder + tokenizer + encoding + backend identity (see `BackendKind::cache_key`).
    pub model_id: String,
    pub d_model: usize,
    pub scenarios: Vec<ScenarioDiskMeta>,
}

pub struct FeatureCache;

impl FeatureCache {
    /// Cache file paths (safetensors + json metadata) for a dataset under a cache key.
    pub fn cache_paths(cache_dir: &Path, dataset_path: &Path, model_id: &str) -> (PathBuf, PathBuf) {
        let stem = dataset_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("dataset");
        let safe_model = model_id
            .replace(['/', '\\', '.', ':', ' '], "_")
            .trim_matches('_')
            .to_string();
        let prefix = format!("{stem}_{safe_model}");
        (
            cache_dir.join(format!("{prefix}.safetensors")),
            cache_dir.join(format!("{prefix}.meta.json")),
        )
    }

    /// True if a cache exists for exactly this dataset content, key and width.
    pub fn is_cache_valid(
        safetensors_path: &Path,
        meta_path: &Path,
        expected_dataset_hash: &str,
        expected_model_id: &str,
        expected_d_model: usize,
    ) -> bool {
        if !safetensors_path.exists() {
            return false;
        }
        let Ok(meta_str) = fs::read_to_string(meta_path) else {
            return false;
        };
        let Ok(meta) = serde_json::from_str::<FeatureCacheMetadata>(&meta_str) else {
            return false;
        };
        meta.version == CACHE_VERSION
            && meta.dataset_hash == expected_dataset_hash
            && meta.model_id == expected_model_id
            && meta.d_model == expected_d_model
    }

    pub fn save_to_disk(
        scenarios: &[CachedScenario],
        safetensors_path: &Path,
        meta_path: &Path,
        dataset_hash: String,
        model_id: String,
        d_model: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut tensors: NamedTensorBytes = Vec::with_capacity(scenarios.len() * 2);
        let mut metas = Vec::with_capacity(scenarios.len());
        for (i, s) in scenarios.iter().enumerate() {
            let f = &s.features;
            if f.d_model != d_model {
                return Err(format!("scenario {} has d_model {}, cache expects {d_model}", s.id, f.d_model).into());
            }
            tensors.push((format!("s_{i}.ctx"), vec![f.ctx_len, d_model], Dtype::F16, f16_bytes(&f.ctx)));
            if f.num_items() > 0 {
                tensors.push((format!("s_{i}.items"), vec![f.num_items(), d_model], Dtype::F16, f16_bytes(&f.items)));
            }
            metas.push(ScenarioDiskMeta {
                id: s.id.clone(),
                targets: s.targets.clone(),
                is_benign: s.is_benign,
                ctx_len: f.ctx_len,
                kinds: f.kinds.clone(),
            });
        }
        write_safetensors(&tensors, None, safetensors_path)?;

        let meta = FeatureCacheMetadata {
            version: CACHE_VERSION,
            dataset_hash,
            model_id,
            d_model,
            scenarios: metas,
        };
        if let Some(parent) = meta_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(meta_path, serde_json::to_string(&meta)?)?;
        Ok(())
    }

    pub fn load_from_disk(
        safetensors_path: &Path,
        meta_path: &Path,
    ) -> Result<Vec<CachedScenario>, Box<dyn std::error::Error>> {
        let meta: FeatureCacheMetadata = serde_json::from_str(&fs::read_to_string(meta_path)?)?;
        let file = fs::File::open(safetensors_path)?;
        // SAFETY: read-only for the duration of the load; every tensor is copied out.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;

        let mut scenarios = Vec::with_capacity(meta.scenarios.len());
        for (i, m) in meta.scenarios.into_iter().enumerate() {
            let ctx = tensor_to_f32_vec(&st.tensor(&format!("s_{i}.ctx"))?)?;
            let items = if m.kinds.is_empty() {
                Vec::new()
            } else {
                tensor_to_f32_vec(&st.tensor(&format!("s_{i}.items"))?)?
            };
            scenarios.push(CachedScenario {
                id: m.id,
                is_benign: m.is_benign,
                targets: m.targets,
                features: ScenarioFeatures {
                    d_model: meta.d_model,
                    ctx,
                    ctx_len: m.ctx_len,
                    items,
                    kinds: m.kinds,
                },
            });
        }
        Ok(scenarios)
    }
}
