//! =====================================================================
//! Persistent Feature Disk Caching for Precomputed Backbone Sequences
//! =====================================================================
//!
//! Stores extracted hidden state tensors ([CLS], candidate states, Noul
//! queries, and Score queries) directly on disk in standard safetensors
//! format with associated metadata JSON.
//!
//! Avoids repeating costly backbone forward passes across training runs,
//! accelerating iterative head tuning and hyperparameter sweeps by 10,000x+.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use burn::tensor::{backend::Backend, Tensor, TensorData};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::model::loader::{param_bytes, tensor_to_f32_vec};
use crate::training::MultiQuestionTargets;

// =====================================================================
// Cached Scenario Representation
// =====================================================================

/// In-memory cached representations of a single scenario for reflex head training.
#[derive(Clone, Debug)]
pub struct CachedScenario<B: Backend> {
    pub id: String,
    pub cls_state: Tensor<B, 2>,
    pub choice_questions: Vec<Tensor<B, 2>>,
    pub noul_states: Option<Tensor<B, 2>>,
    pub score_states: Option<Tensor<B, 2>>,
    pub targets: MultiQuestionTargets,
    pub is_benign: bool,
}

// =====================================================================
// Metadata Schema for Disk Cache
// =====================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScenarioDiskMeta {
    pub id: String,
    pub targets: MultiQuestionTargets,
    pub is_benign: bool,
    pub num_choice_questions: usize,
    pub choice_candidate_counts: Vec<usize>,
    pub num_noul_queries: usize,
    pub num_score_rubrics: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureCacheMetadata {
    pub version: u32,
    pub dataset_hash: String,
    pub model_id: String,
    pub d_model: usize,
    pub scenarios: Vec<ScenarioDiskMeta>,
}

// =====================================================================
// FeatureCache Manager
// =====================================================================

pub struct FeatureCache;

impl FeatureCache {
    /// Computes the SHA-256 of a file's byte contents as lowercase hex.
    ///
    /// Stable across Rust releases and machines, so it doubles as the backbone identity
    /// recorded in heads artifacts.
    pub fn compute_file_hash<P: AsRef<Path>>(path: P) -> Result<String, std::io::Error> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        Ok(hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect())
    }

    /// Determines cache file paths (safetensors + json metadata) from a cache directory and dataset path.
    pub fn cache_paths(
        cache_dir: &Path,
        dataset_path: &Path,
        model_id: &str,
    ) -> (PathBuf, PathBuf) {
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

    /// Checks if a valid, uncorrupted cache exists on disk matching the dataset and model identity.
    pub fn is_cache_valid(
        safetensors_path: &Path,
        meta_path: &Path,
        expected_dataset_hash: &str,
        expected_model_id: &str,
        expected_d_model: usize,
    ) -> bool {
        if !safetensors_path.exists() || !meta_path.exists() {
            return false;
        }

        let meta_str = match fs::read_to_string(meta_path) {
            Ok(s) => s,
            Err(_) => return false,
        };

        let meta: FeatureCacheMetadata = match serde_json::from_str(&meta_str) {
            Ok(m) => m,
            Err(_) => return false,
        };

        meta.version == 1
            && meta.dataset_hash == expected_dataset_hash
            && meta.model_id == expected_model_id
            && meta.d_model == expected_d_model
    }

    /// Serializes cached representations and metadata to disk.
    pub fn save_to_disk<B: Backend>(
        scenarios: &[CachedScenario<B>],
        safetensors_path: &Path,
        meta_path: &Path,
        dataset_hash: String,
        model_id: String,
        d_model: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = safetensors_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if let Some(parent) = meta_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut scenario_metas = Vec::with_capacity(scenarios.len());
        let mut byte_buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

        for (s_idx, s) in scenarios.iter().enumerate() {
            // 1. CLS State: [1, d_model]
            let cls_bytes = param_bytes(s.cls_state.clone());
            byte_buffers.push((format!("s_{s_idx}.cls"), vec![1, d_model], cls_bytes));

            // 2. Choice Questions: each [K_q, d_model]
            let mut choice_cand_counts = Vec::new();
            for (q_idx, cands) in s.choice_questions.iter().enumerate() {
                let dims = cands.dims();
                let k = dims[0];
                choice_cand_counts.push(k);

                let c_bytes = param_bytes(cands.clone());
                byte_buffers.push((format!("s_{s_idx}.choice_{q_idx}"), vec![k, d_model], c_bytes));
            }

            // 3. Noul States: [N_noul, d_model]
            let num_noul = if let Some(ref noul_states) = s.noul_states {
                let dims = noul_states.dims();
                let count = dims[0];
                let n_bytes = param_bytes(noul_states.clone());
                byte_buffers.push((format!("s_{s_idx}.noul"), vec![count, d_model], n_bytes));
                count
            } else {
                0
            };

            // 4. Score States: [N_score, d_model]
            let num_score = if let Some(ref score_states) = s.score_states {
                let dims = score_states.dims();
                let count = dims[0];
                let sc_bytes = param_bytes(score_states.clone());
                byte_buffers.push((format!("s_{s_idx}.score"), vec![count, d_model], sc_bytes));
                count
            } else {
                0
            };

            scenario_metas.push(ScenarioDiskMeta {
                id: s.id.clone(),
                targets: s.targets.clone(),
                is_benign: s.is_benign,
                num_choice_questions: s.choice_questions.len(),
                choice_candidate_counts: choice_cand_counts,
                num_noul_queries: num_noul,
                num_score_rubrics: num_score,
            });
        }

        // Write safetensors binary
        let mut tensors_map = BTreeMap::new();
        for (name, shape, bytes) in &byte_buffers {
            tensors_map.insert(
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            );
        }
        let serialized = safetensors::serialize(&tensors_map, None)?;
        fs::write(safetensors_path, serialized)?;

        // Write metadata JSON
        let meta = FeatureCacheMetadata {
            version: 1,
            dataset_hash,
            model_id,
            d_model,
            scenarios: scenario_metas,
        };
        let meta_json = serde_json::to_string_pretty(&meta)?;
        fs::write(meta_path, meta_json)?;

        Ok(())
    }

    /// Loads precomputed representations directly from safetensors into backend tensors.
    pub fn load_from_disk<B: Backend>(
        safetensors_path: &Path,
        meta_path: &Path,
        device: &B::Device,
    ) -> Result<Vec<CachedScenario<B>>, Box<dyn std::error::Error>> {
        let meta_str = fs::read_to_string(meta_path)?;
        let meta: FeatureCacheMetadata = serde_json::from_str(&meta_str)?;

        let buffer = fs::read(safetensors_path)?;
        let safe = SafeTensors::deserialize(&buffer)?;

        let mut scenarios = Vec::with_capacity(meta.scenarios.len());

        for (s_idx, s_meta) in meta.scenarios.iter().enumerate() {
            // 1. CLS State
            let cls_name = format!("s_{s_idx}.cls");
            let cls_view = safe.tensor(&cls_name)?;
            let cls_f32 = tensor_to_f32_vec(&cls_view)?;
            let cls_state = Tensor::<B, 2>::from_data(
                TensorData::new(cls_f32, cls_view.shape().to_vec()),
                device,
            );

            // 2. Choice Questions
            let mut choice_questions = Vec::with_capacity(s_meta.num_choice_questions);
            for q_idx in 0..s_meta.num_choice_questions {
                let q_name = format!("s_{s_idx}.choice_{q_idx}");
                let q_view = safe.tensor(&q_name)?;
                let q_f32 = tensor_to_f32_vec(&q_view)?;
                let cands = Tensor::<B, 2>::from_data(
                    TensorData::new(q_f32, q_view.shape().to_vec()),
                    device,
                );
                choice_questions.push(cands);
            }

            // 3. Noul States
            let noul_states = if s_meta.num_noul_queries > 0 {
                let n_name = format!("s_{s_idx}.noul");
                let n_view = safe.tensor(&n_name)?;
                let n_f32 = tensor_to_f32_vec(&n_view)?;
                Some(Tensor::<B, 2>::from_data(
                    TensorData::new(n_f32, n_view.shape().to_vec()),
                    device,
                ))
            } else {
                None
            };

            // 4. Score States
            let score_states = if s_meta.num_score_rubrics > 0 {
                let sc_name = format!("s_{s_idx}.score");
                let sc_view = safe.tensor(&sc_name)?;
                let sc_f32 = tensor_to_f32_vec(&sc_view)?;
                Some(Tensor::<B, 2>::from_data(
                    TensorData::new(sc_f32, sc_view.shape().to_vec()),
                    device,
                ))
            } else {
                None
            };

            scenarios.push(CachedScenario {
                id: s_meta.id.clone(),
                cls_state,
                choice_questions,
                noul_states,
                score_states,
                targets: s_meta.targets.clone(),
                is_benign: s_meta.is_benign,
            });
        }

        Ok(scenarios)
    }
}
