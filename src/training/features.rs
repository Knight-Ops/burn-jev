//! Frozen-encoder feature extraction for decision-model training.
//!
//! Runs each scenario through the encoder once (length-bucketed batches), reduces it to
//! [`ScenarioFeatures`], and optionally persists the result with [`FeatureCache`] so repeated
//! training runs skip the encoder entirely.

use std::path::Path;

use burn::tensor::backend::Backend;
use tokenizers::Tokenizer;

use crate::cache::{CachedScenario, FeatureCache};
use crate::dataset::{JevDataset, TokenizedScenario};
use crate::encoding::EncodingConfig;
use crate::model::weights::sha256_file;
use crate::model::ModernBertEncoder;

/// Upper bound on `batch * padded_len` per encoder call; attention memory is
/// `heads * batch * len^2`, so this keeps long sequences in small batches.
const TOKEN_BUDGET: usize = 4096;
const MAX_BATCH: usize = 16;

/// Everything needed to turn dataset records into encoder features.
pub struct FeatureContext<'a, B: Backend> {
    pub encoder: &'a ModernBertEncoder<B>,
    pub tokenizer: &'a Tokenizer,
    pub encoding: &'a EncodingConfig,
}

/// Where and whether to persist computed features.
pub struct CacheSettings<'a> {
    pub dir: &'a Path,
    /// Encoder/tokenizer/encoding/backend identity; see `BackendKind::cache_key`.
    pub model_id: &'a str,
    pub enabled: bool,
}

/// How [`get_or_compute_features`] obtained its features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheOutcome {
    /// Loaded from a valid on-disk cache; the encoder was not run.
    Hit,
    /// Computed through the encoder and written to the cache.
    MissSaved,
    /// Computed through the encoder; caching was disabled.
    Disabled,
}

/// Encodes every record and runs the frozen encoder over them in length-sorted batches.
///
/// `on_progress(done, total)` is called after each encoder batch.
pub fn compute_features<B: Backend>(
    ctx: &FeatureContext<'_, B>,
    dataset: &JevDataset,
    device: &B::Device,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<Vec<CachedScenario>, Box<dyn std::error::Error>> {
    let tokenized: Vec<TokenizedScenario> = dataset
        .records
        .iter()
        .map(|r| r.encode(ctx.tokenizer, ctx.encoding))
        .collect::<Result<_, _>>()?;
    let truncated = tokenized.iter().filter(|t| t.encoded.context_truncated).count();
    if truncated > 0 {
        eprintln!(
            "[features] warning: {truncated} scenario(s) had their context truncated to max_seq_len {}",
            ctx.encoding.max_seq_len
        );
    }

    let mut order: Vec<usize> = (0..tokenized.len()).collect();
    order.sort_by_key(|&i| tokenized[i].encoded.len());

    let mut features = vec![None; tokenized.len()];
    let mut done = 0;
    let mut start = 0;
    while start < order.len() {
        // Sorted ascending, so the last member of a batch sets its padded length.
        let mut end = start + 1;
        while end < order.len()
            && end - start < MAX_BATCH
            && (end - start + 1) * tokenized[order[end]].encoded.len() <= TOKEN_BUDGET
        {
            end += 1;
        }
        let batch: Vec<_> = order[start..end].iter().map(|&i| &tokenized[i].encoded).collect();
        let out = ctx.encoder.scenario_features(&batch, ctx.encoding.pad_id, device);
        for (&i, f) in order[start..end].iter().zip(out) {
            features[i] = Some(f);
        }
        done += end - start;
        on_progress(done, order.len());
        start = end;
    }

    Ok(tokenized
        .into_iter()
        .zip(features)
        .map(|(t, f)| CachedScenario {
            id: t.id,
            is_benign: t.is_benign,
            targets: t.targets,
            features: f.expect("every scenario was encoded"),
        })
        .collect())
}

/// Loads features from the disk cache when it matches the dataset contents and cache key,
/// otherwise computes them (and saves them if caching is enabled).
///
/// `on_progress(done, total)` is only called when the encoder runs.
pub fn get_or_compute_features<B: Backend>(
    ctx: &FeatureContext<'_, B>,
    dataset: &JevDataset,
    dataset_path: &Path,
    cache: &CacheSettings<'_>,
    device: &B::Device,
    on_progress: impl FnMut(usize, usize),
) -> Result<(Vec<CachedScenario>, CacheOutcome), Box<dyn std::error::Error>> {
    let d_model = ctx.encoder.d_model();

    if !cache.enabled {
        let computed = compute_features(ctx, dataset, device, on_progress)?;
        return Ok((computed, CacheOutcome::Disabled));
    }

    let (safetensors_path, meta_path) = FeatureCache::cache_paths(cache.dir, dataset_path, cache.model_id);
    let dataset_hash = sha256_file(dataset_path)?;

    if FeatureCache::is_cache_valid(&safetensors_path, &meta_path, &dataset_hash, cache.model_id, d_model) {
        let loaded = FeatureCache::load_from_disk(&safetensors_path, &meta_path)?;
        return Ok((loaded, CacheOutcome::Hit));
    }

    let computed = compute_features(ctx, dataset, device, on_progress)?;
    FeatureCache::save_to_disk(
        &computed,
        &safetensors_path,
        &meta_path,
        dataset_hash,
        cache.model_id.to_string(),
        d_model,
    )?;
    Ok((computed, CacheOutcome::MissSaved))
}
