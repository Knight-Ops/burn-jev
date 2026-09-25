//! Frozen-backbone feature extraction for head training.
//!
//! Runs each scenario through the backbone once, slices out the delimiter hidden states the
//! heads consume, and optionally persists them with [`FeatureCache`] so repeated training
//! runs skip the backbone entirely.

use std::path::Path;

use burn::tensor::{backend::AutodiffBackend, backend::Backend, Int, Tensor};
use tokenizers::Tokenizer;

use crate::cache::{CachedScenario, FeatureCache};
use crate::dataset::JevDataset;
use crate::delimiters::{
    extract_choice_question_candidates, extract_cls_state, extract_noul_states,
    extract_score_states, CoordinateResolver,
};
use crate::model::BiMamba2Backbone;

/// Everything needed to turn dataset records into backbone features.
pub struct FeatureContext<'a, B: Backend> {
    pub backbone: &'a BiMamba2Backbone<B>,
    pub tokenizer: &'a Tokenizer,
    pub resolver: &'a CoordinateResolver,
}

/// Where and whether to persist computed features.
pub struct CacheSettings<'a> {
    pub dir: &'a Path,
    /// Identity of the backbone; part of the cache key (use the backbone's SHA-256).
    pub model_id: &'a str,
    pub enabled: bool,
}

/// How [`get_or_compute_features`] obtained its features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheOutcome {
    /// Loaded from a valid on-disk cache; the backbone was not run.
    Hit,
    /// Computed through the backbone and written to the cache.
    MissSaved,
    /// Computed through the backbone; caching was disabled.
    Disabled,
}

/// Runs every record through the frozen backbone and extracts delimiter states as
/// autodiff leaves, ready for head training.
///
/// `on_progress(done, total)` is called after each record.
pub fn compute_features_from_backbone<B: AutodiffBackend>(
    ctx: &FeatureContext<'_, B::InnerBackend>,
    dataset: &JevDataset,
    device: &B::Device,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<Vec<CachedScenario<B>>, Box<dyn std::error::Error>> {
    let delimiters = ctx.resolver.config();
    let mut cached = Vec::with_capacity(dataset.len());

    for record in &dataset.records {
        let tokenized = record.encode(ctx.tokenizer, delimiters)?;
        let coords = ctx.resolver.resolve_coordinates(&tokenized.token_ids)?;

        let input_ids =
            Tensor::<B::InnerBackend, 1, Int>::from_data(tokenized.token_ids.as_slice(), device)
                .unsqueeze_dim(0);
        let hidden = ctx.backbone.forward_backbone(input_ids);

        let cls_state = Tensor::from_inner(extract_cls_state(&hidden, 0, &coords));

        let mut choice_questions = Vec::with_capacity(coords.choice_questions.len());
        for q_coords in &coords.choice_questions {
            let cands = extract_choice_question_candidates(&hidden, 0, q_coords, device)?;
            choice_questions.push(Tensor::from_inner(cands));
        }

        let noul_states = extract_noul_states(&hidden, 0, &coords, device).map(Tensor::from_inner);
        let score_states =
            extract_score_states(&hidden, 0, &coords, device).map(Tensor::from_inner);

        cached.push(CachedScenario {
            id: record.id.clone(),
            cls_state,
            choice_questions,
            noul_states,
            score_states,
            targets: tokenized.targets,
            is_benign: record.is_benign,
        });
        on_progress(cached.len(), dataset.len());
    }

    Ok(cached)
}

/// Loads features from the disk cache when it matches the dataset contents and backbone,
/// otherwise computes them (and saves them if caching is enabled).
///
/// `on_progress(done, total)` is called after each record that goes through the backbone;
/// it is never called on a cache hit.
pub fn get_or_compute_features<B: AutodiffBackend>(
    ctx: &FeatureContext<'_, B::InnerBackend>,
    dataset: &JevDataset,
    dataset_path: &Path,
    cache: &CacheSettings<'_>,
    device: &B::Device,
    on_progress: impl FnMut(usize, usize),
) -> Result<(Vec<CachedScenario<B>>, CacheOutcome), Box<dyn std::error::Error>> {
    let d_model = ctx.backbone.d_model();

    if !cache.enabled {
        let computed = compute_features_from_backbone(ctx, dataset, device, on_progress)?;
        return Ok((computed, CacheOutcome::Disabled));
    }

    let (safetensors_path, meta_path) =
        FeatureCache::cache_paths(cache.dir, dataset_path, cache.model_id);
    let dataset_hash = FeatureCache::compute_file_hash(dataset_path)?;

    if FeatureCache::is_cache_valid(
        &safetensors_path,
        &meta_path,
        &dataset_hash,
        cache.model_id,
        d_model,
    ) {
        let loaded = FeatureCache::load_from_disk(&safetensors_path, &meta_path, device)?;
        return Ok((loaded, CacheOutcome::Hit));
    }

    let computed = compute_features_from_backbone(ctx, dataset, device, on_progress)?;
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
