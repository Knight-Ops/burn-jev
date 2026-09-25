use burn::tensor::{
    activation::relu,
    backend::Backend,
    Bool, Int, Tensor,
};

// =====================================================================
// Multi-Task Loss Functions
// =====================================================================

/// Binary Cross-Entropy with Logits for the Noul assertion head.
///
/// Implements the numerically stable formulation:
/// $$\text{BCE}(z, y) = \max(z, 0) - z \cdot y + \log(1 + \exp(-|z|))$$
///
/// # Arguments
/// - `logits`: 1D tensor of assertion logits `[batch_size]`
/// - `targets`: 1D tensor of float targets `[batch_size]` with values in `[0.0, 1.0]`
/// - `mask`: Optional boolean mask `[batch_size]` selecting active items with `<noul_q>` markers
pub fn noul_bce_loss<B: Backend>(
    logits: Tensor<B, 1>,
    targets: Tensor<B, 1>,
    mask: Option<Tensor<B, 1, Bool>>,
    _device: &B::Device,
) -> Tensor<B, 1> {
    let abs_logits = logits.clone().abs();
    let max_zeros = relu(logits.clone());
    let per_element_loss = max_zeros - (logits * targets) + (abs_logits.neg().exp() + 1.0).log();

    match mask {
        Some(m) => {
            let valid_count = m.clone().float().sum();
            let safe_count = valid_count.clamp_min(1.0);
            let masked_loss = per_element_loss.mask_fill(m.bool_not(), 0.0);
            masked_loss.sum() / safe_count
        }
        None => per_element_loss.mean(),
    }
}

/// Categorical Cross-Entropy across candidate coordinates for the In-Context Choice head.
///
/// Supports variable candidate counts $K_i \le 255$ through candidate padding masks.
///
/// # Arguments
/// - `logits`: 2D tensor of candidate logits `[batch_size, max_k]`
/// - `targets`: 1D tensor of ground truth candidate indices `[batch_size]`
/// - `candidate_mask`: 2D boolean mask `[batch_size, max_k]` (true for valid candidate tokens)
/// - `batch_item_mask`: Optional 1D boolean mask `[batch_size]` selecting items with candidate sets
pub fn choice_cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    candidate_mask: Tensor<B, 2, Bool>,
    batch_item_mask: Option<Tensor<B, 1, Bool>>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let [batch_size, max_k] = logits.dims();
    if batch_size == 0 || max_k == 0 {
        return Tensor::<B, 1>::zeros([1], device);
    }

    // Mask out padded candidate positions with large negative values for stable LogSumExp
    let masked_logits = logits.clone().mask_fill(candidate_mask.clone().bool_not(), -1e9);
    let max_val = masked_logits.clone().max_dim(1);
    let exp_shifted = (masked_logits - max_val.clone())
        .exp()
        .mask_fill(candidate_mask.bool_not(), 0.0);
    let lse = max_val + (exp_shifted.sum_dim(1) + 1e-12).log(); // [batch_size, 1]

    // Gather the logit for the target candidate index
    let targets_2d = targets.reshape([batch_size, 1]);
    let target_logits = logits.gather(1, targets_2d); // [batch_size, 1]

    let loss_per_item = (lse - target_logits).squeeze_dim(1); // [batch_size]

    match batch_item_mask {
        Some(m) => {
            let valid_count = m.clone().float().sum();
            let safe_count = valid_count.clamp_min(1.0);
            let masked_loss = loss_per_item.mask_fill(m.bool_not(), 0.0);
            masked_loss.sum() / safe_count
        }
        None => loss_per_item.mean(),
    }
}

/// Cumulative Binary Cross-Entropy across ordinal threshold heads for rubric scores.
///
/// Implements the monotonic ordinal classification loss across the $M - 1$ cumulative heads:
/// Head $k \in \{0, \dots, M-2\}$ evaluates $P(\text{Score} > k + 1)$.
///
/// # Arguments
/// - `cumulative_logits`: 2D tensor of threshold logits `[batch_size, num_rubric_bins - 1]`
/// - `targets`: 1D tensor of scalar target scores `[batch_size]` in range `[1.0, num_rubric_bins]`
/// - `num_rubric_bins`: Total number of rubric levels $M$
/// - `batch_item_mask`: Optional 1D boolean mask `[batch_size]` selecting items with `<score_q>` markers
pub fn ordinal_score_loss<B: Backend>(
    cumulative_logits: Tensor<B, 2>,
    targets: Tensor<B, 1>,
    num_rubric_bins: usize,
    batch_item_mask: Option<Tensor<B, 1, Bool>>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let [batch_size, num_heads] = cumulative_logits.dims();
    if batch_size == 0 || num_heads == 0 {
        return Tensor::<B, 1>::zeros([1], device);
    }

    let expected_heads = num_rubric_bins.saturating_sub(1);
    assert_eq!(
        num_heads, expected_heads,
        "Logit column count ({num_heads}) must equal num_rubric_bins - 1 ({expected_heads})"
    );

    // Build cutoff values [1.0, 2.0, ..., (M-1).0]
    let cutoffs_vec: Vec<f32> = (1..num_rubric_bins).map(|k| k as f32).collect();
    let cutoffs = Tensor::<B, 1>::from_data(cutoffs_vec.as_slice(), device).reshape([1, num_heads]);

    // Continuous/ordinal targets for threshold heads: clamp(y - cutoff, 0.0, 1.0)
    let targets_2d = targets.reshape([batch_size, 1]);
    let threshold_targets = (targets_2d - cutoffs).clamp(0.0, 1.0); // [batch_size, num_heads]

    // Numerically stable BCE across heads
    let abs_logits = cumulative_logits.clone().abs();
    let max_zeros = relu(cumulative_logits.clone());
    let bce_heads =
        max_zeros - (cumulative_logits * threshold_targets) + (abs_logits.neg().exp() + 1.0).log();

    let loss_per_item = bce_heads.mean_dim(1).squeeze_dim(1); // [batch_size]

    match batch_item_mask {
        Some(m) => {
            let valid_count = m.clone().float().sum();
            let safe_count = valid_count.clamp_min(1.0);
            let masked_loss = loss_per_item.mask_fill(m.bool_not(), 0.0);
            masked_loss.sum() / safe_count
        }
        None => loss_per_item.mean(),
    }
}
