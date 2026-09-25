use burn::tensor::{
    activation::sigmoid,
    backend::Backend,
    Bool, Tensor,
};

// =====================================================================
// Proper Scoring Rule: Brier Calibration Loss
// =====================================================================

/// Brier Calibration Loss ($\mathcal{L}_{\text{calibration}}$).
///
/// Enforces statistical alignment between predicted probabilities and empirical
/// binary outcomes through the Brier proper scoring rule:
/// $$\mathcal{L}_{\text{calibration}} = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\sigma(z_i / T) - y_i\right)^2$$
///
/// # Arguments
/// - `logits`: 1D tensor of raw or scaled assertion logits `[batch_size]`
/// - `targets`: 1D tensor of ground truth binary labels `[batch_size]` in `[0.0, 1.0]`
/// - `mask`: Optional 1D boolean mask `[batch_size]` selecting active items
pub fn brier_calibration_loss<B: Backend>(
    calibrated_logits: Tensor<B, 1>,
    targets: Tensor<B, 1>,
    mask: Option<Tensor<B, 1, Bool>>,
    _device: &B::Device,
) -> Tensor<B, 1> {
    let probs = sigmoid(calibrated_logits);
    let squared_diff = (probs - targets).powf_scalar(2.0);

    match mask {
        Some(m) => {
            let valid_count = m.clone().float().sum();
            let safe_count = valid_count.clamp_min(1.0);
            let masked_diff = squared_diff.mask_fill(m.bool_not(), 0.0);
            masked_diff.sum() / safe_count
        }
        None => squared_diff.mean(),
    }
}
