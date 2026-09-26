use burn::tensor::{backend::Backend, Bool, Tensor};

// =====================================================================
// Tenant-Agnostic Contrastive Metric Loss
// =====================================================================

/// Configuration parameters for the contrastive metric loss.
#[derive(Clone, Copy, Debug)]
pub struct MetricLossConfig {
    /// Temperature scaling factor tau for cosine similarities. Default: 0.07.
    pub temperature: f32,
}

impl Default for MetricLossConfig {
    fn default() -> Self {
        Self { temperature: 0.07 }
    }
}

impl MetricLossConfig {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

/// Computes the tenant-agnostic benign-vs-adversarial contrastive metric loss.
///
/// Benign embeddings are pulled together in metric space, while adversarial
/// and outlier queries are pushed away into the exterior metric space.
///
/// # Mathematical Specification
/// For each benign query $i \in \mathcal{B}_{\text{benign}}$:
/// $$\mathcal{L}_i = -\log \frac{\sum_{p \in \mathcal{P}(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in \mathcal{A}(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$
///
/// where:
/// - $\mathbf{z}_i$ is the L2-normalized embedding extracted by the k-NN metric head
/// - $\mathcal{P}(i) = \{ p \ne i \mid \text{is\_benign}[p] = \text{true} \}$
/// - $\mathcal{A}(i) = \{ a \ne i \mid \text{is\_benign}[a] \ne \text{None} \}$ (all labeled peers:
///   benign, cross-tenant, and adversarial)
///
/// Scenarios with unknown benignness (`None`) are neither anchors, positives nor negatives.
/// If a batch contains no valid benign pairs, returns a zero-valued scalar tensor.
pub fn benign_adversarial_metric_loss<B: Backend>(
    embeddings: Tensor<B, 2>,
    is_benign: &[Option<bool>],
    temperature: f32,
    device: &B::Device,
) -> Tensor<B, 1> {
    let [batch_size, _] = embeddings.dims();
    if batch_size < 2 {
        return Tensor::<B, 1>::zeros([1], device);
    }

    // Similarity matrix S_ij = (z_i . z_j) / tau
    let s: Tensor<B, 2> = embeddings.clone().matmul(embeddings.transpose()) / temperature;

    // Construct boolean masks
    let mut pos_mask_data = Vec::with_capacity(batch_size * batch_size);
    let mut all_mask_data = Vec::with_capacity(batch_size * batch_size);
    let mut valid_anchors = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let is_i_benign = is_benign.get(i).copied().flatten() == Some(true);
        let mut row_pos_count = 0;

        for j in 0..batch_size {
            let label_j = is_benign.get(j).copied().flatten();
            let is_j_benign = label_j == Some(true);
            let not_self = i != j;

            all_mask_data.push(not_self && label_j.is_some());

            let is_pos = not_self && is_i_benign && is_j_benign;
            pos_mask_data.push(is_pos);
            if is_pos {
                row_pos_count += 1;
            }
        }

        // An anchor is valid if it is benign and has at least one other benign peer in the batch
        valid_anchors.push(is_i_benign && row_pos_count > 0);
    }

    let valid_count = valid_anchors.iter().filter(|&&v| v).count();
    if valid_count == 0 {
        return Tensor::<B, 1>::zeros([1], device);
    }

    let pos_mask = Tensor::<B, 1, Bool>::from_data(pos_mask_data.as_slice(), device)
        .reshape([batch_size, batch_size]);
    let all_mask = Tensor::<B, 1, Bool>::from_data(all_mask_data.as_slice(), device)
        .reshape([batch_size, batch_size]);
    let valid_mask = Tensor::<B, 1, Bool>::from_data(valid_anchors.as_slice(), device);

    // Numerically stable LogSumExp over positives:
    // logsumexp_p = max_p + log(sum(exp(S - max_p)))
    let pos_s = s.clone().mask_fill(pos_mask.clone().bool_not(), -1e9);
    let pos_max = pos_s.clone().max_dim(1);
    let pos_exp = (pos_s - pos_max.clone())
        .exp()
        .mask_fill(pos_mask.bool_not(), 0.0);
    let pos_lse = pos_max + (pos_exp.sum_dim(1) + 1e-12).log();

    // Numerically stable LogSumExp over all items (excluding self):
    // logsumexp_a = max_a + log(sum(exp(S - max_a)))
    let all_s = s.mask_fill(all_mask.clone().bool_not(), -1e9);
    let all_max = all_s.clone().max_dim(1);
    let all_exp = (all_s - all_max.clone())
        .exp()
        .mask_fill(all_mask.bool_not(), 0.0);
    let all_lse = all_max + (all_exp.sum_dim(1) + 1e-12).log();

    // Loss for anchor i: all_lse - pos_lse
    let loss_per_item = (all_lse - pos_lse).squeeze_dim(1); // [batch_size]

    // Only average across valid anchors that have positive peers
    let valid_losses = loss_per_item.mask_fill(valid_mask.bool_not(), 0.0);
    valid_losses.sum() / (valid_count as f32)
}
