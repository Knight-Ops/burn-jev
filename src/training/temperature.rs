//! Post-hoc temperature scaling of the decision heads.
//!
//! The heads train with their temperatures frozen at 1.0; afterwards each head's temperature
//! is fit on held-out logits by minimizing that head's own training loss (choice cross-entropy,
//! noul BCE, ordinal cumulative BCE) over `logits / T`. Scaling never changes a head's argmax,
//! only how confident it is.

use std::borrow::Borrow;

use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{backend::Backend, Bool, Int, Tensor, TensorData};

use crate::cache::CachedScenario;
use crate::model::{DecisionModel, Head};
use crate::training::tasks::{choice_cross_entropy_loss, noul_bce_loss, ordinal_score_loss};
use crate::training::trainer::{DecisionBatch, DecisionBatcher};

/// Search range for `T`, searched in log space.
const MIN_TEMPERATURE: f64 = 0.05;
const MAX_TEMPERATURE: f64 = 10.0;
const SEARCH_EVALS: usize = 40;

/// Uncalibrated (T = 1) logits and targets for every head, on the host.
#[derive(Clone, Debug, Default)]
pub struct HeadLogits {
    /// Candidate logits, one row per choice question.
    pub choice: Vec<Vec<f32>>,
    pub choice_targets: Vec<usize>,
    pub noul: Vec<f32>,
    pub noul_targets: Vec<f32>,
    /// `[N, num_rubric_bins - 1]` cumulative logits, row-major.
    pub score: Vec<f32>,
    pub score_targets: Vec<f32>,
    pub num_rubric_bins: usize,
}

impl HeadLogits {
    /// Runs `model` over `scenarios` with every temperature reset to 1.
    pub fn collect<B: Backend, I: Borrow<CachedScenario>>(
        model: &DecisionModel<B>,
        scenarios: &[I],
        batch_size: usize,
        device: &B::Device,
    ) -> Self {
        let mut model = model.clone();
        model.heads.set_temperatures(1.0, 1.0, 1.0);
        let mut out = Self {
            num_rubric_bins: model.heads.num_rubric_bins,
            ..Self::default()
        };
        let host = |data: TensorData| data.convert::<f32>().to_vec::<f32>().expect("f32 logits");

        for chunk in scenarios.chunks(batch_size.max(1)) {
            let items: Vec<&CachedScenario> = chunk.iter().map(Borrow::borrow).collect();
            let batch: DecisionBatch<B> = DecisionBatcher.batch(items, device);
            let fb = &batch.features;
            let logits = model.forward(fb);

            if let (Some(l), Some(rows)) = (logits.choice_logits, fb.choice.as_ref()) {
                let flat = host(l.into_data());
                for (q, &count) in rows.counts.iter().enumerate() {
                    let start = q * rows.max_candidates;
                    out.choice.push(flat[start..start + count].to_vec());
                }
                out.choice_targets.extend(batch.choice_targets.iter().map(|&t| t as usize));
            }
            if let Some(l) = logits.noul_logits {
                out.noul.extend(host(l.into_data()));
                out.noul_targets.extend(&batch.noul_targets);
            }
            if let Some(l) = logits.score_logits {
                out.score.extend(host(l.into_data()));
                out.score_targets.extend(&batch.score_targets);
            }
        }
        out
    }

    /// Mean loss of `head` on `logits / t`; `None` if the head has no items.
    pub fn loss<B: Backend>(&self, head: Head, t: f64, device: &B::Device) -> Option<f64> {
        let loss = match head {
            Head::Choice => {
                let q = self.choice.len();
                if q == 0 {
                    return None;
                }
                let c_max = self.choice.iter().map(Vec::len).max().unwrap_or(0);
                let mut logits = vec![0.0f32; q * c_max];
                let mut mask = vec![false; q * c_max];
                for (qi, row) in self.choice.iter().enumerate() {
                    logits[qi * c_max..qi * c_max + row.len()].copy_from_slice(row);
                    mask[qi * c_max..qi * c_max + row.len()].iter_mut().for_each(|m| *m = true);
                }
                let targets: Vec<i64> = self.choice_targets.iter().map(|&t| t as i64).collect();
                choice_cross_entropy_loss(
                    Tensor::<B, 2>::from_data(TensorData::new(logits, [q, c_max]), device) / t,
                    Tensor::<B, 1, Int>::from_data(TensorData::new(targets, [q]), device),
                    Tensor::<B, 2, Bool>::from_data(TensorData::new(mask, [q, c_max]), device),
                    None,
                    device,
                )
            }
            Head::Noul => {
                let n = self.noul.len();
                if n == 0 {
                    return None;
                }
                noul_bce_loss(
                    Tensor::<B, 1>::from_data(TensorData::new(self.noul.clone(), [n]), device) / t,
                    Tensor::<B, 1>::from_data(TensorData::new(self.noul_targets.clone(), [n]), device),
                    None,
                    device,
                )
            }
            Head::Score => {
                let n = self.score_targets.len();
                if n == 0 {
                    return None;
                }
                let heads = self.num_rubric_bins - 1;
                ordinal_score_loss(
                    Tensor::<B, 2>::from_data(TensorData::new(self.score.clone(), [n, heads]), device) / t,
                    Tensor::<B, 1>::from_data(TensorData::new(self.score_targets.clone(), [n]), device),
                    self.num_rubric_bins,
                    None,
                    device,
                )
            }
        };
        Some(loss.into_data().convert::<f32>().to_vec::<f32>().unwrap()[0] as f64)
    }
}

/// The temperature minimizing `head`'s loss on `logits`, by golden-section search over
/// `ln T ∈ [ln 0.05, ln 10]`; 1.0 if the head has no items.
pub fn fit_temperature<B: Backend>(logits: &HeadLogits, head: Head, device: &B::Device) -> f32 {
    let loss = |ln_t: f64| logits.loss::<B>(head, ln_t.exp(), device);
    if loss(0.0).is_none() {
        return 1.0;
    }
    let ln_t = golden_section_min(MIN_TEMPERATURE.ln(), MAX_TEMPERATURE.ln(), SEARCH_EVALS, |x| loss(x).unwrap());
    ln_t.exp() as f32
}

/// Fits `[choice, noul, score]` temperatures for `model` on held-out scenarios. `model`'s
/// own temperatures are ignored (the fit starts from raw logits).
pub fn fit_temperatures<B: Backend, I: Borrow<CachedScenario>>(
    model: &DecisionModel<B>,
    scenarios: &[I],
    batch_size: usize,
    device: &B::Device,
) -> [f32; 3] {
    let logits = HeadLogits::collect(model, scenarios, batch_size, device);
    Head::ALL.map(|head| fit_temperature::<B>(&logits, head, device))
}

/// Minimizer of a unimodal `f` on `[lo, hi]` using `evals` evaluations.
fn golden_section_min(mut lo: f64, mut hi: f64, evals: usize, f: impl Fn(f64) -> f64) -> f64 {
    let r = (5f64.sqrt() - 1.0) / 2.0;
    let mut a = hi - r * (hi - lo);
    let mut b = lo + r * (hi - lo);
    let (mut fa, mut fb) = (f(a), f(b));
    for _ in 2..evals {
        if fa < fb {
            hi = b;
            (b, fb) = (a, fa);
            a = hi - r * (hi - lo);
            fa = f(a);
        } else {
            lo = a;
            (a, fa) = (b, fb);
            b = lo + r * (hi - lo);
            fb = f(b);
        }
    }
    0.5 * (lo + hi)
}
