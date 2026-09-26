//! Per-layer parity of the Burn ModernBERT port against HF transformers.
//!
//! Needs `models/modernbert-base` (see `hf download` in docs) and the fixture from
//! `uv run scripts/modernbert_reference.py`; skipped when either is missing.
//! Compared layer by layer rather than only at the output: the residual stream amplifies
//! f32 noise with depth, so a final-state tolerance alone hides where a divergence starts.

use std::path::Path;

use burn::tensor::{Bool, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_mamba::model::ModernBertLoader;
use safetensors::SafeTensors;

type B = Flex<f32, i32>;

const MODEL_DIR: &str = "models/modernbert-base";
const FIXTURE: &str = "tests/fixtures/modernbert_ref.safetensors";

/// Max |burn - hf| over real tokens, relative to the reference's max magnitude.
const REL_TOL: f32 = 1e-4;
/// The final LayerNorm re-normalizes a residual stream with ~3e4-magnitude outlier
/// dimensions, so f32 summation-order differences in its mean/variance show up as ~1e-4
/// relative error on the small dimensions. Checked with a looser bound plus direction.
const FINAL_REL_TOL: f32 = 1e-3;
/// Every real token's state must point the same way as HF's.
const MIN_COSINE: f64 = 0.999_99;

fn i64s(st: &SafeTensors, name: &str) -> (Vec<i64>, Vec<usize>) {
    let view = st.tensor(name).unwrap_or_else(|_| panic!("fixture missing {name}"));
    (bytemuck::pod_collect_to_vec::<u8, i64>(view.data()), view.shape().to_vec())
}

fn f32s(st: &SafeTensors, name: &str) -> Vec<f32> {
    bytemuck::pod_collect_to_vec::<u8, f32>(st.tensor(name).unwrap().data())
}

#[test]
fn modernbert_matches_hf_per_layer() {
    if !Path::new(MODEL_DIR).join("model.safetensors").exists() || !Path::new(FIXTURE).exists() {
        eprintln!("skipping: need {MODEL_DIR} and {FIXTURE} (run scripts/modernbert_reference.py)");
        return;
    }
    let device = FlexDevice;
    let loaded = ModernBertLoader::load_dir::<B, _>(MODEL_DIR, &device).expect("load encoder");
    let n_layers = loaded.config.num_hidden_layers;
    let d = loaded.config.hidden_size;
    let bytes = std::fs::read(FIXTURE).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();

    let mut worst = 0.0f32;
    for case in ["short", "long", "padded"] {
        let (ids, shape) = i64s(&st, &format!("{case}.input_ids"));
        let (mask, _) = i64s(&st, &format!("{case}.attention_mask"));
        let [b, l] = [shape[0], shape[1]];
        let input_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [b, l]), &device);
        let valid: Vec<bool> = mask.iter().map(|&m| m != 0).collect();
        let attention_mask = Tensor::<B, 2, Bool>::from_data(TensorData::new(valid.clone(), [b, l]), &device);

        let ours = loaded.model.forward_hidden_states(input_ids, attention_mask);
        assert_eq!(ours.len(), n_layers + 2);

        // HF tuple: [emb, layer_0 .. layer_{n-2}, final_norm(layer_{n-1})].
        for i in 0..=n_layers {
            let ours_i = if i == n_layers { &ours[n_layers + 1] } else { &ours[i] };
            let ours_v = ours_i.clone().into_data().to_vec::<f32>().unwrap();
            let hf_v = f32s(&st, &format!("{case}.hidden_{i}"));
            assert_eq!(ours_v.len(), hf_v.len(), "{case} hidden_{i} size");

            let (mut max_err, mut max_ref, mut min_cos) = (0.0f32, 0.0f32, 1.0f64);
            for (pos, is_valid) in valid.iter().enumerate() {
                if !is_valid {
                    continue;
                }
                let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
                for k in pos * d..(pos + 1) * d {
                    max_err = max_err.max((ours_v[k] - hf_v[k]).abs());
                    max_ref = max_ref.max(hf_v[k].abs());
                    let (a, b) = (ours_v[k] as f64, hf_v[k] as f64);
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                min_cos = min_cos.min(dot / (na.sqrt() * nb.sqrt()).max(1e-30));
            }
            let rel = max_err / max_ref.max(1.0);
            let tol = if i == n_layers { FINAL_REL_TOL } else { REL_TOL };
            worst = worst.max(rel);
            eprintln!(
                "{case:>6} hidden_{i:<2} max_abs_err {max_err:.3e}  max_ref {max_ref:9.3}  rel {rel:.3e}  min_cos {min_cos:.8}"
            );
            assert!(rel < tol, "{case} hidden_{i}: rel err {rel:.3e} exceeds {tol:e}");
            assert!(min_cos > MIN_COSINE, "{case} hidden_{i}: min token cosine {min_cos} below {MIN_COSINE}");
        }
    }
    eprintln!("worst relative error {worst:.3e}");
}
