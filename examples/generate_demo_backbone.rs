//! Writes the deterministic synthetic Mamba-2 backbone used for local demos and smoke tests
//! (d_model=256, 4 layers, vocab 50288). It has no pretrained knowledge.
//!
//! Note: `models/mamba2_seeded_demo.safetensors` came from an earlier revision of this
//! generator and is not bit-identical to its current output.
//!
//! `cargo run --release --example generate_demo_backbone -- models/mamba2_demo.safetensors`

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use safetensors::tensor::{Dtype, TensorView};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: generate_demo_backbone <output.safetensors>")?;
    generate_demo_safetensors(&target)
}

/// Generates a valid, self-contained Mamba-2 test safetensors file with coherent weights
fn generate_demo_safetensors(target_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let vocab_size: usize = 50288;
    let d_model: usize = 256;
    let n_layers: usize = 4;
    let d_state: usize = 64;
    let headdim: usize = 64;
    let expand: usize = 2;
    let ngroups: usize = 1;

    let d_inner = d_model * expand; // 512
    let nheads = d_inner / headdim; // 8
    let d_conv = d_inner + 2 * ngroups * d_state; // 512 + 128 = 640
    let in_proj_dim = 2 * d_inner + 2 * ngroups * d_state + nheads; // 1024 + 128 + 8 = 1160

    println!("      Synthesizing coherent Mamba-2 weights: d_model={}, layers={}, heads={}", d_model, n_layers, nheads);

    // 1. Embeddings: normalized sinusoidal features
    let emb_bytes: Vec<u8> = (0..vocab_size * d_model)
        .flat_map(|i| {
            let row = i / d_model;
            let col = i % d_model;
            let val = (row as f32 * 0.05 + col as f32 * 0.1).sin() * 0.02;
            val.to_le_bytes()
        })
        .collect();

    // 2. Linear projection weights
    let in_proj_bytes: Vec<u8> = (0..in_proj_dim * d_model)
        .flat_map(|i| {
            let val = ((i as f32) * 0.001).sin() * 0.01;
            val.to_le_bytes()
        })
        .collect();

    let conv_w_bytes: Vec<u8> = (0..d_conv * 1 * 4)
        .flat_map(|i| {
            let val = ((i as f32) * 0.005).cos() * 0.02;
            val.to_le_bytes()
        })
        .collect();

    let conv_b_bytes: Vec<u8> = vec![0u8; d_conv * 4];

    let dt_bias_bytes: Vec<u8> = (0..nheads)
        .flat_map(|i| {
            let val = 0.1 * (i as f32 + 1.0).ln();
            val.to_le_bytes()
        })
        .collect();

    let a_log_bytes: Vec<u8> = (0..nheads)
        .flat_map(|i| {
            let val = 1.0 + 0.1 * (i as f32);
            val.to_le_bytes()
        })
        .collect();

    let d_skip_bytes: Vec<u8> = (0..nheads)
        .flat_map(|_| 1.0f32.to_le_bytes())
        .collect();

    let inner_norm_bytes: Vec<u8> = (0..d_inner)
        .flat_map(|_| 1.0f32.to_le_bytes())
        .collect();

    let out_proj_bytes: Vec<u8> = (0..d_model * d_inner)
        .flat_map(|i| {
            let val = ((i as f32) * 0.002).sin() * 0.01;
            val.to_le_bytes()
        })
        .collect();

    let norm_layer_bytes: Vec<u8> = (0..d_model)
        .flat_map(|_| 1.0f32.to_le_bytes())
        .collect();

    let norm_f_bytes: Vec<u8> = (0..d_model)
        .flat_map(|_| 1.0f32.to_le_bytes())
        .collect();

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "backbone.embeddings.weight".to_string(),
        TensorView::new(Dtype::F32, vec![vocab_size, d_model], &emb_bytes)?,
    );

    for i in 0..n_layers {
        tensors.insert(
            format!("backbone.layers.{i}.mixer.in_proj.weight"),
            TensorView::new(Dtype::F32, vec![in_proj_dim, d_model], &in_proj_bytes)?,
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.conv1d.weight"),
            TensorView::new(Dtype::F32, vec![d_conv, 1, 4], &conv_w_bytes)?,
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.conv1d.bias"),
            TensorView::new(Dtype::F32, vec![d_conv], &conv_b_bytes)?,
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.dt_bias"),
            TensorView::new(Dtype::F32, vec![nheads], &dt_bias_bytes)?,
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.A_log"),
            TensorView::new(Dtype::F32, vec![nheads], &a_log_bytes)?,
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.D"),
            TensorView::new(Dtype::F32, vec![nheads], &d_skip_bytes)?,
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.norm.weight"),
            TensorView::new(Dtype::F32, vec![d_inner], &inner_norm_bytes)?,
        );
        tensors.insert(
            format!("backbone.layers.{i}.mixer.out_proj.weight"),
            TensorView::new(Dtype::F32, vec![d_model, d_inner], &out_proj_bytes)?,
        );
        tensors.insert(
            format!("backbone.layers.{i}.norm.weight"),
            TensorView::new(Dtype::F32, vec![d_model], &norm_layer_bytes)?,
        );
    }

    tensors.insert(
        "backbone.norm_f.weight".to_string(),
        TensorView::new(Dtype::F32, vec![d_model], &norm_f_bytes)?,
    );

    let serialized = safetensors::serialize(&tensors, None)?;
    fs::write(target_path, serialized)?;
    println!("      Successfully serialized and saved: {}", target_path.display());
    Ok(())
}
