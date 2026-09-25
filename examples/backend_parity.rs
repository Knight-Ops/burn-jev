//! Per-layer CPU vs wgpu parity check for the backbone.
//!
//! Each wgpu layer is fed the CPU layer's input, so the reported error is that layer's own
//! kernel error rather than noise amplified through the stack.
//!
//! `cargo run --release --features wgpu --example backend_parity -- <backbone> <tokenizer> <jsonl>`

use burn::tensor::{Int, Tensor, TensorData};
use tokenizers::Tokenizer;

use burn_mamba::backend::{CpuBackend, FlexDevice, GpuWgpu, WgpuDevice};
use burn_mamba::{DelimiterConfig, JevDataset, Mamba2CheckpointLoader};

fn to_vec<B: burn::tensor::backend::Backend>(t: Tensor<B, 3>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec().unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, backbone, tokenizer, data] = args.as_slice() else {
        return Err("usage: backend_parity <backbone> <tokenizer> <jsonl>".into());
    };
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| e.to_string())?;
    let ds = JevDataset::from_jsonl_file(data)?;
    let tokens = ds.records[0].encode(&tokenizer, &DelimiterConfig::mamba2_reserved())?.token_ids;
    eprintln!("[parity] {} tokens", tokens.len());

    let (cpu_dev, gpu_dev) = (FlexDevice, WgpuDevice::default());
    let cpu = Mamba2CheckpointLoader::load_backbone_file::<CpuBackend, _>(backbone, &cpu_dev)?.model;
    let gpu = Mamba2CheckpointLoader::load_backbone_file::<GpuWgpu, _>(backbone, &gpu_dev)?.model;

    let ids = TensorData::new(tokens.clone(), [1, tokens.len()]);
    let mut x_cpu = cpu.embedding.forward(Tensor::<CpuBackend, 2, Int>::from_data(ids.clone(), &cpu_dev));
    let x_gpu = gpu.embedding.forward(Tensor::<GpuWgpu, 2, Int>::from_data(ids, &gpu_dev));
    report("embedding", &to_vec(x_cpu.clone()), &to_vec(x_gpu));

    for (i, (lc, lg)) in cpu.layers.iter().zip(&gpu.layers).enumerate() {
        let input = Tensor::<GpuWgpu, 3>::from_data(x_cpu.to_data(), &gpu_dev);
        let y_gpu = lg.forward(input);
        x_cpu = lc.forward(x_cpu);
        report(&format!("layer {i:>2}"), &to_vec(x_cpu.clone()), &to_vec(y_gpu));
    }
    Ok(())
}

fn report(name: &str, cpu: &[f32], gpu: &[f32]) {
    let max_abs = cpu.iter().zip(gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let scale = cpu.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    let nan = gpu.iter().filter(|v| !v.is_finite()).count();
    println!("{name}: max|cpu-gpu| {max_abs:.3e}  max|cpu| {scale:.3e}  rel {:.3e}  non-finite {nan}", max_abs / scale.max(1e-12));
}
