//! Per-layer CPU vs wgpu parity check for the ModernBERT encoder.
//!
//! Runs the same encoded scenario through both backends and reports each hidden state's
//! divergence (embedding, every layer, final norm). Errors compound through the stack, so
//! the first layer whose error jumps is where to look.
//!
//! `cargo run --release --features wgpu --example backend_parity -- <encoder_dir> <jsonl>`

use burn::tensor::{Bool, Int, Tensor, TensorData};
use tokenizers::Tokenizer;

use burn_jev::backend::{CpuBackend, FlexDevice, GpuWgpu, WgpuDevice};
use burn_jev::{EncodingConfig, JevDataset, ModernBertLoader};

fn to_vec<B: burn::tensor::backend::Backend>(t: Tensor<B, 3>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec().unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, encoder_dir, data] = args.as_slice() else {
        return Err("usage: backend_parity <encoder_dir> <jsonl>".into());
    };
    let (cpu_dev, gpu_dev) = (FlexDevice, WgpuDevice::default());
    let cpu = ModernBertLoader::load_dir::<CpuBackend, _>(encoder_dir, &cpu_dev)?;
    let gpu = ModernBertLoader::load_dir::<GpuWgpu, _>(encoder_dir, &gpu_dev)?.model;
    let tokenizer_path = cpu.tokenizer_path.clone().ok_or("encoder dir has no tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;

    let ds = JevDataset::from_jsonl_file(data)?;
    let encoding = EncodingConfig::for_encoder(&cpu.config);
    let tokens = ds.records[0].encode(&tokenizer, &encoding)?.encoded.input_ids;
    let l = tokens.len();
    eprintln!("[parity] {l} tokens");

    let ids = TensorData::new(tokens, [1, l]);
    let mask = TensorData::new(vec![true; l], [1, l]);
    let hc = cpu.model.forward_hidden_states(
        Tensor::<CpuBackend, 2, Int>::from_data(ids.clone(), &cpu_dev),
        Tensor::<CpuBackend, 2, Bool>::from_data(mask.clone(), &cpu_dev),
    );
    let hg = gpu.forward_hidden_states(
        Tensor::<GpuWgpu, 2, Int>::from_data(ids, &gpu_dev),
        Tensor::<GpuWgpu, 2, Bool>::from_data(mask, &gpu_dev),
    );
    let n = hc.len();
    for (i, (c, g)) in hc.into_iter().zip(hg).enumerate() {
        let name = match i {
            0 => "embedding".to_string(),
            i if i == n - 1 => "final norm".to_string(),
            i => format!("layer {:>2}", i - 1),
        };
        report(&name, &to_vec(c), &to_vec(g));
    }
    Ok(())
}

fn report(name: &str, cpu: &[f32], gpu: &[f32]) {
    let max_abs = cpu.iter().zip(gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let scale = cpu.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    let nan = gpu.iter().filter(|v| !v.is_finite()).count();
    println!("{name}: max|cpu-gpu| {max_abs:.3e}  max|cpu| {scale:.3e}  rel {:.3e}  non-finite {nan}", max_abs / scale.max(1e-12));
}
