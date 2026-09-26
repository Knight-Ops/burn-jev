//! Tier 1 reflex inference: loads the frozen ModernBERT encoder plus a trained decision
//! artifact and emits one JSON verdict per input scenario.
//!
//! ```text
//! reflex [--backend cpu|wgpu|cuda] --encoder <dir> --artifact <st> [--tokenizer <json>] [--input <jsonl>]   # stdin if omitted
//! ```
//!
//! Each input line is a [`ReflexRequest`] (dataset records work too; labels are ignored).
//! Each output line is `{"id": .., "verdict": ReflexVerdict}` or `{"id": .., "error": ".."}`.
//! The process exits non-zero if any line failed.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use burn::tensor::backend::Backend;
use clap::Parser;
use serde::Serialize;
use tokenizers::Tokenizer;

use burn_jev::backend::{BackendKind, CpuBackend, FlexDevice};
use burn_jev::{load_artifact, sha256_file, ModernBertLoader, ReflexEngine, ReflexRequest, ReflexVerdict};

#[derive(Parser)]
#[command(version, about = "Run Tier 1 reflex inference over JSONL scenarios")]
struct Cli {
    /// Tensor backend to run on.
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,
    /// ModernBERT directory (config.json, model.safetensors, tokenizer.json) the artifact was trained on.
    #[arg(long)]
    encoder: PathBuf,
    /// Decision artifact produced by reflex-train.
    #[arg(long)]
    artifact: PathBuf,
    /// tokenizer.json; defaults to the one in the encoder directory.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// JSONL input; reads stdin when omitted.
    #[arg(long)]
    input: Option<PathBuf>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Output<'a> {
    Verdict { id: Option<&'a str>, verdict: &'a ReflexVerdict },
    Error { id: Option<&'a str>, error: String },
}

fn main() -> ExitCode {
    match dispatch(Cli::parse()) {
        Ok(0) => ExitCode::SUCCESS,
        Ok(failures) => {
            eprintln!("{failures} input(s) failed");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(cli: Cli) -> Result<usize, Box<dyn std::error::Error>> {
    match cli.backend {
        BackendKind::Cpu => run::<CpuBackend>(cli, FlexDevice),
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => {
            run::<burn_jev::backend::GpuWgpu>(cli, burn_jev::backend::WgpuDevice::default())
        }
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => {
            run::<burn_jev::backend::GpuCuda>(cli, burn_jev::backend::CudaDevice::default())
        }
        #[allow(unreachable_patterns)]
        other => Err(other.not_compiled()),
    }
}

fn run<B: Backend>(cli: Cli, device: B::Device) -> Result<usize, Box<dyn std::error::Error>> {
    let loaded = ModernBertLoader::load_dir::<B, _>(&cli.encoder, &device)
        .map_err(|e| format!("failed to load encoder {}: {e}", cli.encoder.display()))?;
    let tokenizer_path = cli
        .tokenizer
        .clone()
        .or(loaded.tokenizer_path.clone())
        .ok_or("no --tokenizer given and the encoder directory has no tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("failed to load tokenizer {}: {e}", tokenizer_path.display()))?;
    let tokenizer_sha = sha256_file(&tokenizer_path)?;

    let (decision, metadata) = load_artifact::<B, _>(&cli.artifact, &loaded.sha256, &tokenizer_sha, &device)
        .map_err(|e| format!("failed to load artifact {}: {e}", cli.artifact.display()))?;
    if metadata.model.reader.d_in != loaded.config.hidden_size {
        return Err(format!(
            "artifact expects encoder width {}, encoder has {}",
            metadata.model.reader.d_in, loaded.config.hidden_size
        )
        .into());
    }
    eprintln!(
        "[reflex] backend {} encoder d_model={} layers={}, artifact {}",
        cli.backend,
        loaded.config.hidden_size,
        loaded.config.num_hidden_layers,
        cli.artifact.display()
    );
    let engine = ReflexEngine::new(loaded.model, decision, metadata.encoding, tokenizer);

    let input: Box<dyn BufRead> = match cli.input {
        Some(ref path) => Box::new(BufReader::new(
            File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?,
        )),
        None => Box::new(io::stdin().lock()),
    };
    let mut out = BufWriter::new(io::stdout().lock());
    let mut failures = 0;

    for (line_no, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request = match serde_json::from_str::<ReflexRequest>(&line) {
            Ok(r) => r,
            Err(e) => {
                failures += 1;
                let error = format!("line {}: invalid request: {e}", line_no + 1);
                serde_json::to_writer(&mut out, &Output::Error { id: None, error })?;
                writeln!(out)?;
                continue;
            }
        };
        let id = request.id.as_deref();
        let result = engine.evaluate(&request, &device).map_err(|e| e.to_string());
        match result {
            Ok(ref verdict) => serde_json::to_writer(&mut out, &Output::Verdict { id, verdict })?,
            Err(error) => {
                failures += 1;
                serde_json::to_writer(&mut out, &Output::Error { id, error })?;
            }
        }
        writeln!(out)?;
        out.flush()?;
    }

    Ok(failures)
}
