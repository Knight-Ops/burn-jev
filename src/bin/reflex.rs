//! Tier 1 reflex inference: loads a backbone plus a heads artifact and emits one JSON
//! verdict per input scenario.
//!
//! ```text
//! reflex [--backend cpu|wgpu] --backbone <st> --heads <st> --tokenizer <json> [--input <jsonl>]   # stdin if omitted
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

use burn_mamba::backend::{BackendKind, CpuBackend, FlexDevice};
use burn_mamba::{CoordinateResolver, Mamba2CheckpointLoader, ReflexEngine, ReflexRequest, ReflexVerdict};

#[derive(Parser)]
#[command(version, about = "Run Tier 1 reflex inference over JSONL scenarios")]
struct Cli {
    /// Tensor backend to run on.
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,
    /// Mamba-2 backbone checkpoint the heads were trained on.
    #[arg(long)]
    backbone: PathBuf,
    /// Heads artifact produced by reflex-train.
    #[arg(long)]
    heads: PathBuf,
    /// HuggingFace tokenizer.json matching the backbone.
    #[arg(long)]
    tokenizer: PathBuf,
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
            run::<burn_mamba::backend::GpuWgpu>(cli, burn_mamba::backend::WgpuDevice::default())
        }
        #[allow(unreachable_patterns)]
        other => Err(other.not_compiled()),
    }
}

fn run<B: Backend>(cli: Cli, device: B::Device) -> Result<usize, Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::from_file(&cli.tokenizer)
        .map_err(|e| format!("failed to load tokenizer {}: {e}", cli.tokenizer.display()))?;

    let loaded = Mamba2CheckpointLoader::load_backbone_file::<B, _>(&cli.backbone, &device)
        .map_err(|e| format!("failed to load backbone {}: {e}", cli.backbone.display()))?;
    let (heads, metadata) =
        Mamba2CheckpointLoader::load_heads_file::<B, _>(&cli.heads, &loaded.sha256, &device)
            .map_err(|e| format!("failed to load heads {}: {e}", cli.heads.display()))?;
    if metadata.d_model != loaded.config.d_model {
        return Err(format!(
            "heads expect d_model={}, backbone has {}",
            metadata.d_model, loaded.config.d_model
        )
        .into());
    }
    let delimiters = metadata.delimiters;
    let mut model = loaded.model;
    model.heads = heads;
    let engine = ReflexEngine::new(model, CoordinateResolver::new(delimiters.clone()));
    eprintln!(
        "[reflex] backend {} backbone d_model={} layers={}, heads {}",
        cli.backend,
        loaded.config.d_model,
        loaded.config.n_layers,
        cli.heads.display()
    );

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
        let result = request
            .encode(&tokenizer, &delimiters)
            .map_err(|e| e.to_string())
            .and_then(|tokens| engine.evaluate(&tokens, &device).map_err(|e| e.to_string()));
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
