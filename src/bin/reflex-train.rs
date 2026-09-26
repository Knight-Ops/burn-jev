//! Trains the Tier 1 decision model (item reader + JEV heads) on a frozen ModernBERT encoder
//! and writes a decision artifact.
//!
//! ```text
//! reflex-train [--backend cpu|wgpu|cuda] --encoder <dir> --train <jsonl> [--train <jsonl> ...] --val <jsonl> [--val <jsonl> ...] --out <st> [options]
//! reflex-train eval [--backend cpu|wgpu|cuda] --encoder <dir> --artifact <st> --data <jsonl> [--baseline-from <jsonl>]
//! ```
//!
//! Training runs on Burn's supervised trainer (TUI when attached to a terminal); logs,
//! metrics and checkpoints land in `--runs-dir/<timestamp>`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use clap::{Args, Parser, Subcommand};
use tokenizers::Tokenizer;

use burn_jev::backend::{BackendKind, CpuBackend, FlexDevice};
use burn_jev::training::{
    evaluate_dataset, get_or_compute_features, train_decision_model, CacheOutcome, CacheSettings,
    FeatureContext, TrainConfig,
};
use burn_jev::{
    load_artifact, save_artifact, sha256_file, ArtifactMetadata, Baselines, CachedScenario,
    DecisionModelConfig, EncodingConfig, EvalReport, ItemReaderConfig, JevDataset, LoadedEncoder,
    ModernBertLoader, ReflexEngine, UnifiedHeadsConfig,
};

type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(version, about = "Train the Tier 1 decision model on a frozen ModernBERT encoder")]
#[command(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    train: TrainArgs,
}

#[derive(Subcommand)]
enum Command {
    /// Evaluate an existing artifact on a labeled dataset without retraining.
    Eval(EvalArgs),
}

#[derive(Args)]
struct EncoderArgs {
    /// Tensor backend to run on.
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,
    /// ModernBERT directory (config.json, model.safetensors, tokenizer.json).
    #[arg(long, required = true)]
    encoder: Option<PathBuf>,
    /// tokenizer.json; defaults to the one in the encoder directory.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
}

#[derive(Args)]
struct TrainArgs {
    #[command(flatten)]
    enc: EncoderArgs,
    /// Training scenarios (JSONL). Repeat to add auxiliary sets (e.g. `data/ext/*.jsonl`); each
    /// file is feature-cached separately and the first one is the reference for baselines.
    #[arg(long, required = true)]
    train: Vec<PathBuf>,
    /// Validation scenarios (JSONL). Repeat to add sets; all are merged into one validation set
    /// that drives early stopping, best-epoch selection and the final report; each file is
    /// feature-cached separately.
    #[arg(long, required = true)]
    val: Vec<PathBuf>,
    /// Output path for the decision artifact.
    #[arg(long, required = true)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = 60)]
    epochs: usize,
    /// Peak learning rate (after warmup).
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 1e-2)]
    weight_decay: f32,
    /// Scenarios per optimizer step.
    #[arg(long, default_value_t = 16, value_parser = parse_positive)]
    batch_size: usize,
    /// Epochs without validation improvement before stopping.
    #[arg(long, default_value_t = 8)]
    patience: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Longest encoder input; only the context is truncated to fit.
    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,
    /// Reader width.
    #[arg(long, default_value_t = 128)]
    reader_dim: usize,
    /// Cross-attention blocks; 0 = mean-pool probe (ablation).
    #[arg(long, default_value_t = 2)]
    reader_blocks: usize,
    #[arg(long, default_value_t = 4)]
    reader_heads: usize,
    /// Dropout in the reader and heads.
    #[arg(long, default_value_t = 0.25)]
    dropout: f64,
    /// Directory for the persistent encoder feature cache.
    #[arg(long, default_value = "data/.cache")]
    cache_dir: PathBuf,
    /// Recompute encoder features and do not write the cache.
    #[arg(long)]
    no_cache: bool,
    /// Parent directory for per-run trainer logs and checkpoints.
    #[arg(long, default_value = "data/.runs")]
    runs_dir: PathBuf,
}

#[derive(Args)]
struct EvalArgs {
    #[command(flatten)]
    enc: EncoderArgs,
    #[arg(long)]
    artifact: PathBuf,
    /// Labeled scenarios (JSONL).
    #[arg(long)]
    data: PathBuf,
    /// Dataset whose label statistics define the baselines (normally the training set);
    /// defaults to `--data` itself.
    #[arg(long)]
    baseline_from: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let backend = match cli.command {
        Some(Command::Eval(ref a)) => a.enc.backend,
        None => cli.train.enc.backend,
    };
    let result = match backend {
        BackendKind::Cpu => dispatch::<CpuBackend>(cli, FlexDevice),
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => dispatch::<burn_jev::backend::GpuWgpu>(cli, burn_jev::backend::WgpuDevice::default()),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => dispatch::<burn_jev::backend::GpuCuda>(cli, burn_jev::backend::CudaDevice::default()),
        #[allow(unreachable_patterns)]
        other => Err(other.not_compiled()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch<B: Backend>(cli: Cli, device: B::Device) -> AppResult<()> {
    match cli.command {
        Some(Command::Eval(args)) => run_eval::<B>(args, device),
        None => run_train::<B>(cli.train, device),
    }
}

fn parse_positive(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(0) => Err("must be at least 1".into()),
        Ok(n) => Ok(n),
        Err(e) => Err(e.to_string()),
    }
}

fn load_dataset(path: &Path) -> AppResult<JevDataset> {
    let ds = JevDataset::from_jsonl_file(path).map_err(|e| format!("failed to load dataset {}: {e}", path.display()))?;
    if ds.is_empty() {
        return Err(format!("dataset {} contains no scenarios", path.display()).into());
    }
    eprintln!("[dataset] {} scenarios from {}", ds.len(), path.display());
    Ok(ds)
}

/// Encoder, tokenizer and their identities.
struct EncoderBundle<B: Backend> {
    loaded: LoadedEncoder<B>,
    tokenizer: Tokenizer,
    tokenizer_sha: String,
}

fn load_encoder<B: Backend>(args: &EncoderArgs, device: &B::Device) -> AppResult<EncoderBundle<B>> {
    let dir = args.encoder.as_ref().expect("clap enforces --encoder");
    eprintln!("[encoder] loading {} (backend {})", dir.display(), args.backend);
    let loaded = ModernBertLoader::load_dir::<B, _>(dir, device)
        .map_err(|e| format!("failed to load encoder {}: {e}", dir.display()))?;
    let tokenizer_path = args
        .tokenizer
        .clone()
        .or(loaded.tokenizer_path.clone())
        .ok_or("no --tokenizer given and the encoder directory has no tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("failed to load tokenizer {}: {e}", tokenizer_path.display()))?;
    let c = &loaded.config;
    eprintln!(
        "[encoder] d_model={} layers={} heads={} vocab={} ({} parameters, sha256 {})",
        c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.vocab_size, loaded.report.parameters_transferred, loaded.sha256
    );
    Ok(EncoderBundle {
        tokenizer_sha: sha256_file(&tokenizer_path)?,
        loaded,
        tokenizer,
    })
}

fn features_for<B: Backend>(
    enc: &EncoderBundle<B>,
    encoding: &EncodingConfig,
    dataset: &JevDataset,
    path: &Path,
    cache: &CacheSettings<'_>,
    device: &B::Device,
) -> AppResult<Vec<CachedScenario>> {
    let ctx = FeatureContext {
        encoder: &enc.loaded.model,
        tokenizer: &enc.tokenizer,
        encoding,
    };
    let start = Instant::now();
    let (features, outcome) = get_or_compute_features(&ctx, dataset, path, cache, device, |done, total| {
        let elapsed = start.elapsed().as_secs_f64();
        let eta = elapsed / done as f64 * (total - done) as f64;
        eprint!("\r[features] {}: {done}/{total} ({elapsed:.1}s, ETA {eta:.0}s)", path.display());
        if done == total {
            eprintln!();
        }
    })?;
    let how = match outcome {
        CacheOutcome::Hit => "loaded from cache",
        CacheOutcome::MissSaved => "computed and cached",
        CacheOutcome::Disabled => "computed (cache disabled)",
    };
    eprintln!("[features] {} scenarios {how} in {:.2}s", features.len(), start.elapsed().as_secs_f64());
    Ok(features)
}

fn run_train<B: Backend>(args: TrainArgs, device: B::Device) -> AppResult<()> {
    // `required = true` guarantees these are present when no subcommand is given.
    let train_paths = args.train.clone();
    let val_paths = args.val.clone();
    let out_path = args.out.clone().unwrap();

    let train_sets = train_paths.iter().map(|p| load_dataset(p)).collect::<AppResult<Vec<_>>>()?;
    let val_sets = val_paths.iter().map(|p| load_dataset(p)).collect::<AppResult<Vec<_>>>()?;
    let enc = load_encoder::<B>(&args.enc, &device)?;
    let encoding = EncodingConfig::for_encoder(&enc.loaded.config).with_max_seq_len(args.max_seq_len);

    let cache_key = args.enc.backend.cache_key(&enc.loaded.sha256, &enc.tokenizer_sha, encoding.max_seq_len);
    let cache = CacheSettings {
        dir: &args.cache_dir,
        model_id: &cache_key,
        enabled: !args.no_cache,
    };
    let mut train_features = Vec::new();
    for (ds, path) in train_sets.iter().zip(&train_paths) {
        train_features.extend(features_for(&enc, &encoding, ds, path, &cache, &device)?);
    }
    let mut val_features = Vec::new();
    for (ds, path) in val_sets.iter().zip(&val_paths) {
        val_features.extend(features_for(&enc, &encoding, ds, path, &cache, &device)?);
    }
    let val_ds = JevDataset {
        records: val_sets.into_iter().flat_map(|ds| ds.records).collect(),
    };
    if val_paths.len() > 1 {
        eprintln!("[dataset] {} validation scenarios merged from {} files", val_ds.len(), val_paths.len());
    }

    let reader = ItemReaderConfig::new(enc.loaded.config.hidden_size)
        .with_d_reader(args.reader_dim)
        .with_n_blocks(args.reader_blocks)
        .with_n_heads(args.reader_heads)
        .with_dropout(args.dropout);
    let model_config = DecisionModelConfig::with_reader(reader, UnifiedHeadsConfig::new(args.reader_dim).with_dropout(args.dropout));
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let train_config = TrainConfig {
        epochs: args.epochs,
        lr: args.lr,
        weight_decay: args.weight_decay,
        batch_size: args.batch_size,
        patience: args.patience,
        seed: args.seed,
        run_dir: args.runs_dir.join(format!("{stamp}-seed{}", args.seed)),
        ..TrainConfig::default()
    };
    eprintln!(
        "\n[train] AdamW lr={} wd={} epochs<={} batch={} patience={} seed={} reader={}x{} blocks, {} heads; run dir {}",
        train_config.lr,
        train_config.weight_decay,
        train_config.epochs,
        train_config.batch_size,
        train_config.patience,
        train_config.seed,
        args.reader_dim,
        args.reader_blocks,
        args.reader_heads,
        train_config.run_dir.display()
    );
    let start = Instant::now();
    let output = train_decision_model::<Autodiff<B>>(train_features, val_features, &model_config, &train_config, &device)?;
    eprintln!(
        "[train] done in {:.1}s; best epoch {} (by validation selection loss)",
        start.elapsed().as_secs_f64(),
        output.best_epoch
    );
    let [tc, tn, ts] = output.temperatures;
    eprintln!("[train] calibration temperatures: choice={tc:.3} noul={tn:.3} score={ts:.3}");

    let mut metadata = ArtifactMetadata::new(model_config, encoding.clone(), enc.loaded.sha256.clone(), enc.tokenizer_sha.clone());
    metadata.best_epoch = Some(output.best_epoch);
    save_artifact(&output.model, &metadata, &out_path)?;
    eprintln!(
        "[export] decision artifact -> {} ({:.2} KB)",
        out_path.display(),
        out_path.metadata()?.len() as f64 / 1024.0
    );

    // End-to-end check through the inference path (encoder → features → decision model).
    let baselines = Baselines::compute(&train_sets[0], &val_ds);
    let engine = ReflexEngine::new(enc.loaded.model, output.model, encoding, enc.tokenizer);
    let report = evaluate_dataset(&engine, &val_ds, &device)?;
    print_report(&report, &baselines, "primary train set");
    Ok(())
}

fn run_eval<B: Backend>(args: EvalArgs, device: B::Device) -> AppResult<()> {
    let dataset = load_dataset(&args.data)?;
    let baselines = match args.baseline_from {
        Some(ref path) => Baselines::compute(&load_dataset(path)?, &dataset),
        None => Baselines::compute(&dataset, &dataset),
    };
    let enc = load_encoder::<B>(&args.enc, &device)?;
    let (decision, metadata) = load_artifact::<B, _>(&args.artifact, &enc.loaded.sha256, &enc.tokenizer_sha, &device)
        .map_err(|e| format!("failed to load artifact {}: {e}", args.artifact.display()))?;
    let engine = ReflexEngine::new(enc.loaded.model, decision, metadata.encoding, enc.tokenizer);
    let report = evaluate_dataset(&engine, &dataset, &device)?;
    let source = if args.baseline_from.is_some() { "reference" } else { "eval set itself" };
    print_report(&report, &baselines, source);
    Ok(())
}

fn pass(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

fn print_report(report: &EvalReport, baselines: &Baselines, baseline_source: &str) {
    println!("\n=== Validation ===");
    for s in &report.scenarios {
        println!("\n--- {} [{}] ---", s.id, s.domain.as_deref().unwrap_or("general"));
        for (i, c) in s.choices.iter().enumerate() {
            println!(
                "  [Choice Q{}] target option {} vs predicted option {} (confidence {:.2}%) -> {}",
                i + 1,
                c.target + 1,
                c.predicted + 1,
                c.confidence * 100.0,
                pass(c.correct())
            );
        }
        for (i, n) in s.nouls.iter().enumerate() {
            println!(
                "  [Noul Q{}] target {} vs predicted {} (prob {:.2}%) -> {}",
                i + 1,
                n.target,
                n.predicted,
                n.probability * 100.0,
                pass(n.correct())
            );
        }
        for (i, o) in s.scores.iter().enumerate() {
            println!(
                "  [Score Q{}] target {:.2} vs predicted {:.2} (diff {:.2})",
                i + 1,
                o.target,
                o.predicted,
                o.abs_error()
            );
        }
    }

    let (cc, ct) = report.choice_counts();
    let (nc, nt) = report.noul_counts();
    let pct = |v: Option<f64>| v.map_or("n/a".to_string(), |v| format!("{:.1}%", v * 100.0));
    let num = |v: Option<f32>| v.map_or("n/a".to_string(), |v| format!("{v:.3}"));
    println!("\n=== Summary (baselines from {baseline_source}) ===");
    println!(
        " Choice accuracy: {:>7} ({cc}/{ct})   uniform {}, longest-candidate {}",
        pct(report.choice_accuracy()),
        pct(baselines.choice_uniform),
        pct(baselines.choice_longest)
    );
    println!(
        " Noul accuracy:   {:>7} ({nc}/{nt})   always-{} {}",
        pct(report.noul_accuracy()),
        baselines.noul_majority_label,
        pct(baselines.noul_majority)
    );
    println!(
        " Score RMSE:      {:>7} ({} rubrics)   predict-mean({:.2}) {}",
        num(report.score_rmse()),
        report.score_count(),
        baselines.score_reference_mean,
        num(baselines.score_mean_rmse)
    );
    println!(
        " Calibration ECE: choice {}, noul {} (10 bins)",
        pct(report.choice_ece()),
        pct(report.noul_ece())
    );

    // A merged evaluation set (several --val files) mixes sources; split it so the core set
    // (our generated data) is visible on its own next to each imported external set.
    let mut by_source: Vec<(String, EvalReport)> = Vec::new();
    for s in &report.scenarios {
        let src = source_of(s.domain.as_deref());
        match by_source.iter_mut().find(|(k, _)| *k == src) {
            Some((_, r)) => r.scenarios.push(s.clone()),
            None => by_source.push((src, EvalReport { scenarios: vec![s.clone()] })),
        }
    }
    if by_source.len() > 1 {
        by_source.sort_by(|a, b| (a.0 != "core", &a.0).cmp(&(b.0 != "core", &b.0)));
        println!("\n=== Per source ===");
        for (src, r) in &by_source {
            let (cc, ct) = r.choice_counts();
            let (nc, nt) = r.noul_counts();
            println!(
                " {src:<30} {:>5} scenarios | choice {:>6} ({cc}/{ct}) | noul {:>6} ({nc}/{nt}) | score RMSE {:>5} ({}) | ECE choice {}, noul {}",
                r.scenarios.len(),
                pct(r.choice_accuracy()),
                pct(r.noul_accuracy()),
                num(r.score_rmse()),
                r.score_count(),
                pct(r.choice_ece()),
                pct(r.noul_ece())
            );
        }
    }
}

/// `core` (our generated set), or the imported external set a domain belongs to
/// (`ext_procedural`, `ext_nemotron_ipi`, `ext_llama_<config>`); see `scripts/import_typed_decisions.py`.
fn source_of(domain: Option<&str>) -> String {
    match domain {
        Some(d) if d.starts_with("ext_") => ["ext_procedural", "ext_nemotron_ipi"]
            .into_iter()
            .find(|p| d.starts_with(p))
            .unwrap_or(d)
            .to_string(),
        _ => "core".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::source_of;

    #[test]
    fn test_source_of() {
        assert_eq!(source_of(None), "core");
        assert_eq!(source_of(Some("security")), "core");
        assert_eq!(source_of(Some("ext_procedural_table_lookup")), "ext_procedural");
        assert_eq!(source_of(Some("ext_nemotron_ipi_healthcare")), "ext_nemotron_ipi");
        assert_eq!(source_of(Some("ext_llama_security_incidents")), "ext_llama_security_incidents");
    }
}
