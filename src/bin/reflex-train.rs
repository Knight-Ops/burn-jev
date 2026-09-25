//! Trains the JEV reflex heads on a frozen Mamba-2 backbone and writes a heads-only artifact.
//!
//! ```text
//! reflex-train [--backend cpu|wgpu] --backbone <st> --tokenizer <json> --train <jsonl> [--val <jsonl>] --out heads.safetensors
//! reflex-train eval [--backend cpu|wgpu] --backbone <st> --heads <st> --tokenizer <json> --data <jsonl>
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use clap::{Args, Parser, Subcommand};
use tokenizers::Tokenizer;

use burn_mamba::backend::{BackendKind, CpuBackend, FlexDevice};
use burn_mamba::training::{
    evaluate_dataset, get_or_compute_features, train_heads, CacheOutcome, CacheSettings,
    FeatureContext, TrainConfig,
};
use burn_mamba::{
    CoordinateResolver, DelimiterConfig, EvalReport, HeadsMetadata, JevDataset,
    Mamba2CheckpointLoader, ReflexEngine,
};

type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(version, about = "Train JEV reflex heads on a frozen Mamba-2 backbone")]
#[command(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    train: TrainArgs,
}

#[derive(Subcommand)]
enum Command {
    /// Evaluate an existing heads artifact on a labeled dataset without retraining.
    Eval(EvalArgs),
}

#[derive(Args)]
struct TrainArgs {
    /// Tensor backend to run on.
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,
    /// Frozen Mamba-2 backbone checkpoint (safetensors).
    #[arg(long, required = true)]
    backbone: Option<PathBuf>,
    /// HuggingFace tokenizer.json matching the backbone.
    #[arg(long, required = true)]
    tokenizer: Option<PathBuf>,
    /// Training scenarios (JSONL).
    #[arg(long, required = true)]
    train: Option<PathBuf>,
    /// Held-out scenarios (JSONL) evaluated after training.
    #[arg(long)]
    val: Option<PathBuf>,
    /// Output path for the heads-only artifact.
    #[arg(long, required = true)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = 50)]
    epochs: usize,
    #[arg(long, default_value_t = 0.02)]
    lr: f64,
    /// Scenarios per optimizer step. The metric loss needs at least 2 benign and 1 adversarial
    /// scenario in a batch to be non-zero.
    #[arg(long, default_value_t = 4, value_parser = parse_batch_size)]
    batch_size: usize,
    /// Seed for head initialization; set it for reproducible runs.
    #[arg(long)]
    seed: Option<u64>,
    /// Directory for the persistent backbone feature cache.
    #[arg(long, default_value = "data/.cache")]
    cache_dir: PathBuf,
    /// Recompute backbone features and do not write the cache.
    #[arg(long)]
    no_cache: bool,
    /// Also write a full checkpoint (backbone + heads) to this path.
    #[arg(long)]
    export_full: Option<PathBuf>,
}

#[derive(Args)]
struct EvalArgs {
    /// Tensor backend to run on.
    #[arg(long, value_enum, default_value_t = BackendKind::Cpu)]
    backend: BackendKind,
    #[arg(long)]
    backbone: PathBuf,
    #[arg(long)]
    heads: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    /// Labeled scenarios (JSONL).
    #[arg(long)]
    data: PathBuf,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Some(Command::Eval(args)) => match args.backend {
            BackendKind::Cpu => run_eval::<CpuBackend>(args, FlexDevice),
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => run_eval::<burn_mamba::backend::GpuWgpu>(
                args,
                burn_mamba::backend::WgpuDevice::default(),
            ),
            #[allow(unreachable_patterns)]
            other => Err(other.not_compiled()),
        },
        None => match cli.train.backend {
            BackendKind::Cpu => run_train::<CpuBackend>(cli.train, FlexDevice),
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => run_train::<burn_mamba::backend::GpuWgpu>(
                cli.train,
                burn_mamba::backend::WgpuDevice::default(),
            ),
            #[allow(unreachable_patterns)]
            other => Err(other.not_compiled()),
        },
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn parse_batch_size(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(0) => Err("batch size must be at least 1".into()),
        Ok(n) => Ok(n),
        Err(e) => Err(e.to_string()),
    }
}

fn load_tokenizer(path: &PathBuf) -> AppResult<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|e| format!("failed to load tokenizer {}: {e}", path.display()).into())
}

fn load_dataset(path: &PathBuf) -> AppResult<JevDataset> {
    let ds = JevDataset::from_jsonl_file(path)
        .map_err(|e| format!("failed to load dataset {}: {e}", path.display()))?;
    if ds.is_empty() {
        return Err(format!("dataset {} contains no scenarios", path.display()).into());
    }
    eprintln!("[dataset] {} scenarios from {}", ds.len(), path.display());
    Ok(ds)
}

fn run_train<B: Backend>(args: TrainArgs, device: B::Device) -> AppResult<()> {
    // `required = true` guarantees these are present when no subcommand is given.
    let backbone_path = args.backbone.unwrap();
    let tokenizer_path = args.tokenizer.unwrap();
    let train_path = args.train.unwrap();
    let out_path = args.out.unwrap();

    let tokenizer = load_tokenizer(&tokenizer_path)?;
    let train_ds = load_dataset(&train_path)?;
    let val_ds = args.val.as_ref().map(load_dataset).transpose()?;

    eprintln!("[backbone] loading {} (backend {})", backbone_path.display(), args.backend);
    let loaded = Mamba2CheckpointLoader::load_backbone_file::<B, _>(&backbone_path, &device)
        .map_err(|e| format!("failed to load backbone {}: {e}", backbone_path.display()))?;
    let config = loaded.config;
    eprintln!(
        "[backbone] d_model={} layers={} vocab={} ({} parameters, sha256 {})",
        config.d_model, config.n_layers, config.vocab_size, loaded.report.parameters_transferred, loaded.sha256
    );

    let delimiters = DelimiterConfig::mamba2_reserved();
    if delimiters.max_token_id() >= config.vocab_size as i64 {
        return Err(format!(
            "delimiter token id {} is outside the backbone vocabulary ({})",
            delimiters.max_token_id(),
            config.vocab_size
        )
        .into());
    }
    let resolver = CoordinateResolver::new(delimiters.clone());

    let ctx = FeatureContext {
        backbone: &loaded.model,
        tokenizer: &tokenizer,
        resolver: &resolver,
    };
    let cache_model_id = args.backend.cache_model_id(&loaded.sha256);
    let cache = CacheSettings {
        dir: &args.cache_dir,
        model_id: &cache_model_id,
        enabled: !args.no_cache,
    };
    let start = Instant::now();
    let mut computed_any = false;
    let (features, outcome) = get_or_compute_features::<Autodiff<B>>(
        &ctx,
        &train_ds,
        &train_path,
        &cache,
        &device,
        |done, total| {
            if !computed_any {
                computed_any = true;
                eprintln!("[features] computing backbone features for {total} scenarios");
            }
            if done % 25 == 0 || done == total {
                let elapsed = start.elapsed().as_secs_f64();
                let eta = elapsed / done as f64 * (total - done) as f64;
                eprint!("\r[features] {done}/{total} ({elapsed:.1}s, ETA {eta:.0}s)");
                if done == total {
                    eprintln!();
                }
            }
        },
    )?;
    let how = match outcome {
        CacheOutcome::Hit => "loaded from cache",
        CacheOutcome::MissSaved => "computed and cached",
        CacheOutcome::Disabled => "computed (cache disabled)",
    };
    eprintln!(
        "[features] {} scenarios {how} in {:.2}s",
        features.len(),
        start.elapsed().as_secs_f64()
    );

    let heads_config = config.heads_config();
    let train_config = TrainConfig {
        epochs: args.epochs,
        lr: args.lr,
        batch_size: args.batch_size,
        seed: args.seed,
        ..TrainConfig::default()
    };
    eprintln!(
        "\n[train] Adam lr={} epochs={} batch={} seed={:?}",
        train_config.lr, train_config.epochs, train_config.batch_size, train_config.seed
    );
    eprintln!(" Epoch | Total Loss | Choice Loss | Noul Loss | Score Loss | Metric Loss | Calib Loss");
    eprintln!("-------+------------+-------------+-----------+------------+-------------+-----------");
    let start = Instant::now();
    let epochs = train_config.epochs;
    let mut zero_metric_epochs = 0usize;
    let output = train_heads(&features, &heads_config, &train_config, &device, |epoch, b| {
        if b.metric_loss == 0.0 {
            zero_metric_epochs += 1;
        }
        if epoch == 1 || epoch % 10 == 0 || epoch == epochs {
            eprintln!(
                " {:>5} | {:>10.4} | {:>11.4} | {:>9.4} | {:>10.4} | {:>11.4} | {:>10.4}",
                epoch, b.total_loss, b.choice_loss, b.noul_loss, b.score_loss, b.metric_loss, b.calibration_loss
            );
        }
    });
    eprintln!("[train] done in {:.2}s", start.elapsed().as_secs_f64());
    if zero_metric_epochs > 0 {
        eprintln!(
            "[train] warning: metric loss was 0 in {zero_metric_epochs}/{epochs} epochs; no batch had \
             2+ benign and 1+ adversarial scenarios, so the k-NN head got no gradient then"
        );
    }

    let metadata = HeadsMetadata::new(&heads_config, delimiters, loaded.sha256.clone());
    Mamba2CheckpointLoader::save_heads_file(&output.heads, &metadata, &out_path)?;
    eprintln!(
        "\n[export] heads -> {} ({:.2} KB)",
        out_path.display(),
        out_path.metadata()?.len() as f64 / 1024.0
    );

    let mut model = loaded.model;
    model.heads = output.heads;
    if let Some(ref full_path) = args.export_full {
        model.save_safetensors_file(full_path)?;
        eprintln!(
            "[export] full model -> {} ({:.2} MB)",
            full_path.display(),
            full_path.metadata()?.len() as f64 / 1_048_576.0
        );
    }

    if let Some(val_ds) = val_ds {
        let engine = ReflexEngine::new(model, resolver);
        let report = evaluate_dataset(&engine, &val_ds, &tokenizer, &device)?;
        print_report(&report);
    }
    Ok(())
}

fn run_eval<B: Backend>(args: EvalArgs, device: B::Device) -> AppResult<()> {
    let tokenizer = load_tokenizer(&args.tokenizer)?;
    let dataset = load_dataset(&args.data)?;

    let loaded = Mamba2CheckpointLoader::load_backbone_file::<B, _>(&args.backbone, &device)
        .map_err(|e| format!("failed to load backbone {}: {e}", args.backbone.display()))?;
    let (heads, metadata) =
        Mamba2CheckpointLoader::load_heads_file::<B, _>(&args.heads, &loaded.sha256, &device)
            .map_err(|e| format!("failed to load heads {}: {e}", args.heads.display()))?;
    if metadata.d_model != loaded.config.d_model {
        return Err(format!(
            "heads expect d_model={}, backbone has {}",
            metadata.d_model, loaded.config.d_model
        )
        .into());
    }

    let mut model = loaded.model;
    model.heads = heads;
    let engine = ReflexEngine::new(model, CoordinateResolver::new(metadata.delimiters));
    let report = evaluate_dataset(&engine, &dataset, &tokenizer, &device)?;
    print_report(&report);
    Ok(())
}

fn pass(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

fn print_report(report: &EvalReport) {
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
    println!("\n=== Summary ===");
    println!(" Choice accuracy: {} ({cc}/{ct})", pct(report.choice_accuracy()));
    println!(" Noul accuracy:   {} ({nc}/{nt})", pct(report.noul_accuracy()));
    println!(
        " Score RMSE:      {} across {} rubric queries",
        report.score_rmse().map_or("n/a".to_string(), |v| format!("{v:.3}")),
        report.score_count()
    );
}
