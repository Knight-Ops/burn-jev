# Tier 1 Reflex Engine: Implementation Checklist & Roadmap

This document serves as the master implementation checklist, architecture tracking ledger, and progress scorecard for the **Tier 1 Reflex Engine** (`burn-mamba`).

---

## Progress Overview

- [x] **Phase 1: Mamba-2 SSD Backbone Refinement & Stability**
- [x] **Phase 2: In-Context Delimiter Parsing & Coordinate Slicing**
- [x] **Phase 3: Typed Decision Layer & Jev Primitives**
- [x] **Phase 4: Multi-Tenant Non-Parametric $k$-NN Anomaly Gating**
- [x] **Phase 5: Tier 1 Pipeline Orchestrator & Tier 2 Handoff Interface**
- [x] **Phase 6: Joint Training Loss & Calibration Engine**
- [x] **Phase 7: Comprehensive Test Suite & Latency Benchmarks**
- [x] **Phase 8: Mamba-2 Pretrained Checkpoint Seeding & Multi-Scale Loader**
- [x] **Phase 9: Multi-Question Single-Pass JEV Primitives & Vectorized Head Evaluation**

---

### Phase 8: Mamba-2 Pretrained Checkpoint Seeding & Multi-Scale Loader

Empower the Tier 1 Reflex Engine to inherit pretrained weights from open-source Mamba-2 checkpoints across arbitrary scales (130M, 370M, 780M, 1.3B, 2.7B+), reducing pretraining compute requirements by ~99.99%.

- [x] **8.1 Architectural Generalization & Mamba-2 Alignment**
  - [x] Grouped State Space (`ngroups`): Support $B$ and $C$ projection sharing across heads ($N_{\text{groups}} = 1$) matching official checkpoints.
  - [x] Trainable `dt_bias`: Incorporated $\Delta t$ bias parameter $\mathbb{R}^{N_{\text{heads}}}$ added prior to softplus discretization.
  - [x] Inner RMSNorm (`inner_norm`): In-mixer normalization on SSD scan outputs prior to gating.
  - [x] Dimension presets: Added `BiMamba2Config::mamba2_130m()`, `mamba2_370m()`, `mamba2_780m()`, `mamba2_1_3b()`, `mamba2_2_7b()`, alongside custom arbitrary dimensions.
- [x] **8.2 Native Safetensors Loader**
  - [x] Zero-Python native Rust checkpoint parser in `src/model/loader.rs` using `safetensors`.
  - [x] Multi-format precision conversion: supports `F32`, `F16`, and `BF16` checkpoint tensors.
  - [x] Automatic tensor alignment: handles Linear weight transposition (`[out, in] -> [in, out]`), Conv1d matching (`[d_conv, 1, 4]`), RMSNorm gammas, embeddings, and log-decay rates.
  - [x] Flexible naming resolution: supports Hugging Face (`backbone.layers...`) and Tri Dao / state-spaces keys.
  - [x] Verification report & safety checks: `LoadReport` tracking layers/parameters transferred, with shape mismatch detection and strict layer bounds.
  - [x] Automatic architecture inference: `Mamba2CheckpointLoader::infer_config` deduces dimensions (`vocab_size`, `d_model`, `n_layers`, `headdim`, `expand`, `d_state`) from raw safetensors files.
  - [x] Verified by `tests/seeding_test.rs`.
- [x] **8.3 Pretrained Inference & Multi-Modal JEV Validation Example**
  - [x] Built full working example in `examples/seeded_jev_inference.rs`.
  - [x] CLI checkpoint ingestion with automatic fallback to coherent self-contained demo checkpoint (`models/mamba2_seeded_demo.safetensors`).
  - [x] Evaluates 3 realistic production JEV scenarios (DevOps tool routing, SQL injection guardrail, Raft consensus logic).
  - [x] Runtime coherence assertions: probability normalization ($\sum p_i = 1.0$), ordinal rubric validity ($S \in [1.0, 5.0]$), and metric state unit sphere norm ($\|\mathbf{z}\|_2 = 1.000$).

---

### Phase 9: Multi-Question Single-Pass JEV Primitives & Vectorized Head Evaluation

Empower the Tier 1 Reflex Engine to parse and evaluate multiple categorical choice questions, boolean assertions, and ordinal rubrics simultaneously in a single forward pass without autoregressive decoding.

- [x] **9.1 Multi-Query Delimiter Grammar & Coordinate Parsing**
  - [x] Added `choice_query_marker_id: i64` (default `7`) and configurable question capacity limits (`max_choice_questions: 32`, `max_noul_questions: 64`, `max_score_questions: 64`) to `DelimiterConfig`.
  - [x] Added `ChoiceQuestionCoordinates` struct tracking question query index and individual candidate coordinates.
  - [x] Upgraded `SequenceCoordinates` to store `choice_questions: Vec<ChoiceQuestionCoordinates>`, `noul_query_indices: Vec<usize>`, and `score_query_indices: Vec<usize>`.
  - [x] Preserved backward compatibility: `candidate_indices`, `noul_query_index`, and `score_query_index` provide backward-compatible single-question accessors.
  - [x] Added helper inquiry methods: `has_noul()`, `has_score()`, `has_choice()`, `choice_question_count()`, `noul_question_count()`, `score_question_count()`.
  - [x] Single-pass $O(N)$ scanning in `CoordinateResolver::resolve_coordinates`: groups candidates per `<choice_q>` and accumulates arbitrary `<noul_q>` and `<score_q>` markers.
- [x] **9.2 Dynamic Tensor Slicing & Extraction**
  - [x] Implemented `extract_noul_states` gathering $[Q_{\text{noul}}, d_{\text{model}}]$ via `Tensor::select(0, indices)`.
  - [x] Implemented `extract_score_states` gathering $[Q_{\text{score}}, d_{\text{model}}]$ via `Tensor::select(0, indices)`.
  - [x] Implemented `extract_choice_question_candidates` gathering $[K_q, d_{\text{model}}]$ for an individual choice question.
  - [x] Preserved existing single-state gather functions for non-breaking API compatibility.
- [x] **9.3 Vectorized JEV Head Evaluation**
  - [x] Added `UnifiedHeads::evaluate_noul_verdicts` and `evaluate_noul_verdicts_default` computing calibrated boolean assertions over $[Q_{\text{noul}}, d_{\text{model}}]$ in a single GEMM pass.
  - [x] Added `UnifiedHeads::evaluate_score_verdicts` evaluating continuous ordinal rubrics over $[Q_{\text{score}}, d_{\text{model}}]$ in a single GEMM pass.
- [x] **9.4 Reflex Pipeline Integration**
  - [x] Updated `ReflexVerdict` to expose `choices: Vec<ChoiceVerdict>`, `nouls: Vec<NoulVerdict>`, and `scores: Vec<ScoreVerdict>`.
  - [x] Preserved backward-compatible primary fields: `choice`, `noul`, and `score`.
  - [x] Updated `ReflexEngine::evaluate` to evaluate all candidate groups, noul queries, and rubric queries in a single pass.
- [x] **9.5 Verification & Real-Tokenized End-to-End Execution**
  - [x] Comprehensive delimiter tests in `tests/delimiters_test.rs` validating multi-query coordinate resolution and tensor extraction.
  - [x] End-to-end pipeline multi-question integration test in `tests/pipeline_test.rs`.
  - [x] All 44 unit and integration tests passing across all test suites.
  - [x] Updated `examples/seeded_jev_inference.rs` to demonstrate multi-choice, multi-noul, and multi-score real BPE inference in a single forward pass.

## Phase 10: Training & Inference Applications
- [x] **10.1 Library-Owned Training Logic**
  - [x] `training::features`: `compute_features_from_backbone` / `get_or_compute_features` (frozen-backbone features, `FeatureCache` keyed by backbone SHA-256).
  - [x] `training::trainer`: `train_heads(scenarios, heads_config, TrainConfig, ..)` with per-epoch `JointLossBreakdown`s; weights in `JointLossConfig::frozen_backbone()`; optional seed.
  - [x] `training::eval`: `evaluate_dataset(&ReflexEngine, &JevDataset, ..) -> EvalReport` (choice/noul accuracy, score RMSE).
  - [x] `ReflexRequest`: unlabeled scenario; single definition of the delimited token layout shared by training and inference.
- [x] **10.2 Heads-Only Artifact**
  - [x] `Mamba2CheckpointLoader::save_heads_file` / `load_heads_file`: `heads.*` tensors plus `HeadsMetadata` (delimiters, dims, backbone SHA-256) in the safetensors header.
  - [x] Loading refuses a backbone whose SHA-256 differs from the one the heads were trained on.
  - [x] `DelimiterConfig::mamba2_reserved()` (IDs 50280..=50286) is the single source of those IDs; inference reads them from the artifact.
- [x] **10.3 Binaries** (`src/bin/`)
  - [x] `reflex-train`: train (+ `--val`), `eval` subcommand, `--export-full` for a combined checkpoint.
  - [x] `reflex`: backbone + heads, JSONL file or stdin in, one JSON verdict per line out.
  - [x] Removed the toy `src/main.rs` and the fallback-heavy examples; `examples/generate_demo_backbone.rs` writes a synthetic demo backbone.
