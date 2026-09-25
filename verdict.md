# Training Failure Analysis & Post-Mortem Verdict

**Run Command:**
```bash
cargo run --release --bin reflex-train -- \
  --backbone models/mamba2-370m-hf.safetensors \
  --tokenizer models/tokenizer.json \
  --train data/reflex_training_data.jsonl \
  --val data/reflex_validation_data.jsonl \
  --out models/reflex_heads.safetensors \
  --seed 42 --epochs 150 --backend wgpu --batch-size 16 --lr 0.01
```

**Run Duration:** ~34,760s (~9.65 hours)
**Dataset:** 1,858 training scenarios, 228 validation scenarios
**Backbone:** `models/mamba2-370m-hf.safetensors` (48 layers, $d_{\text{model}} = 1024$, vocab = 50,288)

---

## 1. Executive Summary & Verdict

The training run was a **total generalization failure caused by the unconstrained memorization of corrupted, uninformative feature vectors**.

While training loss dropped by **93.0%** overall (with Choice loss collapsing from **1.7785 down to 0.0048**, representing near-100% training accuracy), validation performance collapsed completely to **pure random chance**:

| Task Metric | Validation Result | Theoretical Random Baseline | Interpretation |
| :--- | :--- | :--- | :--- |
| **Choice Accuracy** | **22.7%** (55 / 242) | **25.0%** (4-way categorical) | Worse than random guessing |
| **Noul Accuracy** | **51.0%** (292 / 573) | **50.0%** (Binary True/False) | Pure coin flip |
| **Score RMSE** | **1.711** | ~1.4 – 1.7 (Uniform on $[1, 5]$) | Uninformative / mean guessing |

Furthermore, the model exhibited **extreme overconfidence on erroneous predictions** (e.g., 91.9%, 99.8%, and 100.0% confidence on incorrect candidates). This is the canonical diagnostic indicator of an over-parameterized classifier fitting static noise without learning invariant semantic representations.

---

## 2. Root Cause Analysis

Investigation of the codebase, tokenizer configuration, delimiter placement, and model checkpoint weights identified four compounding structural defects:

### A. Pretrained Causal Backbone Forcibly Run with Bidirectional Weight-Sharing
The loaded checkpoint [`models/mamba2-370m-hf.safetensors`](file:///home/carl/Documents/projects/burn-mamba/models/mamba2-370m-hf.safetensors) is an open-source Mamba-2 checkpoint (`state-spaces/mamba2-370m`) pretrained exclusively as an **autoregressive, causal (left-to-right) language model**.

In [`src/model/ssd.rs`](file:///home/carl/Documents/projects/burn-mamba/src/model/ssd.rs#L350-L368), the engine attempts bidirectional modeling by running the same causal weights over the reversed sequence and fusing them:

```rust
// Forward chronological scan
let out_fwd = self.forward_pass(x.clone());

// Backward outcome scan (reverse time, scan, reverse back)
let x_rev = self.reverse_sequence(x);
let out_bwd_rev = self.forward_pass(x_rev);
let out_bwd = self.reverse_sequence(out_bwd_rev);

// Fuse both temporal trajectories
let fused = out_fwd + out_bwd;
self.norm.forward(residual + fused)
```

**Why this breaks:**
1. **Asymmetric 1D Convolutions:** The depthwise `conv1d` layer is configured with causal explicit padding `(3, 0)`. Convolving backward-reversed tokens with filters trained on forward-ordered natural language produces nonsensical activations.
2. **Causal SSM State Decay:** The SSD scan parameters ($A_{\log}, \Delta t, B, C$) are parameterized for forward time accumulation. Backward scanning with forward-decay dynamics destroys state coherence.
3. **Multi-layer Compounding:** Because `fused = out_fwd + out_bwd` is computed inside every block, layer 0 outputs corrupted representations into layer 1. Propagated through all 48 layers, the pretrained semantic space of Mamba-2 is completely degraded into pseudo-random representations.

---

### B. Delimiter Placement & Causal Coordinate Slicing Blindness
In [`src/dataset.rs`](file:///home/carl/Documents/projects/burn-mamba/src/dataset.rs#L295-L318), the token sequence layout is constructed as:

```text
[CLS] <context> [SEP] <choice_q> <prompt> <cand> <option 0> <cand> <option 1> ... <noul_q> <assertion> ... <score_q> <rubric> ... [EOS]
```

In [`src/delimiters.rs`](file:///home/carl/Documents/projects/burn-mamba/src/delimiters.rs#L40-L45), the coordinate resolver operates under `CoordinateTarget::MarkerToken` by default:

* **Choice Candidates:** The slice coordinate is taken at the `<cand>` marker index **before** `<option k>` text has appeared. In causal directionality, the hidden state at `<cand> 0` contains the prompt, but 0 tokens of Option 0. Slicing at `<cand> 1` contains Option 0, but 0 tokens of Option 1.
* **Noul Queries:** Sliced at `<noul_q>` **before** the assertion text is ingested.
* **Rubric Scores:** Sliced at `<score_q>` **before** the rubric prompt text is ingested.

Because the forward pass at the marker position has not yet ingested the query payload, and the backward pass generates out-of-distribution noise, the gathered representations at those markers contain **zero discriminative information** about the actual options or assertions.

---

### C. Uninitialized Delimiter Embeddings on a Frozen Backbone
In [`src/delimiters.rs`](file:///home/carl/Documents/projects/burn-mamba/src/delimiters.rs#L55-L68), special delimiter IDs `50280..=50286` were designated for `[CLS]`, `[SEP]`, `[EOS]`, `<cand>`, `<noul_q>`, `<score_q>`, and `<choice_q>`.

Inspection of the binary checkpoint weights revealed:
* Real BPE token embeddings (e.g. token 50276) have an average L2 norm of **~5.91**.
* Rows `50280..50286` sit in the unused padding rows above the 50,277-token vocabulary; their weights are raw uninitialized values with an abnormally large L2 norm of **~13.56**.
* Because [`src/bin/reflex-train.rs`](file:///home/carl/Documents/projects/burn-mamba/src/bin/reflex-train.rs#L180-L240) trains heads on a **frozen backbone**, these uninitialized token embeddings are never updated or adapted. Every marker injects a large, arbitrary perturbation directly into the attention/SSM mixer.

---

### D. Extreme Over-Parameterization & Aggressive Optimizer Settings
The head architecture and training regime set up the exact conditions for memorization of uninformative features:

* **Trainable Parameters:** The [`UnifiedHeads`](file:///home/carl/Documents/projects/burn-mamba/src/model/heads.rs#L30-L70) module contains ~2.2 million parameters across its multi-layer perceptrons (`1024 -> 512 -> 1/4`).
* **Dataset Size:** Only 1,858 training scenarios.
* **Optimization Regime:** 150 epochs, learning rate **0.01** using standard Adam with no learning rate warmup, no cosine decay, and no early stopping.
* **Regularization:** Zero dropout, zero weight decay, zero data augmentation.

An unregularized 2.2M-parameter MLP can trivially memorize 1,858 static 1,024-dimensional vectors (even if drawn from pure Gaussian noise). Over 150 epochs at $\text{lr} = 0.01$, the heads learned arbitrary decision boundaries separating the 1,858 training samples, yielding a near-zero training loss and zero generalization capability.

---

## 3. Remediation Roadmap

To convert the Tier 1 Reflex Engine into a functional, generalizing system, the following architectural fixes should be implemented:

1. **Restore Strictly Causal Forward Propagation:**
   * In [`src/model/ssd.rs`](file:///home/carl/Documents/projects/burn-mamba/src/model/ssd.rs), remove the `out_bwd` reverse pass in `forward()` unless bidirectional pretraining is performed from scratch. Causal Mamba-2 must run purely chronologically.

2. **Fix Delimiter Slicing / Coordinate Extraction:**
   * Modify [`src/delimiters.rs`](file:///home/carl/Documents/projects/burn-mamba/src/delimiters.rs) and [`src/dataset.rs`](file:///home/carl/Documents/projects/burn-mamba/src/dataset.rs) so that coordinates are gathered **after** the content has been ingested (e.g. slicing the final token of each candidate or assertion, or applying mean pooling across candidate spans).
   * Alternatively, format the prompt so that evaluation tokens follow the full payload (e.g. `<cand> text </cand>`, slicing at `</cand>`).

3. **Initialize Delimiter Token Embeddings:**
   * Rather than referencing uninitialized vocabulary rows `50280..=50286`, either map delimiters to existing semantic tokens (e.g. `[`, `]`, `:`, `?`), or allow embedding rows to update during training.

4. **Regularize Head Training:**
   * Introduce Dropout ($p \in [0.1, 0.2]$) in [`UnifiedHeads`](file:///home/carl/Documents/projects/burn-mamba/src/model/heads.rs).
   * Lower the learning rate from `0.01` to `1e-4` – `5e-4`.
   * Add weight decay ($1\text{e-}4$).
   * Reduce the epoch count to 10–25 and implement validation evaluation checkpoints for early stopping.
