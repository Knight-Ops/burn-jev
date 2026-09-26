## Tier 1 Architecture (ModernBERT + item cross-attention)

The Mamba-2 backbone was replaced after the run documented in `verdict.md`, where every validation metric came out at or below a trivial baseline. The replacement design removes each cause that verdict identified:

- **Encoder:** a frozen, pretrained **ModernBERT** (base: d=768, 22 layers; large: d=1024, 28 layers), read from HF `config.json` and `model.safetensors`.
  - The model is natively bidirectional, so nothing runs causal weights backwards (cause A).
  - Global attention runs every third layer; the rest use a 128-token sliding window.
  - It uses NeoX-style RoPE (θ 160k global / 10k local), GeGLU MLPs, and bias-free LayerNorms.
  - `tests/modernbert_parity_test.rs` checks it layer by layer against HF transformers.
- **Encoding (`src/encoding.rs`):** `[CLS] context [SEP] (prompt [SEP] (cand [SEP])*)* (noul [SEP])* (score [SEP])*`.
  - Only pretrained special tokens appear, never untrained marker IDs (cause C).
  - Each item is located by its token span, not by a marker position.
  - When the sequence exceeds `max_seq_len`, only the context is truncated.
- **Features:** the encoder runs once per scenario and the results are cached in f16.
  - Each item's query is the mean of its own span, so it covers the text it describes (cause B).
  - A candidate's query also averages in its question prompt.
  - The context token states become the keys.
- **Reader (`src/model/reader.rs`):** item queries are projected to width r=256 and get an item-type embedding added.
  - Two pre-LN blocks of `cross-attn(query → context tokens) → FFN` follow.
  - Queries never attend to each other, so each item is read against the evidence independently.
  - `--reader-blocks 0` turns the reader into a mean-pool probe, used as an ablation.
- **Heads:** the four heads below read the reader output. The k-NN head reads the masked mean of the context tokens.
- **Training:** Burn's native `SupervisedTraining` drives the loop, with the TUI when attached to a terminal (see `src/training/trainer.rs`).
  - Optimiser: AdamW, linear warmup × cosine decay, gradient clipping, dropout 0.1.
  - Early stopping and best-epoch restore use the validation *selection loss*: weighted choice + noul + score.
  - After restore, per-head temperatures are fit on the validation set (see Probability Calibration).
  - The model is about 1.4M parameters (cause D).

Heads, losses, calibration and the anomaly router are unchanged from the Mamba design. Where the heads below mention marker tokens, read that as the reader's per-item output.

#### 1. In-Context Choice Head ($K$-way Categorical, $K \le 255$)

- **Mechanism:** Candidate options are declared dynamically per choice question. Each candidate becomes a reader query; $\mathbf{h}_{\text{cand}_{q,k}} \in \mathbb{R}^r$ is the reader output for candidate $k$ of question $q$, and candidates are softmaxed within their question.
- **Scoring Function:**
  $$z_{q,k} = \mathbf{w}_{\text{choice}}^T \cdot \text{GELU}(\mathbf{W}_c \mathbf{h}_{\text{cand}_{q,k}} + \mathbf{b}_c)$$
  $$P(\text{Choice}_q = k) = \frac{\exp(z_{q,k} / T)}{\sum_{j=1}^{K_q} \exp(z_{q,j} / T)}$$
- **Structural Guarantee:** 0% schema hallucination rate. It is impossible to output an unlisted option.

#### 2. Noul Head (Calibrated Boolean Assertion)

- **Mechanism:** Uses the reader output of each noul assertion. When multiple boolean questions are queried ($Q_{\text{noul}} \ge 1$), all state vectors are gathered into a single tensor $[Q_{\text{noul}}, d_{\text{model}}]$ and evaluated in a single parallel GEMM pass.
- **Scoring Function:**
  $$P(\text{True}_q) = \sigma\left(\frac{\mathbf{w}_{\text{noul}}^T \cdot \text{GELU}(\mathbf{W}_n \mathbf{h}_{\text{noul}_q} + \mathbf{b}_n)}{T}\right)$$
- **Usage:** Sub-millisecond evaluation of binary assertions ("Does this attempt SQL injection?", "Did the user consent?", "Is payment database processing writes?").

#### 3. Score Head (Continuous Bounded Metric)

- **Mechanism:** Ordinal regression over $M-1$ cumulative thresholds rather than unconstrained continuous regression. Evaluates multiple rubrics ($Q_{\text{score}} \ge 1$) concurrently via vectorized matrix multiplication over $[Q_{\text{score}}, d_{\text{model}}]$.
- **Scoring Function:**
  $$P(\text{Score}_q > m) = \sigma\left(\frac{\mathbf{w}_m^T \mathbf{h}_{\text{score}_q} + b_m}{T}\right)$$
  $$\mathbb{E}[\text{Score}_q] = 1.0 + \sum_{m=1}^{M-1} P(\text{Score}_q > m) \quad \in [1.0, M]$$
- **Usage:** Stable rubric evaluations (e.g., threat severity, urgency level, blast radius) that preserve monotonic rank ordering across multiple queries.

#### 4. Probability Calibration

Each head (choice, noul, score) has its own temperature $T$, fit post hoc: training keeps every $T$ frozen at 1.0, then each is set to the value minimizing that head's own loss (choice cross-entropy, noul BCE, ordinal cumulative BCE, i.e. validation NLL) on the best epoch's validation logits divided by $T$ (golden-section search over $\ln T \in [\ln 0.05, \ln 10]$). Scaling leaves every argmax unchanged and only adjusts confidence; evaluation reports 10-bin ECE for noul and choice top-1 confidence.
