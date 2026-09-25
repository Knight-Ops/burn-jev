#### 1. In-Context Choice Head ($K$-way Categorical, $K \le 255$)

- **Mechanism:** Candidate options are declared dynamically in the prompt delimited by `<cand>` tags. For sequences with multiple choice questions, questions are demarcated with `<choice_q>` tokens, grouping candidate states per question. The model gathers the hidden states $\mathbf{h}_{\text{cand}_{q,k}} \in \mathbb{R}^D$ directly at those token coordinates.
- **Scoring Function:**
  $$z_{q,k} = \mathbf{w}_{\text{choice}}^T \cdot \text{GELU}(\mathbf{W}_c \mathbf{h}_{\text{cand}_{q,k}} + \mathbf{b}_c)$$
  $$P(\text{Choice}_q = k) = \frac{\exp(z_{q,k} / T)}{\sum_{j=1}^{K_q} \exp(z_{q,j} / T)}$$
- **Structural Guarantee:** 0% schema hallucination rate. It is impossible to output an unlisted option.

#### 2. Noul Head (Calibrated Boolean Assertion)

- **Mechanism:** Slices the hidden vectors at one or more dedicated `<noul_q>` question markers. When multiple boolean questions are queried ($Q_{\text{noul}} \ge 1$), all state vectors are gathered into a single tensor $[Q_{\text{noul}}, d_{\text{model}}]$ and evaluated in a single parallel GEMM pass.
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

All heads are calibrated using a learnable Platt scaling temperature parameter $T$, trained via strictly proper scoring rules (Brier score / Expected Calibration Error minimization).
