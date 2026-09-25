use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Linear, LinearConfig,
        RmsNorm, RmsNormConfig,
    },
    tensor::{
        activation::{silu, softplus},
        backend::Backend,
        Bool, Tensor,
    },
};

// =====================================================================
// Bidirectional Mamba-2 Layer (State Space Duality / SSD)
// =====================================================================

#[derive(Config, Debug)]
pub struct Mamba2SSDConfig {
    pub d_model: usize,
    #[config(default = "64")]
    pub d_state: usize,
    #[config(default = "64")]
    pub headdim: usize,
    #[config(default = "2")]
    pub expand: usize,
    #[config(default = "1")]
    pub ngroups: usize,
    #[config(default = "true")]
    pub use_dt_bias: bool,
    #[config(default = "true")]
    pub use_inner_norm: bool,
}

#[derive(Module, Debug)]
pub struct Mamba2SSDBlock<B: Backend> {
    pub in_proj: Linear<B>,
    pub conv1d: Conv1d<B>,
    pub out_proj: Linear<B>,
    pub norm: RmsNorm<B>,
    pub a_log: Param<Tensor<B, 1>>,
    pub d_skip: Param<Tensor<B, 1>>,
    pub dt_bias: Option<Param<Tensor<B, 1>>>,
    pub inner_norm: Option<RmsNorm<B>>,
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub headdim: usize,
    pub nheads: usize,
    pub ngroups: usize,
    pub d_conv: usize,
}

impl Mamba2SSDConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2SSDBlock<B> {
        let d_inner = self.d_model * self.expand;
        assert_eq!(d_inner % self.headdim, 0, "d_inner must be divisible by headdim");
        let nheads = d_inner / self.headdim;
        let d_conv = d_inner + 2 * self.ngroups * self.d_state;
        let in_proj_dim = 2 * d_inner + 2 * self.ngroups * self.d_state + nheads;

        let in_proj = LinearConfig::new(self.d_model, in_proj_dim)
            .with_bias(false)
            .init(device);

        let conv1d = Conv1dConfig::new(d_conv, d_conv, 4)
            .with_padding(burn::nn::PaddingConfig1d::Explicit(3, 0))
            .with_groups(d_conv)
            .with_bias(true)
            .init(device);

        let out_proj = LinearConfig::new(d_inner, self.d_model)
            .with_bias(false)
            .init(device);

        let norm = RmsNormConfig::new(self.d_model).init(device);

        // Mamba-2 State Space Duality: scalar log-decay parameter A per head.
        // Initialized with log-spaced decay rates from 1.0 to 16.0 (a_h = exp(alpha_h))
        let a_log_vec: Vec<f32> = if nheads == 1 {
            vec![1.0f32.ln()]
        } else {
            let min_log = 1.0f32.ln();
            let max_log = 16.0f32.ln();
            (0..nheads)
                .map(|h| min_log + (h as f32 / (nheads - 1) as f32) * (max_log - min_log))
                .collect()
        };
        let a_log = Param::from_tensor(Tensor::<B, 1>::from_data(a_log_vec.as_slice(), device));

        // Direct feedthrough parameter D per head, initialized to 1.0
        let d_skip = Param::from_tensor(Tensor::<B, 1>::ones([nheads], device));

        // Optional dt_bias parameter per head, initialized to zeros
        let dt_bias = if self.use_dt_bias {
            Some(Param::from_tensor(Tensor::<B, 1>::zeros([nheads], device)))
        } else {
            None
        };

        // Optional inner RMSNorm on SSD scan output (dimension: d_inner)
        let inner_norm = if self.use_inner_norm {
            Some(RmsNormConfig::new(d_inner).init(device))
        } else {
            None
        };

        Mamba2SSDBlock {
            in_proj,
            conv1d,
            out_proj,
            norm,
            a_log,
            d_skip,
            dt_bias,
            inner_norm,
            d_model: self.d_model,
            d_inner,
            d_state: self.d_state,
            headdim: self.headdim,
            nheads,
            ngroups: self.ngroups,
            d_conv,
        }
    }
}

/// Chunk length for the chunked SSD scan. Sequential work is `ceil(L / SSD_CHUNK)` steps.
pub const SSD_CHUNK: usize = 64;

impl<B: Backend> Mamba2SSDBlock<B> {
    /// Broadcast per-group B/C projections to per-head: [B, L, G, N] -> [B, L, H, N].
    /// Head h reads group h / (H / G), matching the reference Mamba-2 layout.
    fn expand_groups(&self, t: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, seq_len, ngroups, d_state] = t.dims();
        if ngroups == self.nheads {
            return t;
        }
        let heads_per_group = self.nheads / ngroups;
        t.reshape([batch, seq_len, ngroups, 1, d_state])
            .expand([batch, seq_len, ngroups, heads_per_group, d_state])
            .reshape([batch, seq_len, self.nheads, d_state])
    }

    /// Chunked SSD (state space duality) scan for Mamba-2.
    ///
    /// Within each chunk of length Q the recurrence is evaluated as masked attention-like
    /// matmuls; only the chunk-boundary states are carried sequentially. Numerically
    /// equivalent to [`Self::ssd_scan_recurrent`].
    pub fn ssd_scan(
        &self,
        x: Tensor<B, 4>,  // [Batch, SeqLen, Heads, HeadDim]
        dt: Tensor<B, 3>, // [Batch, SeqLen, Heads]
        b: Tensor<B, 4>,  // [Batch, SeqLen, Groups, State]
        c: Tensor<B, 4>,  // [Batch, SeqLen, Groups, State]
    ) -> Tensor<B, 4> {
        let [batch, seq_len, nheads, headdim] = x.dims();
        let d_state = self.d_state;
        let device = x.device();
        if seq_len == 0 {
            return x;
        }

        let b = self.expand_groups(b); // [B, L, H, N]
        let c = self.expand_groups(c); // [B, L, H, N]

        // 1. Pad L up to a multiple of Q with zero dt/x/B/C (trailing, so causal outputs are unaffected)
        let q = SSD_CHUNK.min(seq_len);
        let n_chunks = seq_len.div_ceil(q);
        let padded_len = n_chunks * q;
        let pad = padded_len - seq_len;
        let (x, dt, b, c) = if pad > 0 {
            (
                Tensor::cat(vec![x, Tensor::zeros([batch, pad, nheads, headdim], &device)], 1),
                Tensor::cat(vec![dt, Tensor::zeros([batch, pad, nheads], &device)], 1),
                Tensor::cat(vec![b, Tensor::zeros([batch, pad, nheads, d_state], &device)], 1),
                Tensor::cat(vec![c, Tensor::zeros([batch, pad, nheads, d_state], &device)], 1),
            )
        } else {
            (x, dt, b, c)
        };

        // 2. Reshape into chunks, heads before the intra-chunk axis
        let x = x.reshape([batch, n_chunks, q, nheads, headdim]).swap_dims(2, 3); // [B, c, H, Q, P]
        let dt = dt.reshape([batch, n_chunks, q, nheads]).swap_dims(2, 3); // [B, c, H, Q]
        let b = b.reshape([batch, n_chunks, q, nheads, d_state]).swap_dims(2, 3); // [B, c, H, Q, N]
        let c = c.reshape([batch, n_chunks, q, nheads, d_state]).swap_dims(2, 3); // [B, c, H, Q, N]

        // 3. Cumulative log-decay within each chunk: cs_i = sum_{j<=i} dt_j * A
        let a = self.a_log.val().exp().neg().reshape([1, 1, nheads, 1]);
        let cs = (dt.clone() * a).cumsum(3); // [B, c, H, Q]
        let cs_last = cs.clone().slice([0..batch, 0..n_chunks, 0..nheads, q - 1..q]); // [B, c, H, 1]

        let x_dt = x.clone() * dt.unsqueeze_dim(4); // [B, c, H, Q, P]

        // 4. Intra-chunk decay L[i, j] = exp(cs_i - cs_j) for i >= j, else 0.
        //    Mask to -inf before exp so the upper triangle never overflows.
        let seg = cs.clone().unsqueeze_dim::<5>(4) - cs.clone().unsqueeze_dim::<5>(3); // [B, c, H, Q, Q]
        let upper = Tensor::<B, 2, Bool>::tril_mask([q, q], 0, &device)
            .reshape([1, 1, 1, q, q])
            .expand([batch, n_chunks, nheads, q, q]);
        let decay = seg.mask_fill(upper, f32::NEG_INFINITY).exp();

        // 5. Diagonal (intra-chunk) outputs
        let scores = c.clone().matmul(b.clone().swap_dims(3, 4)); // [B, c, H, Q, Q]
        let y_diag = (scores * decay).matmul(x_dt.clone()); // [B, c, H, Q, P]

        // 6. State contributed by each chunk at its end: S_k = (x dt exp(cs_last - cs))^T B
        let state_decay = (cs_last.clone() - cs.clone()).exp().unsqueeze_dim::<5>(4); // [B, c, H, Q, 1]
        let chunk_states = (x_dt * state_decay).swap_dims(3, 4).matmul(b); // [B, c, H, P, N]

        // 7. Inter-chunk recurrence over only ceil(L / Q) steps
        let chunk_decay = cs_last.exp(); // [B, c, H, 1]
        let mut h = Tensor::<B, 4>::zeros([batch, nheads, headdim, d_state], &device);
        let mut h_in = Vec::with_capacity(n_chunks);
        for k in 0..n_chunks {
            h_in.push(h.clone());
            let decay_k = chunk_decay
                .clone()
                .slice([0..batch, k..k + 1, 0..nheads, 0..1])
                .reshape([batch, nheads, 1, 1]);
            let s_k = chunk_states
                .clone()
                .slice([0..batch, k..k + 1, 0..nheads, 0..headdim, 0..d_state])
                .reshape([batch, nheads, headdim, d_state]);
            h = h * decay_k + s_k;
        }
        let h_in = Tensor::stack::<5>(h_in, 1); // [B, c, H, P, N]

        // 8. Off-diagonal outputs from the carried-in state, plus direct feedthrough D
        let y_off = c.matmul(h_in.swap_dims(3, 4)) * cs.exp().unsqueeze_dim::<5>(4); // [B, c, H, Q, P]
        let d_skip = self.d_skip.val().reshape([1, 1, nheads, 1, 1]);
        let y = y_diag + y_off + x * d_skip;

        y.swap_dims(2, 3)
            .reshape([batch, padded_len, nheads, headdim])
            .slice([0..batch, 0..seq_len, 0..nheads, 0..headdim])
    }

    /// Token-by-token recurrent scan for Mamba-2 SSD. Reference implementation for [`Self::ssd_scan`].
    pub fn ssd_scan_recurrent(
        &self,
        x: Tensor<B, 4>,  // [Batch, SeqLen, Heads, HeadDim]
        dt: Tensor<B, 3>, // [Batch, SeqLen, Heads]
        b: Tensor<B, 4>,  // [Batch, SeqLen, Groups, State]
        c: Tensor<B, 4>,  // [Batch, SeqLen, Groups, State]
    ) -> Tensor<B, 4> {
        let [batch, seq_len, nheads, headdim] = x.dims();
        let d_state = self.d_state;
        let device = x.device();

        let b = self.expand_groups(b); // [B, L, H, N]
        let c = self.expand_groups(c); // [B, L, H, N]

        // Loop invariants: dA = exp(dt * A), dt * x and D * x for the whole sequence
        let a = self.a_log.val().exp().neg().reshape([1, 1, nheads]);
        let d_a = (dt.clone() * a).exp(); // [B, L, H]
        let x_dt = x.clone() * dt.unsqueeze_dim(3); // [B, L, H, P]
        let d_x = x * self.d_skip.val().reshape([1, 1, nheads, 1]); // [B, L, H, P]

        let mut h = Tensor::<B, 4>::zeros([batch, nheads, headdim, d_state], &device);
        let mut y_steps: Vec<Tensor<B, 4>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let d_a_t = d_a.clone().slice([0..batch, t..t + 1, 0..nheads])
                .reshape([batch, nheads, 1, 1]);
            let x_dt_t = x_dt.clone().slice([0..batch, t..t + 1, 0..nheads, 0..headdim])
                .reshape([batch, nheads, headdim, 1]);
            let b_t = b.clone().slice([0..batch, t..t + 1, 0..nheads, 0..d_state])
                .reshape([batch, nheads, 1, d_state]);
            let c_t = c.clone().slice([0..batch, t..t + 1, 0..nheads, 0..d_state])
                .reshape([batch, nheads, 1, d_state]);

            // h_t = h_{t-1} * dA_t + dt_t * (x_t ⊗ B_t)
            h = h * d_a_t + x_dt_t * b_t;

            // y_t = sum_N(h_t * C_t)
            let y_t = (h.clone() * c_t).sum_dim(3).reshape([batch, 1, nheads, headdim]);
            y_steps.push(y_t);
        }

        Tensor::cat(y_steps, 1) + d_x // [B, L, H, P]
    }

    /// Numerically stable softplus preventing f32 exp overflow
    pub fn stable_softplus<const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
        // For x > 20.0, softplus(x) = ln(1 + e^x) ≈ x (difference < 2e-9)
        // Clamping input to [-80.0, 20.0] before softplus and blending prevents +inf in f32 exp.
        let clamped = x.clone().clamp(-80.0, 20.0);
        let sp = burn::tensor::activation::softplus(clamped, 1.0);
        // If x > 20.0, use x directly:
        let mask = x.clone().greater_elem(20.0);
        sp.mask_where(mask, x)
    }

    /// Single directional forward pass through convolutions and SSD
    pub fn forward_pass(&self, u: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = u.dims();

        // 1. In-projection
        let projected = self.in_proj.forward(u);
        let x_conv_in = projected.clone().slice([0..batch, 0..seq_len, 0..self.d_conv]);
        let z_total = projected.slice([0..batch, 0..seq_len, self.d_conv..self.d_conv + self.nheads + self.d_inner]);

        let dt_raw = z_total.clone().slice([0..batch, 0..seq_len, 0..self.nheads]);
        let z_gate = z_total.slice([0..batch, 0..seq_len, self.nheads..self.nheads + self.d_inner]);

        // 2. 1D Depthwise Convolution: swap to [B, C, L] then back to [B, L, C]
        let conv_in = x_conv_in.swap_dims(1, 2);
        let conv_out = self.conv1d.forward(conv_in);
        let conv_sliced = conv_out.slice([0..batch, 0..self.d_conv, 0..seq_len]).swap_dims(1, 2);
        let x_act = silu(conv_sliced);

        // 3. Unpack tensors
        let x_vec = x_act.clone().slice([0..batch, 0..seq_len, 0..self.d_inner])
            .reshape([batch, seq_len, self.nheads, self.headdim]);
        let b_vec = x_act.clone().slice([0..batch, 0..seq_len, self.d_inner..self.d_inner + self.ngroups * self.d_state])
            .reshape([batch, seq_len, self.ngroups, self.d_state]);
        let c_vec = x_act.slice([0..batch, 0..seq_len, self.d_inner + self.ngroups * self.d_state..self.d_conv])
            .reshape([batch, seq_len, self.ngroups, self.d_state]);

        // Numerically stable discretization with optional dt_bias
        let dt_biased = match &self.dt_bias {
            Some(bias) => dt_raw + bias.val().reshape([1, 1, self.nheads]),
            None => dt_raw,
        };
        let dt_clamped = dt_biased.clamp(-20.0, 20.0);
        let dt = softplus(dt_clamped, 1.0);

        // 4. SSD Scan Core
        let y = self.ssd_scan(x_vec, dt, b_vec, c_vec)
            .reshape([batch, seq_len, self.d_inner]);

        // 5. Multiplicative gate with optional inner_norm & output projection
        let y_normed = match &self.inner_norm {
            Some(norm) => norm.forward(y),
            None => y,
        };
        let gated = y_normed * silu(z_gate);
        self.out_proj.forward(gated)
    }

    /// Helper to reverse tensor along sequence axis (dimension 1) via native GPU flip
    pub fn reverse_sequence(&self, tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        tensor.flip([1])
    }

    /// Bidirectional forward execution with residual norm
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();

        // Forward chronological scan
        let out_fwd = self.forward_pass(x.clone());

        // Backward outcome scan (reverse time, scan, reverse back)
        let x_rev = self.reverse_sequence(x);
        let out_bwd_rev = self.forward_pass(x_rev);
        let out_bwd = self.reverse_sequence(out_bwd_rev);

        // Fuse both temporal trajectories
        let fused = out_fwd + out_bwd;
        self.norm.forward(residual + fused)
    }
}
