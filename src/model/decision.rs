//! The trainable Tier 1 decision model (reader + heads) and the host-side features it reads.
//!
//! The frozen encoder's outputs are reduced once per scenario to [`ScenarioFeatures`]: the
//! context token states (the reader's keys) and one pooled vector per item (its query). Both
//! training (from the feature cache) and inference (straight from the encoder) go through
//! [`FeatureBatch`] and [`DecisionModel`], so the two paths cannot drift apart.

use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Bool, Int, Tensor, TensorData},
};

use crate::encoding::{candidate_counts, EncodedScenario, ItemKind};
use crate::model::heads::{UnifiedHeads, UnifiedHeadsConfig};
use crate::model::jev::{ChoiceVerdict, JevError, NoulVerdict, ScoreVerdict};
use crate::model::modernbert::ModernBertEncoder;
use crate::model::reader::{ItemReader, ItemReaderConfig};
use crate::training::JointLossConfig;

// =====================================================================
// Host-side features
// =====================================================================

/// One scenario's encoder features, row-major f32 on the host.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioFeatures {
    pub d_model: usize,
    /// `[ctx_len, d_model]` context token states.
    pub ctx: Vec<f32>,
    pub ctx_len: usize,
    /// `[kinds.len(), d_model]` pooled item queries, in [`EncodedScenario::items`] order.
    pub items: Vec<f32>,
    pub kinds: Vec<ItemKind>,
}

impl ScenarioFeatures {
    /// Reduces one sequence's hidden states (`hidden: [seq_len, d_model]`, row-major) to
    /// features: context rows are copied; each item is the mean of its span, and a candidate
    /// additionally averages in its question prompt's mean.
    pub fn from_hidden(hidden: &[f32], d_model: usize, encoded: &EncodedScenario) -> Self {
        let rows = |range: &std::ops::Range<usize>| &hidden[range.start * d_model..range.end * d_model];
        let mean = |range: &std::ops::Range<usize>| {
            let mut acc = vec![0.0f32; d_model];
            for row in rows(range).chunks_exact(d_model) {
                for (a, &v) in acc.iter_mut().zip(row) {
                    *a += v;
                }
            }
            let n = range.len().max(1) as f32;
            acc.iter_mut().for_each(|a| *a /= n);
            acc
        };

        let mut items = Vec::with_capacity(encoded.items.len() * d_model);
        for item in &encoded.items {
            let mut q = mean(&item.range);
            if let Some(ref prompt) = item.prompt_range {
                for (a, p) in q.iter_mut().zip(mean(prompt)) {
                    *a = 0.5 * (*a + p);
                }
            }
            items.extend(q);
        }
        Self {
            d_model,
            ctx: rows(&encoded.context).to_vec(),
            ctx_len: encoded.context.len(),
            items,
            kinds: encoded.items.iter().map(|i| i.kind).collect(),
        }
    }

    pub fn num_items(&self) -> usize {
        self.kinds.len()
    }

    pub fn choice_candidate_counts(&self) -> Vec<usize> {
        candidate_counts(self.kinds.iter().copied())
    }
}

impl<B: Backend> ModernBertEncoder<B> {
    /// Runs a padded batch of encoded scenarios through the encoder and reduces each to its
    /// [`ScenarioFeatures`]. Padding uses `pad_id` and is masked out of attention.
    pub fn scenario_features(
        &self,
        encoded: &[&EncodedScenario],
        pad_id: i64,
        device: &B::Device,
    ) -> Vec<ScenarioFeatures> {
        if encoded.is_empty() {
            return Vec::new();
        }
        let b = encoded.len();
        let l_max = encoded.iter().map(|e| e.len()).max().unwrap();
        let mut ids = vec![pad_id; b * l_max];
        let mut mask = vec![false; b * l_max];
        for (bi, e) in encoded.iter().enumerate() {
            ids[bi * l_max..bi * l_max + e.len()].copy_from_slice(&e.input_ids);
            mask[bi * l_max..bi * l_max + e.len()].iter_mut().for_each(|m| *m = true);
        }
        let hidden = self.forward(
            Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [b, l_max]), device),
            Tensor::<B, 2, Bool>::from_data(TensorData::new(mask, [b, l_max]), device),
        );
        let d = self.d_model();
        let hidden = hidden.into_data().convert::<f32>().to_vec::<f32>().expect("f32 hidden states");
        encoded
            .iter()
            .enumerate()
            .map(|(bi, e)| {
                let seq = &hidden[bi * l_max * d..(bi * l_max + e.len()) * d];
                ScenarioFeatures::from_hidden(seq, d, e)
            })
            .collect()
    }
}

// =====================================================================
// Padded device batch
// =====================================================================

/// Row indices into the flattened reader output `[B * K_max, r]` for one item type.
pub struct ItemRows<B: Backend> {
    pub rows: Tensor<B, 1, Int>,
    pub count: usize,
}

/// Choice candidates grouped per question, padded to the largest question.
pub struct ChoiceRows<B: Backend> {
    /// `[Q * C_max]`; padding points at row 0 and is masked.
    pub rows: Tensor<B, 1, Int>,
    /// `[Q, C_max]`, true for real candidates.
    pub mask: Tensor<B, 2, Bool>,
    pub questions: usize,
    pub max_candidates: usize,
    /// Candidates per question, in batch order.
    pub counts: Vec<usize>,
}

pub struct FeatureBatch<B: Backend> {
    pub ctx: Tensor<B, 3>,
    /// `[B, L_max]`, true = padding.
    pub ctx_pad: Tensor<B, 2, Bool>,
    /// `[B, L_max]` mean-pool weights over real context tokens.
    pub ctx_pool: Tensor<B, 2>,
    pub queries: Tensor<B, 3>,
    pub types: Tensor<B, 2, Int>,
    pub max_items: usize,
    pub choice: Option<ChoiceRows<B>>,
    pub noul: Option<ItemRows<B>>,
    pub score: Option<ItemRows<B>>,
}

impl<B: Backend> FeatureBatch<B> {
    /// Pads scenarios into one batch. Items keep their per-scenario order, so the `i`-th
    /// choice question / noul / score in the batch is the `i`-th in scenario order.
    pub fn new(scenarios: &[&ScenarioFeatures], device: &B::Device) -> Self {
        assert!(!scenarios.is_empty(), "empty batch");
        let d = scenarios[0].d_model;
        let b = scenarios.len();
        let l_max = scenarios.iter().map(|s| s.ctx_len).max().unwrap_or(0).max(1);
        let k_max = scenarios.iter().map(|s| s.num_items()).max().unwrap_or(0).max(1);

        let mut ctx = vec![0.0f32; b * l_max * d];
        let mut pad = vec![true; b * l_max];
        let mut pool = vec![0.0f32; b * l_max];
        let mut queries = vec![0.0f32; b * k_max * d];
        let mut types = vec![0i64; b * k_max];

        let mut choice_groups: Vec<Vec<i64>> = Vec::new();
        let mut noul_rows = Vec::new();
        let mut score_rows = Vec::new();

        for (bi, s) in scenarios.iter().enumerate() {
            assert_eq!(s.d_model, d, "mixed d_model in batch");
            ctx[bi * l_max * d..(bi * l_max + s.ctx_len) * d].copy_from_slice(&s.ctx);
            for t in 0..s.ctx_len {
                pad[bi * l_max + t] = false;
                pool[bi * l_max + t] = 1.0 / s.ctx_len as f32;
            }
            queries[bi * k_max * d..(bi * k_max + s.num_items()) * d].copy_from_slice(&s.items);

            let first_group = choice_groups.len();
            for (ki, kind) in s.kinds.iter().enumerate() {
                let row = (bi * k_max + ki) as i64;
                types[bi * k_max + ki] = kind.type_id() as i64;
                match *kind {
                    ItemKind::Choice { question, .. } => {
                        let g = first_group + question;
                        if choice_groups.len() <= g {
                            choice_groups.resize(g + 1, Vec::new());
                        }
                        choice_groups[g].push(row);
                    }
                    ItemKind::Noul { .. } => noul_rows.push(row),
                    ItemKind::Score { .. } => score_rows.push(row),
                }
            }
        }

        let item_rows = |rows: Vec<i64>| {
            (!rows.is_empty()).then(|| ItemRows {
                count: rows.len(),
                rows: Tensor::from_data(TensorData::new(rows.clone(), [rows.len()]), device),
            })
        };
        let choice = (!choice_groups.is_empty()).then(|| {
            let q = choice_groups.len();
            let c_max = choice_groups.iter().map(Vec::len).max().unwrap();
            let mut rows = vec![0i64; q * c_max];
            let mut mask = vec![false; q * c_max];
            for (qi, group) in choice_groups.iter().enumerate() {
                for (ci, &row) in group.iter().enumerate() {
                    rows[qi * c_max + ci] = row;
                    mask[qi * c_max + ci] = true;
                }
            }
            ChoiceRows {
                rows: Tensor::from_data(TensorData::new(rows, [q * c_max]), device),
                mask: Tensor::from_data(TensorData::new(mask, [q, c_max]), device),
                questions: q,
                max_candidates: c_max,
                counts: choice_groups.iter().map(Vec::len).collect(),
            }
        });

        Self {
            ctx: Tensor::from_data(TensorData::new(ctx, [b, l_max, d]), device),
            ctx_pad: Tensor::from_data(TensorData::new(pad, [b, l_max]), device),
            ctx_pool: Tensor::from_data(TensorData::new(pool, [b, l_max]), device),
            queries: Tensor::from_data(TensorData::new(queries, [b, k_max, d]), device),
            types: Tensor::from_data(TensorData::new(types, [b, k_max]), device),
            max_items: k_max,
            choice,
            noul: item_rows(noul_rows),
            score: item_rows(score_rows),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.ctx.dims()[0]
    }
}

// =====================================================================
// Decision model
// =====================================================================

#[derive(Config, Debug)]
pub struct DecisionModelConfig {
    pub reader: ItemReaderConfig,
    pub heads: UnifiedHeadsConfig,
}

impl DecisionModelConfig {
    /// Default reader and heads for an encoder of width `d_in`.
    pub fn for_encoder(d_in: usize) -> Self {
        let reader = ItemReaderConfig::new(d_in);
        let heads = UnifiedHeadsConfig::new(reader.d_reader);
        Self { reader, heads }
    }

    /// Keeps the heads' width tied to the reader's output width.
    pub fn with_reader(reader: ItemReaderConfig, heads: UnifiedHeadsConfig) -> Self {
        let heads = UnifiedHeadsConfig { d_model: reader.d_reader, ..heads };
        Self { reader, heads }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> DecisionModel<B> {
        assert_eq!(self.heads.d_model, self.reader.d_reader, "heads must read the reader width");
        DecisionModel {
            reader: self.reader.init(device),
            heads: self.heads.init(device),
            loss_weights: JointLossConfig::frozen_backbone(),
        }
    }
}

#[derive(Module, Debug)]
pub struct DecisionModel<B: Backend> {
    pub reader: ItemReader<B>,
    pub heads: UnifiedHeads<B>,
    /// Multi-task loss weights used by the train/valid steps; not persisted.
    #[module(skip)]
    pub loss_weights: JointLossConfig,
}

/// Calibrated logits for every item in a batch.
pub struct DecisionOutputs<B: Backend> {
    /// `[B, knn_dim]`, L2-normalized.
    pub embedding: Tensor<B, 2>,
    /// `[Q, C_max]` temperature-scaled logits (mask with [`ChoiceRows::mask`]).
    pub choice_logits: Option<Tensor<B, 2>>,
    /// `[N_noul]` temperature-scaled logits.
    pub noul_logits: Option<Tensor<B, 1>>,
    /// `[N_score, bins - 1]` temperature-scaled cumulative logits.
    pub score_logits: Option<Tensor<B, 2>>,
}

/// Typed decisions for one scenario.
pub struct ScenarioDecisions<B: Backend> {
    pub choices: Vec<ChoiceVerdict>,
    pub nouls: Vec<NoulVerdict>,
    pub scores: Vec<ScoreVerdict>,
    /// `[1, knn_dim]`.
    pub embedding: Tensor<B, 2>,
}

impl<B: Backend> DecisionModel<B> {
    pub fn with_loss_weights(mut self, loss_weights: JointLossConfig) -> Self {
        self.loss_weights = loss_weights;
        self
    }

    pub fn forward(&self, batch: &FeatureBatch<B>) -> DecisionOutputs<B> {
        let read = self.reader.forward(
            batch.queries.clone(),
            batch.types.clone(),
            batch.ctx.clone(),
            batch.ctx_pad.clone(),
        );
        let [b, k, r] = read.dims();
        let flat = read.reshape([b * k, r]);
        let embedding = self
            .heads
            .extract_knn_embedding(self.reader.context_embedding(batch.ctx.clone(), batch.ctx_pool.clone()));

        let choice_logits = batch.choice.as_ref().map(|c| {
            let states = flat.clone().select(0, c.rows.clone());
            self.heads
                .forward_choice_logits(states)
                .reshape([c.questions, c.max_candidates])
        });
        let noul_logits = batch
            .noul
            .as_ref()
            .map(|n| self.heads.forward_noul_logits(flat.clone().select(0, n.rows.clone())));
        let score_logits = batch
            .score
            .as_ref()
            .map(|s| self.heads.forward_score_logits(flat.clone().select(0, s.rows.clone())));

        DecisionOutputs {
            embedding,
            choice_logits,
            noul_logits,
            score_logits,
        }
    }

    /// Runs one scenario and converts the logits into typed verdicts.
    pub fn decide(&self, features: &ScenarioFeatures, device: &B::Device) -> Result<ScenarioDecisions<B>, JevError> {
        let batch = FeatureBatch::new(&[features], device);
        let read = self.reader.forward(batch.queries.clone(), batch.types.clone(), batch.ctx.clone(), batch.ctx_pad.clone());
        let [_, k, r] = read.dims();
        let flat = read.reshape([k, r]);
        let embedding = self
            .heads
            .extract_knn_embedding(self.reader.context_embedding(batch.ctx.clone(), batch.ctx_pool.clone()));

        let mut choices = Vec::new();
        if let Some(ref c) = batch.choice {
            let states = flat.clone().select(0, c.rows.clone()).reshape([c.questions, c.max_candidates, r]);
            for (qi, &count) in c.counts.iter().enumerate() {
                let q = states.clone().slice([qi..qi + 1, 0..count]).reshape([count, r]);
                choices.push(self.heads.evaluate_choice_verdict(q)?);
            }
        }
        let nouls = match batch.noul {
            Some(ref n) => self.heads.evaluate_noul_verdicts_default(flat.clone().select(0, n.rows.clone()))?,
            None => Vec::new(),
        };
        let scores = match batch.score {
            Some(ref s) => self.heads.evaluate_score_verdicts(flat.select(0, s.rows.clone()))?,
            None => Vec::new(),
        };
        Ok(ScenarioDecisions {
            choices,
            nouls,
            scores,
            embedding,
        })
    }
}
