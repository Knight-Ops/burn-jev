use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::anomaly::embedding::AnomalyEmbedding;
use crate::anomaly::traits::{AnomalyError, AnomalyIndex};

/// Default cold-start sample horizon (N = 1000) for weight alpha interpolation
pub const DEFAULT_COLD_START_HORIZON: usize = 1000;

/// Default number of nearest neighbors (k = 10)
pub const DEFAULT_KNN_K: usize = 10;

/// Default anomaly distance threshold (tau = 0.75)
pub const DEFAULT_ANOMALY_THRESHOLD: f32 = 0.75;

/// Reserved namespace for global safe anchor baseline embeddings
pub const GLOBAL_ANCHOR_NAMESPACE: &str = "__global_anchor__";

/// Typed result of an anomaly gating evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct AnomalyVerdict {
    /// True if effective distance exceeds tenant threshold (S(z) > tau)
    pub is_anomaly: bool,
    /// Effective interpolated distance score S(z)
    pub effective_distance: f32,
    /// Tenant anomaly threshold tau
    pub threshold: f32,
    /// Cold-start weighting factor alpha in [0.0, 1.0]
    pub cold_start_alpha: f32,
    /// Mean k-NN distance within the tenant namespace (None if N_samples == 0)
    pub tenant_distance: Option<f32>,
    /// Mean k-NN distance within the global safe anchor namespace
    pub global_distance: f32,
    /// k parameter used for nearest-neighbor aggregation
    pub k: usize,
}

/// Tenant configuration holding detection parameters and threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct TenantConfig {
    pub tenant_id: String,
    pub threshold: f32,
    pub k: usize,
    pub cold_start_horizon: usize,
}

impl TenantConfig {
    pub fn new(tenant_id: impl Into<String>, threshold: f32, k: usize) -> Result<Self, AnomalyError> {
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(AnomalyError::InvalidThreshold { threshold });
        }
        if k == 0 {
            return Err(AnomalyError::InsufficientSamples { needed: 1, found: 0 });
        }

        Ok(Self {
            tenant_id: tenant_id.into(),
            threshold,
            k,
            cold_start_horizon: DEFAULT_COLD_START_HORIZON,
        })
    }

    pub fn with_cold_start_horizon(mut self, horizon: usize) -> Self {
        self.cold_start_horizon = horizon.max(1);
        self
    }
}

/// Calculates the empirical distance threshold tau from a validation set of benign k-NN distances.
///
/// Formula:
/// tau = Percentile({ d_k(v) | v in V_benign }, (1 - FPR) * 100)
pub fn calibrate_threshold_from_distances(
    distances: &[f32],
    fpr: f32,
) -> Result<f32, AnomalyError> {
    if !fpr.is_finite() || fpr <= 0.0 || fpr >= 1.0 {
        return Err(AnomalyError::InvalidFpr { fpr });
    }
    if distances.is_empty() {
        return Err(AnomalyError::InsufficientSamples { needed: 1, found: 0 });
    }

    let mut sorted = distances.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile = (1.0 - fpr) * 100.0;
    let n = sorted.len();

    if n == 1 {
        return Ok(sorted[0]);
    }

    // Linear interpolation between closest ranks
    let rank = (percentile / 100.0) * (n - 1) as f32;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    let weight = rank - low as f32;

    let threshold = sorted[low] * (1.0 - weight) + sorted[high] * weight;
    Ok(threshold)
}

/// Multi-tenant anomaly gating engine and lifecycle coordinator.
///
/// Coordinates multi-tenant isolation, cold-start interpolation, sliding-window
/// ring buffers, and threshold gating over an underlying `AnomalyIndex` backend.
pub struct TenantRegistry {
    index: Arc<dyn AnomalyIndex>,
    tenants: RwLock<HashMap<String, TenantConfig>>,
    global_namespace: String,
    default_k: usize,
    default_threshold: f32,
    default_cold_start_horizon: usize,
}

impl TenantRegistry {
    /// Initialize a new TenantRegistry with an underlying pluggable `AnomalyIndex`.
    pub fn new(index: Arc<dyn AnomalyIndex>) -> Self {
        Self {
            index,
            tenants: RwLock::new(HashMap::new()),
            global_namespace: GLOBAL_ANCHOR_NAMESPACE.to_string(),
            default_k: DEFAULT_KNN_K,
            default_threshold: DEFAULT_ANOMALY_THRESHOLD,
            default_cold_start_horizon: DEFAULT_COLD_START_HORIZON,
        }
    }

    /// Populate the global safe baseline anchors.
    pub fn set_global_anchors(&self, anchors: &[AnomalyEmbedding]) -> Result<(), AnomalyError> {
        self.index.clear(&self.global_namespace)?;
        for anchor in anchors {
            self.index.insert_benign(&self.global_namespace, anchor.clone())?;
        }
        Ok(())
    }

    /// Register a specific tenant configuration.
    pub fn register_tenant(&self, config: TenantConfig) -> Result<(), AnomalyError> {
        let mut tenants = self.tenants.write().unwrap();
        tenants.insert(config.tenant_id.clone(), config);
        Ok(())
    }

    /// Retrieve a tenant configuration, or construct a default configuration if not explicitly registered.
    pub fn get_or_create_config(&self, tenant_id: &str) -> TenantConfig {
        let tenants = self.tenants.read().unwrap();
        if let Some(cfg) = tenants.get(tenant_id) {
            cfg.clone()
        } else {
            TenantConfig {
                tenant_id: tenant_id.to_string(),
                threshold: self.default_threshold,
                k: self.default_k,
                cold_start_horizon: self.default_cold_start_horizon,
            }
        }
    }

    /// Calibrate a tenant's anomaly threshold using held-out benign validation embeddings.
    pub fn calibrate_tenant_threshold(
        &self,
        tenant_id: &str,
        benign_val_samples: &[AnomalyEmbedding],
        fpr: f32,
        k: usize,
    ) -> Result<f32, AnomalyError> {
        if benign_val_samples.is_empty() {
            return Err(AnomalyError::InsufficientSamples { needed: 1, found: 0 });
        }

        let mut distances = Vec::with_capacity(benign_val_samples.len());
        for sample in benign_val_samples {
            let knn_distances = self.index.search_knn(tenant_id, sample, k)?;
            if knn_distances.is_empty() {
                // If tenant is empty during calibration, query against global baseline
                let global_distances = self.index.search_knn(&self.global_namespace, sample, k)?;
                if global_distances.is_empty() {
                    return Err(AnomalyError::InsufficientSamples { needed: 1, found: 0 });
                }
                let mean_d = global_distances.iter().sum::<f32>() / (global_distances.len() as f32);
                distances.push(mean_d);
            } else {
                let mean_d = knn_distances.iter().sum::<f32>() / (knn_distances.len() as f32);
                distances.push(mean_d);
            }
        }

        let calibrated_tau = calibrate_threshold_from_distances(&distances, fpr)?;

        // Update tenant config with calibrated threshold
        let mut tenants = self.tenants.write().unwrap();
        let config = tenants
            .entry(tenant_id.to_string())
            .or_insert_with(|| TenantConfig {
                tenant_id: tenant_id.to_string(),
                threshold: calibrated_tau,
                k,
                cold_start_horizon: self.default_cold_start_horizon,
            });
        config.threshold = calibrated_tau;
        config.k = k;

        Ok(calibrated_tau)
    }

    /// Evaluate a query embedding against the multi-tenant anomaly gating engine.
    ///
    /// Computes:
    /// 1. alpha = min(1.0, N_samples / cold_start_horizon)
    /// 2. S(z) = alpha * D_tenant(z) + (1 - alpha) * D_global(z)
    /// 3. is_anomaly = S(z) > tau_tenant
    pub fn evaluate(
        &self,
        tenant_id: &str,
        query: &AnomalyEmbedding,
    ) -> Result<AnomalyVerdict, AnomalyError> {
        let config = self.get_or_create_config(tenant_id);
        let k = config.k;

        // Query global baseline
        let global_knn = match self.index.search_knn(&self.global_namespace, query, k) {
            Ok(d) if !d.is_empty() => d,
            _ => {
                return Err(AnomalyError::InsufficientSamples {
                    needed: 1,
                    found: self.index.sample_count(&self.global_namespace),
                });
            }
        };
        let global_distance = global_knn.iter().sum::<f32>() / (global_knn.len() as f32);

        // Query tenant index
        let tenant_count = self.index.sample_count(tenant_id);
        let (alpha, tenant_distance, effective_distance) = if tenant_count == 0 {
            (0.0f32, None, global_distance)
        } else {
            let tenant_knn = self.index.search_knn(tenant_id, query, k)?;
            let mean_tenant_d = if tenant_knn.is_empty() {
                global_distance
            } else {
                tenant_knn.iter().sum::<f32>() / (tenant_knn.len() as f32)
            };

            let alpha = (tenant_count as f32 / config.cold_start_horizon as f32).min(1.0);
            let s_z = alpha * mean_tenant_d + (1.0 - alpha) * global_distance;
            (alpha, Some(mean_tenant_d), s_z)
        };

        let is_anomaly = effective_distance > config.threshold;

        Ok(AnomalyVerdict {
            is_anomaly,
            effective_distance,
            threshold: config.threshold,
            cold_start_alpha: alpha,
            tenant_distance,
            global_distance,
            k,
        })
    }

    /// Record a verified benign query vector into the tenant's sliding window index.
    pub fn record_benign(
        &self,
        tenant_id: &str,
        vector: AnomalyEmbedding,
    ) -> Result<(), AnomalyError> {
        self.index.insert_benign(tenant_id, vector)
    }

    /// Number of samples currently in the tenant's buffer.
    pub fn tenant_sample_count(&self, tenant_id: &str) -> usize {
        self.index.sample_count(tenant_id)
    }

    /// Number of global anchor baseline vectors.
    pub fn global_sample_count(&self) -> usize {
        self.index.sample_count(&self.global_namespace)
    }
}
