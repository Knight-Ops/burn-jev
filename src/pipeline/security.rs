use std::sync::Arc;

use burn::tensor::backend::Backend;

use crate::anomaly::{AnomalyVerdict, TenantRegistry};
use crate::dataset::ReflexRequest;
use crate::pipeline::engine::{ReflexEngine, ReflexError, ReflexVerdict};

/// Default threat threshold on calibrated Noul probability (P_threat > 0.85)
pub const DEFAULT_THREAT_THRESHOLD: f32 = 0.85;

/// Specific reason triggering Tier 2 escalation handoff.
#[derive(Clone, Debug, PartialEq)]
pub enum EscalationReason {
    /// Non-parametric k-NN distance exceeded tenant outlier threshold (d_k > tau)
    OutlierAnomaly { distance: f32, threshold: f32 },
    /// Calibrated Noul threat assertion exceeded threshold (P_threat > 0.85)
    HighThreatVerdict { threat_probability: f32, threshold: f32 },
    /// Both an outlier embedding and a high threat probability were triggered simultaneously
    DualAnomalyThreat { distance: f32, threat_probability: f32 },
}

/// Routing decision emitted by the Tier 1 Reflex Security Guardrail.
#[derive(Clone, Debug, PartialEq)]
pub enum Tier1Routing {
    /// Query passed all reflex gates (~95% of traffic) and proceeds directly to target LLM
    FastPathPass {
        verdict: ReflexVerdict,
        anomaly_verdict: Option<AnomalyVerdict>,
    },
    /// Query flagged as anomalous or high-risk (~5% of traffic) and diverted to Tier 2 mechanistic audit
    EscalateToTier2 {
        reason: EscalationReason,
        verdict: ReflexVerdict,
        anomaly_verdict: Option<AnomalyVerdict>,
        /// The original request, handed to the Tier 2 audit.
        request: ReflexRequest,
    },
}

impl Tier1Routing {
    /// Returns true if the query was escalated to Tier 2.
    pub fn is_escalated(&self) -> bool {
        matches!(self, Self::EscalateToTier2 { .. })
    }

    /// Access the underlying generic ReflexVerdict regardless of routing decision.
    pub fn reflex_verdict(&self) -> &ReflexVerdict {
        match self {
            Self::FastPathPass { verdict, .. } => verdict,
            Self::EscalateToTier2 { verdict, .. } => verdict,
        }
    }
}

/// Specialized security guardrail and router built on top of `ReflexEngine`.
///
/// Combines the generic neural reflex outputs with multi-tenant anomaly gating
/// and threat thresholding to implement the asymmetric Tier 1 / Tier 2 defense.
pub struct ReflexSecurityRouter<B: Backend> {
    pub engine: ReflexEngine<B>,
    pub registry: Arc<TenantRegistry>,
    pub threat_threshold: f32,
    pub auto_record_benign: bool,
}

impl<B: Backend> ReflexSecurityRouter<B> {
    /// Construct a new ReflexSecurityRouter.
    pub fn new(engine: ReflexEngine<B>, registry: Arc<TenantRegistry>) -> Self {
        Self {
            engine,
            registry,
            threat_threshold: DEFAULT_THREAT_THRESHOLD,
            auto_record_benign: true,
        }
    }

    /// Configure custom threat probability threshold.
    pub fn with_threat_threshold(mut self, threshold: f32) -> Self {
        self.threat_threshold = threshold.clamp(0.01, 0.99);
        self
    }

    /// Configure whether clean/benign queries should automatically be recorded into tenant sliding window.
    pub fn with_auto_record_benign(mut self, auto_record: bool) -> Self {
        self.auto_record_benign = auto_record;
        self
    }

    /// Evaluates the request and executes the complete Tier 1 reflex gating logic.
    pub fn route(
        &self,
        tenant_id: &str,
        request: &ReflexRequest,
        device: &B::Device,
    ) -> Result<Tier1Routing, ReflexError> {
        // Step 1: Run the fast reflex pass through the encoder and decision model
        let verdict = self.engine.evaluate(request, device)?;

        // Step 2: Query tenant vector index for outlier anomaly distance
        let anomaly_verdict = self.registry.evaluate(tenant_id, &verdict.embedding)?;

        // Step 3: Evaluate calibrated threat probability from Noul assertion (if present)
        let threat_prob = verdict.noul.as_ref().map(|n| n.probability).unwrap_or(0.0);
        let is_high_threat = threat_prob > self.threat_threshold;
        let is_anomaly = anomaly_verdict.is_anomaly;

        // Step 4: Gating decision
        if is_anomaly && is_high_threat {
            Ok(Tier1Routing::EscalateToTier2 {
                reason: EscalationReason::DualAnomalyThreat {
                    distance: anomaly_verdict.effective_distance,
                    threat_probability: threat_prob,
                },
                verdict,
                anomaly_verdict: Some(anomaly_verdict),
                request: request.clone(),
            })
        } else if is_anomaly {
            Ok(Tier1Routing::EscalateToTier2 {
                reason: EscalationReason::OutlierAnomaly {
                    distance: anomaly_verdict.effective_distance,
                    threshold: anomaly_verdict.threshold,
                },
                verdict,
                anomaly_verdict: Some(anomaly_verdict),
                request: request.clone(),
            })
        } else if is_high_threat {
            Ok(Tier1Routing::EscalateToTier2 {
                reason: EscalationReason::HighThreatVerdict {
                    threat_probability: threat_prob,
                    threshold: self.threat_threshold,
                },
                verdict,
                anomaly_verdict: Some(anomaly_verdict),
                request: request.clone(),
            })
        } else {
            // Clean traffic: pass directly to target LLM with zero overhead
            if self.auto_record_benign {
                // Record into tenant's sliding window ring buffer
                let _ = self.registry.record_benign(tenant_id, verdict.embedding.clone());
            }

            Ok(Tier1Routing::FastPathPass {
                verdict,
                anomaly_verdict: Some(anomaly_verdict),
            })
        }
    }
}
