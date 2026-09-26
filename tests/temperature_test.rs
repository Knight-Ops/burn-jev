//! Post-hoc temperature fitting recovers a known over-confidence factor.

use burn_flex::{Flex, FlexDevice};
use burn_jev::training::{fit_temperature, HeadLogits};
use burn_jev::Head;

type B = Flex<f32, i32>;

struct Lcg(u64);
impl Lcg {
    /// Uniform in [0, 1).
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

const SCALE: f32 = 5.0;

/// Calibrated logits (labels drawn from their own probabilities), stored ×`SCALE`.
fn overconfident_logits() -> HeadLogits {
    let mut rng = Lcg(7);
    let mut out = HeadLogits {
        num_rubric_bins: 5,
        ..HeadLogits::default()
    };
    for _ in 0..4000 {
        let z = rng.next() * 8.0 - 4.0;
        out.noul.push(z * SCALE);
        out.noul_targets.push((rng.next() < 1.0 / (1.0 + (-z).exp())) as u8 as f32);

        let k = 2 + (rng.next() * 3.0) as usize;
        let z: Vec<f32> = (0..k).map(|_| rng.next() * 6.0 - 3.0).collect();
        let sum: f32 = z.iter().map(|v| v.exp()).sum();
        let (mut u, mut target) = (rng.next() * sum, k - 1);
        for (i, v) in z.iter().enumerate() {
            u -= v.exp();
            if u < 0.0 {
                target = i;
                break;
            }
        }
        out.choice.push(z.iter().map(|v| v * SCALE).collect());
        out.choice_targets.push(target);
    }
    out
}

#[test]
fn recovers_the_logit_scale() {
    let logits = overconfident_logits();
    for head in [Head::Noul, Head::Choice] {
        let t = fit_temperature::<B>(&logits, head, &FlexDevice);
        assert!((t - SCALE).abs() < 0.5, "{head:?}: fit T = {t}, expected ≈ {SCALE}");
    }
}

#[test]
fn heads_without_items_keep_unit_temperature() {
    let logits = HeadLogits {
        num_rubric_bins: 5,
        ..HeadLogits::default()
    };
    for head in Head::ALL {
        assert_eq!(fit_temperature::<B>(&logits, head, &FlexDevice), 1.0);
    }
}
