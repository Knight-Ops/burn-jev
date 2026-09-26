use burn_jev::{CachedScenario, FeatureCache, ItemKind, MultiQuestionTargets, ScenarioFeatures};

fn scenario(id: &str, ctx_len: usize, kinds: Vec<ItemKind>, d: usize) -> CachedScenario {
    let n = kinds.len();
    CachedScenario {
        id: id.into(),
        is_benign: id.ends_with('b'),
        targets: MultiQuestionTargets::new().with_choice_target(1).with_noul_target(0.25).with_score_target(3.5),
        features: ScenarioFeatures {
            d_model: d,
            ctx: (0..ctx_len * d).map(|i| (i as f32 * 0.37).sin() * 3.0).collect(),
            ctx_len,
            items: (0..n * d).map(|i| (i as f32 * 0.11).cos()).collect(),
            kinds,
        },
    }
}

#[test]
fn test_feature_cache_roundtrip() {
    let d = 8;
    let dir = std::env::temp_dir().join(format!("burn_jev_cache_test_{}", std::process::id()));
    let (st_path, meta_path) = FeatureCache::cache_paths(&dir, std::path::Path::new("train.jsonl"), "enc/abc:1");
    let scenarios = vec![
        scenario(
            "s1b",
            5,
            vec![
                ItemKind::Choice { question: 0, candidate: 0 },
                ItemKind::Choice { question: 0, candidate: 1 },
                ItemKind::Noul { index: 0 },
                ItemKind::Score { index: 0 },
            ],
            d,
        ),
        scenario("s2a", 3, vec![], d),
    ];

    FeatureCache::save_to_disk(&scenarios, &st_path, &meta_path, "hash".into(), "enc/abc:1".into(), d).unwrap();
    assert!(FeatureCache::is_cache_valid(&st_path, &meta_path, "hash", "enc/abc:1", d));
    assert!(!FeatureCache::is_cache_valid(&st_path, &meta_path, "other", "enc/abc:1", d));
    assert!(!FeatureCache::is_cache_valid(&st_path, &meta_path, "hash", "enc/xyz", d));
    assert!(!FeatureCache::is_cache_valid(&st_path, &meta_path, "hash", "enc/abc:1", d + 1));

    let loaded = FeatureCache::load_from_disk(&st_path, &meta_path).unwrap();
    assert_eq!(loaded.len(), 2);
    for (a, b) in scenarios.iter().zip(&loaded) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.is_benign, b.is_benign);
        assert_eq!(a.targets, b.targets);
        assert_eq!(a.features.kinds, b.features.kinds);
        assert_eq!(a.features.ctx_len, b.features.ctx_len);
        // Stored as f16.
        for (x, y) in a.features.ctx.iter().chain(&a.features.items).zip(b.features.ctx.iter().chain(&b.features.items)) {
            assert!((x - y).abs() <= 2e-3 * x.abs().max(1.0), "{x} vs {y}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
