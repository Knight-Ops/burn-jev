#!/usr/bin/env python3
"""Merge, de-bias, validate, dedup and write the reflex-head train/val JSONL files.

Inputs:
  data/seed/train_seed.jsonl, data/seed/val_seed.jsonl   (original hand-written records)
  data/gen/<cluster>/train_part_*.jsonl                  (generated train shards)
  data/gen/<cluster>/val_part_*.jsonl                    (generated val shards)
Outputs:
  data/reflex_training_data.jsonl, data/reflex_validation_data.jsonl

Stdlib only; deterministic for a given --seed.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
KEYS = {"id", "domain", "context", "is_benign", "choice_questions", "noul_queries", "score_rubrics"}
JACCARD_THRESHOLD = 0.8
SHINGLE_N = 5


# ---------------------------------------------------------------- loading


def load_jsonl(path):
    out = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append((f"{os.path.relpath(path, ROOT)}:{ln}", json.loads(line)))
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{ln}: JSON error: {e}")
    return out


def load_split(split):
    seed = load_jsonl(os.path.join(DATA, "seed", f"{split}_seed.jsonl"))
    gen = []
    for path in sorted(glob.glob(os.path.join(DATA, "gen", "*", f"{split}_part_*.jsonl"))):
        gen.extend(load_jsonl(path))
    return [(loc, r, True) for loc, r in seed] + [(loc, r, False) for loc, r in gen]


# ---------------------------------------------------------------- validation


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(r):
    """Mirrors JevScenarioRecord::validate plus score <= 5 and non-empty text fields."""
    errs = []
    if set(r) - KEYS:
        errs.append(f"unknown keys {sorted(set(r) - KEYS)}")
    for k in ("id", "context"):
        if not isinstance(r.get(k), str) or not r[k].strip():
            errs.append(f"empty {k}")
    if "domain" in r and r["domain"] is not None and not isinstance(r["domain"], str):
        errs.append("domain must be a string")
    if not isinstance(r.get("is_benign", False), bool):
        errs.append("is_benign must be bool")
    for i, q in enumerate(r.get("choice_questions", [])):
        c = q.get("candidates", [])
        if not isinstance(q.get("prompt"), str) or not q["prompt"].strip():
            errs.append(f"choice {i}: empty prompt")
        if len(c) < 2 or not all(isinstance(x, str) and x.strip() for x in c):
            errs.append(f"choice {i}: needs >= 2 non-empty candidates")
        if len(set(c)) != len(c):
            errs.append(f"choice {i}: duplicate candidates")
        t = q.get("target")
        if not isinstance(t, int) or isinstance(t, bool) or not 0 <= t < len(c):
            errs.append(f"choice {i}: target {t!r} out of range")
    for i, n in enumerate(r.get("noul_queries", [])):
        if not isinstance(n.get("assertion"), str) or not n["assertion"].strip():
            errs.append(f"noul {i}: empty assertion")
        if not _num(n.get("target")) or not 0.0 <= n["target"] <= 1.0:
            errs.append(f"noul {i}: target {n.get('target')!r} not in [0,1]")
    for i, s in enumerate(r.get("score_rubrics", [])):
        if not isinstance(s.get("prompt"), str) or not s["prompt"].strip():
            errs.append(f"score {i}: empty prompt")
        if not _num(s.get("target")) or not 1.0 <= s["target"] <= 5.0:
            errs.append(f"score {i}: target {s.get('target')!r} not in [1,5]")
    if not (r.get("choice_questions") or r.get("noul_queries") or r.get("score_rubrics")):
        errs.append("no questions at all")
    return errs


# ---------------------------------------------------------------- dedup


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def shingles(text):
    words = normalize(text).split()
    if len(words) < SHINGLE_N:
        return {" ".join(words)}
    return {" ".join(words[i : i + SHINGLE_N]) for i in range(len(words) - SHINGLE_N + 1)}


def dedup(records):
    """records: list of (split, loc, rec, is_seed), in priority order. Returns kept list + drop log."""
    kept, dropped = [], []
    exact = {}
    index = defaultdict(list)  # shingle -> kept indices
    kept_sh = []
    for split, loc, r, is_seed in records:
        norm = normalize(r["context"])
        h = hashlib.sha256(norm.encode()).hexdigest()
        if h in exact and not is_seed:
            dropped.append((loc, r["id"], f"exact duplicate of {exact[h]}"))
            continue
        sh = shingles(r["context"])
        overlap = Counter()
        for s in sh:
            for j in index.get(s, ()):
                overlap[j] += 1
        dup_of = None
        for j, inter in overlap.most_common(5):
            union = len(sh) + len(kept_sh[j]) - inter
            if union and inter / union > JACCARD_THRESHOLD:
                dup_of = j
                break
        if dup_of is not None and not is_seed:
            other = kept[dup_of]
            dropped.append((loc, r["id"], f"near-duplicate (>{JACCARD_THRESHOLD}) of {other[2]['id']} [{other[0]}]"))
            continue
        idx = len(kept)
        kept.append((split, loc, r, is_seed))
        kept_sh.append(sh)
        exact.setdefault(h, r["id"])
        for s in sh:
            index[s].append(idx)
    return kept, dropped


# ---------------------------------------------------------------- de-biasing


def rebalance_choice_targets(records, rng):
    """Permute candidates so the correct answer lands uniformly across positions per candidate count.

    Positions are drawn from a shuffled round-robin pool per candidate count k, giving an
    (almost) exactly uniform target histogram instead of a merely random one.
    """
    pools = defaultdict(list)
    for r in records:
        for q in r.get("choice_questions", []):
            k = len(q["candidates"])
            if not pools[k]:
                pool = list(range(k))
                rng.shuffle(pool)
                pools[k] = pool
            new_pos = pools[k].pop()
            correct = q["candidates"][q["target"]]
            others = [c for i, c in enumerate(q["candidates"]) if i != q["target"]]
            rng.shuffle(others)
            others.insert(new_pos, correct)
            q["candidates"] = others
            q["target"] = new_pos


# ---------------------------------------------------------------- report


def histogram(values, edges):
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[-1]):
                counts[i] += 1
                break
    return counts


def report(name, records):
    n = len(records)
    print(f"\n=== {name}: {n} records ===")
    if not n:
        return
    doms = Counter(r.get("domain") or "<none>" for r in records)
    print("domains:", ", ".join(f"{d}={c}" for d, c in doms.most_common()))
    benign = sum(r["is_benign"] for r in records)
    print(f"benign ratio: {benign / n:.3f} ({benign}/{n})")
    by_dom = defaultdict(list)
    for r in records:
        by_dom[r.get("domain") or "<none>"].append(r["is_benign"])
    print("benign by domain:", ", ".join(f"{d}={sum(v) / len(v):.2f}" for d, v in sorted(by_dom.items())))

    tgt = defaultdict(Counter)
    len_ok, len_bad, longest_is_correct, nq = [], [], 0, 0
    for r in records:
        for q in r.get("choice_questions", []):
            k = len(q["candidates"])
            tgt[k][q["target"]] += 1
            lens = [len(c) for c in q["candidates"]]
            len_ok.append(lens[q["target"]])
            len_bad.extend(l for i, l in enumerate(lens) if i != q["target"])
            longest_is_correct += lens[q["target"]] == max(lens)
            nq += 1
    for k in sorted(tgt):
        print(f"choice target hist (k={k}):", [tgt[k][i] for i in range(k)])
    if nq:
        print(
            f"candidate length: correct avg {sum(len_ok) / len(len_ok):.1f}, "
            f"incorrect avg {sum(len_bad) / len(len_bad):.1f}, correct-is-longest {longest_is_correct / nq:.2f}"
        )
        exp = sum(1 / len(q["candidates"]) for r in records for q in r.get("choice_questions", [])) / nq
        print(f"  (chance that correct is longest if length were uninformative ~{exp:.2f})")

    nouls = [n_["target"] for r in records for n_ in r.get("noul_queries", [])]
    if nouls:
        graded = sum(0 < v < 1 for v in nouls)
        print(
            f"noul: n={len(nouls)} mean={sum(nouls) / len(nouls):.3f} "
            f"ones={sum(v == 1 for v in nouls)} zeros={sum(v == 0 for v in nouls)} graded={graded}"
        )
    scores = [s["target"] for r in records for s in r.get("score_rubrics", [])]
    if scores:
        edges = [1, 2, 3, 4, 5.0001]
        print(f"score: n={len(scores)} mean={sum(scores) / len(scores):.2f} hist[1-2,2-3,3-4,4-5]={histogram(scores, edges)}")
    ctx = [len(r["context"]) for r in records]
    print(f"context chars: min={min(ctx)} avg={sum(ctx) / n:.0f} max={max(ctx)}")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--dry-run", action="store_true", help="validate and report without writing outputs")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    train_in, val_in = load_split("train"), load_split("val")

    errors = []
    for loc, r, _ in train_in + val_in:
        for e in validate(r):
            errors.append(f"{loc} ({r.get('id')}): {e}")
    ids = Counter(r.get("id") for _, r, _ in train_in + val_in)
    errors += [f"duplicate id {i!r} ({c}x)" for i, c in ids.items() if c > 1]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(f"{len(errors)} validation errors; fix shards and rerun")

    # Dedup priority: seeds, then val (keep held-out split intact), then train.
    ordered = (
        [("val", l, r, s) for l, r, s in val_in if s]
        + [("train", l, r, s) for l, r, s in train_in if s]
        + [("val", l, r, s) for l, r, s in val_in if not s]
        + [("train", l, r, s) for l, r, s in train_in if not s]
    )
    kept, dropped = dedup(ordered)
    print(f"loaded train={len(train_in)} val={len(val_in)}; dropped {len(dropped)} duplicates")
    for loc, rid, why in dropped:
        print(f"  drop {rid} ({loc}): {why}")

    train = [r for split, _, r, _ in kept if split == "train"]
    val = [r for split, _, r, _ in kept if split == "val"]
    train.sort(key=lambda r: r["id"])
    val.sort(key=lambda r: r["id"])
    rebalance_choice_targets(train, rng)
    rebalance_choice_targets(val, rng)
    rng.shuffle(train)
    rng.shuffle(val)

    report("train", train)
    report("val", val)

    if args.dry_run:
        return
    for name, recs in (("reflex_training_data.jsonl", train), ("reflex_validation_data.jsonl", val)):
        with open(os.path.join(DATA, name), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"wrote data/{name} ({len(recs)} records)")


if __name__ == "__main__":
    main()
