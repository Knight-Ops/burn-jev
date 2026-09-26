#!/usr/bin/env python3
"""Quick validator for a single generated reflex-data part file (mirrors JevScenarioRecord::validate)."""
import json
import sys


def check_record(r):
    errs = []
    for k in ("id", "context"):
        if not isinstance(r.get(k), str) or not r[k].strip():
            errs.append(f"missing/empty {k}")
    if not isinstance(r.get("is_benign"), bool):
        errs.append("is_benign must be bool")
    if "domain" in r and not isinstance(r["domain"], str):
        errs.append("domain must be string")
    extra = set(r) - {"id", "domain", "context", "is_benign", "choice_questions", "noul_queries", "score_rubrics"}
    if extra:
        errs.append(f"unknown keys {sorted(extra)}")
    cq = r.get("choice_questions", [])
    nq = r.get("noul_queries", [])
    sr = r.get("score_rubrics", [])
    if not (1 <= len(cq) <= 3):
        errs.append(f"{len(cq)} choice questions (want 1-3)")
    if not (2 <= len(nq) <= 4):
        errs.append(f"{len(nq)} noul queries (want 2-4)")
    if not (1 <= len(sr) <= 3):
        errs.append(f"{len(sr)} score rubrics (want 1-3)")
    for i, q in enumerate(cq):
        c = q.get("candidates", [])
        t = q.get("target")
        if not isinstance(q.get("prompt"), str) or not q["prompt"].strip():
            errs.append(f"choice {i} empty prompt")
        if not (2 <= len(c) <= 50) or not all(isinstance(x, str) and x.strip() for x in c):
            errs.append(f"choice {i} needs 2-50 non-empty candidates")
        if len(set(c)) != len(c):
            errs.append(f"choice {i} duplicate candidates")
        if not isinstance(t, int) or isinstance(t, bool) or not (0 <= t < len(c)):
            errs.append(f"choice {i} bad target {t!r}")
    for i, n in enumerate(nq):
        t = n.get("target")
        if not isinstance(n.get("assertion"), str) or not n["assertion"].strip():
            errs.append(f"noul {i} empty assertion")
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not (0.0 <= t <= 1.0):
            errs.append(f"noul {i} bad target {t!r}")
    for i, s in enumerate(sr):
        t = s.get("target")
        if not isinstance(s.get("prompt"), str) or not s["prompt"].strip():
            errs.append(f"score {i} empty prompt")
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not (1.0 <= t <= 5.0):
            errs.append(f"score {i} bad target {t!r}")
    return errs


def main():
    bad = 0
    total = 0
    ids = set()
    for path in sys.argv[1:]:
        with open(path) as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    print(f"{path}:{ln}: blank line")
                    bad += 1
                    continue
                total += 1
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"{path}:{ln}: JSON error: {e}")
                    bad += 1
                    continue
                errs = check_record(r)
                if r.get("id") in ids:
                    errs.append("duplicate id")
                ids.add(r.get("id"))
                for e in errs:
                    print(f"{path}:{ln} ({r.get('id')}): {e}")
                bad += bool(errs)
    print(f"{total} records, {bad} with errors")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
