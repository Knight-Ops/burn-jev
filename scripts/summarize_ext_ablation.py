#!/usr/bin/env python3
"""Summarize scripts/run_ext_ablation.sh logs: mean±std over seeds per config, plus per-domain.

  scripts/summarize_ext_ablation.py [--logs data/.runs/ext-ablation] [--out docs/ext_data_ablation.md]

Stdlib only. Parses the "=== Summary" block and the per-scenario lines of `reflex-train` output.
The validation set is merged (in-domain + external held-out sets), so the headline summary mixes
sources; the per-source tables (from per-scenario lines) separate in-domain from each ext set.
"""
import argparse
import glob
import math
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ORDER = ["baseline", "procedural", "llama_security", "both", "nemotron"]

RE_SCEN = re.compile(r"^--- (\S+) \[(.+)\] ---$")
RE_CHOICE = re.compile(r"\[Choice Q\d+\].*-> (PASS|FAIL)")
RE_NOUL = re.compile(r"\[Noul Q\d+\].*-> (PASS|FAIL)")
RE_SCORE = re.compile(r"\[Score Q\d+\].*\(diff ([\d.]+)\)")
# Values may be "n/a" (e.g. a held-out set without score rubrics) -> None.
RE_SUMMARY = {
    "choice_acc": re.compile(r"Choice accuracy:\s+([\d.]+|n/a)"),
    "noul_acc": re.compile(r"Noul accuracy:\s+([\d.]+|n/a)"),
    "score_rmse": re.compile(r"Score RMSE:\s+([\d.]+|n/a)"),
    "choice_ece": re.compile(r"Calibration ECE: choice ([\d.]+|n/a)"),
    "noul_ece": re.compile(r"noul ([\d.]+%|n/a) \(10 bins\)"),
}
RE_BEST = re.compile(r"best epoch (\d+)")
METRICS = [
    ("choice_acc", "Choice acc %", True),
    ("noul_acc", "Noul acc %", True),
    ("score_rmse", "Score RMSE", False),
    ("choice_ece", "Choice ECE %", False),
    ("noul_ece", "Noul ECE %", False),
]


def source_of(domain):
    """in_domain, or the external set a domain came from (ext_procedural, ext_nemotron_ipi, ...)."""
    if not domain.startswith("ext_"):
        return "in_domain"
    for prefix in ("ext_procedural", "ext_nemotron_ipi"):
        if domain.startswith(prefix):
            return prefix
    return domain


def item_metrics(v):
    return {
        "choice_acc": 100 * sum(v["choice"]) / len(v["choice"]) if v["choice"] else None,
        "noul_acc": 100 * sum(v["noul"]) / len(v["noul"]) if v["noul"] else None,
        "score_rmse": math.sqrt(sum(e * e for e in v["score"]) / len(v["score"])) if v["score"] else None,
    }


def parse_log(path):
    summary, domains, best = {}, defaultdict(lambda: defaultdict(list)), None
    domain = None
    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n").split("\r")[-1]
            if m := RE_SCEN.match(line):
                domain = m.group(2)
                continue
            if domain is not None:
                if m := RE_CHOICE.search(line):
                    domains[domain]["choice"].append(m.group(1) == "PASS")
                elif m := RE_NOUL.search(line):
                    domains[domain]["noul"].append(m.group(1) == "PASS")
                elif m := RE_SCORE.search(line):
                    domains[domain]["score"].append(float(m.group(1)))
            for key, rx in RE_SUMMARY.items():
                if m := rx.search(line):
                    v = m.group(1).rstrip("%")
                    summary[key] = None if v == "n/a" else float(v)
            if m := RE_BEST.search(line):
                best = int(m.group(1))
    if len(summary) < len(RE_SUMMARY):
        return None
    summary["best_epoch"] = best
    per_domain = {d: item_metrics(v) for d, v in domains.items()}
    sources = defaultdict(lambda: defaultdict(list))
    for d, v in domains.items():
        for kind, xs in v.items():
            sources[source_of(d)][kind].extend(xs)
    per_source = {src: item_metrics(v) for src, v in sources.items()}
    return summary, per_domain, per_source


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s


def fmt(m, s, digits):
    return "n/a" if m is None else f"{m:.{digits}f} ± {s:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.join(ROOT, "data/.runs/ext-ablation"))
    ap.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    args = ap.parse_args()

    runs = defaultdict(dict)  # config -> seed -> (summary, per_domain, per_source)
    for path in sorted(glob.glob(os.path.join(args.logs, "*-seed*.log"))):
        m = re.match(r"(.+)-seed(\d+)\.log$", os.path.basename(path))
        if m is None:
            continue
        parsed = parse_log(path)
        if parsed is None:
            print(f"[skip] incomplete log {path}", file=sys.stderr)
            continue
        runs[m.group(1)][int(m.group(2))] = parsed
    configs = [c for c in CONFIG_ORDER if c in runs] + sorted(c for c in runs if c not in CONFIG_ORDER)
    if not configs:
        sys.exit("no complete logs found")

    out = ["Headline (merged validation set; all sources mixed):", ""]
    out.append("| Config | Seeds | " + " | ".join(label for _, label, _ in METRICS) + " | Best epoch |")
    out.append("|---" * (len(METRICS) + 3) + "|")
    agg = {}
    for c in configs:
        seeds = runs[c]
        row = [c, ",".join(str(s) for s in sorted(seeds))]
        agg[c] = {}
        for key, _, _ in METRICS:
            agg[c][key] = mean_std([seeds[s][0][key] for s in seeds])
            row.append(fmt(*agg[c][key], 3 if key == "score_rmse" else 1))
        row.append(fmt(*mean_std([seeds[s][0]["best_epoch"] for s in seeds]), 1))
        out.append("| " + " | ".join(row) + " |")

    if "baseline" in agg:
        out.append("")
        out.append("Delta vs baseline (mean over seeds; + = better):")
        out.append("")
        out.append("| Config | " + " | ".join(label for _, label, _ in METRICS) + " |")
        out.append("|---" * (len(METRICS) + 1) + "|")
        for c in configs:
            if c == "baseline":
                continue
            cells = []
            for key, _, higher in METRICS:
                b, x = agg["baseline"][key][0], agg[c][key][0]
                d = (x - b) if higher else (b - x)
                cells.append(f"{d:+.3f}" if key == "score_rmse" else f"{d:+.1f}")
            out.append(f"| {c} | " + " | ".join(cells) + " |")

    domains = sorted({d for c in configs for s in runs[c].values() for d in s[1] if source_of(d) == "in_domain"})
    for key, label, _ in METRICS[:3]:
        out.append("")
        out.append(f"Per in-domain domain {label} (mean over seeds):")
        out.append("")
        out.append("| Domain | " + " | ".join(configs) + " |")
        out.append("|---" * (len(configs) + 1) + "|")
        for d in domains:
            cells = []
            for c in configs:
                m, _ = mean_std([s[1].get(d, {}).get(key) for s in runs[c].values()])
                cells.append("n/a" if m is None else (f"{m:.3f}" if key == "score_rmse" else f"{m:.1f}"))
            out.append(f"| {d} | " + " | ".join(cells) + " |")

    # Per source: in-domain first, then each external held-out set.
    sources = sorted({src for c in configs for r in runs[c].values() for src in r[2]}, key=lambda x: (x != "in_domain", x))
    for src in sources:
        out.append("")
        out.append(f"Source `{src}` (mean ± std over seeds):")
        out.append("")
        out.append("| Config | " + " | ".join(label for _, label, _ in METRICS[:3]) + " |")
        out.append("|---" * 4 + "|")
        for c in configs:
            cells = [
                fmt(*mean_std([r[2].get(src, {}).get(key) for r in runs[c].values()]), 3 if key == "score_rmse" else 1)
                for key, _, _ in METRICS[:3]
            ]
            out.append(f"| {c} | " + " | ".join(cells) + " |")

    text = "\n".join(out) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
