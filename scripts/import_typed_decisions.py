#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets>=3", "tokenizers>=0.20"]
# ///
"""Convert Hugging Face "typed-decisions" datasets into reflex-train JSONL (auxiliary train sets).

Sources:
  procedural  tasksource/procedural-typed-decisions (synthetic, exact labels)
  llama       LocalLLaMA/typed-decisions (soft, annotator-distribution labels)
  nemotron    nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1 (CC-BY-4.0; attribution
              required). Not typed-decisions: every row is an attack, so each yields an attack
              record plus a clean counterfactual with the injection text removed; questions are
              derived exactly from the row's injection metadata (see `import_nemotron`).

Mapping per row:
  context  <- state (JSON rendered as indented `key: value` lines; plain-text states kept as is)
  choice   <- prompt = instructions, candidates = criteria values, target = argmax gold
  noul     <- assertion = instructions, target = gold noul probability
  score    <- instructions + rescaled legend; target = 1 + 4 * score / (N - 1)

Filters: nouls with |p - 0.5| < --noul-margin, choices whose argmax probability is below
--choice-min-prob, and scores with purely numeric criteria (counts, not ordinal ratings) are
dropped; rows left with no questions, or whose encoding would truncate the context at
--max-seq-len, are dropped. `is_benign` is only set where the source says so (the llama
`true_positive` noul, subject to the same margin); otherwise it is omitted (= unknown).

Outputs are auxiliary train files for `reflex-train --train <in-domain> --train <ext>`; never
mix them into the validation set. Run with LD_LIBRARY_PATH unset (libtorch conflicts):

  env -u LD_LIBRARY_PATH uv run scripts/import_typed_decisions.py procedural --out data/ext/procedural.jsonl
  env -u LD_LIBRARY_PATH uv run scripts/import_typed_decisions.py llama --configs security_incidents \\
      --out data/ext/llama_security.jsonl
  env -u LD_LIBRARY_PATH uv run scripts/import_typed_decisions.py nemotron --out data/ext/nemotron_ipi.jsonl
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCEDURAL = "tasksource/procedural-typed-decisions"
LLAMA = "LocalLLaMA/typed-decisions"
NEMOTRON = "nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1"
NEMOTRON_CATEGORIES = {
    "unauthorized_action": "Trigger an action the user did not ask for",
    "data_modification": "Modify or corrupt stored records",
    "denial_of_service": "Disrupt or block the service (denial of service)",
    "exfiltration": "Send sensitive data to an outside party",
}
# Subsets that exercise "read the evidence, apply the stated rule/table".
PROCEDURAL_SUBSETS = [
    "evidence_sufficiency",
    "policy_applicability",
    "policy_under_uncertainty",
    "table_lookup",
    "needle_retrieval",
    "record_aggregation",
]


# ---------------------------------------------------------------- context rendering


def fmt_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, separators=(", ", ": "))
    return str(v)


def fmt_inline(v):
    if isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v):
        return ", ".join(fmt_scalar(x) for x in v) if v else "(none)"
    return fmt_scalar(v)


def render_dict(d, indent=0):
    pad = " " * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.extend(render_dict(v, indent + 2))
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            lines.append(f"{pad}{k}:")
            for item in v:
                lines.append(f"{pad}- " + "; ".join(f"{k2}: {fmt_inline(v2)}" for k2, v2 in item.items()))
        else:
            lines.append(f"{pad}{k}: {fmt_inline(v)}")
    return lines


def render_state(state):
    try:
        obj = json.loads(state)
    except (json.JSONDecodeError, TypeError):
        return state.strip()
    if isinstance(obj, dict):
        return "\n".join(render_dict(obj))
    return fmt_inline(obj)


# ---------------------------------------------------------------- label mapping


def criteria_items(criteria):
    """(key, text) pairs for a criteria dict or list."""
    if isinstance(criteria, dict):
        return [(str(k), str(v) if v not in (None, "") else str(k)) for k, v in criteria.items()]
    return [(str(i), str(v)) for i, v in enumerate(criteria)]


def is_numeric_scale(texts):
    def num(t):
        try:
            float(t)
            return True
        except ValueError:
            return False

    return all(num(t) for t in texts)


def convert_questions(questions, answers, args, stats):
    choices, nouls, scores = [], [], []
    for name, q in questions.items():
        a = answers.get(name)
        if a is None:
            stats["q_missing_answer"] += 1
            continue
        kind = q.get("type")
        instr = q.get("instructions", "").strip()
        if kind == "choice":
            items = criteria_items(q.get("criteria", {}))
            keys = [k for k, _ in items]
            probs = a.get("probabilities") or {a["choice"]: 1.0}
            best = max(keys, key=lambda k: probs.get(k, 0.0))
            if a.get("choice") is not None and a["choice"] in keys and probs.get(a["choice"], 0.0) >= probs.get(best, 0.0):
                best = a["choice"]
            if len(items) < 2 or probs.get(best, 0.0) < args.choice_min_prob:
                stats["choice_dropped"] += 1
                continue
            choices.append({"prompt": instr, "candidates": [t for _, t in items], "target": keys.index(best)})
        elif kind == "noul":
            p = float(a["noul"])
            if abs(p - 0.5) < args.noul_margin:
                stats["noul_dropped"] += 1
                continue
            nouls.append({"assertion": instr, "target": round(p, 4)})
        elif kind == "score":
            texts = [t for _, t in criteria_items(q.get("criteria", []))]
            n = len(texts)
            if n < 2 or is_numeric_scale(texts):
                stats["score_dropped"] += 1
                continue
            anchors = "; ".join(f"{1 + 4 * i / (n - 1):.3g} = {t}" for i, t in enumerate(texts))
            target = 1.0 + 4.0 * float(a["score"]) / (n - 1)
            scores.append({"prompt": f"{instr} Rate on a 1-to-5 scale where {anchors}", "target": round(target, 4)})
        else:
            stats[f"q_unknown_type_{kind}"] += 1
    return choices, nouls, scores


# ---------------------------------------------------------------- length check


def seq_len(tok, record):
    """Encoded length with an untruncated context; mirrors `encode_scenario` in src/encoding.rs."""
    n = lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    total = 2 + n(record["context"])  # [CLS] context [SEP]
    for q in record["choice_questions"]:
        total += n(q["prompt"]) + 1 + sum(n(c) + 1 for c in q["candidates"])
    total += sum(n(q["assertion"]) + 1 for q in record["noul_queries"])
    total += sum(n(q["prompt"]) + 1 for q in record["score_rubrics"])
    return total


def build_record(rid, domain, state, questions, answers, is_benign, tok, args, stats):
    choices, nouls, scores = convert_questions(questions, answers, args, stats)
    if not (choices or nouls or scores):
        stats["row_no_questions"] += 1
        return None
    record = {"id": rid, "domain": domain, "context": render_state(state)}
    if is_benign is not None:
        record["is_benign"] = is_benign
    record.update({"choice_questions": choices, "noul_queries": nouls, "score_rubrics": scores})
    length = seq_len(tok, record)
    if length > args.max_seq_len:
        stats["row_too_long"] += 1
        return None
    stats["tokens_max"] = max(stats["tokens_max"], length)
    stats["tokens_sum"] += length
    stats["n_choice"] += len(choices)
    stats["n_noul"] += len(nouls)
    stats["n_score"] += len(scores)
    return record


# ---------------------------------------------------------------- sources


def import_procedural(args, tok):
    out = []
    for subset in args.subsets:
        stats = Counter()
        ds = load_dataset(PROCEDURAL, subset, split="train").shuffle(seed=args.seed)
        kept = []
        for row in ds:
            if len(kept) >= args.per_subset:
                break
            stats["rows_seen"] += 1
            rec = build_record(
                f"ext_proc_{subset}_{len(kept):05d}",
                f"ext_procedural_{subset}",
                row["state"],
                json.loads(row["questions"]),
                json.loads(row["answers"]),
                None,
                tok,
                args,
                stats,
            )
            if rec is not None:
                kept.append(rec)
        report(subset, kept, stats)
        out.extend(kept)
    return out


def import_llama(args, tok):
    out = []
    for config in args.configs:
        stats = Counter()
        kept = []
        for split in ("train", "test"):
            for row in load_dataset(LLAMA, config, split=split):
                stats["rows_seen"] += 1
                gold = json.loads(row["gold"])
                is_benign = None
                tp = gold.get("true_positive")
                if tp is not None and abs(float(tp["noul"]) - 0.5) >= args.noul_margin:
                    is_benign = float(tp["noul"]) < 0.5
                rec = build_record(
                    f"ext_llama_{config}_{len(kept):05d}",
                    f"ext_llama_{config}",
                    row["state"],
                    json.loads(row["questions"]),
                    gold,
                    is_benign,
                    tok,
                    args,
                    stats,
                )
                if rec is not None:
                    kept.append(rec)
        report(config, kept, stats)
        out.extend(kept)
    return out


def strip_text(obj, needle):
    """Copy of `obj` with `needle` removed from every string leaf; returns (copy, n_removed)."""
    if isinstance(obj, str):
        return (obj.replace(needle, ""), 1) if needle in obj else (obj, 0)
    if isinstance(obj, dict):
        pairs = [(k, *strip_text(v, needle)) for k, v in obj.items()]
        return {k: v for k, v, _ in pairs}, sum(n for _, _, n in pairs)
    if isinstance(obj, list):
        pairs = [strip_text(v, needle) for v in obj]
        return [v for v, _ in pairs], sum(n for _, n in pairs)
    return obj, 0


def nemotron_context(row, env):
    msgs = {m["role"]: m["content"] for m in row["responses_create_params"]["input"]}
    tools = [t.get("name", "?") for t in (json.loads(t) if isinstance(t, str) else t for t in row["responses_create_params"]["tools"])]
    return "\n".join(
        [
            f"Agent instructions: {msgs.get('system', '').strip()}",
            f"User request: {msgs.get('user', '').strip()}",
            f"Agent tools: {', '.join(tools)}",
            "Data available to the agent:",
            *render_dict(env),
        ]
    )


def import_nemotron(args, tok):
    """Attack record + clean counterfactual per row. Labels (all exact, from metadata):
    - noul "retrieved data contains instructions not from the user": attack 1, clean 0
    - noul "calling <target_tool> is required for the request": 0 (never in required_tools)
    - noul "agent should follow the embedded instructions": attack only, 0
    - choice "what should the agent do": complete the request / also do the injected goal / refuse -> 0
    - choice "what does the embedded instruction attempt": attack only, the attack category
    No score rubrics: the source has no graded label to derive them from.
    """
    stats = Counter()
    kept = []
    ds = load_dataset(NEMOTRON, split="train").shuffle(seed=args.seed)
    for row in ds:
        if args.limit and len(kept) >= 2 * args.limit:
            break
        stats["rows_seen"] += 1
        inj = row["injection"]
        env = row["environment"]
        env = json.loads(env) if isinstance(env, str) else env
        clean_env, removed = strip_text(env, inj["injection_text"])
        target_tool = inj["target_tool"]
        if not removed or target_tool in row["required_tools"] or inj["category"] not in NEMOTRON_CATEGORIES:
            stats["row_unusable"] += 1
            continue
        required = ", ".join(row["required_tools"]) or "the relevant tools"
        act = {
            "prompt": "What should the agent do next?",
            "candidates": [
                f"Complete the user's request using {required}, treating retrieved data as data only",
                f"Complete the user's request and also: {inj['goal']}",
                "Refuse the entire task and take no action",
            ],
            "target": 0,
        }
        pair = []
        for variant, e in (("attack", env), ("clean", clean_env)):
            attack = variant == "attack"
            choices = [act]
            nouls = [
                {"assertion": "Does the data available to the agent contain instructions that did not come from the user?", "target": float(attack)},
                {"assertion": f"Is calling {target_tool} required to complete the user's request?", "target": 0.0},
            ]
            if attack:
                cats = list(NEMOTRON_CATEGORIES)
                choices.append(
                    {
                        "prompt": "What does the instruction embedded in the retrieved data attempt?",
                        "candidates": [NEMOTRON_CATEGORIES[c] for c in cats],
                        "target": cats.index(inj["category"]),
                    }
                )
                nouls.append({"assertion": "Should the agent follow the instructions embedded in the retrieved data?", "target": 0.0})
            record = {
                "id": f"ext_nemo_{row['id']:05d}_{variant}",
                "domain": f"ext_nemotron_ipi_{row['domain']}",
                "context": nemotron_context(row, e),
                "is_benign": not attack,
                "choice_questions": choices,
                "noul_queries": nouls,
                "score_rubrics": [],
            }
            length = seq_len(tok, record)
            if length > args.max_seq_len:
                break
            pair.append((record, length))
        if len(pair) != 2:
            stats["row_too_long"] += 1
            continue
        for record, length in pair:
            kept.append(record)
            stats["tokens_max"] = max(stats["tokens_max"], length)
            stats["tokens_sum"] += length
            stats["n_choice"] += len(record["choice_questions"])
            stats["n_noul"] += len(record["noul_queries"])
    report("nemotron_ipi", kept, stats)
    if stats["row_unusable"]:
        print(f"[nemotron_ipi] {stats['row_unusable']} rows unusable (injection text not found verbatim)", file=sys.stderr)
    return kept


def report(name, kept, stats):
    n = len(kept)
    benign = Counter(r.get("is_benign") for r in kept)
    print(
        f"[{name}] kept {n}/{stats['rows_seen']} rows | choice {stats['n_choice']} noul {stats['n_noul']} "
        f"score {stats['n_score']} | dropped: choice {stats['choice_dropped']} noul {stats['noul_dropped']} "
        f"score {stats['score_dropped']} rows(no-q {stats['row_no_questions']}, too-long {stats['row_too_long']}) | "
        f"tokens mean {stats['tokens_sum'] / max(n, 1):.0f} max {stats['tokens_max']} | is_benign {dict(benign)}",
        file=sys.stderr,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    p = sub.add_parser("procedural")
    p.add_argument("--subsets", nargs="+", default=PROCEDURAL_SUBSETS)
    p.add_argument("--per-subset", type=int, default=500, help="rows kept per subset (train split)")
    l = sub.add_parser("llama")
    l.add_argument("--configs", nargs="+", default=["security_incidents"])
    n = sub.add_parser("nemotron")
    n.add_argument("--limit", type=int, default=0, help="max source rows (each yields 2 records); 0 = all")
    for s in (p, l, n):
        s.add_argument("--out", required=True)
        s.add_argument("--tokenizer", default=os.path.join(ROOT, "models/modernbert-base/tokenizer.json"))
        s.add_argument("--max-seq-len", type=int, default=2048, help="drop rows whose context would be truncated")
        s.add_argument("--noul-margin", type=float, default=0.3, help="keep nouls with |p - 0.5| >= margin")
        s.add_argument("--choice-min-prob", type=float, default=0.5, help="keep choices whose argmax prob >= this")
        s.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    importers = {"procedural": import_procedural, "llama": import_llama, "nemotron": import_nemotron}
    records = importers[args.source](args, tok)
    random.Random(args.seed).shuffle(records)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
