# Reflex-head training data generation spec

You are generating labeled scenarios for training small classifier heads (choice / noul / score)
on top of a Mamba backbone. The heads read a telemetry `context` plus questions and must predict the
labels. Quality and label correctness matter more than anything else. Diversity matters second.

## Style references
Read `data/seed/train_seed.jsonl` and `data/seed/val_seed.jsonl` (15 records). Match their style:
dense, concrete operational telemetry (service names, IPs, ASNs, CIDRs, ports, percentages, latencies,
counts, timestamps, RBAC roles, policy text). Do NOT copy or lightly paraphrase those scenarios.

## Record schema (one JSON object per line, JSONL)
```
{"id": "<cluster>_<nnnn>",            // unique, zero-padded, e.g. websec_0007; val: <cluster>_val_<nnnn>
 "domain": "<domain string>",         // the domain you are given
 "context": "<300-800 chars of telemetry>",
 "is_benign": true|false,
 "choice_questions": [ {"prompt": "...", "candidates": ["...", "..."], "target": <int index>} ],  // 1-3 questions, 2-4 candidates each
 "noul_queries": [ {"assertion": "...?", "target": <float in [0,1]>} ],                          // 2-4 queries
 "score_rubrics": [ {"prompt": "Rate ... on a 1-to-5 scale", "target": <float in [1,5]>} ] }   // 1-3 rubrics
```

## Label conventions
- `is_benign` = there is NO adversarial actor or malicious intent. Operational failures (OOM, deadlock,
  lag, misconfig, bad deploy, hardware fault, honest user error) are `true`. Attacks, abuse, fraud,
  insider misuse, and deliberate policy evasion are `false`. Authorized, scoped red-team / pentest activity
  is `true` (the actor is sanctioned) — but only if the context shows the authorization evidence.
- Labels must be derivable from the context. Every noul target must be answerable from what the context
  says. Every choice target must be the option a competent senior engineer/analyst would pick.

## Quotas (per file you write, approximately)
- ~50% `is_benign: true`, ~50% `false`.
- >=30% low-urgency / routine / informational scenarios where the main severity/urgency rubrics are 1.0-2.5
  (e.g. healthy-but-noisy systems, a single failed login, a scheduled job running slow but within SLA,
  a blocked scanner with zero impact, a cert expiring in 60 days). Score targets should spread over the full
  1-5 range; use one decimal (e.g. 1.3, 2.7, 3.4). Aim for roughly uniform coverage of 1-5 across all rubrics.
- >=15% hard negatives / look-alikes: benign that looks like an attack (authorized pentest with ticket,
  legitimate bulk export by data team with approval, backup job hammering storage, noisy vuln scanner run
  by the security team, load test) AND attacks that look normal (attacks from internal IPs or valid
  credentials, low-and-slow exfiltration, living-off-the-land binaries, a malicious but well-formed request).
- Noul targets: about half 1.0 and half 0.0 overall. Use graded values (0.25 / 0.5 / 0.75) in ~10% of
  queries, only where the context is genuinely ambiguous/partial. Mix positively and negatively phrased
  assertions so "yes" is not always the scary answer.
- Choice candidates: every distractor must be plausible, similarly specific, and of SIMILAR LENGTH to the
  correct answer. The correct answer must NOT systematically be the longest or most detailed. Include a
  "reasonable but suboptimal" option (e.g. too aggressive, too slow, right action wrong order, overkill
  for a low-severity event). In low-severity scenarios the correct answer is often the measured/minimal
  one ("log and monitor", "open a ticket for next sprint") and the distractor is the over-reaction.
  Place the correct answer at varying positions (it will also be shuffled downstream).
- Vary the question prompts, rubric prompts (urgency, blast radius, confidence of malicious intent,
  data sensitivity, customer impact, remediation complexity, likelihood of false positive, etc.) and
  context formats (log lines, alert JSON-ish text, metrics summaries, audit trails, ticket excerpts).
- Every scenario must be meaningfully different: different systems, numbers, actors, and failure modes.
  Do not produce templated near-duplicates that only swap an IP or a name.

## Output mechanics
- Write in parts of ~25 records each using the Write tool: `data/gen/<cluster>/train_part_NN.jsonl`
  (NN = 01..08, ~200 train records total) and `data/gen/<cluster>/val_part_01.jsonl` (~25 val records).
  One compact JSON object per line, no trailing commas, no comments, no blank lines.
- After each part, validate it:
  `python3 scripts/check_part.py data/gen/<cluster>/<file>` and fix any errors it reports.
- Only write files under `data/gen/<cluster>/`. Do not touch anything else in the repo.
- When done, reply with: record counts, benign ratio, the held-out sub-topics you used for val, and any concerns.
