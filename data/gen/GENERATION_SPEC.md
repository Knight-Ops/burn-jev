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
 "choice_questions": [ {"prompt": "...", "candidates": ["...", "..."], "target": <int index>} ],  // 1-3 questions; candidates mostly 2-4, occasionally 10 or 50 (see Quotas)
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
  context formats (log lines, alert JSON-ish text, metrics summaries, audit trails, ticket excerpts,
  embedded scoring/lookup tables — see "In-context scoring matrices" below).
- Every scenario must be meaningfully different: different systems, numbers, actors, and failure modes.
  Do not produce templated near-duplicates that only swap an IP or a name.
- Choice-candidate-count variance: ~90-95% of `choice_questions` keep the usual 2-4 candidates
  (2-candidate/binary questions are allowed and should appear sometimes — don't default to 3-4 only).
  The remaining ~5-10% should use 10 or 50 candidates instead, to exercise the head's handling of
  larger option sets. At 10/50 candidates the options can be shorter and more schematic than usual
  (e.g. a list of hostnames, employee IDs, ticket numbers, CVE IDs, error codes, port numbers) —
  you do not need full detailed prose for 50 distractors — but the target must still be unambiguously
  derivable from the context, candidates must stay unique, and the correct answer must not be
  identifiable merely by length or position.

## In-context scoring matrices
A subset of records should embed an explicit lookup/scoring table directly in `context` — e.g. a
likelihood x impact risk matrix, an SLA-tier x downtime-duration penalty table, a severity-lookup
grid keyed by affected-record-count bucket, an escalation table keyed by role x incident-class.
Render the table as plain-text rows/columns inside the context string (it's still one JSON string —
use `\n` and aligned spacing or a simple `label: value` grid, whatever reads as authentic ops
documentation pasted into a ticket or runbook).

Rules for these records:
- The `score_rubrics` target (and/or `choice_questions`/`noul_queries` targets, where relevant) must
  be the value a competent analyst gets by correctly looking up the right row/column in the embedded
  table against the scenario's stated facts (e.g. "likelihood: high" x "impact: moderate" -> table
  says 3.5). Compute the lookup yourself and double-check the target actually matches the table
  before writing the record — this is a hard correctness requirement, not a style flourish.
- Vary table shapes and axes across records (don't reuse the same 3x3 risk matrix every time); vary
  which axis values the scenario's facts land on, including edge cells (min/max rows/columns) and at
  least a few off-by-one traps (e.g. a distractor choice/rubric value one cell away from the correct
  lookup) so the head can't just guess the table's central tendency.
- These records may run a bit longer than the usual 300-800 char guidance to fit the table — that's
  fine, don't strip detail to hit a length target.
- Still mix `is_benign` true/false, urgency levels, and hard negatives as usual; the matrix is a
  context-format device, not a separate topic.

## Topics already covered
Existing clusters/domains (do not duplicate their scenarios): `security` (cloudiam, websec, malware
— cloud IAM, web app security, malware/EDR), `networking` (network), `llm_security` (llmsec),
`fintech` (fintech), `distributed_systems` (distsys), `devops` (devops), `data_ml` (dataml),
`supply_chain` (supplychain).

## New clusters for this round
| cluster dir | domain string | topic scope |
|---|---|---|
| `datalabel` | `data_labeling` | automated/crowdsourced data labeling pipelines, annotation QA, active learning loops, weak supervision (Snorkel-style labeling functions), RLHF preference-data collection, label-poisoning by malicious annotators, PII leakage in labeled datasets |
| `icsot` | `industrial_control` | SCADA/PLC telemetry, Modbus/DNP3 traffic, water-treatment & manufacturing safety interlocks, substation automation, building/HVAC automation, PLC firmware updates |
| `healthcare` | `healthcare` | EHR access logs, HIPAA audit trails, clinical device telemetry (infusion pumps, imaging), break-glass emergency access, pharmacy inventory, medical billing/claims fraud |
| `retail` | `retail` | POS systems, e-commerce checkout fraud, chargebacks/disputes, loyalty-program abuse, promo-code abuse, inventory shrinkage, card-present vs card-not-present signals |
| `telecom` | `telecom` | SIM-swap fraud, carrier billing systems, number-porting fraud, robocall/spam detection, roaming anomalies, IoT SIM fleets |
| `insider` | `insider_threat` | HR/offboarding workflows, DLP alerts, badge+VPN correlation, departing-employee exfil, whistleblower vs malicious-insider distinctions, privileged access reviews |
| `trustsafety` | `trust_safety` | platform content-moderation queues, policy-violation triage (spam/harassment/scam accounts — policy-level, not graphic content), account-takeover vs organic abuse, bot-network detection, appeals process |
| `riskmatrix` | `risk_scoring` | every record embeds an in-context scoring matrix (see section above); scenarios span multiple existing operational domains (security, devops, fintech, healthcare-style, etc.) purely as flavor — the actual skill under test is correct table lookup, not a new subject area |

## Output mechanics
- Write in parts of ~25 records each using the Write tool: `data/gen/<cluster>/train_part_NN.jsonl`
  (NN = 01..08, ~200 train records total) and `data/gen/<cluster>/val_part_01.jsonl` (~25 val records).
  One compact JSON object per line, no trailing commas, no comments, no blank lines.
- After each part, validate it:
  `python3 scripts/check_part.py data/gen/<cluster>/<file>` and fix any errors it reports.
- Only write files under `data/gen/<cluster>/`. Do not touch anything else in the repo.
- When done, reply with: record counts, benign ratio, the held-out sub-topics you used for val, and any concerns.
