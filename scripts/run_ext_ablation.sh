#!/usr/bin/env bash
# Ablation: does adding imported typed-decisions data (data/ext/) help the Tier 1 model?
# Every config trains on the in-domain set plus zero or more auxiliary files, validates on the
# unchanged in-domain validation set, and is repeated over several seeds (val metrics are
# seed-noisy). Summarize with scripts/summarize_ext_ablation.py.
#
#   scripts/run_ext_ablation.sh [config ...]     # default: all configs
#   ENCODER=models/modernbert-large SEEDS="42 1 2 3" scripts/run_ext_ablation.sh baseline
set -euo pipefail
cd "$(dirname "$0")/.."

ENCODER=${ENCODER:-models/modernbert-base}
SEEDS=${SEEDS:-"42 1 2"}
BACKEND=${BACKEND:-wgpu}
LOGS=${LOGS:-data/.runs/ext-ablation}
TRAIN=data/reflex_training_data.jsonl
VAL=data/reflex_validation_data.jsonl

declare -A EXTRA=(
  [baseline]=""
  [procedural]="data/ext/procedural.jsonl"
  [llama_security]="data/ext/llama_security.jsonl"
  [both]="data/ext/procedural.jsonl data/ext/llama_security.jsonl"
  [nemotron]="data/ext/nemotron_ipi.jsonl"
)
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(baseline procedural llama_security both nemotron)

cargo build --release --bin reflex-train
mkdir -p "$LOGS" models/ext-ablation

for config in "${CONFIGS[@]}"; do
  [[ -v "EXTRA[$config]" ]] || { echo "unknown config: $config" >&2; exit 1; }
  extra_args=()
  for f in ${EXTRA[$config]}; do extra_args+=(--train "$f"); done
  for seed in $SEEDS; do
    log="$LOGS/$config-seed$seed.log"
    if grep -q "^ Score RMSE" "$log" 2>/dev/null; then
      echo "[skip] $config seed $seed (done: $log)"
      continue
    fi
    echo "[run] $config seed $seed -> $log"
    ./target/release/reflex-train --backend "$BACKEND" --encoder "$ENCODER" \
      --train "$TRAIN" "${extra_args[@]}" --val "$VAL" \
      --out "models/ext-ablation/$config-seed$seed.safetensors" --seed "$seed" \
      > "$log" 2>&1 </dev/null
  done
done
