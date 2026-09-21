#!/usr/bin/env bash
# Experiment 3 — fine-tune small encoder decision models on the netops rows (grouped split).
# Runs on the GPU host. Usage: bash train/run_finetune.sh [epochs]
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-$HOME/projects/jevlet-work}
TD=$WORK/typed-decisions/src
MODELS=$WORK/models
EPOCHS=${1:-4}
export USE_TF=0 TORCHDYNAMO_DISABLE=1 TORCH_DISABLE_NATIVE_JIT=1 CPATH="$WORK/pyhdr/usr/include/python3.12:$WORK/pyhdr/usr/include/aarch64-linux-gnu/python3.12:${CPATH:-}" PYTHONPATH="$TD:${PYTHONPATH:-}"

python -m train.make_td_data --out data/td-desc --gate --seed 0
python -m train.make_td_data --out data/td-bare --gate --seed 0 --no-desc

run() {  # name backbone data max_len extra...
  local name=$1 bb=$2 data=$3 maxlen=$4; shift 4
  echo "=== $name"
  python -m typed_decisions.train_encoder --model "$bb" --data "$data" --out "runs/$name" \
      --epochs "$EPOCHS" --batch 16 --lr 3e-5 --head-lr 1e-3 --max-state 256 --max-len "$maxlen" \
      --augment 0.5 --bench-n 1,4 --save "$@" 2>&1 | grep -vE "Warning|warn|Loading weights" | tail -40
}

run modernbert-base-desc  "$MODELS/answerdotai_ModernBERT-base"   data/td-desc 2048 --attn sdpa
run deberta-v3-base-bare  "$MODELS/microsoft_deberta-v3-base"     data/td-bare 512  --attn eager
run deberta-v3-large-openjev-desc "$MODELS/com-kotobalabs_open-jev-deberta-v3-large" data/td-bare 512 --attn eager
echo E3_TRAIN_DONE
