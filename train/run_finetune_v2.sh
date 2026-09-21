#!/usr/bin/env bash
# Experiment 3b — (a) ModernBERT-base, longer, with question-order shuffling; (b) a dedicated,
# class-balanced confirm-gate model. Then packed evaluation of every bundle on the held-out groups.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-$HOME/projects/jevlet-work}
TD=$WORK/typed-decisions/src
MODELS=$WORK/models
export USE_TF=0 TORCHDYNAMO_DISABLE=1 TORCH_DISABLE_NATIVE_JIT=1 PYTHONPATH="$TD:${PYTHONPATH:-}"

python -m train.make_td_data --out data/td-desc-shuf --seed 0 --shuffle-questions 1
python -m train.make_td_data --out data/td-gate --seed 0 --gate-only

train() {  # name backbone data max_len epochs extra...
  local name=$1 bb=$2 data=$3 maxlen=$4 ep=$5; shift 5
  echo "=== $name"
  python -m typed_decisions.train_encoder --model "$bb" --data "$data" --out "runs/$name" \
      --epochs "$ep" --batch 16 --lr 3e-5 --head-lr 1e-3 --max-state 256 --max-len "$maxlen" \
      --augment 0.5 --bench-n 1,4 --save "$@" 2>&1 | grep -vE "Warning|warn|Loading weights" | tail -25
}
train modernbert-base-desc-shuf-e8 "$MODELS/answerdotai_ModernBERT-base" data/td-desc-shuf 2048 8 --attn sdpa
train modernbert-base-gate-e6      "$MODELS/answerdotai_ModernBERT-base" data/td-gate      1024 6 --attn sdpa

evalb() {  # name manifest [--no-desc]
  python -m train.eval_bundle --bundle "runs/$1/bundle" --pkg "$TD" --split-manifest "$2" --tag "ft-$1" --device cuda "${@:3}" 2>&1 | grep -vE "Warning|warn|Loading" | tail -1
}
evalb modernbert-base-desc data/td-desc/split_manifest.json
evalb modernbert-base-desc-shuf-e8 data/td-desc-shuf/split_manifest.json
evalb modernbert-base-gate-e6 data/td-gate/split_manifest.json
[ -d runs/deberta-v3-base-bare/bundle ] && evalb deberta-v3-base-bare data/td-bare/split_manifest.json --no-desc
[ -d runs/deberta-v3-large-openjev-desc/bundle ] && evalb deberta-v3-large-openjev-desc data/td-bare/split_manifest.json --no-desc
echo E3B_DONE
