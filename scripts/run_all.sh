#!/usr/bin/env bash
# Run the remaining experiment pipeline end-to-end (cache-aware; safe to re-run).
#
#   scripts/run_all.sh [stage]
#
# Stages (default: all):
#   data      prepare train/val (+ gonogo already done), generate candidates
#   gonogo    full E0 evaluation on gonogo
#   trainval  score/baselines/select on train (+val), fit mode-B weights
#   test      prepare/generate/score/baselines/select/evaluate on test (Faz 5 gate)
#   analyses  E2/E3/E4/E5 + panels on train/val/test
#
# GPU-heavy stages can run overnight; every stage is resumable.
set -euo pipefail

STAGE="${1:-all}"
CFG="configs/default.yaml"
run() { echo "==> $*"; uv run "$@"; }

case "$STAGE" in
  data|all)
    run counterpart prepare  --split train --workers 8
    run counterpart prepare  --split val   --workers 8
    run counterpart generate --split train
    run counterpart generate --split val
    ;;&
  gonogo|all)
    run counterpart evaluate --split gonogo
    run counterpart experiments --split gonogo
    ;;&
  trainval|all)
    for split in train val; do
      run counterpart score     --split "$split" --workers 8
      run counterpart baselines --split "$split" --methods "" --clip --dino
      run counterpart evaluate  --split "$split"
      run counterpart select    --split "$split"
      run counterpart experiments --split "$split"
    done
    run counterpart fit-weights --split train
    ;;&
  test|all)
    run counterpart prepare   --split test --workers 8 --i-know
    run counterpart generate  --split test --i-know
    run counterpart score     --split test --workers 8 --i-know
    run counterpart baselines --split test --methods "" --clip --dino --i-know
    run counterpart evaluate  --split test --i-know
    run counterpart select    --split test --i-know
    run counterpart experiments --split test --i-know
    ;;&
  analyses|all)
    run counterpart e6 --split val
    run counterpart viz --split gonogo --kind panel --n 20
    run counterpart viz --split test --kind panel --n 20
    ;;&
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
echo "done: $STAGE"
