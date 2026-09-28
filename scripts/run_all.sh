#!/usr/bin/env bash
# Run the experiment pipeline end-to-end (cache-aware, resumable, safe to re-run).
#
#   scripts/run_all.sh [stage]        # stages: gonogo data trainval test analyses all
#
# Execution order for `all` follows the case arms below: E0 first (fast decision),
# then the train/val caches, then the test-split experiments (Faz 5 gate).
# Every stage is idempotent: rerunning only fills in what is missing.
set -euo pipefail

STAGE="${1:-all}"
CFG="configs/default.yaml"
run() { echo "==> $*"; uv run "$@"; }

case "$STAGE" in
  gonogo|all)
    run counterpart generate    --split gonogo          # resumable, ~1.5 h left
    run counterpart evaluate    --split gonogo          # full E0 report
    run counterpart experiments --split gonogo          # E2/E3/E4/E5 on gonogo
    ;;&
  data|all)
    run counterpart prepare  --split train --workers 8  # resumable (102/300 done)
    run counterpart prepare  --split val   --workers 8
    run counterpart generate --split train              # ~9 h GPU
    run counterpart generate --split val                # ~3 h GPU
    ;;&
  trainval|all)
    for split in train val; do
      run counterpart score       --split "$split" --workers 8
      run counterpart baselines   --split "$split" --methods "" --clip --dino
      run counterpart evaluate    --split "$split"
      run counterpart select      --split "$split"
      run counterpart experiments --split "$split"
    done
    run counterpart fit-weights --split train           # mode-B weights
    ;;&
  test|all)
    run counterpart prepare   --split test --workers 8 --i-know
    run counterpart generate  --split test --i-know     # ~9 h GPU
    run counterpart score     --split test --workers 8 --i-know
    run counterpart baselines --split test --methods "" --clip --dino --i-know
    run counterpart evaluate  --split test --i-know
    run counterpart select    --split test --i-know
    run counterpart experiments --split test --i-know   # E1/E2/E3/E4/E5 (test)
    ;;&
  analyses|all)
    run counterpart e6  --split val
    run counterpart viz --split gonogo --kind panel --n 20
    run counterpart viz --split test   --kind panel --n 20
    ;;&
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
echo "done: $STAGE"
