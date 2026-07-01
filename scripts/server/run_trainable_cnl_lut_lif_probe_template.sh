#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$GPU" ]]; then
  echo "[trainable-cnl-lut-lif] set GPU or CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the CIFAR data root}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the retained QKFormer checkpoint}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/trainable_cnl_lut_lif_smoke}"
RUN_MODE="${RUN_MODE:-smoke}"

mkdir -p "$RESULT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

COMMON=(
  --root "$REPO_ROOT"
  --data-dir "$DATA_DIR"
  --checkpoint "$CHECKPOINT"
  --result-dir "$RESULT_DIR"
  --family "${FAMILY:-cifar100}"
  --local-cuda-index 0
  --time-step "${TIME_STEP:-4}"
  --state-bits "${STATE_BITS:-6}"
  --input-bits "${INPUT_BITS:-6}"
  --seed "${SEED:-42}"
  --max-drop "${MAX_DROP:-1.25}"
)

if [[ "$RUN_MODE" == "smoke" ]]; then
  EXTRA=(--batch-size 4 --workers 2 --calib-batches 1 --epochs 1 --max-train-batches 1 --max-eval-batches 1 --no-amp)
elif [[ "$RUN_MODE" == "full" ]]; then
  EXTRA=(--batch-size 32 --workers 4 --calib-batches "${CALIB_BATCHES:-32}" --epochs "${EPOCHS:-3}" --max-train-batches "${MAX_TRAIN_BATCHES:-128}" --amp)
else
  echo "[trainable-cnl-lut-lif] RUN_MODE must be smoke or full, got: $RUN_MODE" >&2
  exit 2
fi

"$PYTHON_BIN" tools/trainable_cnl_lut_lif_probe.py "${COMMON[@]}" "${EXTRA[@]}"
