#!/usr/bin/env bash
set -euo pipefail

mode="${1:-full_c100_t4}"

: "${REPO_ROOT:?Set REPO_ROOT to the anonymous repository root}"
: "${DATA_DIR:?Set DATA_DIR to the CIFAR data root}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the retained checkpoint directory}"
: "${RESULT_DIR:?Set RESULT_DIR to the output directory}"
: "${PYTHON:=python}"
: "${CUDA_VISIBLE_DEVICES:=0}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

case "$mode" in
  smoke_c100_t4)
    "$PYTHON" tools/lut_if_poc.py \
      --root "$REPO_ROOT" \
      --data-dir "$DATA_DIR" \
      --checkpoint "$CHECKPOINT_DIR/qkformer_cifar100_t4_seed42_model_best.pth.tar" \
      --result-dir "$RESULT_DIR/lut_if_poc_c100_t4_smoke" \
      --local-cuda-index 0 \
      --time-step 4 \
      --state-bits 6 \
      --input-bits 6 \
      --seed 42 \
      --batch-size 4 \
      --workers 2 \
      --calib-batches 1 \
      --epochs 1 \
      --max-train-batches 1 \
      --max-eval-batches 1
    ;;
  full_c100_t4)
    "$PYTHON" tools/lut_if_poc.py \
      --root "$REPO_ROOT" \
      --data-dir "$DATA_DIR" \
      --checkpoint "$CHECKPOINT_DIR/qkformer_cifar100_t4_seed42_model_best.pth.tar" \
      --result-dir "$RESULT_DIR/lut_if_poc_c100_t4_full" \
      --local-cuda-index 0 \
      --time-step 4 \
      --state-bits 6 \
      --input-bits 6 \
      --seed 42 \
      --batch-size 32 \
      --workers 4 \
      --calib-batches 32 \
      --epochs 3 \
      --max-train-batches 128 \
      --max-drop 1.0
    ;;
  *)
    echo "unknown mode: $mode" >&2
    echo "valid modes: smoke_c100_t4, full_c100_t4" >&2
    exit 2
    ;;
esac
