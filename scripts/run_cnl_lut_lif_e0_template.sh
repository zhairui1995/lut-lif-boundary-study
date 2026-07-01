#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke_c100_t1_8bit}"

: "${REPO_ROOT:?Set REPO_ROOT to the anonymous repository root}"
: "${DATA_DIR:?Set DATA_DIR to the CIFAR data root}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the retained checkpoint directory}"
: "${RESULT_DIR:?Set RESULT_DIR to the output directory}"
: "${PYTHON:=python}"
: "${CUDA_VISIBLE_DEVICES:=0}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

case "$mode" in
  smoke_c100_t1_8bit)
    "$PYTHON" tools/cnl_lut_lif_e0.py \
      --root "$REPO_ROOT" \
      --data-dir "$DATA_DIR" \
      --checkpoint "$CHECKPOINT_DIR/qkformer_cifar100_t1_seed42_model_best.pth.tar" \
      --result-dir "$RESULT_DIR/cnl_lut_lif_c100_t1_bit8_smoke" \
      --family cifar100 \
      --num-classes 100 \
      --local-cuda-index 0 \
      --time-step 1 \
      --state-bits 8 \
      --input-bits 8 \
      --seed 42 \
      --drop-threshold 4.79 \
      --batch-size 4 \
      --workers 2 \
      --calib-batches 1 \
      --max-eval-batches 1
    ;;
  full_c100_t1_8bit)
    "$PYTHON" tools/cnl_lut_lif_e0.py \
      --root "$REPO_ROOT" \
      --data-dir "$DATA_DIR" \
      --checkpoint "$CHECKPOINT_DIR/qkformer_cifar100_t1_seed42_model_best.pth.tar" \
      --result-dir "$RESULT_DIR/cnl_lut_lif_c100_t1_bit8_full" \
      --family cifar100 \
      --num-classes 100 \
      --local-cuda-index 0 \
      --time-step 1 \
      --state-bits 8 \
      --input-bits 8 \
      --seed 42 \
      --drop-threshold 4.79 \
      --batch-size 32 \
      --workers 4 \
      --calib-batches 128
    ;;
  full_c100_t4_6bit)
    "$PYTHON" tools/cnl_lut_lif_e0.py \
      --root "$REPO_ROOT" \
      --data-dir "$DATA_DIR" \
      --checkpoint "$CHECKPOINT_DIR/qkformer_cifar100_t4_seed42_model_best.pth.tar" \
      --result-dir "$RESULT_DIR/cnl_lut_lif_c100_t4_6bit_full" \
      --family cifar100 \
      --num-classes 100 \
      --local-cuda-index 0 \
      --time-step 4 \
      --state-bits 6 \
      --input-bits 6 \
      --seed 42 \
      --drop-threshold 2.06 \
      --batch-size 32 \
      --workers 4 \
      --calib-batches 128 \
      --include-failed-dense-e1
    ;;
  *)
    echo "unknown mode: $mode" >&2
    echo "valid modes: smoke_c100_t1_8bit, full_c100_t1_8bit, full_c100_t4_6bit" >&2
    exit 2
    ;;
esac
