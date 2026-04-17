#!/bin/bash
# run_training.sh — v5 training launcher
#
# S-3 FIX: Replaced hardcoded absolute paths with env-var overrides.
# S-3 FIX: Device auto-detected (CUDA if available, else CPU).
# F-3 FIX: early_stopping_patience increased to 20 (was 10).
# F-4 FIX: well_consistency_weight lowered to 0.2 (was 0.5).
#
# Usage:
#   bash run_training.sh                      # use defaults
#   DATA_DIR=/my/data bash run_training.sh    # override data path
#   DEVICE=cuda:0 bash run_training.sh        # force GPU

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/pipette_well_dataset}"
LABELS="${LABELS:-${DATA_DIR}/labels.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints}"

# ── Device auto-detect ─────────────────────────────────────────────────────
if [ -z "${DEVICE:-}" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE="cuda"
        echo "[run_training.sh] CUDA available — training on GPU"
    else
        DEVICE="cpu"
        echo "[run_training.sh] No CUDA — training on CPU"
    fi
fi

# ── Training hyper-params ──────────────────────────────────────────────────
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_FRAMES="${NUM_FRAMES:-4}"
VAL_SPLIT="${VAL_SPLIT:-0.2}"
IMG_SIZE="${IMG_SIZE:-448}"
PATIENCE="${PATIENCE:-20}"            # F-3: was 10
WELL_CONSISTENCY_WEIGHT="${WELL_CONSISTENCY_WEIGHT:-0.2}"  # F-4: was 0.5

# ── Launch ─────────────────────────────────────────────────────────────────
echo "[run_training.sh] Starting training v5"
echo "  DATA_DIR : ${DATA_DIR}"
echo "  LABELS   : ${LABELS}"
echo "  OUTPUT   : ${OUTPUT_DIR}"
echo "  DEVICE   : ${DEVICE}"
echo "  EPOCHS   : ${EPOCHS}  PATIENCE: ${PATIENCE}"
echo "  IMG_SIZE : ${IMG_SIZE}  FRAMES: ${NUM_FRAMES}  BATCH: ${BATCH_SIZE}"

python "${REPO_ROOT}/train.py" \
  --data_dir      "${DATA_DIR}" \
  --labels        "${LABELS}" \
  --epochs        "${EPOCHS}" \
  --batch_size    "${BATCH_SIZE}" \
  --num_frames    "${NUM_FRAMES}" \
  --val_split     "${VAL_SPLIT}" \
  --output        "${OUTPUT_DIR}" \
  --device        "${DEVICE}" \
  --img_size      "${IMG_SIZE}" \
  --patience      "${PATIENCE}" \
  --well_consistency_weight "${WELL_CONSISTENCY_WEIGHT}" \
  2>&1 | tee "${REPO_ROOT}/training.log"
