#!/bin/bash
# run_training.sh — v7 training launcher
#
# S-3 FIX: Replaced hardcoded absolute paths with env-var overrides.
# S-3 FIX: Device auto-detected (CUDA if available, else CPU).
# F-3 FIX: early_stopping_patience increased to 20 (was 10).
# F-4 FIX: well_consistency_weight lowered to 0.2 (was 0.5).
# v6:  LoRA gradient flow fix (removed no_grad from backbone calls).
#       focal_gamma=0, col_weight=2.0, lora_rank=4, temporal_layers=1.
# v7:  val_threshold 0.3→0.4 (diagnostic showed 0.3 masked 60% exact match).
#       epochs 50→80 (model still improving at epoch 50, no early stop triggered).
#       resume from v6 best.pt checkpoint (epoch 28, val_loss=0.4294) so v7
#       continues fine-tuning rather than restarting from scratch.
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
PYTHON="${PYTHON:-python3}"

if [ -z "${DEVICE:-}" ]; then
    if "${PYTHON}" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE="cuda"
        echo "[run_training.sh] CUDA available — training on GPU"
    else
        DEVICE="cpu"
        echo "[run_training.sh] No CUDA — training on CPU"
    fi
fi

# ── Training hyper-params ──────────────────────────────────────────────────
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_FRAMES="${NUM_FRAMES:-4}"
VAL_SPLIT="${VAL_SPLIT:-0.2}"
IMG_SIZE="${IMG_SIZE:-448}"
PATIENCE="${PATIENCE:-20}"                      # F-3: was 10
WELL_CONSISTENCY_WEIGHT="${WELL_CONSISTENCY_WEIGHT:-0.2}"  # F-4: was 0.5
FOCAL_GAMMA="${FOCAL_GAMMA:-0.0}"              # v6: plain weighted BCE (was 2.0 focal)
COL_WEIGHT="${COL_WEIGHT:-2.0}"                # v6: upweight column head
LORA_RANK="${LORA_RANK:-4}"                    # v6: was 8
TEMPORAL_LAYERS="${TEMPORAL_LAYERS:-1}"        # v6: was 2
RESUME="${RESUME:-${OUTPUT_DIR}/best.pt}"      # v7: resume from v6 checkpoint

# ── Launch ─────────────────────────────────────────────────────────────────
echo "[run_training.sh] Starting training v7"
echo "  DATA_DIR       : ${DATA_DIR}"
echo "  LABELS         : ${LABELS}"
echo "  OUTPUT         : ${OUTPUT_DIR}"
echo "  DEVICE         : ${DEVICE}"
echo "  EPOCHS         : ${EPOCHS}  PATIENCE: ${PATIENCE}"
echo "  IMG_SIZE       : ${IMG_SIZE}  FRAMES: ${NUM_FRAMES}  BATCH: ${BATCH_SIZE}"
echo "  FOCAL_GAMMA    : ${FOCAL_GAMMA}  COL_WEIGHT: ${COL_WEIGHT}"
echo "  LORA_RANK      : ${LORA_RANK}  TEMPORAL_LAYERS: ${TEMPORAL_LAYERS}"

"${PYTHON}" "${REPO_ROOT}/train.py" \
  --data_dir                "${DATA_DIR}" \
  --labels                  "${LABELS}" \
  --epochs                  "${EPOCHS}" \
  --batch_size              "${BATCH_SIZE}" \
  --num_frames              "${NUM_FRAMES}" \
  --val_split               "${VAL_SPLIT}" \
  --output                  "${OUTPUT_DIR}" \
  --device                  "${DEVICE}" \
  --img_size                "${IMG_SIZE}" \
  --patience                "${PATIENCE}" \
  --well_consistency_weight "${WELL_CONSISTENCY_WEIGHT}" \
  --focal_gamma             "${FOCAL_GAMMA}" \
  --col_weight              "${COL_WEIGHT}" \
  --lora_rank               "${LORA_RANK}" \
  --temporal_layers         "${TEMPORAL_LAYERS}" \
  --resume                  "${RESUME}" \
  2>&1 | tee "${REPO_ROOT}/training.log"
