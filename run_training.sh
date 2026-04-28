#!/bin/bash
# run_training.sh — training launcher
#
# Training runs are identified by YYYYMMDD.<git-hash> (same scheme as releases).
# TRAINING_VERS is auto-computed from the current date + HEAD hash unless overridden.
# Log files are written to training_results/training_<TRAINING_VERS>.log.
#
# Changelog (historical — pre-date+hash versioning):
#   S-3: Replaced hardcoded absolute paths with env-var overrides.
#   S-3: Device auto-detected (CUDA if available, else CPU).
#   F-3: early_stopping_patience increased to 20 (was 10).
#   F-4: well_consistency_weight lowered to 0.2 (was 0.5).
#   v6:  LoRA gradient flow fix; focal_gamma=0, col_weight=2.0, lora_rank=4, temporal_layers=1.
#   v7:  val_threshold 0.3→0.4 (diagnostic showed 0.3 masked 60% exact match).
#   v8:  Clip-type head: 3-class (single/row/col) + type_loss_weight=1.0.
#        Architecture change — cannot resume from v7; fresh training from DINOv2.
#   v9:  Fix 3: val decoder → threshold=0.4. Fix 4: checkpoint on Jaccard. Fix 5: NUM_FRAMES 4→8.
#   v10: Hybrid checkpoint criterion (Jaccard OR val_loss). Temporal jitter. Type-conditioned val
#        decoder + train metrics. Diagnosis: 70% train Jaccard but ~0% val Jaccard — overfitting.
#   v11: Leak-free synthetic split (real-only val, real+synth-of-train for train). Val no longer
#        augmented (was previously sharing the augment=True parent dataset). Removed broken
#        HorizontalFlip aug that flipped images without remapping col labels. Added MotionBlur
#        and ImageCompression augs. Effective dataset: 80 train_real + 560 train_synth + 20 val_real.
#        Fresh start (no resume) — augmented data manifold differs from v10 epoch-39 weights.
#
# Usage:
#   bash run_training.sh                      # auto-versioned, real labels only
#   USE_SYNTHETIC=1 bash run_training.sh      # use leak-free real+synthetic split (v11+ default)
#   DATA_DIR=/my/data bash run_training.sh    # override data path
#   DEVICE=cuda:0 bash run_training.sh        # force GPU
#   TRAINING_VERS=custom bash run_training.sh # override auto-version
#   RESUME=  bash run_training.sh             # train from scratch (empty RESUME)

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/pipette_well_dataset}"
LABELS="${LABELS:-${DATA_DIR}/labels.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints}"
TRAINING_OUTPUT_DIR="${TRAINING_OUTPUT_DIR:-${REPO_ROOT}/training_results}"
_GIT_HASH="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
_DATE="$(date -u +%Y%m%d)"
TRAINING_VERS="${TRAINING_VERS:-${_DATE}.${_GIT_HASH}}"
RESUME="${RESUME:-${REPO_ROOT}/checkpoints/best.pt}"

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
NUM_FRAMES="${NUM_FRAMES:-8}"
VAL_SPLIT="${VAL_SPLIT:-0.2}"
IMG_SIZE="${IMG_SIZE:-518}"
PATIENCE="${PATIENCE:-20}"                      # F-3: was 10
WELL_CONSISTENCY_WEIGHT="${WELL_CONSISTENCY_WEIGHT:-0.2}"  # F-4: was 0.5
FOCAL_GAMMA="${FOCAL_GAMMA:-0.0}"              # v6: plain weighted BCE (was 2.0 focal)
COL_WEIGHT="${COL_WEIGHT:-2.0}"                # v6: upweight column head
LORA_RANK="${LORA_RANK:-2}"                    # v13: 4→2, fewer params to reduce overfitting
TEMPORAL_LAYERS="${TEMPORAL_LAYERS:-1}"        # v6: was 2
TYPE_LOSS_WEIGHT="${TYPE_LOSS_WEIGHT:-1.0}"    # v8: clip-type head weight
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"           # v13: 10× increase (was 1e-3) to combat overfitting
DROPOUT="${DROPOUT:-0.5}"                      # v13: fusion MLP + temporal attention dropout (was 0.3)
NUM_FOLDS="${NUM_FOLDS:-5}"                    # v13: K-fold CV number of folds
FOLD_INDEX="${FOLD_INDEX:-0}"                  # v13: which fold to hold out as validation

# RESUME from v9 best.pt — same architecture, compatible checkpoint

# ── Launch ─────────────────────────────────────────────────────────────────
echo "[run_training.sh] Starting training ${TRAINING_VERS}"
echo "  DATA_DIR       : ${DATA_DIR}"
echo "  LABELS         : ${LABELS}"
echo "  OUTPUT         : ${OUTPUT_DIR}"
echo "  DEVICE         : ${DEVICE}"
echo "  EPOCHS         : ${EPOCHS}  PATIENCE: ${PATIENCE}"
echo "  IMG_SIZE       : ${IMG_SIZE}  FRAMES: ${NUM_FRAMES}  BATCH: ${BATCH_SIZE}"
echo "  FOCAL_GAMMA    : ${FOCAL_GAMMA}  COL_WEIGHT: ${COL_WEIGHT}"
echo "  LORA_RANK      : ${LORA_RANK}  TEMPORAL_LAYERS: ${TEMPORAL_LAYERS}"
echo "  TYPE_LOSS_WEIGHT: ${TYPE_LOSS_WEIGHT}"
echo "  WEIGHT_DECAY   : ${WEIGHT_DECAY}  DROPOUT: ${DROPOUT}"
echo "  KFOLD          : ${NUM_FOLDS}  FOLD: ${FOLD_INDEX}"
echo "  RESUME         : ${RESUME:-none}"

# Synthetic data uses a leak-free split (real-only val, real+synth-of-train).
# Set USE_SYNTHETIC=1 (default) to enable; USE_SYNTHETIC=0 forces real-only training.
SYNTHETIC_LABELS_PATH="${DATA_DIR}/labels_synthetic.json"
USE_SYNTHETIC="${USE_SYNTHETIC:-1}"
SYNTHETIC_ARG=""
if [ "${USE_SYNTHETIC}" = "1" ] && [ -f "${SYNTHETIC_LABELS_PATH}" ]; then
    SYNTHETIC_ARG="--synthetic_labels ${SYNTHETIC_LABELS_PATH}"
    echo "  SYNTHETIC      : ${SYNTHETIC_LABELS_PATH} (leak-free split: real val, real+synth train)"
else
    echo "  SYNTHETIC      : disabled (real-only)"
fi

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
  --type_loss_weight        "${TYPE_LOSS_WEIGHT}" \
  --weight_decay            "${WEIGHT_DECAY}" \
  --dropout                 "${DROPOUT}" \
  --kfold                   "${NUM_FOLDS}" \
  --fold                    "${FOLD_INDEX}" \
  ${SYNTHETIC_ARG} \
  ${RESUME:+--resume "${RESUME}"} \
  2>&1 | tee "${TRAINING_OUTPUT_DIR}/training_${TRAINING_VERS}.log"
