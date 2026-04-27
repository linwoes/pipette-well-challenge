#!/bin/bash
# Push or version the Kaggle kernel (notebook).
#
# Stages the notebook + a kernel-metadata.json into a staging directory
# and runs `kaggle kernels push`. The notebook's REPO_REF is left at
# whatever the file specifies (default 'main'); the kernel will git
# clone that ref each time it runs, so code changes flow through GitHub.
#
# Usage:
#   bash scripts/kaggle_push_notebook.sh
#
# Env overrides:
#   KAGGLE_KERNEL_SLUG        (default: pipette-well-training)
#   KAGGLE_DATA_SLUG          (default: pipette-well-dataset)
#   KAGGLE_CHECKPOINT_SLUG    (default: pipette-well-checkpoint;
#                              set to empty to skip — fresh DINOv2 init)
#   KAGGLE_KERNEL_PRIVATE     (default: true)
#   KAGGLE_KERNEL_GPU         (default: true)
#   KAGGLE_USERNAME           (default: from ~/.kaggle/kaggle.json)

source "$(dirname "$0")/_kaggle_common.sh"

NB_SRC="${REPO_ROOT}/notebooks/kaggle_train.ipynb"
KERNEL_ID="${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}"
STAGE="${STAGE_ROOT}/kernel"
PRIVATE="${KAGGLE_KERNEL_PRIVATE:-true}"
GPU="${KAGGLE_KERNEL_GPU:-true}"

if [ ! -f "$NB_SRC" ]; then
    echo "ERROR: notebook not found: $NB_SRC" >&2
    exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE"
cp "$NB_SRC" "$STAGE/kaggle_train.ipynb"

# Build dataset_sources list. The Python helper renders proper JSON so
# we don't have to escape arrays by hand.
python3 - "$STAGE/kernel-metadata.json" <<EOF
import json, os, sys
out = sys.argv[1]
sources = ["${KAGGLE_USERNAME}/${KAGGLE_DATA_SLUG}"]
ckpt = "${KAGGLE_CHECKPOINT_SLUG}"
if ckpt:
    sources.append("${KAGGLE_USERNAME}/" + ckpt)
meta = {
    "id": "${KERNEL_ID}",
    "title": "Pipette Well Training",
    "code_file": "kaggle_train.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "${PRIVATE}",
    "enable_gpu": "${GPU}",
    "enable_internet": "true",
    "dataset_sources": sources,
    "competition_sources": [],
    "kernel_sources": [],
}
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
print("Wrote", out)
print(json.dumps(meta, indent=2))
EOF

echo
echo "Pushing kernel $KERNEL_ID..."
kaggle kernels push -p "$STAGE"

echo
echo "Kernel URL: https://www.kaggle.com/code/${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}"
echo "  Click 'Run All' on Kaggle, or use:  bash scripts/kaggle_run.sh"
