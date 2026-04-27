#!/bin/bash
# Push or version the Kaggle kernel (notebook).
#
# WARNING: this push will reset the kernel's accelerator to your Kaggle
# account default. The Kaggle SaveKernel API exposes only
# enable_gpu=true/false, not GPU TYPE. If your default is P100 and you
# wanted T4, you must re-select T4 in the Kaggle UI after this push.
#
# Use this only when you actually need to push the notebook structure
# (first-time kernel creation, dataset-attachment changes, cell layout
# changes). For code-only updates, just `git push` and trigger a run
# from the Kaggle UI — the notebook git-pulls at runtime.
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
echo
echo "** ACCELERATOR NOTICE **"
echo "The push reset the kernel's accelerator to your Kaggle account default."
echo "If you want T4 (recommended): open the kernel above, click Edit,"
echo "Settings panel → Accelerator → 'GPU T4 x2', then Save & Run All."
echo "Triggering a run via CLI right now would use whichever GPU your account"
echo "defaults to, which is currently P100 (sm_60, not supported by Kaggle's"
echo "PyTorch image)."
