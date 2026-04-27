#!/bin/bash
# Pull the kernel's output (best.pt + training log) back to local.
#
# `kaggle kernels output` downloads everything written to /kaggle/working/
# during the most recent run of the kernel.
#
# Usage:
#   bash scripts/kaggle_pull_checkpoint.sh                # pull to checkpoints/kaggle/
#   bash scripts/kaggle_pull_checkpoint.sh /path/to/dst   # pull to specified dir
#
# After pulling, promote the checkpoint with:
#   cp checkpoints/kaggle/checkpoints/best.pt checkpoints/best.pt
#   python make_release.py --notes "Kaggle GPU run, jaccard X" --deploy

source "$(dirname "$0")/_kaggle_common.sh"

OUT_DIR="${1:-${REPO_ROOT}/checkpoints/kaggle}"
KERNEL_ID="${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}"

mkdir -p "$OUT_DIR"

echo "Pulling latest output of $KERNEL_ID → $OUT_DIR"
kaggle kernels output "$KERNEL_ID" -p "$OUT_DIR"

echo
echo "Files retrieved:"
find "$OUT_DIR" -type f -exec ls -la {} \; | head -20

# Surface the most useful path
ckpt_path="$(find "$OUT_DIR" -name 'best.pt' -type f | head -1)"
if [ -n "$ckpt_path" ]; then
    echo
    echo "Best checkpoint: $ckpt_path"
    python3 - "$ckpt_path" <<'EOF' 2>/dev/null || true
import sys, torch
ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(f"  epoch        : {ckpt.get('epoch', '?')}")
print(f"  val_loss     : {ckpt.get('val_loss', float('nan')):.4f}")
print(f"  jaccard      : {ckpt.get('jaccard', float('nan')):.4f}")
print(f"  exact_match  : {ckpt.get('exact_match', float('nan')):.4f}")
EOF
    echo
    echo "To promote:"
    echo "  cp '$ckpt_path' checkpoints/best.pt"
    echo "  python make_release.py --notes \"Kaggle run\" --deploy"
fi
