#!/bin/bash
# Common helpers for kaggle_*.sh scripts. Sourced, not executed directly.
#
# Resolves KAGGLE_USERNAME from (in order):
#   1. existing env var
#   2. ~/.kaggle/kaggle.json
# Errors out with a useful message if neither is available.
#
# Defines:
#   REPO_ROOT
#   KAGGLE_USERNAME
#   KAGGLE_DATA_SLUG       (default: pipette-well-dataset)
#   KAGGLE_CHECKPOINT_SLUG (default: pipette-well-checkpoint; can be empty)
#   KAGGLE_KERNEL_SLUG     (default: pipette-well-training)
#   STAGE_ROOT             (.kaggle_stage under repo root, gitignored)
#
# Usage from another script:
#   source "$(dirname "$0")/_kaggle_common.sh"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "ERROR: kaggle CLI not found. Install with: pip install kaggle" >&2
    exit 1
fi

# Resolve KAGGLE_USERNAME
if [ -z "${KAGGLE_USERNAME:-}" ]; then
    if [ -f "$HOME/.kaggle/kaggle.json" ]; then
        KAGGLE_USERNAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["username"])' "$HOME/.kaggle/kaggle.json" 2>/dev/null || echo "")"
    fi
fi
if [ -z "${KAGGLE_USERNAME:-}" ]; then
    cat >&2 <<EOM
ERROR: KAGGLE_USERNAME is not set and ~/.kaggle/kaggle.json was not found.

To authenticate the Kaggle CLI:
  1. Sign in at https://www.kaggle.com/ and go to Account.
  2. Click "Create New API Token" — downloads kaggle.json.
  3. Place it at ~/.kaggle/kaggle.json with mode 600:
       mkdir -p ~/.kaggle
       mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json
EOM
    exit 1
fi
export KAGGLE_USERNAME

KAGGLE_DATA_SLUG="${KAGGLE_DATA_SLUG:-pipette-well-dataset}"
KAGGLE_CHECKPOINT_SLUG="${KAGGLE_CHECKPOINT_SLUG:-pipette-well-checkpoint}"
KAGGLE_KERNEL_SLUG="${KAGGLE_KERNEL_SLUG:-pipette-well-training}"
STAGE_ROOT="${REPO_ROOT}/.kaggle_stage"

mkdir -p "$STAGE_ROOT"
