#!/bin/bash
# Poll a Kaggle kernel run and pull its output when done.
#
# IMPORTANT: this script DEFAULTS to NOT pushing the notebook (SKIP_PUSH=1).
# `kaggle kernels push` resets the kernel's accelerator to your account
# default — if that's P100 and you want T4, the push silently undoes your
# UI change. The recommended workflow is:
#   1. Trigger the run manually from the Kaggle UI (preserves T4)
#   2. Then run this script to poll + pull
#
# Set SKIP_PUSH=0 to re-enable the auto-push behavior (only useful if your
# Kaggle account default GPU is the one you actually want, e.g. T4).
#
# Usage:
#   bash scripts/kaggle_run.sh                   # poll + pull (no push)
#   SKIP_PUSH=0 bash scripts/kaggle_run.sh       # push + poll + pull
#
# Env overrides:
#   POLL_INTERVAL_S    seconds between status polls (default 60)
#   MAX_WAIT_S         max wait before giving up (default 43200 = 12h, the
#                      Kaggle session limit)
#   SKIP_PUSH          1 (default) = skip push, just poll/pull;
#                      0 = also push the notebook (resets accelerator!)

source "$(dirname "$0")/_kaggle_common.sh"

POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"
MAX_WAIT_S="${MAX_WAIT_S:-43200}"
KERNEL_ID="${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}"

if [ "${SKIP_PUSH:-1}" != "1" ]; then
    echo "[1/3] Pushing notebook..."
    echo "WARNING: this will reset the kernel's accelerator to your Kaggle"
    echo "         account default. If you wanted T4 and your default is"
    echo "         P100, re-select T4 in the UI before this run starts."
    bash "$(dirname "$0")/kaggle_push_notebook.sh"
    echo
else
    echo "[1/3] SKIP_PUSH=1 — assuming a run is already queued/active on Kaggle."
fi

echo "[2/3] Polling kernel status (every ${POLL_INTERVAL_S}s, max ${MAX_WAIT_S}s)..."
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT_S" ]; do
    # Kaggle status output looks like:
    #   "<owner>/<slug> has status \"KernelWorkerStatus.RUNNING\""
    # We want the bit after the dot. The `|| true` and `: ${var:=…}` pattern
    # keeps a failed extraction from killing the loop under set -euo pipefail.
    status_json="$(kaggle kernels status "$KERNEL_ID" 2>&1 || true)"
    status="$(printf '%s' "$status_json" | sed -nE 's/.*KernelWorkerStatus\.([A-Za-z]+).*/\1/p' | head -1 || true)"
    : "${status:=unknown}"
    printf '  [%4ds] status=%s\n' "$elapsed" "$status"
    case "$status" in
        COMPLETE|complete|completed|Complete)
            echo "Kernel completed."
            break
            ;;
        ERROR|error|CANCELACKNOWLEDGED|CANCELREQUESTED|cancelAcknowledged|cancelRequested|Failed|FAILED)
            echo "Kernel ended in non-success state: $status"
            echo "$status_json"
            exit 2
            ;;
    esac
    sleep "$POLL_INTERVAL_S"
    elapsed=$((elapsed + POLL_INTERVAL_S))
done
if [ "$elapsed" -ge "$MAX_WAIT_S" ]; then
    echo "Timed out waiting for kernel. The run may still finish — pull later with kaggle_pull_checkpoint.sh."
    exit 3
fi

echo
echo "[3/3] Pulling kernel output..."
bash "$(dirname "$0")/kaggle_pull_checkpoint.sh"
