#!/bin/bash
# Full deployment loop: push notebook → trigger run → poll → pull output.
#
# `kaggle kernels push` not only updates the kernel code but also queues
# a new run, so this script just needs to push, poll status, and pull.
#
# Usage:
#   bash scripts/kaggle_run.sh
#
# Env overrides:
#   POLL_INTERVAL_S    seconds between status polls (default 60)
#   MAX_WAIT_S         max wait before giving up (default 43200 = 12h, the
#                      Kaggle session limit; if hit, the kernel may still
#                      complete and you can re-run kaggle_pull_checkpoint.sh
#                      later)
#   SKIP_PUSH          if set to 1, skip the kernel push and just poll the
#                      currently-running version (use this if you launched
#                      the run manually on the Kaggle UI)

source "$(dirname "$0")/_kaggle_common.sh"

POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"
MAX_WAIT_S="${MAX_WAIT_S:-43200}"
KERNEL_ID="${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}"

if [ "${SKIP_PUSH:-0}" != "1" ]; then
    echo "[1/3] Pushing notebook..."
    bash "$(dirname "$0")/kaggle_push_notebook.sh"
    echo
fi

echo "[2/3] Polling kernel status (every ${POLL_INTERVAL_S}s, max ${MAX_WAIT_S}s)..."
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT_S" ]; do
    status_json="$(kaggle kernels status "$KERNEL_ID" 2>&1 || true)"
    status="$(echo "$status_json" | grep -oE 'status[":[:space:]]+"?[a-zA-Z]+' | head -1 | grep -oE '[a-zA-Z]+$' || echo unknown)"
    printf '  [%4ds] status=%s\n' "$elapsed" "$status"
    case "$status" in
        complete|completed)
            echo "Kernel completed."
            break
            ;;
        error|cancelAcknowledged|cancelRequested)
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
