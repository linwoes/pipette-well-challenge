#!/bin/bash
# Push or version the data Kaggle Dataset.
#
# Creates the dataset on the first run; subsequent runs version it with
# the current contents of data/pipette_well_dataset/. Files are symlinked
# into a staging directory so the upload sees the actual paths but the
# repo is left clean.
#
# Usage:
#   bash scripts/kaggle_push_data.sh
#   bash scripts/kaggle_push_data.sh "Refreshed synthetic clips"
#
# Env overrides:
#   DATA_DIR              (default: data/pipette_well_dataset)
#   KAGGLE_DATA_SLUG      (default: pipette-well-dataset)
#   KAGGLE_USERNAME       (default: from ~/.kaggle/kaggle.json)
#
# Note: Kaggle re-uploads everything; this is ~1.4 GB for the full
# real+synthetic set. Run only when data actually changes.

source "$(dirname "$0")/_kaggle_common.sh"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/pipette_well_dataset}"
MESSAGE="${1:-Update $(date -u +%Y-%m-%dT%H:%M:%SZ)}"
DATASET_ID="${KAGGLE_USERNAME}/${KAGGLE_DATA_SLUG}"
STAGE="${STAGE_ROOT}/data"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR not found: $DATA_DIR" >&2
    exit 1
fi
if [ ! -f "$DATA_DIR/labels.json" ]; then
    echo "ERROR: $DATA_DIR/labels.json missing — wrong DATA_DIR?" >&2
    exit 1
fi

echo "Staging $DATA_DIR → $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE"

cat > "$STAGE/dataset-metadata.json" <<EOF
{
  "title": "Pipette Well Dataset",
  "id": "${DATASET_ID}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Symlink everything (labels + .mp4); kaggle CLI follows symlinks.
for f in "$DATA_DIR"/*; do
    ln -s "$f" "$STAGE/$(basename "$f")"
done

n_files="$(ls "$STAGE" | wc -l | tr -d ' ')"
echo "Staged $n_files files for upload as $DATASET_ID"

# Detect whether the dataset already exists. The CLI returns non-zero on
# a not-found dataset, so we use it as the existence test.
if kaggle datasets metadata -p /tmp "$DATASET_ID" >/dev/null 2>&1; then
    echo "Dataset $DATASET_ID exists — pushing new version."
    kaggle datasets version -p "$STAGE" -m "$MESSAGE"
else
    echo "Dataset $DATASET_ID does not exist — creating."
    kaggle datasets create -p "$STAGE"
fi

echo
echo "Done. Dataset URL: https://www.kaggle.com/datasets/${KAGGLE_USERNAME}/${KAGGLE_DATA_SLUG}"
