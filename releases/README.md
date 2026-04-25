# Model Releases

Each release is a directory named `YYYYMMDD.<git-hash>` — sortable by date, pinned to the exact commit that produced the weights.

## Structure

```
releases/
  index.json                  # registry of all releases with metrics
  latest -> YYYYMMDD.<hash>/  # symlink to most recent release
  YYYYMMDD.<hash>/
    model.pt                  # model weights (DualViewFusion checkpoint)
    config.json               # model config, metrics, and provenance
    RELEASE_NOTES.md          # what changed and known limitations
```

## Using a release

```bash
# Clone the repo at the exact commit this release was built from
git clone https://github.com/linwoes/pipette-well-challenge
git checkout <hash>          # hash is in config.json

# Run inference
python inference.py \
  --model releases/<version>/model.pt \
  --fpv path/to/fpv.mp4 \
  --topview path/to/topview.mp4
```

Or use `releases/latest/` to always point at the most recent release:

```bash
python inference.py --model releases/latest/model.pt --fpv fpv.mp4 --topview top.mp4
```

## Creating a release

```bash
# Package current best.pt as a release
python make_release.py

# With custom release notes
python make_release.py --notes "Improved Jaccard from 0.06 to 0.42 after 40 epochs"

# Package a specific checkpoint
python make_release.py --checkpoint checkpoints/epoch_40.pt
```

## Version scheme

`YYYYMMDD.<7-char-git-hash>`

- Date is UTC date of packaging, not training date.
- Hash is the HEAD commit at time of release — ensures the inference code
  and weights are always paired to the same source revision.
- `index.json` records all releases with their key metrics so you can compare
  without loading weights.
