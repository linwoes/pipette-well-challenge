# Model Releases

## Version scheme

All releases and training runs use the format **`YYYYMMDD.<7-char-git-hash>`** — sortable by date, pinned to the exact source commit.

Examples: `20260425.edb3173`, `20260512.a3f91bc`

This scheme applies everywhere:
- `releases/<version>/` — packaged model release
- `training_results/training_<version>.log` — training run log
- `releases/index.json` — release registry

## Directory structure

```
releases/
  index.json                  # registry of all releases with metrics
  latest    -> YYYYMMDD.<hash>/  # most recently packaged release
  deployed  -> YYYYMMDD.<hash>/  # highest-performing validated release
  YYYYMMDD.<hash>/
    model.pt                  # model weights (DualViewFusion checkpoint, via Git LFS)
    config.json               # model config, metrics, git provenance
    RELEASE_NOTES.md          # what changed, known limitations
```

## Symlinks

| Symlink | Meaning |
|---|---|
| `latest` | Most recently packaged release. Updated automatically by `make_release.py`. |
| `deployed` | Highest-performing release that has been validated and approved for use. Updated manually with `--deploy` or `--deploy-only`. |

Use `deployed` in production — it only advances when a human has validated the model:

```bash
python inference.py --model releases/deployed/model.pt --fpv fpv.mp4 --topview top.mp4
```

## Using a release

```bash
# Clone and pin to the exact commit this release was built from
git clone https://github.com/linwoes/pipette-well-challenge
git checkout <hash>    # hash is in releases/<version>/config.json

# Run inference against a specific release
python inference.py --model releases/<version>/model.pt --fpv fpv.mp4 --topview top.mp4

# Or always use the deployed model
python inference.py --model releases/deployed/model.pt --fpv fpv.mp4 --topview top.mp4
```

Git LFS is used for `model.pt` files. Run `git lfs pull` after cloning to download weights.

## Creating a release

```bash
# Package current best.pt (updates releases/latest)
python make_release.py --notes "Jaccard improved to 0.42 after 40 epochs"

# Package and immediately mark as deployed (updates both symlinks)
python make_release.py --notes "Validated on held-out set" --deploy

# Mark an existing release as deployed without re-packaging
python make_release.py --deploy-only 20260512.a3f91bc

# Preview without writing
python make_release.py --dry-run
```

## index.json

`releases/index.json` is a flat registry of all releases, newest first. Each entry includes:

```json
{
  "version": "20260425.edb3173",
  "packaged_at": "2026-04-25T19:36:43Z",
  "git_hash": "edb3173",
  "epoch": 4,
  "metrics": { "jaccard": 0.0607, "val_loss": 0.6498, ... },
  "deployed": false,
  "notes": "..."
}
```

Exactly one entry will have `"deployed": true` at any time.
