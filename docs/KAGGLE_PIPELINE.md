# Kaggle deployment pipeline

End-to-end flow: **local dev → GitHub → Kaggle (T4 GPU) → checkpoint back**.

```
┌──────────────┐  git push   ┌──────────────┐  push notebook   ┌────────────┐
│  local dev   │ ──────────► │   GitHub     │ ───────────────► │   Kaggle   │
│              │             │              │  (Action or CLI) │  T4 GPU    │
│  edit code   │             │              │                  │  runs      │
└──────────────┘             └──────────────┘                  └────────────┘
       ▲                                                              │
       │                kaggle kernels output                          │
       └──────────────────────────────────────────────────────────────┘
                       (pull best.pt back)
```

## Why this exists

CPU training runs at ~4.3 hours/epoch on the current laptop. A Kaggle T4 brings that to ~10–15 minutes/epoch. The free tier gives 30 GPU hours/week — enough for one full 80-epoch run with patience-based early stopping.

## One-time setup

### 1. Local Kaggle CLI

```bash
pip install kaggle
# Sign in at kaggle.com → Account → "Create New API Token" → downloads kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Verify: `kaggle datasets list -m` should print without errors.

### 2. GitHub Actions secrets (for auto-deploy)

In the GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**

| Secret | Value |
|---|---|
| `KAGGLE_USERNAME` | your Kaggle username |
| `KAGGLE_KEY` | the `key` field from `kaggle.json` |

Optionally, in **Variables**, set non-default slugs:

| Variable | Default | Purpose |
|---|---|---|
| `KAGGLE_KERNEL_SLUG` | `pipette-well-training` | Kaggle kernel ID slug |
| `KAGGLE_DATA_SLUG` | `pipette-well-dataset` | Data dataset slug |
| `KAGGLE_CHECKPOINT_SLUG` | `pipette-well-checkpoint` | Resume checkpoint slug; set empty to disable resume |

### 3. Push the data dataset (one-time, ~1.4 GB)

```bash
bash scripts/kaggle_push_data.sh
```

Creates `<your-username>/pipette-well-dataset` on Kaggle. Re-run only when the dataset changes (e.g. after re-running `generate_synthetic_data.py`).

## Day-to-day use

### Standard loop (laptop)

```bash
# 1. Make code changes locally
git add ... && git commit -m '...' && git push

# 2. Push the notebook + queue a run (single command)
bash scripts/kaggle_run.sh
```

`kaggle_run.sh`:
1. Pushes the notebook (which queues a Kaggle run by default)
2. Polls kernel status every 60s
3. Pulls the kernel output (best.pt + log) to `checkpoints/kaggle/`

If the run takes longer than the polling timeout (default 12 h, the Kaggle session limit), the script exits with status 3. The Kaggle run keeps going; pull later with:

```bash
bash scripts/kaggle_pull_checkpoint.sh
```

### Hands-off loop (GitHub Actions)

When you push a change to `notebooks/kaggle_train.ipynb` on `main`, the **Deploy notebook to Kaggle** workflow runs automatically and updates your Kaggle kernel. By default `kaggle kernels push` also queues a new run — if that's not what you want for a particular update, cancel from the Kaggle UI.

To trigger from GitHub: **Actions → Deploy notebook to Kaggle → Run workflow**.

### Promoting a Kaggle checkpoint to a release

After `kaggle_pull_checkpoint.sh` finishes:

```bash
# Inspect what came back
ls -la checkpoints/kaggle/
# Replace local best.pt
cp checkpoints/kaggle/checkpoints/best.pt checkpoints/best.pt
# Tag and deploy
python make_release.py \
  --notes "Kaggle T4 run, jaccard 0.42, epoch 38" \
  --deploy
```

## Scripts reference

| Script | Purpose |
|---|---|
| `scripts/_kaggle_common.sh` | Shared helpers (CLI check, username resolution). Sourced, not run directly. |
| `scripts/kaggle_push_data.sh` | Create or version the Kaggle Dataset of videos + labels. |
| `scripts/kaggle_push_notebook.sh` | Update the kernel code on Kaggle. |
| `scripts/kaggle_pull_checkpoint.sh` | Download whatever the latest kernel run produced in `/kaggle/working/`. |
| `scripts/kaggle_run.sh` | Push notebook + queue run + poll until done + pull output. The closed-loop entry point. |

All scripts respect these env overrides:

```bash
KAGGLE_USERNAME           # default: from ~/.kaggle/kaggle.json
KAGGLE_KERNEL_SLUG        # default: pipette-well-training
KAGGLE_DATA_SLUG          # default: pipette-well-dataset
KAGGLE_CHECKPOINT_SLUG    # default: pipette-well-checkpoint (empty = no resume)
```

## What lives where on Kaggle

| Kaggle resource | Slug (default) | Created by | Updated by |
|---|---|---|---|
| Dataset (videos + labels) | `<user>/pipette-well-dataset` | `kaggle_push_data.sh` | re-run the same script |
| Dataset (resume checkpoint) | `<user>/pipette-well-checkpoint` | manual: `kaggle datasets create` from a folder containing `best.pt` | `kaggle datasets version` or re-create |
| Kernel (notebook) | `<user>/pipette-well-training` | `kaggle_push_notebook.sh` | same script (or GH Action) |

## Common failure modes

- **`AcceleratorError: CUDA error: no kernel image is available for execution on the device`** — Kaggle assigned a **P100 GPU** (compute capability sm_60), but the default PyTorch image is compiled for sm_70+. **Fix**: open the kernel on the Kaggle UI → Settings panel on the right → Accelerator → switch to **GPU T4 x2** → re-run all cells. The notebook's smoke test 1 now detects this case and aborts with the same instruction before training starts.
- **"Kaggle credentials not found"** — `~/.kaggle/kaggle.json` missing or wrong perms. `chmod 600` it.
- **"Dataset not found"** during notebook run — the notebook auto-detects datasets by walking `/kaggle/input/`, but you must attach them in the Kaggle UI: *Notebook → Add data → Your Datasets*. Once attached, `notebooks/kaggle_train.ipynb` finds them automatically.
- **Smoke tests fail before training starts** — that's the point. Read which test failed; fix locally; push to GitHub; the Action redeploys; re-run on Kaggle.
- **`kaggle kernels push` errors with "Invalid dataset_sources"** — the dataset slugs in `kernel-metadata.json` must point at datasets that actually exist for your account. Run `kaggle_push_data.sh` first.
- **Session timeout (12 h)** — checkpoint at `/kaggle/working/checkpoints/best.pt` is preserved. Pull it, save it back as the resume dataset (or attach it as `pipette-well-checkpoint`), and start a new session.

## Cost / quota

| Tier | GPU hours / week | Session limit |
|---|---|---|
| Kaggle Free | 30 | 12 h |
| Kaggle GPU+ | 30 (T4 x2 / P100) | 12 h |

A full v11 run from scratch is ~80 epochs × ~12 min = ~16 GPU hours. With a resume from the v10 best.pt (which already has the type head trained perfectly), it's typically much shorter.
