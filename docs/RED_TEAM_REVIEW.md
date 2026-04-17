# Red Team Technical Review — Pipette Well Detection Stack

**Date:** 2026-04-16  
**Initiated by:** PM  
**Review scope:** All artifacts in the public repository — source code, documentation, test suite, training logs, and inference outputs. No access to the development team was granted or used.  
**Status:** FINAL

---

## PM Framing

This review was commissioned because the system has completed two full training runs (v2, v3) and still shows 0% exact match on every validation epoch. Before authorising a v4 training run, I needed an independent assessment of whether the root cause is data volume, architecture, or implementation defects — and whether the fixes applied after v3 QA actually resolve the problem.

The red team was given no briefings, no access to the development team, and no explanations beyond the repo itself. Their charter: find anything that could plausibly cause or mask a failure to learn. Each reviewer worked from their own domain lens.

---

## Reviewers

| Pseudonym | Level | Role |
|-----------|-------|------|
| **Lovelace** | Staff | ML Researcher — model architecture, loss function correctness, training dynamics |
| **Turing** | Senior | Data Engineer — data pipeline, label schema, preprocessing assumptions |
| **Shannon** | Senior | MLOps Engineer — inference pipeline, production readiness, failure modes |
| **Hopper** | Mid | QA Engineer — test coverage, evaluation methodology, metric validity |

---

## PM Summary

The development team correctly identified and patched three bugs after v3 (focal loss alpha, loss/metric disconnect, threshold collapse). Those fixes are real and necessary. However, this review found that they are not sufficient. There are five additional issues that will prevent meaningful exact match even with the bug fixes applied, and two of them are severe enough to corrupt every training run that has happened so far.

**The single most damaging finding** (Lovelace, Finding L-1): the system documented as implementing "Late Fusion (Mandated)" actually implements feature-level concatenation. These are architecturally different things. The docs describe a constraint the code does not satisfy, which means neither the team nor any external evaluator can reason about what the model actually does from the documentation alone.

**The second most damaging finding** (Shannon, Finding S-1): the DINOv2 backbone silently falls back to ResNet18 with no dimension guard. This has already caused a checkpoint loading failure in production (described in the v3 QA report). The fix applied was downstream workaround, not upstream prevention.

**Priority order for v4:**

1. Resolve the late fusion vs. concatenation discrepancy — decide which is mandated, make code match docs.
2. Add a hard dimension guard to the backbone fallback path before any training runs.
3. Add the overfitting smoke test (5 examples, 50 epochs) as a CI gate before any full training run.
4. Fix `configs/default.yaml` to reflect the actual production configuration.
5. Address the `jaccard_similarity(∅, ∅) = 1.0` metric bug.

---

## Lovelace — Staff ML Researcher

*Scope: `src/models/fusion.py`, `src/models/backbone.py`, `train.py`, training logs*

### L-1 · CRITICAL — Late fusion mandate violated

`ARCHITECTURE.md` designates late fusion as mandated architecture. The code in `fusion.py` implements simple feature concatenation:

```python
# fusion.py line 284
fused = torch.cat([fpv_temporal, topview_temporal], dim=1)
```

Late fusion means each view produces independent logits which are then combined. Concatenation is mid-level feature fusion. These are not equivalent. The factorized row/column heads downstream of the concatenation cannot recover view-independent predictions. This is not a naming issue — it materially affects what the gradient signal looks like and how view disagreement is handled. Every training decision justified by the late fusion constraint (e.g., independent view confidence) is being applied to a model that doesn't satisfy it.

**Action:** Decide. If concatenation is the chosen design, update `ARCHITECTURE.md` and remove the "Mandated" designation. If late fusion is genuinely required, implement it: give each view its own row/column heads, combine logits with a learned or fixed fusion gate.

---

### L-2 · HIGH — Backbone fallback silently corrupts temporal dimension

In `backbone.py`, if DINOv2 fails to load (torch.hub timeout, timm version change, network restriction), the code sets `self.use_fallback = True` and uses a ResNet18 that produces 512-dim features. The temporal transformer and fusion MLP are initialised with 768-dim expectations. The fallback path resets `self.temporal_dim = 512` — but this reassignment happens *after* the temporal modules are already constructed.

The v3 QA report describes exactly this failure occurring in production. The downstream fix (auto-detecting backbone type from checkpoint keys) is a workaround. The root cause — constructing modules before confirming backbone output dimension — is still in the code. The next time DINOv2 loads successfully, the checkpoint will be 768-dim, and the workaround will select DINOv2 initialization and load cleanly. But if DINOv2 fails mid-run on a different machine, the training produces a 512-dim checkpoint that is silently incompatible with any other environment where DINOv2 loads.

**Action:** Move the `temporal_dim` decision to the top of `__init__`, before any module construction. Add a hard assertion after backbone load: `assert self.backbone_fpv.output_dim == self.temporal_dim`.

---

### L-3 · HIGH — Well consistency loss assumes factorized ground truth

The new well consistency loss (added in the v3 fix commit) computes:

```python
gt_wells = torch.bmm(row_targets.unsqueeze(2).float(),
                     col_targets.unsqueeze(1).float())  # (B, 8, 12)
```

This generates the Cartesian product of row and column targets and treats it as ground truth for the 8×12 well grid. This is only correct if the real wells are exactly the Cartesian product of their row and column marginals — i.e., if every combination of active rows and active columns is an active well.

Looking at `data/synthetic_val/labels.json`, the multi-well scatter entries (clips 013, 016, 022) include wells like `(C,3), (F,5), (G,7), (H,10)` — spanning four different rows and four different columns. The Cartesian product of those rows and columns is 16 wells, not 4. The consistency loss will penalise the model for *not* predicting the 12 phantom wells that are in the product but not in the ground truth. This is a direct incorrect supervision signal for any non-rectangular well pattern.

**Action:** The consistency loss is only valid for rectangular well patterns (full rows, full columns, single wells). Add a per-sample guard: if the ground truth is not a Cartesian product of its row/col marginals, set `well_consistency_weight = 0` for that sample. Or drop the consistency loss and rely on the corrected focal loss alone.

---

### L-4 · MEDIUM — Training uses alpha=0.75 but loss default is alpha=0.25

`train.py` line 497 documents that `focal_alpha` defaults to 0.75. The `WellDetectionLoss.__init__` in `fusion.py` defaults to 0.25. These are different. The comment in `fusion.py` line 304 still references 0.25 as the "Focal Loss paper" default. 

For a plate where ~2 wells are active out of 96, the class imbalance ratio is roughly 47:1. At alpha=0.75, the positive-class weight ratio is 0.75/0.25 = 3:1 in favour of positives — this partially corrects imbalance but does not fully compensate for 47:1. At alpha=0.25, positives are actually downweighted relative to negatives (0.25/0.75 = 0.33:1), which is the wrong direction.

The training script uses 0.75 (correct direction), but the default in the loss class is 0.25 (wrong direction). Any code that instantiates `WellDetectionLoss()` directly, without passing alpha, will silently use the wrong value.

**Action:** Set the default in `WellDetectionLoss.__init__` to 0.75 to match training intent. Remove the comment referencing the paper default as it is misleading in this context.

---

## Turing — Senior Data Engineer

*Scope: `data/`, `src/preprocessing/video_loader.py`, label schema, `data/synthetic_val/labels.json`*

### T-1 · CRITICAL — Real dataset is a Git LFS pointer, not data

`data/pipette_well_dataset.tar.gz` is 134 bytes on disk — a Git LFS pointer, not the actual archive. The `data/README.md` and `data/download_dataset.py` exist, but the download script attempts to pull from GitHub LFS. In the sandbox environment this is blocked. 

Every training run documented (v2, v3) used the real data on the author's machine, but the checkpoints committed to the repo were trained in a different environment. The red team cannot reproduce any training result from this repository. More importantly, any evaluator or collaborator cloning this repo cannot train or validate anything without separately obtaining the dataset through a channel not described in the repository.

**Action:** Either replace the LFS pointer with a direct download link (cloud storage, DOI), or add explicit documentation of the data access process. The `data/README.md` should state clearly: "This file is a Git LFS pointer. If LFS is unavailable, obtain the dataset from [X]."

---

### T-2 · HIGH — Synthetic validation data is not representative

The `data/synthetic_val/` clips are programmatically generated solid-colour rectangles with no visual similarity to actual pipette-and-well-plate video. The v3 QA report correctly flags this as a limitation but still uses the synthetic metrics as the primary QA output. The inference results show 0% IoU and 0% exact match — but this is uninterpretable because the model was never exposed to this kind of input during training (whether the real training data or the synthetic training data used in the sandbox).

Evaluating a vision model trained on real lab video against synthetic colour blocks tells you nothing about model quality. It only tells you the model doesn't generalise to out-of-distribution inputs, which is expected and uninformative.

**Action:** Synthetic QA data is only valid for pipeline integration testing — confirming that videos can be loaded, tensors shaped correctly, and output JSON produced. It should never be cited as an accuracy metric. Separate the QA report into pipeline smoke tests (synthetic data OK) and model accuracy evaluation (requires real data).

---

### T-3 · HIGH — Label schema allows invalid well combinations

The `labels.json` schema stores row and column as separate string fields. There is no validation that the combination is a real well on the 96-well plate. Wells are `A–H × 1–12`. A label like `{"well_row": "I", "well_column": 13}` would pass schema validation as currently implemented. `validate_output()` in `output_formatter.py` checks that columns are 1–12 integers and rows are A–H strings — but only on *output*, not on ground truth labels loaded by the training loop.

**Action:** Add a `validate_labels()` function that runs at dataset load time and rejects any label where `well_row ∉ {A..H}` or `well_column ∉ {1..12}`. The training loop should call this before the first epoch.

---

### T-4 · MEDIUM — `frame_sampling_rate` in config is ambiguous

`configs/default.yaml` line 24: `frame_sampling_rate: 5`. It is not documented whether this means "sample every 5th frame" or "sample 5 frames total". `video_loader.py` uses `max_frames=8` as its parameter name — sampling a fixed count. The config key name implies a rate (every Nth frame), not a count.

If a 36-frame clip is sampled "every 5th frame" the result is 7 frames. If it is sampled "5 frames total" (evenly spaced) the result is 5 frames. With the temporal transformer's `pos_embed` fixed to `max_frames=8`, either interpretation can produce a sequence shorter than 8 — which is silently truncated/padded with different behaviour depending on implementation path.

**Action:** Rename the config key to `num_frames` or `frame_stride` as appropriate, add a comment, and add an assertion in `video_loader.py` that `len(frames) == num_frames` after sampling.

---

## Shannon — Senior MLOps Engineer

*Scope: `inference.py`, `tools/visualizer.py`, `run_training.sh`, `configs/default.yaml`, checkpoint management*

### S-1 · CRITICAL — `configs/default.yaml` describes a different system than what is deployed

The canonical configuration file specifies:

```yaml
model:
  backbone: "resnet18"
  input_size: [224, 224]
  fusion_type: "concatenation"

checkpoint:
  path: "models/best_checkpoint.pth"
  device: "cuda"
```

The actual inference code uses DINOv2 (falling back to ResNet18), `img_size=448`, and loads from `checkpoints/best.pt`. None of these match the config file. The config file is not loaded by `inference.py` at all — `PipetteWellDetector.__init__` takes hardcoded defaults. This means the config file is documentation that does not drive behaviour. Anyone using the config to understand or reproduce the system will form a false picture.

**Action:** Either wire `inference.py` and `train.py` to read from `configs/default.yaml`, or mark the config file clearly as a reference document with a warning that it is not loaded by the current codebase.

---

### S-2 · HIGH — No checkpoint versioning or reproducibility metadata

`checkpoints/best.pt` is overwritten each training run. The v3 QA report notes that `checkpoints/best.pt` contains epoch 4 data from a synthetic training run, not the epoch 14 checkpoint from the actual v3 training run. This means the checkpoint in the repo does not correspond to any of the training logs in `training_results/`. There is no metadata in `best.pt` linking it to a specific training run, configuration, or data version.

**Action:** Save checkpoints as `checkpoints/run_{uid}_epoch_{n}.pt`. Write a `checkpoints/manifest.json` that maps each checkpoint file to its training run ID, config snapshot, and data version. Never overwrite `best.pt` in place.

---

### S-3 · HIGH — `run_training.sh` hardcodes environment-specific paths and forces CPU

```bash
# run_training.sh
--data_dir /sessions/jolly-cool-einstein/data/pipette_well_dataset
--device cpu
```

The absolute path will fail on any machine other than the development sandbox. `--device cpu` disables GPU even if available, making a training run that took ~4 hours on CPU an unnecessary constraint. This script is the only documented way to launch training. Any collaborator running it will produce results that are not comparable to the logged v2/v3 runs due to the device difference (numerical precision differences between CPU and GPU float32 can affect weight initialisation and convergence).

**Action:** Replace hardcoded path with `${DATA_DIR:-data/pipette_well_dataset}`. Replace `--device cpu` with auto-detection. Add a `--dry-run` flag that prints resolved config without starting training.

---

### S-3 · MEDIUM — No graceful handling of inference failure mid-batch

`inference.py` wraps the entire pipeline in a single try/except that re-raises on failure. If one clip in a batch fails (corrupted video, wrong frame count, filesystem error), the entire batch fails and the caller receives no partial results. For the visualizer's rank command, which processes 30 clips, a single bad clip aborts the run.

**Action:** In `PipetteWellDetector.infer_and_predict`, catch per-clip exceptions and return a structured error result (not re-raise) so the caller can continue with remaining clips.

---

## Hopper — Mid QA Engineer

*Scope: `tests/`, `src/utils/metrics.py`, `docs/QA_STRATEGY.md`, `inference_results/`*

### H-1 · HIGH — `jaccard_similarity(∅, ∅)` returns 1.0

```python
# metrics.py
def jaccard_similarity(pred_wells, gt_wells):
    ...
    if not union:
        return 1.0  # Both empty — treat as perfect match
```

When both prediction and ground truth are empty, the function returns a Jaccard score of 1.0 — "perfect match." This will inflate mean Jaccard scores in any evaluation set where the model predicts no wells and ground truth is also empty (even if that never occurs in practice, the semantic is wrong). More concretely: if the adaptive post-processor's fallback path produces an empty prediction list against a ground truth with wells, the union is non-empty and Jaccard is 0. But if some edge case causes *both* to be empty, the model gets credit for a correct prediction it did not make.

**Action:** Return `0.0` for empty union. Add a specific `null_prediction_rate` metric to flag when the model produces no wells.

---

### H-2 · HIGH — Test suite does not validate against the documented backbone

`tests/test_training_setup.py` line 75 creates a model with `backbone: "resnet18"`. `ARCHITECTURE.md` mandates DINOv2. The test suite therefore validates a configuration that is explicitly not the production configuration. A CI run that passes all tests is not evidence that the DINOv2-based system works.

**Action:** Add a separate test suite (`tests/test_dinov2_stack.py`) that validates the full DINOv2 path end-to-end, including LoRA injection, forward pass shape, and checkpoint save/load. Gate this test on network availability; skip gracefully if DINOv2 cannot be downloaded, but log a clear warning.

---

### H-3 · HIGH — No overfitting smoke test exists

The QA strategy document (`docs/QA_STRATEGY.md`) does not include a memorisation test. The most basic diagnostic for a model that shows 0% exact match across 20 epochs is: can it memorise 5 training examples in 50 epochs? If it cannot, the architecture or loss function has a fundamental bug regardless of data volume. This test does not exist anywhere in the repository.

The v3 bug fixes addressed three specific issues. Without a memorisation test, there is no way to confirm the fixes are sufficient before launching a full training run costing ~4 hours of compute.

**Action:** Add `tests/test_overfit_smoke.py`. Test: train `DualViewFusion` with `use_dinov2=False` on 5 randomly-seeded synthetic examples for 100 epochs. Assert `exact_match ≥ 0.8` at epoch 100. This test should run in under 2 minutes on CPU and should be required to pass before any full training run is authorised.

---

### H-4 · MEDIUM — Evaluation uses synthetic data but reports metric names used for real data

The `docs/QA_REPORT_v3.md` reports "Exact Match: 0 / 30 (0%)" and "Mean Jaccard: 0.000" under a heading that does not clearly distinguish these as synthetic-data results. These metric names are the same as those used in the training logs, which measure performance on real data. A reader scanning the report could interpret the 0% exact match as the model's performance on real-world data, when it is actually the model's performance on programmatically generated colour blocks that no training run has ever seen.

**Action:** Rename synthetic evaluation metrics in QA reports: "Synthetic Pipeline Exact Match" and "Synthetic Pipeline Jaccard" to make clear they measure pipeline correctness, not model accuracy.

---

### H-5 · LOW — `lfs_upload_url.txt` is committed to the repository

The file `lfs_upload_url.txt` is present in the repository root. This file appears to contain a pre-signed URL for a Git LFS upload operation. Pre-signed URLs are time-limited but represent real authenticated access to storage. Even if expired, committing any URL with embedded credentials or signatures is a security hygiene violation.

**Action:** Add `lfs_upload_url.txt` to `.gitignore` immediately. Rotate any credentials associated with the signed URL as a precaution.

---

## Consolidated Finding Register

| ID | Reviewer | Severity | File(s) | Finding |
|----|----------|----------|---------|---------|
| L-1 | Lovelace | CRITICAL | `fusion.py`, `ARCHITECTURE.md` | "Late fusion" mandate violated — code uses concatenation |
| L-2 | Lovelace | HIGH | `backbone.py` | ResNet18 fallback constructed with wrong temporal dim before guard |
| L-3 | Lovelace | HIGH | `fusion.py` | Well consistency loss produces incorrect supervision for non-rectangular well patterns |
| L-4 | Lovelace | MEDIUM | `fusion.py`, `train.py` | Alpha default mismatch: loss default 0.25, training uses 0.75 |
| T-1 | Turing | CRITICAL | `data/` | Real dataset inaccessible — LFS pointer only, no reproduction path |
| T-2 | Turing | HIGH | `data/synthetic_val/` | Synthetic QA data cited as accuracy metric; it only validates pipeline |
| T-3 | Turing | HIGH | `labels.json` schema | No validation of well coordinates against legal plate positions at load time |
| T-4 | Turing | MEDIUM | `configs/default.yaml`, `video_loader.py` | `frame_sampling_rate` semantics ambiguous; possible sequence length mismatch |
| S-1 | Shannon | CRITICAL | `configs/default.yaml`, `inference.py` | Config file describes a different system; is not loaded by inference or training |
| S-2 | Shannon | HIGH | `checkpoints/` | No checkpoint versioning; `best.pt` does not correspond to any training log |
| S-3 | Shannon | HIGH | `run_training.sh` | Hardcoded sandbox paths and forced CPU; not portable |
| S-4 | Shannon | MEDIUM | `inference.py` | No per-clip error isolation; single failure aborts full batch |
| H-1 | Hopper | HIGH | `metrics.py` | `jaccard_similarity(∅, ∅) = 1.0` — empty prediction incorrectly scored as perfect |
| H-2 | Hopper | HIGH | `tests/` | Test suite validates ResNet18 config, not DINOv2 production config |
| H-3 | Hopper | HIGH | `tests/` | No memorisation smoke test; fixes cannot be verified without full 4-hour run |
| H-4 | Hopper | MEDIUM | `docs/QA_REPORT_v3.md` | Synthetic metrics reported under same names as real-data metrics |
| H-5 | Hopper | LOW | `lfs_upload_url.txt` | Pre-signed URL committed to repository root |

---

## PM Disposition

**Block v4 training on:** L-3 (consistency loss bug) and H-3 (no smoke test). These are the only two items that will directly determine whether a v4 training run produces meaningful signal. Everything else can run in parallel with or after v4.

**Assign to next sprint:** L-1 (architecture documentation), S-1 (config file), T-2 (QA report language), H-1 (Jaccard metric), H-5 (LFS URL).

**Assign to backlog:** L-4, T-3, T-4, S-2, S-3, S-4, H-2, H-4.

**Do not block on:** T-1 (dataset access). Real data is available on the development machine. The team can train and evaluate. The documentation gap is a collaboration risk, not a model quality risk.
