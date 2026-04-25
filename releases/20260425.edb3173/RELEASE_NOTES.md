# Release 20260425.edb3173

**Packaged:** 2026-04-25T19:36:43.080499+00:00
**Commit:** `edb31732452df2fe4d866c1b2e930ea21ddea4c1` (main)
**Message:** Add versioned release system with make_release.py
**WARNING:** repository had uncommitted changes at time of packaging.

## Metrics (val set)

| Metric | Value |
|---|---|
| Jaccard | 0.0607 |
| Exact Match | 0.0000 |
| Val Loss | 0.6498 |
| Cardinality Acc | 0.3000 |
| Epoch | 5 |

## Model config

| Parameter | Value |
|---|---|
| img_size | 448 |
| lora_rank | 4 |
| temporal_layers | 1 |
| shared_backbone | True |
| use_lora | True |

## Notes

v9 epoch-5 baseline; Jaccard 0.06 — model in early spatial learning phase, used as resume seed for v10 hybrid-criterion training.

## Usage

```bash
git checkout edb31732452df2fe4d866c1b2e930ea21ddea4c1
python inference.py --model releases/20260425.edb3173/model.pt --fpv fpv.mp4 --topview top.mp4
```
