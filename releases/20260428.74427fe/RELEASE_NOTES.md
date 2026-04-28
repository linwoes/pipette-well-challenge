# Release 20260428.74427fe

**Packaged:** 2026-04-28T19:24:47.947987+00:00
**Commit:** `74427fe3abfd1224b23d108f7c7dfd83e29ff7ff` (main)
**Message:** feat: enable synthetic data by default (USE_SYNTHETIC=1)
**WARNING:** repository had uncommitted changes at time of packaging.

## Metrics (val set)

| Metric | Value |
|---|---|
| Jaccard | 0.0000 |
| Exact Match | 0.0000 |
| Val Loss | 0.4992 |
| Cardinality Acc | 1.0000 |
| Epoch | 4 |

## Model config

| Parameter | Value |
|---|---|
| img_size | 448 |
| lora_rank | 4 |
| temporal_layers | 1 |
| shared_backbone | True |
| use_lora | True |

## Notes

518 run, epoch 43, val_loss Y

## Usage

```bash
git checkout 74427fe3abfd1224b23d108f7c7dfd83e29ff7ff
python inference.py --model releases/20260428.74427fe/model.pt --fpv fpv.mp4 --topview top.mp4
```
