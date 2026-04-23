import torch
import numpy as np
import json
from pathlib import Path
from src.models.fusion import DualViewFusion
from src.preprocessing.video_loader import load_video, preprocess_frame
from src.postprocessing.output_formatter import logits_to_wells_adaptive, logits_to_wells
from src.utils.metrics import exact_match, jaccard_similarity, cardinality_accuracy

device = 'cpu'
checkpoint = torch.load('checkpoints/best.pt', weights_only=False)

# Load model config from checkpoint; fall back to v6 defaults if not present
cfg = checkpoint.get('model_config', {})
model = DualViewFusion(
    num_rows=cfg.get('num_rows', 8),
    num_columns=cfg.get('num_columns', 12),
    shared_backbone=cfg.get('shared_backbone', True),
    use_lora=cfg.get('use_lora', True),
    lora_rank=cfg.get('lora_rank', 4),
    temporal_layers=cfg.get('temporal_layers', 1),
    use_dinov2=cfg.get('use_dinov2', True),
    img_size=cfg.get('img_size', 448),
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")
print(f"Arch: lora_rank={cfg.get('lora_rank', 4)}, temporal_layers={cfg.get('temporal_layers', 1)}")

# Load labels
labels = json.load(open('data/pipette_well_dataset/labels.json'))
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(path, n=4):
    frames = load_video(path, max_frames=n)
    proc = np.array([preprocess_frame(f, size=(448, 448)) for f in frames])
    proc = (proc - mean) / std
    t = torch.from_numpy(proc.transpose(0,3,1,2)).float()
    return t.unsqueeze(0)

# Threshold sweep
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
results = {t: {'exact': [], 'jaccard': [], 'card': []} for t in thresholds + ['adaptive']}

# Per-clip type breakdown
type_results = {'single': {'adaptive': {'exact': [], 'jaccard': []}},
                'row':    {'adaptive': {'exact': [], 'jaccard': []}},
                'col':    {'adaptive': {'exact': [], 'jaccard': []}}}

data_dir = Path('data/pipette_well_dataset')
processed = 0
skipped = 0

# Collect raw sigmoid stats across all clips
all_row_max = []
all_col_max = []
all_row_2nd = []
all_col_2nd = []

for clip in labels[:20]:   # approximate val set (last 20)
    fpv_key = clip.get('clip_id_FPV') or clip.get('fpv_clip_id') or clip.get('clip_id')
    top_key = clip.get('clip_id_Topview') or clip.get('topview_clip_id') or clip.get('top_clip_id')
    if fpv_key is None or top_key is None:
        print(f"WARNING: could not find clip id keys in: {list(clip.keys())}")
        break

    fpv_path = data_dir / f"{fpv_key}.mp4"
    top_path = data_dir / f"{top_key}.mp4"
    if not fpv_path.exists():
        skipped += 1
        continue

    fpv_t = preprocess(str(fpv_path))
    top_t = preprocess(str(top_path))

    with torch.no_grad():
        row_logits, col_logits = model(fpv_t, top_t)

    row_arr = row_logits.squeeze(0).numpy()
    col_arr = col_logits.squeeze(0).numpy()
    row_sig = 1 / (1 + np.exp(-row_arr))
    col_sig = 1 / (1 + np.exp(-col_arr))

    gt_wells = [{'well_row': w['well_row'], 'well_column': int(w['well_column'])}
                for w in clip['wells_ground_truth']]
    n_gt = len(gt_wells)

    # Collect sigmoid stats
    sorted_row = np.sort(row_sig)[::-1]
    sorted_col = np.sort(col_sig)[::-1]
    all_row_max.append(sorted_row[0])
    all_col_max.append(sorted_col[0])
    all_row_2nd.append(sorted_row[1] if len(sorted_row) > 1 else 0)
    all_col_2nd.append(sorted_col[1] if len(sorted_col) > 1 else 0)

    for t in thresholds:
        pred = logits_to_wells(row_arr, col_arr, threshold=t)
        results[t]['exact'].append(exact_match(pred, gt_wells))
        results[t]['jaccard'].append(jaccard_similarity(pred, gt_wells))
        results[t]['card'].append(cardinality_accuracy(pred, gt_wells))

    adaptive_pred = logits_to_wells_adaptive(row_arr, col_arr)
    em = exact_match(adaptive_pred, gt_wells)
    jc = jaccard_similarity(adaptive_pred, gt_wells)
    results['adaptive']['exact'].append(em)
    results['adaptive']['jaccard'].append(jc)
    results['adaptive']['card'].append(cardinality_accuracy(adaptive_pred, gt_wells))

    clip_type = 'single' if n_gt == 1 else ('row' if n_gt == 8 else 'col')
    type_results[clip_type]['adaptive']['exact'].append(em)
    type_results[clip_type]['adaptive']['jaccard'].append(jc)

    processed += 1

print(f"\nProcessed {processed} clips, skipped {skipped}")
print(f"\n{'Method':<12} {'Exact%':>8} {'Jaccard':>9} {'Card%':>8}")
for key in thresholds + ['adaptive']:
    r = results[key]
    if not r['exact']:
        print(f"{str(key):<12}  (no data)")
        continue
    print(f"{str(key):<12} {np.mean(r['exact'])*100:>7.1f}% "
          f"{np.mean(r['jaccard']):>9.4f} {np.mean(r['card'])*100:>7.1f}%")

print("\n--- Adaptive breakdown by clip type ---")
for ctype, data in type_results.items():
    ex = data['adaptive']['exact']
    jc = data['adaptive']['jaccard']
    n = len(ex)
    if n == 0:
        print(f"  {ctype:<8}: no clips")
    else:
        print(f"  {ctype:<8}: n={n}  exact={np.mean(ex)*100:.1f}%  jaccard={np.mean(jc):.4f}")

print("\n--- Sigmoid distribution (approximate val set) ---")
print(f"Row max sigmoid:  mean={np.mean(all_row_max):.4f}  min={np.min(all_row_max):.4f}  max={np.max(all_row_max):.4f}")
print(f"Row 2nd sigmoid:  mean={np.mean(all_row_2nd):.4f}  min={np.min(all_row_2nd):.4f}  max={np.max(all_row_2nd):.4f}")
print(f"Col max sigmoid:  mean={np.mean(all_col_max):.4f}  min={np.min(all_col_max):.4f}  max={np.max(all_col_max):.4f}")
print(f"Col 2nd sigmoid:  mean={np.mean(all_col_2nd):.4f}  min={np.min(all_col_2nd):.4f}  max={np.max(all_col_2nd):.4f}")
print(f"\nMax/2nd ratio (row): {np.mean(np.array(all_row_max)/np.maximum(np.array(all_row_2nd), 1e-6)):.2f}x")
print(f"Max/2nd ratio (col): {np.mean(np.array(all_col_max)/np.maximum(np.array(all_col_2nd), 1e-6)):.2f}x")
