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
model = DualViewFusion(num_rows=8, num_columns=12, use_dinov2=True, img_size=448)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

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

data_dir = Path('data/pipette_well_dataset')
processed = 0
skipped = 0
for clip in labels[:20]:   # val set approximate
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
    gt_wells = [{'well_row': w['well_row'], 'well_column': int(w['well_column'])}
                for w in clip['wells_ground_truth']]

    for t in thresholds:
        pred = logits_to_wells(row_arr, col_arr, threshold=t)
        results[t]['exact'].append(exact_match(pred, gt_wells))
        results[t]['jaccard'].append(jaccard_similarity(pred, gt_wells))
        results[t]['card'].append(cardinality_accuracy(pred, gt_wells))

    adaptive_pred = logits_to_wells_adaptive(row_arr, col_arr)
    results['adaptive']['exact'].append(exact_match(adaptive_pred, gt_wells))
    results['adaptive']['jaccard'].append(jaccard_similarity(adaptive_pred, gt_wells))
    results['adaptive']['card'].append(cardinality_accuracy(adaptive_pred, gt_wells))
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

# Also print raw sigmoid ranges to diagnose threshold calibration
print("\n--- Logit diagnostics (last sample) ---")
print(f"Row logits (sigmoid): min={1/(1+np.exp(-row_arr)).min():.4f}  max={1/(1+np.exp(-row_arr)).max():.4f}  mean={1/(1+np.exp(-row_arr)).mean():.4f}")
print(f"Col logits (sigmoid): min={1/(1+np.exp(-col_arr)).min():.4f}  max={1/(1+np.exp(-col_arr)).max():.4f}  mean={1/(1+np.exp(-col_arr)).mean():.4f}")
