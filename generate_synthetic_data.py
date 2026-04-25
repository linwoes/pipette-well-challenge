#!/usr/bin/env python3
"""
Synthetic data generator for pipette well detection.

Generates augmented video pairs from real clips using ffmpeg filters.

Augmentation strategies
-----------------------
Photometric (labels unchanged):
  bright    — +10% brightness, +10% saturation
  dark      — -10% brightness, -10% contrast
  noise     — per-frame Gaussian noise (sigma ~12 DN)
  contrast  — +30% contrast, +20% saturation

Geometric (labels remapped):
  flip_h    — horizontal flip: col X → col (13-X)
  flip_v    — vertical flip:   row → mirror row (A↔H, B↔G, C↔F, D↔E)
  flip_hv   — both flips combined

Output
------
  data/pipette_well_dataset/Synth_*_FPV.mp4
  data/pipette_well_dataset/Synth_*_Topview.mp4
  data/pipette_well_dataset/labels_synthetic.json
  data/pipette_well_dataset/labels_combined.json

Usage
-----
  python3 generate_synthetic_data.py                   # all augmentations
  python3 generate_synthetic_data.py --dry-run         # plan only
  python3 generate_synthetic_data.py --aug flip_h flip_v  # subset
  python3 generate_synthetic_data.py --skip-existing   # resume interrupted run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / 'data' / 'pipette_well_dataset'
LABELS_PATH = DATA_DIR / 'labels.json'
FFMPEG = '/opt/homebrew/bin/ffmpeg'

# ── Label remapping ────────────────────────────────────────────────────────
ROW_MIRROR: Dict[str, str] = {
    'A': 'H', 'B': 'G', 'C': 'F', 'D': 'E',
    'E': 'D', 'F': 'C', 'G': 'B', 'H': 'A',
}


def remap_hflip(wells: List[Dict]) -> List[Dict]:
    """Horizontal flip: col X → col (13-X). Rows unchanged."""
    return [{'well_row': w['well_row'],
             'well_column': str(13 - int(w['well_column']))} for w in wells]


def remap_vflip(wells: List[Dict]) -> List[Dict]:
    """Vertical flip: row → mirror row (A↔H, B↔G, ...). Cols unchanged."""
    return [{'well_row': ROW_MIRROR[w['well_row']],
             'well_column': w['well_column']} for w in wells]


def remap_hvflip(wells: List[Dict]) -> List[Dict]:
    return remap_vflip(remap_hflip(wells))


# ── Augmentation definitions ───────────────────────────────────────────────
AUGMENTATIONS = [
    {
        'name': 'bright',
        'vfilter': 'eq=brightness=0.10:saturation=1.1',
        'remap': None,
        'description': '+10% brightness, +10% saturation',
    },
    {
        'name': 'dark',
        'vfilter': 'eq=brightness=-0.10:contrast=0.9',
        'remap': None,
        'description': '-10% brightness, -10% contrast',
    },
    {
        'name': 'noise',
        'vfilter': 'noise=alls=12:allf=t',
        'remap': None,
        'description': 'Temporal Gaussian noise (sigma ~12 DN)',
    },
    {
        'name': 'contrast',
        'vfilter': 'eq=contrast=1.3:saturation=1.2',
        'remap': None,
        'description': '+30% contrast, +20% saturation',
    },
    {
        'name': 'flip_h',
        'vfilter': 'hflip',
        'remap': remap_hflip,
        'description': 'Horizontal flip — col X → col (13-X)',
    },
    {
        'name': 'flip_v',
        'vfilter': 'vflip',
        'remap': remap_vflip,
        'description': 'Vertical flip — row → mirror row',
    },
    {
        'name': 'flip_hv',
        'vfilter': 'hflip,vflip',
        'remap': remap_hvflip,
        'description': 'Horizontal + vertical flip',
    },
]

AUG_BY_NAME = {a['name']: a for a in AUGMENTATIONS}


# ── Core helpers ───────────────────────────────────────────────────────────

def synth_stem(original_stem: str, aug_name: str) -> str:
    """
    Build a synthetic clip stem from original stem + aug name.

    'Plate_10_clip_0001_FPV'  + 'bright' → 'Synth_Plate_10_clip_0001_bright_FPV'
    'Plate_10_clip_0001_Topview' + 'bright' → 'Synth_Plate_10_clip_0001_bright_Topview'
    """
    # Split off the trailing _FPV or _Topview suffix
    parts = original_stem.rsplit('_', 1)   # ['Plate_10_clip_0001', 'FPV']
    base, view = parts[0], parts[1]
    return f'Synth_{base}_{aug_name}_{view}'


def run_ffmpeg(src: Path, dst: Path, vfilter: str, dry_run: bool) -> bool:
    cmd = [
        FFMPEG, '-y', '-loglevel', 'error',
        '-i', str(src),
        '-vf', vfilter,
        '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
        '-an',
        str(dst),
    ]
    if dry_run:
        print(f'    [dry] ffmpeg -i {src.name} -vf "{vfilter}" → {dst.name}')
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'    ERROR: {result.stderr[-400:]}', file=sys.stderr)
        return False
    return True


def verify_remap_roundtrip():
    """Sanity-check: two hflips → identity."""
    wells = [{'well_row': 'C', 'well_column': '3'}, {'well_row': 'A', 'well_column': '12'}]
    assert remap_hflip(remap_hflip(wells)) == wells, 'hflip roundtrip failed'
    assert remap_vflip(remap_vflip(wells)) == wells, 'vflip roundtrip failed'
    assert remap_hvflip(remap_hvflip(wells)) == wells, 'hvflip roundtrip failed'


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training clips')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan without writing any files')
    parser.add_argument('--aug', nargs='+', metavar='NAME',
                        choices=list(AUG_BY_NAME.keys()),
                        help='Run only these augmentation(s)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip output pairs that already exist (default: on)')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                        help='Regenerate even if outputs already exist')
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    parser.add_argument('--labels', default=str(LABELS_PATH))
    args = parser.parse_args()

    verify_remap_roundtrip()

    data_dir = Path(args.data_dir)
    labels_path = Path(args.labels)
    synthetic_labels_path = data_dir / 'labels_synthetic.json'
    combined_labels_path = data_dir / 'labels_combined.json'

    real_labels: List[Dict] = json.load(open(labels_path))

    augs = [AUG_BY_NAME[n] for n in args.aug] if args.aug else AUGMENTATIONS

    total_synthetic = len(real_labels) * len(augs)
    print(f'Real clips      : {len(real_labels)}')
    print(f'Augmentations   : {[a["name"] for a in augs]}')
    print(f'Synthetic target: {total_synthetic}')
    print(f'Combined total  : {len(real_labels) + total_synthetic}')
    print()

    synthetic_labels: List[Dict] = []
    n_generated = n_skipped = n_errors = 0

    for clip_idx, clip in enumerate(real_labels):
        fpv_src = data_dir / f'{clip["clip_id_FPV"]}.mp4'
        top_src = data_dir / f'{clip["clip_id_Topview"]}.mp4'

        if not fpv_src.exists() or not top_src.exists():
            print(f'[{clip_idx+1:3d}/{len(real_labels)}] MISSING source — skipping '
                  f'{clip["clip_id_FPV"]}')
            continue

        for aug in augs:
            fpv_id = synth_stem(clip['clip_id_FPV'], aug['name'])
            top_id = synth_stem(clip['clip_id_Topview'], aug['name'])
            fpv_dst = data_dir / f'{fpv_id}.mp4'
            top_dst = data_dir / f'{top_id}.mp4'

            if args.skip_existing and fpv_dst.exists() and top_dst.exists():
                n_skipped += 1
                remapped = aug['remap'](clip['wells_ground_truth']) if aug['remap'] else clip['wells_ground_truth']
                synthetic_labels.append({
                    'clip_id_FPV': fpv_id,
                    'clip_id_Topview': top_id,
                    'wells_ground_truth': remapped,
                    'synthetic': True,
                    'aug_type': aug['name'],
                    'source_fpv': clip['clip_id_FPV'],
                })
                continue

            print(f'[{clip_idx+1:3d}/{len(real_labels)}] {aug["name"]:10s}  '
                  f'{clip["clip_id_FPV"]}')

            ok_fpv = run_ffmpeg(fpv_src, fpv_dst, aug['vfilter'], args.dry_run)
            ok_top = run_ffmpeg(top_src, top_dst, aug['vfilter'], args.dry_run)

            if ok_fpv and ok_top:
                remapped = (aug['remap'](clip['wells_ground_truth'])
                            if aug['remap'] else clip['wells_ground_truth'])
                synthetic_labels.append({
                    'clip_id_FPV': fpv_id,
                    'clip_id_Topview': top_id,
                    'wells_ground_truth': remapped,
                    'synthetic': True,
                    'aug_type': aug['name'],
                    'source_fpv': clip['clip_id_FPV'],
                })
                n_generated += 1
            else:
                n_errors += 1
                fpv_dst.unlink(missing_ok=True)
                top_dst.unlink(missing_ok=True)

    print()
    print(f'Generated : {n_generated}')
    print(f'Skipped   : {n_skipped}  (already existed)')
    print(f'Errors    : {n_errors}')

    if args.dry_run:
        print('\n[dry-run] No files written.')
        return

    if not synthetic_labels:
        print('No synthetic labels to write.')
        return

    with open(synthetic_labels_path, 'w') as f:
        json.dump(synthetic_labels, f, indent=2)
    print(f'\nWrote {synthetic_labels_path}  ({len(synthetic_labels)} entries)')

    combined = real_labels + synthetic_labels
    with open(combined_labels_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f'Wrote {combined_labels_path}  ({len(combined)} total entries)')
    print()
    print('To train on combined dataset:')
    print(f'  LABELS={combined_labels_path} bash run_training.sh')


if __name__ == '__main__':
    main()
