#!/usr/bin/env python3
"""
Package a trained checkpoint into a versioned release.

Version format: YYYYMMDD.<7-char-git-hash>
  - Sortable by date, pinned to exact source commit.
  - Date is UTC date of packaging.

Symlinks maintained in releases/:
  latest    → most recently packaged release
  deployed  → highest-performing validated release (set with --deploy)

Usage:
  python make_release.py                            # package checkpoints/best.pt
  python make_release.py --checkpoint path/to/x.pt
  python make_release.py --notes "Reached 0.42 Jaccard after 40 epochs"
  python make_release.py --deploy                   # also mark as deployed
  python make_release.py --deploy-only <version>    # mark existing release as deployed
  python make_release.py --dry-run                  # preview without writing

Output:
  releases/<version>/model.pt
  releases/<version>/config.json
  releases/<version>/RELEASE_NOTES.md
  releases/latest    -> <version>  (always updated)
  releases/deployed  -> <version>  (only updated with --deploy or --deploy-only)
  releases/index.json              (registry updated, deployed field set)
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent
RELEASES_DIR = REPO_ROOT / 'releases'
DEFAULT_CHECKPOINT = REPO_ROOT / 'checkpoints' / 'best.pt'


def git_info() -> dict:
    def run(cmd):
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()

    try:
        return {
            'hash': run(['git', 'rev-parse', 'HEAD']),
            'short_hash': run(['git', 'rev-parse', '--short', 'HEAD']),
            'branch': run(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
            'commit_message': run(['git', 'log', '-1', '--format=%s']),
            'commit_date': run(['git', 'log', '-1', '--format=%cI']),
            'dirty': run(['git', 'status', '--porcelain']) != '',
        }
    except subprocess.CalledProcessError as e:
        print(f'WARNING: git info unavailable: {e}', file=sys.stderr)
        return {'hash': 'unknown', 'short_hash': 'unknown', 'branch': 'unknown',
                'commit_message': '', 'commit_date': '', 'dirty': False}


def make_version(short_hash: str) -> str:
    date = datetime.now(timezone.utc).strftime('%Y%m%d')
    return f'{date}.{short_hash}'


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return {
        'epoch': ckpt.get('epoch'),
        'metrics': {
            'jaccard': ckpt.get('jaccard'),
            'val_loss': ckpt.get('val_loss'),
            'exact_match': ckpt.get('exact_match'),
            'cardinality_acc': ckpt.get('cardinality_acc'),
        },
        'model_config': ckpt.get('model_config', {}),
    }


def update_symlink(name: str, version: str, dry_run: bool):
    link = RELEASES_DIR / name
    if not dry_run:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(version)
    print(f'{"[dry-run] Would update" if dry_run else "Updated  :"} releases/{name} -> {version}')


def update_index(version: str, config: dict, deployed: bool, dry_run: bool):
    index_path = RELEASES_DIR / 'index.json'
    entries = json.loads(index_path.read_text()) if index_path.exists() else []

    entry = {
        'version': version,
        'packaged_at': config['packaged_at'],
        'git_hash': config['git']['short_hash'],
        'epoch': config['checkpoint']['epoch'],
        'metrics': config['checkpoint']['metrics'],
        'deployed': deployed,
        'notes': config.get('notes', ''),
    }

    # Replace existing entry for same version, otherwise append
    existing = {e['version']: e for e in entries}
    if version in existing:
        existing[version].update(entry)
    else:
        existing[version] = entry

    # If marking as deployed, clear deployed flag on all others
    if deployed:
        for v, e in existing.items():
            if v != version:
                e['deployed'] = False

    entries = sorted(existing.values(), key=lambda e: e['version'], reverse=True)

    if not dry_run:
        index_path.write_text(json.dumps(entries, indent=2) + '\n')
    else:
        print(f'[dry-run] Would update {index_path}')


def write_release_notes(path: Path, version: str, config: dict, notes: str, dry_run: bool):
    m = config['checkpoint']['metrics']
    mc = config['checkpoint']['model_config']
    git = config['git']

    content = f"""# Release {version}

**Packaged:** {config['packaged_at']}
**Commit:** `{git['hash']}` ({git['branch']})
**Message:** {git['commit_message']}
{"**WARNING:** repository had uncommitted changes at time of packaging." if git['dirty'] else ""}

## Metrics (val set)

| Metric | Value |
|---|---|
| Jaccard | {m['jaccard']:.4f} |
| Exact Match | {m['exact_match']:.4f} |
| Val Loss | {m['val_loss']:.4f} |
| Cardinality Acc | {m['cardinality_acc']:.4f} |
| Epoch | {config['checkpoint']['epoch'] + 1} |

## Model config

| Parameter | Value |
|---|---|
| img_size | {mc.get('img_size', 448)} |
| lora_rank | {mc.get('lora_rank', 4)} |
| temporal_layers | {mc.get('temporal_layers', 1)} |
| shared_backbone | {mc.get('shared_backbone', True)} |
| use_lora | {mc.get('use_lora', True)} |

## Notes

{notes if notes else '_No release notes provided._'}

## Usage

```bash
git checkout {git['hash']}
python inference.py --model releases/{version}/model.pt --fpv fpv.mp4 --topview top.mp4
```
"""
    if not dry_run:
        path.write_text(content)
    else:
        print(f'[dry-run] Would write {path}')
        print(content)


def cmd_deploy_only(version: str, dry_run: bool):
    release_dir = RELEASES_DIR / version
    if not release_dir.exists():
        print(f'ERROR: release not found: {release_dir}', file=sys.stderr)
        sys.exit(1)

    config_path = release_dir / 'config.json'
    config = json.loads(config_path.read_text())
    m = config['checkpoint']['metrics']
    print(f'Deploying : {version}')
    print(f'Jaccard   : {m["jaccard"]:.4f}  |  Val Loss: {m["val_loss"]:.4f}')

    update_symlink('deployed', version, dry_run)
    update_index(version, config, deployed=True, dry_run=dry_run)
    if not dry_run:
        print(f'\nreleases/deployed now points to {version}')


def main():
    parser = argparse.ArgumentParser(description='Package a checkpoint as a versioned release')
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT),
                        help='Path to checkpoint file (default: checkpoints/best.pt)')
    parser.add_argument('--notes', default='',
                        help='Release notes (what changed, known issues, etc.)')
    parser.add_argument('--deploy', action='store_true',
                        help='Also update the deployed symlink to this release')
    parser.add_argument('--deploy-only', metavar='VERSION',
                        help='Mark an existing release as deployed without re-packaging')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without writing any files')
    args = parser.parse_args()

    if args.deploy_only:
        cmd_deploy_only(args.deploy_only, args.dry_run)
        return

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f'ERROR: checkpoint not found: {checkpoint_path}', file=sys.stderr)
        sys.exit(1)

    git = git_info()
    if git['dirty']:
        print('WARNING: repository has uncommitted changes — release will be marked dirty.')

    version = make_version(git['short_hash'])
    release_dir = RELEASES_DIR / version

    print(f'Version   : {version}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Output    : {release_dir}')
    print(f'Commit    : {git["hash"]} ({git["branch"]})')
    if git['dirty']:
        print('Dirty     : YES — uncommitted changes present')
    if args.deploy:
        print('Deploy    : YES — will update releases/deployed')

    ckpt_meta = load_checkpoint_metadata(checkpoint_path)
    m = ckpt_meta['metrics']
    print(f'Epoch     : {ckpt_meta["epoch"] + 1}')
    print(f'Jaccard   : {m["jaccard"]:.4f}  |  Val Loss: {m["val_loss"]:.4f}')
    print(f'Exact Match: {m["exact_match"]:.4f}  |  Cardinality Acc: {m["cardinality_acc"]:.4f}')

    config = {
        'version': version,
        'packaged_at': datetime.now(timezone.utc).isoformat(),
        'git': git,
        'checkpoint': ckpt_meta,
        'notes': args.notes,
        'inference': {
            'command': f'python inference.py --model releases/{version}/model.pt --fpv <fpv.mp4> --topview <top.mp4>',
            'threshold': 0.4,
            'img_size': ckpt_meta['model_config'].get('img_size', 448),
        },
    }

    if args.dry_run:
        print('\n[dry-run] No files written.')
        write_release_notes(release_dir / 'RELEASE_NOTES.md', version, config, args.notes, dry_run=True)
        update_index(version, config, deployed=args.deploy, dry_run=True)
        return

    release_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_path, release_dir / 'model.pt')
    print(f'\nCopied    : model.pt ({checkpoint_path.stat().st_size / 1e6:.1f} MB)')

    (release_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Wrote     : config.json')

    write_release_notes(release_dir / 'RELEASE_NOTES.md', version, config, args.notes, dry_run=False)
    print(f'Wrote     : RELEASE_NOTES.md')

    update_symlink('latest', version, dry_run=False)
    if args.deploy:
        update_symlink('deployed', version, dry_run=False)

    update_index(version, config, deployed=args.deploy, dry_run=False)
    print(f'Updated  : releases/index.json')

    print(f'\nRelease {version} ready.')
    if args.deploy:
        print(f'  releases/deployed -> {version}')
    print(f'  python inference.py --model releases/{version}/model.pt --fpv fpv.mp4 --topview top.mp4')


if __name__ == '__main__':
    main()
