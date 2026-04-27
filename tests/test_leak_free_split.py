"""
Tests for build_leak_free_split — the function that partitions real and
synthetic labels into train/val with the guarantee that no synthetic
augmentation of a val real clip ever appears in training.

If this guarantee breaks, the model would essentially train on its
validation set, and we would get optimistic val metrics that bear no
relation to held-out performance. These tests run on every PR.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import build_leak_free_split, _assert_no_synthetic_leakage


def make_real(n: int):
    return [
        {
            'clip_id_FPV': f'Plate_1_clip_{i:04d}_FPV',
            'clip_id_Topview': f'Plate_1_clip_{i:04d}_Topview',
            'wells_ground_truth': [{'well_row': 'A', 'well_column': '1'}],
        }
        for i in range(n)
    ]


def make_synth(real_labels, augs=('bright', 'dark', 'flip_h')):
    out = []
    for r in real_labels:
        base = r['clip_id_FPV'].replace('_FPV', '')
        for aug in augs:
            out.append({
                'clip_id_FPV': f'Synth_{base}_{aug}_FPV',
                'clip_id_Topview': f'Synth_{base}_{aug}_Topview',
                'wells_ground_truth': r['wells_ground_truth'],
                'synthetic': True,
                'aug_type': aug,
                'source_fpv': r['clip_id_FPV'],
            })
    return out


def test_basic_split_sizes():
    real = make_real(100)
    synth = make_synth(real, augs=('bright', 'dark', 'flip_h'))  # 300

    train, val = build_leak_free_split(real, synth, val_split=0.2, seed=42)

    assert len(val) == 20  # 20% of real
    assert len(train) == 80 + 80 * 3  # 80 train real + 240 synth (3 augs each)


def test_no_synthetic_in_val():
    real = make_real(50)
    synth = make_synth(real)
    train, val = build_leak_free_split(real, synth, val_split=0.2, seed=0)
    assert all(not v.get('synthetic') for v in val), "val must be real-only"


def test_no_train_synthetic_derived_from_val_clip():
    """The critical leak guarantee: no train synthetic can be an
    augmentation of a val real clip."""
    real = make_real(30)
    synth = make_synth(real)
    train, val = build_leak_free_split(real, synth, val_split=0.2, seed=7)

    val_ids = {v['clip_id_FPV'] for v in val}
    for t in train:
        if t.get('synthetic'):
            assert t['source_fpv'] not in val_ids, (
                f"LEAK: train synthetic {t['clip_id_FPV']} was derived from val "
                f"clip {t['source_fpv']}"
            )


def test_no_clip_id_overlap():
    real = make_real(40)
    synth = make_synth(real)
    train, val = build_leak_free_split(real, synth, val_split=0.25, seed=3)
    train_ids = {t['clip_id_FPV'] for t in train}
    val_ids = {v['clip_id_FPV'] for v in val}
    assert not (train_ids & val_ids)


def test_deterministic_with_seed():
    real = make_real(50)
    synth = make_synth(real)
    a_train, a_val = build_leak_free_split(real, synth, val_split=0.2, seed=99)
    b_train, b_val = build_leak_free_split(real, synth, val_split=0.2, seed=99)
    assert [v['clip_id_FPV'] for v in a_val] == [v['clip_id_FPV'] for v in b_val]
    assert [t['clip_id_FPV'] for t in a_train] == [t['clip_id_FPV'] for t in b_train]


def test_different_seed_changes_split():
    real = make_real(50)
    synth = make_synth(real)
    a_val = {v['clip_id_FPV'] for v in build_leak_free_split(real, synth, 0.2, seed=1)[1]}
    b_val = {v['clip_id_FPV'] for v in build_leak_free_split(real, synth, 0.2, seed=2)[1]}
    assert a_val != b_val


def test_synth_referencing_unknown_real_is_dropped():
    """Synthetic entries whose source_fpv is unknown end up in neither
    split — they're orphans, not leaks."""
    real = make_real(20)
    synth = make_synth(real)
    # Add an orphan synth that references a non-existent real clip
    synth.append({
        'clip_id_FPV': 'Synth_GHOST_bright_FPV',
        'clip_id_Topview': 'Synth_GHOST_bright_Topview',
        'wells_ground_truth': [{'well_row': 'A', 'well_column': '1'}],
        'synthetic': True,
        'aug_type': 'bright',
        'source_fpv': 'Plate_99_clip_0099_FPV',  # not in real
    })
    train, val = build_leak_free_split(real, synth, val_split=0.2, seed=5)
    train_ids = {t['clip_id_FPV'] for t in train}
    assert 'Synth_GHOST_bright_FPV' not in train_ids


def test_assert_catches_synthetic_in_val():
    """The defensive assertion should raise if val accidentally contains
    a synthetic entry."""
    train = [{'clip_id_FPV': 'A_FPV'}]
    val = [{'clip_id_FPV': 'B_FPV', 'synthetic': True, 'source_fpv': 'X'}]
    with pytest.raises(AssertionError, match="synthetic"):
        _assert_no_synthetic_leakage(train, val)


def test_assert_catches_train_derived_from_val():
    train = [{
        'clip_id_FPV': 'Synth_X_bright_FPV',
        'synthetic': True,
        'source_fpv': 'X_FPV',
    }]
    val = [{'clip_id_FPV': 'X_FPV'}]
    with pytest.raises(AssertionError, match="derived from val"):
        _assert_no_synthetic_leakage(train, val)


def test_assert_catches_clip_id_overlap():
    train = [{'clip_id_FPV': 'X_FPV'}]
    val = [{'clip_id_FPV': 'X_FPV'}]
    with pytest.raises(AssertionError, match="overlap"):
        _assert_no_synthetic_leakage(train, val)


def test_real_world_distribution_proportions():
    """100 real clips × 7 augmentations = 800 combined. With val_split=0.2
    we expect train_real=80, val_real=20, train_synth=80*7=560, total
    train=640. This matches the actual production setup."""
    real = make_real(100)
    synth = make_synth(real, augs=('bright', 'dark', 'noise', 'contrast', 'flip_h', 'flip_v', 'flip_hv'))
    assert len(synth) == 700

    train, val = build_leak_free_split(real, synth, val_split=0.2, seed=42)
    assert len(val) == 20
    assert len(train) == 80 + 80 * 7  # 640
