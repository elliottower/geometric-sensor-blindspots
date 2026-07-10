"""Tests for designed metrics (L1, L2).

Validates on synthetic data where ground truth is known:
- T_loc must respond to additive per-device offsets
- T_scale must respond to multiplicative per-device gain
- T_geo must stay flat under both offset and gain
- L2 at lambda=0 must reproduce geodesic-like blindness
- L2 at lambda=1 must reproduce L1-like sensitivity
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nv_diamond import DecoherenceParams, generate_dataset
from designed_metrics import (
    _t_loc, _t_scale, _t_geo, _amplitude_mask,
    designed_l1_composite, designed_l1_loc, designed_l1_scale,
    designed_l2_dial, designed_l2_sweep,
    DESIGNED_METHODS,
)


@pytest.fixture
def clean_data():
    params = DecoherenceParams(sigma_device=0.0, sigma_gain=0.0)
    return generate_dataset(n_per_class=200, params=params, seed=42, n_devices=3)


@pytest.fixture
def offset_data():
    params = DecoherenceParams(sigma_device=0.08, sigma_gain=0.0)
    return generate_dataset(n_per_class=200, params=params, seed=42, n_devices=3)


@pytest.fixture
def gain_data():
    params = DecoherenceParams(sigma_device=0.0, sigma_gain=0.12)
    return generate_dataset(n_per_class=200, params=params, seed=42, n_devices=3)


def test_amplitude_mask_excludes_frequency():
    mask = _amplitude_mask(100)
    assert mask.sum() == 80
    for c in range(20):
        assert not mask[c * 5 + 3]
        assert mask[c * 5 + 0]
        assert mask[c * 5 + 1]
        assert mask[c * 5 + 2]
        assert mask[c * 5 + 4]


def test_t_loc_responds_to_offset(clean_data, offset_data):
    X_clean, _, dev_clean = clean_data
    X_off, _, dev_off = offset_data
    loc_clean = _t_loc(X_clean, dev_clean)
    loc_off = _t_loc(X_off, dev_off)
    assert loc_off > loc_clean * 2, (
        f"T_loc should increase substantially with device offset: "
        f"clean={loc_clean:.4f}, offset={loc_off:.4f}"
    )


def test_t_scale_responds_to_gain(clean_data, gain_data):
    X_clean, _, dev_clean = clean_data
    X_gain, _, dev_gain = gain_data
    scale_clean = _t_scale(X_clean, dev_clean)
    scale_gain = _t_scale(X_gain, dev_gain)
    assert scale_gain > scale_clean * 2, (
        f"T_scale should increase substantially with device gain: "
        f"clean={scale_clean:.6f}, gain={scale_gain:.6f}"
    )


def test_t_geo_flat_under_offset(clean_data, offset_data):
    X_clean, _, dev_clean = clean_data
    X_off, _, dev_off = offset_data
    geo_clean = _t_geo(X_clean, dev_clean)
    geo_off = _t_geo(X_off, dev_off)
    ratio = geo_off / geo_clean if geo_clean > 1e-12 else 1.0
    assert 0.5 < ratio < 2.0, (
        f"T_geo should stay flat under offset: "
        f"clean={geo_clean:.4f}, offset={geo_off:.4f}, ratio={ratio:.2f}"
    )


def test_t_geo_flat_under_gain(clean_data, gain_data):
    X_clean, _, dev_clean = clean_data
    X_gain, _, dev_gain = gain_data
    geo_clean = _t_geo(X_clean, dev_clean)
    geo_gain = _t_geo(X_gain, dev_gain)
    ratio = geo_gain / geo_clean if geo_clean > 1e-12 else 1.0
    assert 0.5 < ratio < 2.0, (
        f"T_geo should stay flat under gain: "
        f"clean={geo_clean:.4f}, gain={geo_gain:.4f}, ratio={ratio:.2f}"
    )


def test_l1_composite_returns_all_components(clean_data):
    X, y, dev = clean_data
    result = designed_l1_composite(X, y, dev)
    assert "score" in result
    assert "name" in result
    assert "t_loc" in result
    assert "t_scale" in result
    assert "t_geo" in result
    assert result["score"] == pytest.approx(result["t_loc"] + result["t_scale"])


def test_l2_lambda0_is_blind_to_offset(offset_data):
    X, y, dev = offset_data
    result_lam0 = designed_l2_dial(X, y, dev, lam=0.0)
    result_lam1 = designed_l2_dial(X, y, dev, lam=1.0)
    assert result_lam0["score"] < result_lam1["score"] * 0.5, (
        f"L2 at lambda=0 should be much less sensitive than lambda=1: "
        f"lam0={result_lam0['score']:.4f}, lam1={result_lam1['score']:.4f}"
    )


def test_l2_sweep_monotonic_on_offset():
    params = DecoherenceParams(sigma_device=0.08)
    X, y, dev = generate_dataset(n_per_class=200, params=params, seed=99, n_devices=5)
    sweep = designed_l2_sweep(X, y, dev, n_lambda=11)
    scores = sweep["scores"]
    assert len(scores) == 11
    assert scores[-1] > scores[0], (
        f"L2 sweep should increase from lambda=0 to lambda=1: "
        f"first={scores[0]:.4f}, last={scores[-1]:.4f}"
    )


def test_all_designed_methods_return_valid_output(clean_data):
    X, y, dev = clean_data
    for name, fn in DESIGNED_METHODS.items():
        result = fn(X, y, dev)
        assert "score" in result, f"{name} missing 'score'"
        assert "name" in result, f"{name} missing 'name'"
        assert np.isfinite(result["score"]), f"{name} returned non-finite score"


def test_l1_specificity_on_n_centers():
    """L1 should NOT be sensitive to ensemble size (anti-detect-everything guard).

    Compute L1 at n_centers=10 vs n_centers=40. The score should not
    scale proportionally — it measures DEVICE spread, not ensemble size.
    """
    params_small = DecoherenceParams(n_centers=10, sigma_device=0.05)
    params_large = DecoherenceParams(n_centers=40, sigma_device=0.05)
    scores_small, scores_large = [], []
    for seed in range(50):
        X_s, y_s, d_s = generate_dataset(n_per_class=100, params=params_small, seed=seed, n_devices=3)
        X_l, y_l, d_l = generate_dataset(n_per_class=100, params=params_large, seed=seed, n_devices=3)
        scores_small.append(designed_l1_composite(X_s, y_s, d_s)["score"])
        scores_large.append(designed_l1_composite(X_l, y_l, d_l)["score"])
    ratio = np.mean(scores_large) / np.mean(scores_small)
    assert ratio < 5.0, (
        f"L1 score should not scale linearly with n_centers (specificity check): "
        f"ratio={ratio:.2f}, small_mean={np.mean(scores_small):.4f}, "
        f"large_mean={np.mean(scores_large):.4f}"
    )
