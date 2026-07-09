"""Tests for NV-diamond simulation oracle."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nv_diamond import (
    DecoherenceParams,
    generate_nv_sample,
    generate_dataset,
    sweep_single_axis,
    TEMP_HEALTHY,
    TEMP_TUMOR,
    FREQ_SENSITIVITY,
    N_FEATURES_PER_CENTER,
)


def test_sample_shape():
    params = DecoherenceParams(n_centers=20)
    rng = np.random.default_rng(0)
    sample = generate_nv_sample(TEMP_HEALTHY, params, device_offset=0.0, rng=rng)
    assert sample.shape == (20 * N_FEATURES_PER_CENTER,)


def test_dataset_shape():
    params = DecoherenceParams(n_centers=10)
    X, y, dev = generate_dataset(n_per_class=50, params=params, seed=0, n_devices=3)
    assert X.shape == (3 * 2 * 50, 10 * N_FEATURES_PER_CENTER)
    assert y.shape == (300,)
    assert dev.shape == (300,)
    assert set(y) == {0, 1}
    assert set(dev) == {0, 1, 2}


def test_deterministic_given_seed():
    params = DecoherenceParams()
    X1, y1, _ = generate_dataset(n_per_class=20, params=params, seed=12345)
    X2, y2, _ = generate_dataset(n_per_class=20, params=params, seed=12345)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_different_seeds_differ():
    params = DecoherenceParams()
    X1, _, _ = generate_dataset(n_per_class=20, params=params, seed=1)
    X2, _, _ = generate_dataset(n_per_class=20, params=params, seed=2)
    assert not np.allclose(X1, X2)


def test_frequency_encodes_temperature():
    params = DecoherenceParams(sigma_temp=0.0, sigma_device=0.0, sigma_surface=0.0)
    X, y, _ = generate_dataset(n_per_class=500, params=params, seed=42, n_devices=1)
    freq_col_indices = [i * N_FEATURES_PER_CENTER + 3 for i in range(params.n_centers)]
    mean_freq_healthy = X[y == 0][:, freq_col_indices].mean()
    mean_freq_tumor = X[y == 1][:, freq_col_indices].mean()
    expected_diff = FREQ_SENSITIVITY * (TEMP_TUMOR - TEMP_HEALTHY)
    actual_diff = mean_freq_tumor - mean_freq_healthy
    assert actual_diff == pytest.approx(expected_diff, rel=0.1)


def test_t2_degrades_with_alpha():
    t2_col_indices_fn = lambda nc: [i * N_FEATURES_PER_CENTER for i in range(nc)]
    params_clean = DecoherenceParams(alpha_T2=0.0)
    params_dirty = DecoherenceParams(alpha_T2=0.8)
    X_clean, _, _ = generate_dataset(n_per_class=200, params=params_clean, seed=99, n_devices=1)
    X_dirty, _, _ = generate_dataset(n_per_class=200, params=params_dirty, seed=99, n_devices=1)
    mean_t2_clean = X_clean[:, t2_col_indices_fn(20)].mean()
    mean_t2_dirty = X_dirty[:, t2_col_indices_fn(20)].mean()
    assert mean_t2_clean > mean_t2_dirty * 2


def test_device_variation_adds_offset():
    params_no_var = DecoherenceParams(sigma_device=0.0, sigma_temp=0.0)
    params_var = DecoherenceParams(sigma_device=0.05, sigma_temp=0.0)
    X_no, _, dev_no = generate_dataset(n_per_class=200, params=params_no_var, seed=55, n_devices=3)
    X_var, _, dev_var = generate_dataset(n_per_class=200, params=params_var, seed=55, n_devices=3)
    freq_cols = [i * N_FEATURES_PER_CENTER + 3 for i in range(20)]
    dev0_freq_no = X_no[dev_no == 0][:, freq_cols].mean()
    dev1_freq_no = X_no[dev_no == 1][:, freq_cols].mean()
    dev0_freq_var = X_var[dev_var == 0][:, freq_cols].mean()
    dev1_freq_var = X_var[dev_var == 1][:, freq_cols].mean()
    cross_device_spread_no = abs(dev0_freq_no - dev1_freq_no)
    cross_device_spread_var = abs(dev0_freq_var - dev1_freq_var)
    assert cross_device_spread_var > cross_device_spread_no


def test_sweep_single_axis_returns_correct_levels():
    results = sweep_single_axis("alpha_T2", n_per_class=10, seed=0, n_devices=1)
    assert len(results) == 10
    levels = [r[0] for r in results]
    assert levels[0] == pytest.approx(0.0)
    assert levels[-1] == pytest.approx(1.0)
    for level, X, y, dev in results:
        assert X.shape[0] == 2 * 10
        assert X.shape[1] == 20 * N_FEATURES_PER_CENTER


def test_sweep_n_centers_varies_feature_dim():
    results = sweep_single_axis("n_centers", n_per_class=10, seed=0, n_devices=1)
    dims = [r[1].shape[1] for r in results]
    assert dims[0] == 5 * N_FEATURES_PER_CENTER
    assert dims[-1] == 50 * N_FEATURES_PER_CENTER
    assert len(set(dims)) > 1


def test_surface_noise_increases_linewidth_variance():
    params_clean = DecoherenceParams(sigma_surface=0.0)
    params_noisy = DecoherenceParams(sigma_surface=0.4)
    X_clean, _, _ = generate_dataset(n_per_class=300, params=params_clean, seed=77, n_devices=1)
    X_noisy, _, _ = generate_dataset(n_per_class=300, params=params_noisy, seed=77, n_devices=1)
    lw_cols = [i * N_FEATURES_PER_CENTER + 1 for i in range(20)]
    var_clean = X_clean[:, lw_cols].var()
    var_noisy = X_noisy[:, lw_cols].var()
    assert var_noisy > var_clean * 1.5
