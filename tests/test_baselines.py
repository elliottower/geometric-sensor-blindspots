"""Tests for scalar baselines."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nv_diamond import DecoherenceParams, generate_dataset
from baselines import BASELINES


@pytest.fixture
def nv_data():
    params = DecoherenceParams(alpha_T2=0.2, sigma_surface=0.05, sigma_device=0.02)
    X, y, dev = generate_dataset(n_per_class=100, params=params, seed=42, n_devices=3)
    return X, y, dev


@pytest.mark.parametrize("name", sorted(BASELINES.keys()))
def test_baseline_returns_score(nv_data, name):
    X, y, dev = nv_data
    result = BASELINES[name](X, y, dev)
    assert "score" in result
    assert "name" in result
    assert isinstance(result["score"], (int, float, np.integer, np.floating))
    assert np.isfinite(result["score"])


def test_ttest_detects_signal():
    params = DecoherenceParams(sigma_temp=0.0, sigma_device=0.0)
    X, y, dev = generate_dataset(n_per_class=200, params=params, seed=55, n_devices=2)
    result = BASELINES["ttest"](X, y, dev)
    assert result["score"] > 2.0


def test_lasso_above_chance():
    params = DecoherenceParams(sigma_temp=0.0, sigma_device=0.0)
    X, y, dev = generate_dataset(n_per_class=200, params=params, seed=66, n_devices=2)
    result = BASELINES["lasso"](X, y, dev)
    assert result["score"] > 0.5
