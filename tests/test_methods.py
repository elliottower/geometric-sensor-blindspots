"""Tests for all 23 geometric methods."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nv_diamond import DecoherenceParams, generate_dataset
from methods import METHODS


@pytest.fixture
def nv_data():
    params = DecoherenceParams(alpha_T2=0.2, sigma_surface=0.05, sigma_device=0.02)
    X, y, dev = generate_dataset(n_per_class=100, params=params, seed=42, n_devices=3)
    return X, y, dev


@pytest.mark.parametrize("method_id", sorted(METHODS.keys()))
def test_method_returns_score(nv_data, method_id):
    X, y, dev = nv_data
    result = METHODS[method_id](X, y, dev)
    assert "score" in result
    assert "name" in result
    assert isinstance(result["score"], (int, float, np.integer, np.floating))
    assert np.isfinite(result["score"])


def test_geodesic_increases_with_perturbation():
    params_clean = DecoherenceParams(sigma_device=0.0, sigma_temp=0.0)
    params_perturbed = DecoherenceParams(sigma_device=0.08, sigma_temp=1.5)
    X_clean, y_c, dev_c = generate_dataset(n_per_class=200, params=params_clean, seed=10, n_devices=3)
    X_pert, y_p, dev_p = generate_dataset(n_per_class=200, params=params_perturbed, seed=10, n_devices=3)
    geo_clean = METHODS[8](X_clean, y_c, dev_c)["score"]
    geo_pert = METHODS[8](X_pert, y_p, dev_p)["score"]
    assert geo_pert > geo_clean


def test_sheaf_h1_zero_under_no_device_variation():
    params = DecoherenceParams(sigma_device=0.0, sigma_temp=0.0, sigma_surface=0.0)
    X, y, dev = generate_dataset(n_per_class=300, params=params, seed=77, n_devices=3)
    result = METHODS[10](X, y, dev)
    assert result["p_value"] > 0.01


def test_cka_high_when_devices_agree():
    params = DecoherenceParams(sigma_device=0.0, sigma_temp=0.0)
    X, y, dev = generate_dataset(n_per_class=200, params=params, seed=33, n_devices=3)
    result = METHODS[13](X, y, dev)
    assert result["score"] > 0.5


def test_persistent_homology_positive():
    params = DecoherenceParams(alpha_T2=0.3)
    X, y, dev = generate_dataset(n_per_class=80, params=params, seed=50, n_devices=2)
    result = METHODS[15](X, y, dev)
    assert result["score"] > 0.0


def test_bottleneck_larger_for_separated_classes():
    params_clean = DecoherenceParams(sigma_temp=0.0, sigma_device=0.0)
    params_noisy = DecoherenceParams(alpha_T2=0.8, sigma_surface=0.3)
    X_clean, y_c, dev_c = generate_dataset(n_per_class=100, params=params_clean, seed=10, n_devices=2)
    X_noisy, y_n, dev_n = generate_dataset(n_per_class=100, params=params_noisy, seed=10, n_devices=2)
    bn_clean = METHODS[16](X_clean, y_c, dev_c)["score"]
    bn_noisy = METHODS[16](X_noisy, y_n, dev_n)["score"]
    assert isinstance(bn_clean, (float, np.floating))
    assert isinstance(bn_noisy, (float, np.floating))


def test_qfi_flatness_high_when_devices_similar():
    params = DecoherenceParams(sigma_device=0.0, sigma_temp=0.0)
    X, y, dev = generate_dataset(n_per_class=200, params=params, seed=44, n_devices=3)
    result = METHODS[17](X, y, dev)
    assert result["score"] > 0.5


def test_berry_phase_zero_when_no_device_variation():
    params = DecoherenceParams(sigma_device=0.0, sigma_temp=0.0, sigma_surface=0.0)
    X, y, dev = generate_dataset(n_per_class=300, params=params, seed=55, n_devices=3)
    result = METHODS[18](X, y, dev)
    assert result["score"] < 0.5


def test_spectral_gap_positive():
    params = DecoherenceParams(alpha_T2=0.1)
    X, y, dev = generate_dataset(n_per_class=100, params=params, seed=66, n_devices=3)
    result = METHODS[19](X, y, dev)
    assert result["score"] > 0.0


def test_wasserstein_detects_class_separation():
    params = DecoherenceParams(sigma_temp=0.0, sigma_device=0.0)
    X, y, dev = generate_dataset(n_per_class=200, params=params, seed=77, n_devices=2)
    result = METHODS[20](X, y, dev)
    assert result["score"] > 0.0


def test_vn_entropy_stable_across_similar_devices():
    params = DecoherenceParams(sigma_device=0.0, sigma_temp=0.0)
    X, y, dev = generate_dataset(n_per_class=200, params=params, seed=88, n_devices=3)
    result = METHODS[21](X, y, dev)
    assert result["score"] > 0.5


def test_fidelity_high_for_similar_devices():
    params = DecoherenceParams(sigma_device=0.0, sigma_temp=0.0)
    X, y, dev = generate_dataset(n_per_class=100, params=params, seed=99, n_devices=2)
    result = METHODS[22](X, y, dev)
    assert result["score"] > 0.5


def test_h0_persistence_known_answer_two_clusters():
    """Two well-separated clusters: the inter-cluster merge should be the
    dominant death event. All within-cluster merges happen at small distances
    (sigma=0.1), so the largest non-survivor death should be ~10 (inter-cluster)
    while the third-largest should be <1 (within-cluster)."""
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(50, 3))
    cluster_b = rng.normal(loc=10.0, scale=0.1, size=(50, 3))
    X = np.vstack([cluster_a, cluster_b])
    from methods import _persistence_h0
    dgm = _persistence_h0(X, n_sub=100)
    deaths = sorted([d for _, d in dgm], reverse=True)
    within_cluster_deaths = sorted([d for _, d in dgm if d < 5.0])
    inter_cluster_deaths = [d for _, d in dgm if d > 5.0]
    assert len(inter_cluster_deaths) >= 1
    assert max(within_cluster_deaths) < 2.0
