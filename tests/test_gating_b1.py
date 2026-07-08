"""Tests for B1 experiment (edge vs loop monotonicity).

Run with: uv run --no-project --with numpy --with scipy --with scikit-learn -m pytest tests/test_gating_b1.py -xvs
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "biomedical-cohort-transfer" / "src"))

from gating_b1_edge_vs_loop import (
    mean_edge_geodesics,
    compute_delta_rho,
    bootstrap_median_ci,
    run_seed,
    benjamini_hochberg,
    ONSET_WINDOW_IDX,
)
from gating_h7b_v3 import (
    generate_latent,
    generate_base_embedding,
    generate_perturbations,
    sensor_embedding,
    generate_sensor_data,
    D, K_LATENT, OBS_NOISE_BASE, PCA_K, EPSILON_RANGE,
    N_SAMPLES_PER_CLASS,
)
from transportability import top_k_subspace


# ======================================================================
# mean_edge_geodesics
# ======================================================================

def test_mean_edge_geodesics_returns_three_edges():
    rng = np.random.default_rng(42)
    Xs = [rng.standard_normal((200, D)) for _ in range(3)]
    mean_geo, per_edge = mean_edge_geodesics(Xs, PCA_K)
    assert len(per_edge) == 3
    assert mean_geo == pytest.approx(np.mean(per_edge))


def test_mean_edge_geodesics_identical_data_near_zero():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, D))
    mean_geo, per_edge = mean_edge_geodesics([X, X, X], PCA_K)
    assert mean_geo == pytest.approx(0.0, abs=1e-6)


def test_mean_edge_geodesics_increases_with_perturbation():
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U_base = generate_base_embedding(np.random.default_rng(100))
    perts = generate_perturbations(3, np.random.default_rng(200))

    geo_small, _ = mean_edge_geodesics(
        [generate_sensor_data(z, sensor_embedding(U_base, p, 0.05),
                              OBS_NOISE_BASE, np.random.default_rng(300 + i))
         for i, p in enumerate(perts)], PCA_K)

    geo_large, _ = mean_edge_geodesics(
        [generate_sensor_data(z, sensor_embedding(U_base, p, 0.8),
                              OBS_NOISE_BASE, np.random.default_rng(300 + i))
         for i, p in enumerate(perts)], PCA_K)

    assert geo_large > geo_small


# ======================================================================
# compute_delta_rho
# ======================================================================

def test_delta_rho_monotone_edge_noisy_loop():
    rng = np.random.default_rng(42)
    eps = list(range(20))
    edges = [float(x) + rng.normal(0, 0.1) for x in range(20)]
    loops = [rng.normal(5.0, 1.0) for _ in range(20)]
    dr, re, rl, pe, pl = compute_delta_rho(eps, loops, edges)
    assert re > 0.9
    assert abs(rl) < 0.5
    assert dr > 0.5


def test_delta_rho_both_monotone_near_zero():
    eps = list(range(20))
    edges = [float(x) for x in range(20)]
    loops = [float(x) for x in range(20)]
    dr, re, rl, pe, pl = compute_delta_rho(eps, loops, edges)
    assert dr == pytest.approx(0.0, abs=0.01)


def test_delta_rho_returns_correct_components():
    eps = [0, 1, 2, 3, 4]
    edges = [1, 2, 3, 4, 5]
    loops = [5, 4, 3, 2, 1]
    dr, re, rl, pe, pl = compute_delta_rho(eps, loops, edges)
    assert re == pytest.approx(1.0)
    assert rl == pytest.approx(-1.0)
    assert dr == pytest.approx(2.0)


# ======================================================================
# bootstrap_median_ci
# ======================================================================

def test_bootstrap_ci_positive_values():
    values = np.ones(50) + np.random.default_rng(42).normal(0, 0.1, 50)
    med, lo, hi = bootstrap_median_ci(values)
    assert lo > 0
    assert hi > lo
    assert lo < med < hi


def test_bootstrap_ci_centered_at_zero():
    values = np.random.default_rng(42).normal(0, 1, 50)
    med, lo, hi = bootstrap_median_ci(values)
    assert lo < 0 < hi


def test_bootstrap_ci_reproducible():
    values = np.random.default_rng(42).normal(0.5, 0.3, 50)
    r1 = bootstrap_median_ci(values)
    r2 = bootstrap_median_ci(values)
    assert r1 == r2


# ======================================================================
# Onset window is by index
# ======================================================================

def test_onset_window_selects_five_points():
    eps_selected = EPSILON_RANGE[ONSET_WINDOW_IDX]
    assert len(eps_selected) == 5
    assert eps_selected[0] == pytest.approx(0.0)
    assert eps_selected[-1] == pytest.approx(4.0 / 19.0)


# ======================================================================
# run_seed integration
# ======================================================================

def test_run_seed_returns_all_fields():
    result = run_seed(99999, identical_perturbation=False)
    assert "delta_rho" in result
    assert "rho_edge" in result
    assert "rho_loop" in result
    assert "onset_slope" in result
    assert len(result["epsilons"]) == 20
    assert len(result["loop_holonomies"]) == 20
    assert len(result["mean_geodesics"]) == 20


def test_run_seed_reproducible():
    r1 = run_seed(12345, identical_perturbation=False)
    r2 = run_seed(12345, identical_perturbation=False)
    assert r1["delta_rho"] == r2["delta_rho"]
    assert r1["rho_edge"] == r2["rho_edge"]


def test_identical_perturbation_small_delta_rho():
    results = [run_seed(90000 + i, identical_perturbation=True) for i in range(5)]
    delta_rhos = [r["delta_rho"] for r in results]
    assert abs(np.mean(delta_rhos)) < 0.5, \
        f"Identical perturbation should have Δρ near 0, got mean={np.mean(delta_rhos):.3f}"


# ======================================================================
# Negative control: identical perturbation → no edge-loop divergence
# ======================================================================

def test_negative_control_edge_geodesics_near_zero():
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U_base = generate_base_embedding(np.random.default_rng(100))
    same_pert = np.random.default_rng(200).standard_normal((D, K_LATENT))

    for eps in [0.0, 0.3, 0.7]:
        U_s = sensor_embedding(U_base, same_pert, eps)
        Xs = [generate_sensor_data(z, U_s, OBS_NOISE_BASE, np.random.default_rng(300 + i))
              for i in range(3)]
        mean_geo, _ = mean_edge_geodesics(Xs, PCA_K)
        assert mean_geo < 2.0, \
            f"Identical perturbation at eps={eps} should have small geodesics, got {mean_geo:.3f}"


# ======================================================================
# benjamini_hochberg
# ======================================================================

def test_bh_all_significant():
    reject, adj = benjamini_hochberg([0.001, 0.002, 0.003], alpha=0.05)
    assert all(reject)


def test_bh_none_significant():
    reject, adj = benjamini_hochberg([0.5, 0.6, 0.7], alpha=0.05)
    assert not any(reject)


def test_bh_preserves_order():
    reject, adj = benjamini_hochberg([0.01, 0.5, 0.03], alpha=0.05)
    assert reject[0] is True
    assert reject[1] is False
    assert reject[2] is True
