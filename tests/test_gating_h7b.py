"""Tests for gating experiment v2.

Validates the fixes from DEVIATION_LOG.md and core experiment logic.
Run with: uv run --no-project --with numpy --with scipy --with scikit-learn -m pytest tests/ -xvs
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "biomedical-cohort-transfer" / "src"))

from gating_h7b_v2 import (
    _shared_base,
    generate_sensor_a,
    generate_sensor_b,
    generate_sensor_c,
    within_sensor_auc,
    compute_holonomy,
    pool_resplit_null,
    analyze_seed,
    GENERATORS,
    D,
    SIGNAL_DIM,
    SIGNAL_STRENGTH,
    N_SAMPLES_PER_CLASS,
    PCA_K,
)
from transportability import top_k_subspace, cocycle_holonomy


# ======================================================================
# DEV-001 regression: row-permutation null is degenerate
# ======================================================================

def test_row_permutation_is_noop_for_pca():
    rng = np.random.default_rng(999)
    X = rng.standard_normal((200, 50))
    U_orig, _ = top_k_subspace(X, 3)
    U_perm, _ = top_k_subspace(X[rng.permutation(200)], 3)
    assert np.linalg.norm(U_orig - U_perm) == pytest.approx(0.0, abs=1e-10)


def test_row_permutation_null_has_zero_variance():
    rng = np.random.default_rng(111)
    Xs = [rng.standard_normal((100, 40)) for _ in range(3)]
    subspaces = [top_k_subspace(X, 3)[0] for X in Xs]
    _, obs = cocycle_holonomy(subspaces)

    null_norms = []
    for _ in range(50):
        perm_sub = [top_k_subspace(X[rng.permutation(100)], 3)[0] for X in Xs]
        _, n = cocycle_holonomy(perm_sub)
        null_norms.append(n)

    assert np.std(null_norms) == pytest.approx(0.0, abs=1e-10), \
        "Row-perm null should have zero variance (it's a no-op for PCA)"


# ======================================================================
# DEV-001 fix: pool-resplit null is NOT degenerate
# ======================================================================

def test_pool_resplit_null_has_nonzero_variance():
    rng = np.random.default_rng(222)
    Xs = [rng.standard_normal((100, 40)) for _ in range(3)]
    null_norms = pool_resplit_null(Xs, 3, 200, rng)
    assert np.std(null_norms) > 0.01, \
        "Pool-resplit null must produce a non-degenerate distribution"


def test_pool_resplit_null_produces_lower_holonomy_for_similar_data():
    rng = np.random.default_rng(333)
    shared = rng.standard_normal((300, 40))
    Xs = [shared[:100], shared[100:200], shared[200:300]]
    null_norms = pool_resplit_null(Xs, 3, 100, rng)

    hol, _ = compute_holonomy(Xs, 3)
    assert hol < np.percentile(null_norms, 95), \
        "Subsets of same data should not have significantly high holonomy"


# ======================================================================
# DEV-002 regression: different dimensions must crash
# ======================================================================

def test_cocycle_holonomy_crashes_on_dimension_mismatch():
    rng = np.random.default_rng(444)
    U_100 = top_k_subspace(rng.standard_normal((100, 100)), 3)[0]
    U_80 = top_k_subspace(rng.standard_normal((100, 80)), 3)[0]
    with pytest.raises(ValueError, match="mismatch"):
        cocycle_holonomy([U_100, U_80])


# ======================================================================
# DEV-002 fix: all generators produce same dimension
# ======================================================================

def test_all_generators_produce_same_dimension():
    rng = np.random.default_rng(555)
    for gen in GENERATORS:
        for eps in [0.0, 0.5, 1.0]:
            X, y = gen(50, eps, rng)
            assert X.shape == (100, D), f"{gen.__name__} eps={eps}: got {X.shape}"
            assert y.shape == (100,)
            assert set(np.unique(y)) == {0.0, 1.0}


def test_holonomy_computable_across_all_sensors():
    rng = np.random.default_rng(666)
    Xs = [gen(50, 0.5, rng)[0] for gen in GENERATORS]
    hol, pairwise = compute_holonomy(Xs, PCA_K)
    assert isinstance(hol, float)
    assert hol >= 0
    assert len(pairwise) == 3


# ======================================================================
# Signal validity: clean sensors should classify, noisy should degrade
# ======================================================================

def test_clean_sensors_classify_above_threshold():
    rng = np.random.default_rng(777)
    for gen in GENERATORS:
        X, y = gen(N_SAMPLES_PER_CLASS, 0.0, rng)
        auc = within_sensor_auc(X, y)
        assert auc > 0.75, f"{gen.__name__} at eps=0 has AUC={auc:.3f} < 0.75"


def test_high_decoherence_degrades_at_least_one_sensor():
    rng = np.random.default_rng(888)
    aucs_clean = [within_sensor_auc(*gen(N_SAMPLES_PER_CLASS, 0.0, rng))
                  for gen in GENERATORS]
    aucs_noisy = [within_sensor_auc(*gen(N_SAMPLES_PER_CLASS, 1.0, rng))
                  for gen in GENERATORS]
    degradation = [c - n for c, n in zip(aucs_clean, aucs_noisy)]
    assert max(degradation) > 0.05, \
        f"At eps=1.0 at least one sensor should degrade significantly, got {degradation}"


# ======================================================================
# Holonomy trajectory: should increase with decoherence
# ======================================================================

def test_holonomy_increases_with_decoherence():
    rng = np.random.default_rng(1234)
    hol_0 = compute_holonomy([gen(100, 0.0, rng)[0] for gen in GENERATORS], PCA_K)[0]
    hol_1 = compute_holonomy([gen(100, 1.0, rng)[0] for gen in GENERATORS], PCA_K)[0]
    assert hol_1 > hol_0, \
        f"Holonomy should increase: eps=0 got {hol_0:.4f}, eps=1 got {hol_1:.4f}"


# ======================================================================
# Reproducibility
# ======================================================================

def test_generators_are_reproducible():
    for gen in GENERATORS:
        X1, y1 = gen(50, 0.3, np.random.default_rng(42))
        X2, y2 = gen(50, 0.3, np.random.default_rng(42))
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# ======================================================================
# Analysis logic
# ======================================================================

def test_analyze_seed_h7b_confirmed_when_holonomy_fires_before_auc_drops():
    rows = [
        {"epsilon": 0.0, "min_auc": 0.95, "holonomy_p": 0.50},
        {"epsilon": 0.2, "min_auc": 0.85, "holonomy_p": 0.01},
        {"epsilon": 0.4, "min_auc": 0.80, "holonomy_p": 0.001},
        {"epsilon": 0.6, "min_auc": 0.70, "holonomy_p": 0.001},
    ]
    result = analyze_seed(rows)
    assert result["H7b_confirmed"] is True
    assert result["H7c_confirmed"] is True
    assert result["epsilon_crit_holonomy"] == pytest.approx(0.2)
    assert result["epsilon_crit_single_sensor"] == pytest.approx(0.6)


def test_analyze_seed_h7b_fails_when_holonomy_never_fires():
    rows = [
        {"epsilon": 0.0, "min_auc": 0.95, "holonomy_p": 0.50},
        {"epsilon": 0.5, "min_auc": 0.60, "holonomy_p": 0.30},
        {"epsilon": 1.0, "min_auc": 0.50, "holonomy_p": 0.20},
    ]
    result = analyze_seed(rows)
    assert result["H7b_confirmed"] is False
    assert result["H7c_confirmed"] is False


def test_analyze_seed_h7b_true_h7c_false_when_auc_drops_first():
    rows = [
        {"epsilon": 0.0, "min_auc": 0.95, "holonomy_p": 0.50},
        {"epsilon": 0.2, "min_auc": 0.70, "holonomy_p": 0.50},
        {"epsilon": 0.4, "min_auc": 0.60, "holonomy_p": 0.01},
    ]
    result = analyze_seed(rows)
    assert result["H7b_confirmed"] is False, \
        "H7b requires holonomy significant WHILE AUC still above threshold"
    assert result["H7c_confirmed"] is False


def test_analyze_seed_h7c_requires_holonomy_before_auc():
    rows = [
        {"epsilon": 0.0, "min_auc": 0.95, "holonomy_p": 0.50},
        {"epsilon": 0.3, "min_auc": 0.76, "holonomy_p": 0.50},
        {"epsilon": 0.5, "min_auc": 0.74, "holonomy_p": 0.04},
        {"epsilon": 0.7, "min_auc": 0.60, "holonomy_p": 0.001},
    ]
    result = analyze_seed(rows)
    assert result["epsilon_crit_holonomy"] == pytest.approx(0.5)
    assert result["epsilon_crit_single_sensor"] == pytest.approx(0.5)
    assert result["H7b_confirmed"] is False, \
        "At eps=0.5 min_auc=0.74 < 0.75 so no row has both conditions met"


# ======================================================================
# Edge cases
# ======================================================================

def test_shared_base_labels_balanced():
    rng = np.random.default_rng(42)
    X, y = _shared_base(100, rng)
    assert X.shape == (200, 100)
    assert y.shape == (200,)
    assert np.sum(y == 0) == 100
    assert np.sum(y == 1) == 100


def test_shared_base_has_signal_in_first_features():
    rng = np.random.default_rng(42)
    X, y = _shared_base(500, rng)
    mean_0 = X[y == 0].mean(axis=0)
    mean_1 = X[y == 1].mean(axis=0)
    diff = mean_1 - mean_0
    assert np.all(np.abs(diff[:SIGNAL_DIM]) > 0.3), \
        "Signal features should show class separation"
    assert np.all(np.abs(diff[SIGNAL_DIM:]) < 0.2), \
        "Non-signal features should have no class separation"


def test_epsilon_zero_sensor_a_matches_base():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    X_a, _ = generate_sensor_a(50, 0.0, rng1)
    X_base, _ = _shared_base(50, rng2)
    np.testing.assert_allclose(X_a, X_base, rtol=1e-10,
                               err_msg="At eps=0, sensor A should match base")
