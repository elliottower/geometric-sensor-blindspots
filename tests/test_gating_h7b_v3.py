"""Tests for gating experiment v3 (latent-projection design).

Run with: uv run --no-project --with numpy --with scipy --with scikit-learn -m pytest tests/test_gating_h7b_v3.py -xvs
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "biomedical-cohort-transfer" / "src"))

from gating_h7b_v3 import (
    generate_latent,
    generate_base_embedding,
    generate_perturbations,
    sensor_embedding,
    generate_sensor_data,
    within_sensor_auc,
    compute_holonomy,
    bootstrap_baseline,
    analyze_seed,
    D,
    K_LATENT,
    LATENT_SIGNAL,
    OBS_NOISE_BASE,
    OBS_NOISE_RATE,
    PERTURBATION_SCALE,
    PCA_K,
    N_SAMPLES_PER_CLASS,
    AUC_THRESHOLD,
    HOLONOMY_ALPHA,
)
from transportability import top_k_subspace, cocycle_holonomy


# ======================================================================
# Latent model
# ======================================================================

def test_latent_has_class_separation():
    rng = np.random.default_rng(42)
    z, y = generate_latent(500, rng)
    mean_0 = z[y == 0].mean(axis=0)
    mean_1 = z[y == 1].mean(axis=0)
    assert mean_1[0] - mean_0[0] > 3.0


def test_latent_labels_balanced():
    rng = np.random.default_rng(42)
    z, y = generate_latent(100, rng)
    assert z.shape == (200, K_LATENT)
    assert np.sum(y == 0) == 100
    assert np.sum(y == 1) == 100


# ======================================================================
# Embedding
# ======================================================================

def test_base_embedding_orthonormal():
    U = generate_base_embedding(np.random.default_rng(42))
    assert U.shape == (D, K_LATENT)
    np.testing.assert_allclose(U.T @ U, np.eye(K_LATENT), atol=1e-10)


def test_sensor_embedding_orthonormal():
    U = generate_base_embedding(np.random.default_rng(42))
    perts = generate_perturbations(3, np.random.default_rng(100))
    for p in perts:
        for eps in [0.0, 0.3, 0.7, 1.0]:
            U_s = sensor_embedding(U, p, eps)
            np.testing.assert_allclose(U_s.T @ U_s, np.eye(K_LATENT), atol=1e-10)


def test_sensor_embedding_identity_at_eps_zero():
    U = generate_base_embedding(np.random.default_rng(42))
    perts = generate_perturbations(3, np.random.default_rng(100))
    for p in perts:
        U_s = sensor_embedding(U, p, 0.0)
        np.testing.assert_allclose(U_s, U, atol=1e-14)


def test_sensor_embeddings_diverge():
    U = generate_base_embedding(np.random.default_rng(42))
    perts = generate_perturbations(3, np.random.default_rng(100))
    Us_small = [sensor_embedding(U, p, 0.1) for p in perts]
    Us_large = [sensor_embedding(U, p, 0.8) for p in perts]
    diff_small = np.linalg.norm(Us_small[0] - Us_small[1])
    diff_large = np.linalg.norm(Us_large[0] - Us_large[1])
    assert diff_large > diff_small


# ======================================================================
# Dimension consistency
# ======================================================================

def test_all_sensor_data_same_dimension():
    rng = np.random.default_rng(42)
    z, y = generate_latent(50, rng)
    U = generate_base_embedding(np.random.default_rng(100))
    perts = generate_perturbations(3, np.random.default_rng(200))
    for pert in perts:
        U_s = sensor_embedding(U, pert, 0.5)
        X = generate_sensor_data(z, U_s, 1.0, np.random.default_rng(300))
        assert X.shape == (100, D)


def test_holonomy_computable():
    rng = np.random.default_rng(42)
    z, y = generate_latent(100, rng)
    U = generate_base_embedding(np.random.default_rng(100))
    perts = generate_perturbations(3, np.random.default_rng(200))
    Us = [sensor_embedding(U, p, 0.5) for p in perts]
    Xs = [generate_sensor_data(z, U_s, 1.0, np.random.default_rng(300 + i))
          for i, U_s in enumerate(Us)]
    hol, pw = compute_holonomy(Xs, PCA_K)
    assert isinstance(hol, float) and hol >= 0
    assert len(pw) == 3


# ======================================================================
# Core hypothesis mechanics
# ======================================================================

def test_holonomy_near_zero_at_eps_zero():
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U = generate_base_embedding(np.random.default_rng(100))
    Xs = [generate_sensor_data(z, U, OBS_NOISE_BASE, np.random.default_rng(42 + i))
          for i in range(3)]
    hol, _ = compute_holonomy(Xs, PCA_K)
    assert hol < 1.5, f"Holonomy at eps=0 should be small, got {hol:.4f}"


def test_holonomy_increases_with_perturbation():
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U = generate_base_embedding(np.random.default_rng(100))
    perts = generate_perturbations(3, np.random.default_rng(200))

    hols = []
    for eps in [0.0, 0.3, 0.7, 1.0]:
        Us = [sensor_embedding(U, p, eps) for p in perts]
        Xs = [generate_sensor_data(z, U_s, OBS_NOISE_BASE, np.random.default_rng(42 + i))
              for i, U_s in enumerate(Us)]
        hol, _ = compute_holonomy(Xs, PCA_K)
        hols.append(hol)

    assert hols[-1] > hols[0] + 0.3, \
        f"Holonomy should increase: {hols}"


# ======================================================================
# Signal calibration
# ======================================================================

def test_clean_sensor_classifies_well():
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U = generate_base_embedding(np.random.default_rng(100))
    X = generate_sensor_data(z, U, OBS_NOISE_BASE, np.random.default_rng(200))
    auc = within_sensor_auc(X, y)
    assert auc > AUC_THRESHOLD, f"Clean sensor AUC={auc:.3f}"


def test_noisy_sensor_degrades():
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U = generate_base_embedding(np.random.default_rng(100))
    noise_high = OBS_NOISE_BASE + OBS_NOISE_RATE * 1.0
    X = generate_sensor_data(z, U, noise_high, np.random.default_rng(200))
    auc = within_sensor_auc(X, y)
    assert auc < 0.85, f"High-noise sensor should degrade: AUC={auc:.3f}"


# ======================================================================
# Bootstrap null calibration
# ======================================================================

def test_bootstrap_has_variance():
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U = generate_base_embedding(np.random.default_rng(100))
    boot = bootstrap_baseline(z, y, U, 100, np.random.default_rng(300))
    assert boot.std() > 0.001


def test_eps_zero_not_significant_vs_bootstrap():
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U = generate_base_embedding(np.random.default_rng(100))
    boot = bootstrap_baseline(z, y, U, 200, np.random.default_rng(300))

    Xs = [generate_sensor_data(z, U, OBS_NOISE_BASE, np.random.default_rng(42 + i))
          for i in range(3)]
    hol, _ = compute_holonomy(Xs, PCA_K)
    p = float(np.mean(boot >= hol))
    assert p > HOLONOMY_ALPHA, f"eps=0 should not be significant (p={p:.3f})"


# ======================================================================
# NEGATIVE CONTROL: identical perturbations → H7b must fail
# ======================================================================

def test_negative_control_identical_perturbation():
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U = generate_base_embedding(np.random.default_rng(100))
    boot = bootstrap_baseline(z, y, U, 200, np.random.default_rng(300))
    same_pert = np.random.default_rng(400).standard_normal((D, K_LATENT))

    rows = []
    for eps in [0.0, 0.3, 0.5, 0.7, 1.0]:
        U_s = sensor_embedding(U, same_pert, eps)
        obs_noise = OBS_NOISE_BASE + OBS_NOISE_RATE * eps
        Xs = [generate_sensor_data(z, U_s, obs_noise, np.random.default_rng(42 + i))
              for i in range(3)]
        hol, _ = compute_holonomy(Xs, PCA_K)
        p = float(np.mean(boot >= hol))
        auc_min = min(within_sensor_auc(X, y) for X in Xs)
        rows.append({"epsilon": eps, "min_auc": auc_min, "holonomy_p": p})

    verdict = analyze_seed(rows)
    assert verdict["H7b_confirmed"] is False, \
        "Identical perturbation (no inconsistency) must NOT confirm H7b"


# ======================================================================
# Analysis logic
# ======================================================================

def test_analyze_h7b_confirmed():
    rows = [
        {"epsilon": 0.0, "min_auc": 0.95, "holonomy_p": 0.60},
        {"epsilon": 0.2, "min_auc": 0.88, "holonomy_p": 0.02},
        {"epsilon": 0.5, "min_auc": 0.78, "holonomy_p": 0.001},
        {"epsilon": 0.8, "min_auc": 0.65, "holonomy_p": 0.001},
    ]
    result = analyze_seed(rows)
    assert result["H7b_confirmed"] is True
    assert result["H7c_confirmed"] is True


def test_analyze_h7b_fails():
    rows = [
        {"epsilon": 0.0, "min_auc": 0.95, "holonomy_p": 0.60},
        {"epsilon": 0.5, "min_auc": 0.55, "holonomy_p": 0.30},
    ]
    result = analyze_seed(rows)
    assert result["H7b_confirmed"] is False


# ======================================================================
# Reproducibility
# ======================================================================

def test_latent_reproducible():
    z1, y1 = generate_latent(50, np.random.default_rng(42))
    z2, y2 = generate_latent(50, np.random.default_rng(42))
    np.testing.assert_array_equal(z1, z2)


def test_embedding_reproducible():
    U1 = generate_base_embedding(np.random.default_rng(42))
    U2 = generate_base_embedding(np.random.default_rng(42))
    np.testing.assert_array_equal(U1, U2)
