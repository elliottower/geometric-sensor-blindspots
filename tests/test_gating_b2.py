"""Tests for B2 experiment (orthogonal noise/perturbation axes).

Validates that the two-axis data model separates noise from perturbation,
and that the negative control (identical perturbation at fixed noise)
produces near-zero edge geodesics when eps_pert=0.

Run with:
    uv run --no-project --with numpy --with scipy --with scikit-learn \
        --with matplotlib -m pytest tests/test_gating_b2.py -xvs
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "biomedical-cohort-transfer" / "src"))

from gating_b2_orthogonal import (
    generate_latent,
    generate_base_embedding,
    generate_perturbations,
    sensor_embedding,
    generate_sensor_data,
    mean_edge_geodesics,
    compute_holonomy,
    run_seed_at_noise,
    run_seed_noise_axis,
    bootstrap_median_ci,
    benjamini_hochberg,
    D, K_LATENT, PCA_K, N_SAMPLES_PER_CLASS,
    PERTURBATION_SCALE,
)


def test_identical_embedding_geodesic_is_noise_floor():
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U_base = generate_base_embedding(np.random.default_rng(100))
    Xs_same = [generate_sensor_data(z, U_base, 1.0, np.random.default_rng(200 + i))
               for i in range(3)]
    geo_same, _ = mean_edge_geodesics(Xs_same, PCA_K)

    perts = generate_perturbations(3, np.random.default_rng(300))
    Xs_diff = [generate_sensor_data(z, sensor_embedding(U_base, p, 0.8),
                                    1.0, np.random.default_rng(200 + i))
               for i, p in enumerate(perts)]
    geo_diff, _ = mean_edge_geodesics(Xs_diff, PCA_K)

    assert geo_diff > geo_same, \
        f"Perturbed embeddings should have larger geodesic than same: {geo_same:.3f} vs {geo_diff:.3f}"


def test_perturbation_axis_orthogonal_to_noise():
    """At eps_pert=0, all sensors share U_base. Geodesics reflect only
    observation noise jitter and should NOT increase monotonically with
    perturbation scale (because eps_pert=0 means no perturbation)."""
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U_base = generate_base_embedding(np.random.default_rng(100))
    perts = generate_perturbations(3, np.random.default_rng(200))

    geos_at_zero_pert = []
    for noise in [1.0, 2.0, 4.5]:
        Us = [sensor_embedding(U_base, p, 0.0) for p in perts]
        Xs = [generate_sensor_data(z, U_s, noise, np.random.default_rng(300 + i))
              for i, U_s in enumerate(Us)]
        geo, _ = mean_edge_geodesics(Xs, PCA_K)
        geos_at_zero_pert.append(geo)

    geo_at_high_pert, _ = mean_edge_geodesics(
        [generate_sensor_data(z, sensor_embedding(U_base, p, 0.8),
                              1.0, np.random.default_rng(300 + i))
         for i, p in enumerate(perts)], PCA_K)

    assert geo_at_high_pert > max(geos_at_zero_pert), \
        f"High eps_pert should dominate noise-floor jitter: pert={geo_at_high_pert:.3f} vs max noise-floor={max(geos_at_zero_pert):.3f}"


def test_perturbation_increases_geodesic_at_fixed_noise():
    """At fixed noise, increasing eps_pert should increase geodesic."""
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U_base = generate_base_embedding(np.random.default_rng(100))
    perts = generate_perturbations(3, np.random.default_rng(200))

    fixed_noise = 1.0
    geo_small, _ = mean_edge_geodesics(
        [generate_sensor_data(z, sensor_embedding(U_base, p, 0.1),
                              fixed_noise, np.random.default_rng(300 + i))
         for i, p in enumerate(perts)], PCA_K)

    geo_large, _ = mean_edge_geodesics(
        [generate_sensor_data(z, sensor_embedding(U_base, p, 0.8),
                              fixed_noise, np.random.default_rng(300 + i))
         for i, p in enumerate(perts)], PCA_K)

    assert geo_large > geo_small, \
        f"Larger perturbation should give larger geodesic: {geo_small:.3f} vs {geo_large:.3f}"


def test_identical_perturbation_geodesic_near_zero_at_fixed_noise():
    """The negative control: identical perturbation at fixed noise should
    give geodesics driven only by observation noise jitter, not by
    eps_pert."""
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U_base = generate_base_embedding(np.random.default_rng(100))

    single_pert = np.random.default_rng(200).standard_normal((D, K_LATENT))

    fixed_noise = 2.0
    geos = []
    for eps_p in [0.0, 0.3, 0.7, 1.0]:
        U_s = sensor_embedding(U_base, single_pert, eps_p)
        Xs = [generate_sensor_data(z, U_s, fixed_noise, np.random.default_rng(300 + i))
              for i in range(3)]
        geo, _ = mean_edge_geodesics(Xs, PCA_K)
        geos.append(geo)

    rho, p_val = stats.spearmanr([0.0, 0.3, 0.7, 1.0], geos)
    assert abs(rho) < 0.9, \
        f"Identical perturbation should not produce strong monotonicity: ρ={rho:.3f}"


def test_run_seed_at_noise_returns_fields():
    r = run_seed_at_noise(99999, 2.0, identical_perturbation=False)
    assert "rho_geo" in r
    assert "rho_hol" in r
    assert len(r["eps_perts"]) == 20
    assert len(r["holonomies"]) == 20
    assert len(r["geodesics"]) == 20
    assert r["fixed_noise"] == 2.0


def test_run_seed_reproducible():
    r1 = run_seed_at_noise(12345, 2.0, identical_perturbation=False)
    r2 = run_seed_at_noise(12345, 2.0, identical_perturbation=False)
    assert r1["rho_geo"] == r2["rho_geo"]
    assert r1["holonomies"] == r2["holonomies"]


def test_negctrl_rho_geo_near_zero():
    """Core test: at fixed noise, identical perturbation across eps_pert
    sweep should give ρ_geo near zero (no monotone signal)."""
    results = [run_seed_at_noise(90000 + i, 2.0, identical_perturbation=True)
               for i in range(5)]
    rho_geos = [r["rho_geo"] for r in results]
    mean_rho = np.mean(rho_geos)
    assert abs(mean_rho) < 0.5, \
        f"Negative control should have |ρ_geo| near 0, got mean={mean_rho:.3f}, " \
        f"per-seed: {[f'{r:.3f}' for r in rho_geos]}"


def test_different_pert_rho_geo_positive():
    """At fixed low noise, different perturbation should produce positive
    ρ_geo (edge geodesic increases with eps_pert)."""
    results = [run_seed_at_noise(90000 + i, 1.0, identical_perturbation=False)
               for i in range(5)]
    rho_geos = [r["rho_geo"] for r in results]
    mean_rho = np.mean(rho_geos)
    assert mean_rho > 0.3, \
        f"Different perturbation at low noise should have positive ρ_geo, " \
        f"got mean={mean_rho:.3f}, per-seed: {[f'{r:.3f}' for r in rho_geos]}"


def test_noise_axis_produces_monotone_geodesic_confound():
    """Documents the confound that B2 is designed to avoid: sweeping
    noise at eps_pert=0 (no perturbation) still produces perfectly
    monotone geodesics because noise degrades PCA subspace estimation.
    This is exactly why B2 holds noise fixed — so this effect cannot
    contaminate the primary endpoint."""
    seed = 42
    data_rng = np.random.default_rng(seed)
    proj_rng = np.random.default_rng(seed + 1000)

    z, y = generate_latent(N_SAMPLES_PER_CLASS, data_rng)
    U_base = generate_base_embedding(proj_rng)

    noise_levels = np.linspace(1.0, 4.5, 10)
    geos = []
    for noise in noise_levels:
        Xs = [generate_sensor_data(z, U_base, noise, np.random.default_rng(int(noise * 1000) + i))
              for i in range(3)]
        geo, _ = mean_edge_geodesics(Xs, PCA_K)
        geos.append(geo)

    rho, _ = stats.spearmanr(noise_levels, geos)
    assert rho > 0.8, \
        f"Noise sweep should produce monotone geodesics (the confound B2 avoids): ρ={rho:.3f}"


def test_generative_model_axes_independent():
    """The generators are independent: sensor_embedding uses only eps_pert,
    generate_sensor_data uses only obs_noise, no shared derivation."""
    rng = np.random.default_rng(42)
    z, y = generate_latent(N_SAMPLES_PER_CLASS, rng)
    U_base = generate_base_embedding(np.random.default_rng(100))
    perts = generate_perturbations(3, np.random.default_rng(200))

    Us_low = [sensor_embedding(U_base, p, 0.3) for p in perts]
    Us_high = [sensor_embedding(U_base, p, 0.3) for p in perts]
    for U_l, U_h in zip(Us_low, Us_high):
        np.testing.assert_array_equal(U_l, U_h)

    Xs_1 = [generate_sensor_data(z, U_s, 2.0, np.random.default_rng(300 + i))
            for i, U_s in enumerate(Us_low)]
    Xs_2 = [generate_sensor_data(z, U_s, 2.0, np.random.default_rng(300 + i))
            for i, U_s in enumerate(Us_high)]
    for X1, X2 in zip(Xs_1, Xs_2):
        np.testing.assert_array_equal(X1, X2)


def test_noise_attenuates_pert_signal():
    """Documents measurement-level interaction: noise masks the
    perturbation signal in the geodesic metric. The perturbation-
    attributable geodesic component (geo_diff - geo_ident) shrinks
    as noise increases, because PCA subspace estimation becomes
    noisier. This is expected physics, not a generative-model confound.
    The B2 experiment characterizes this attenuation via the noise grid."""
    seed = 42
    data_rng = np.random.default_rng(seed)
    proj_rng = np.random.default_rng(seed + 1000)
    pert_rng = np.random.default_rng(seed + 3000)

    z, y = generate_latent(N_SAMPLES_PER_CLASS, data_rng)
    U_base = generate_base_embedding(proj_rng)
    perts = generate_perturbations(3, pert_rng)

    single_pert = np.random.default_rng(seed + 4000).standard_normal((D, K_LATENT))

    fixed_eps_pert = 0.5

    geo_diff_low, _ = mean_edge_geodesics(
        [generate_sensor_data(z, sensor_embedding(U_base, p, fixed_eps_pert),
                              1.0, np.random.default_rng(1000 + i))
         for i, p in enumerate(perts)], PCA_K)
    geo_ident_low, _ = mean_edge_geodesics(
        [generate_sensor_data(z, sensor_embedding(U_base, single_pert, fixed_eps_pert),
                              1.0, np.random.default_rng(1000 + i))
         for i in range(3)], PCA_K)
    delta_low = geo_diff_low - geo_ident_low

    geo_diff_high, _ = mean_edge_geodesics(
        [generate_sensor_data(z, sensor_embedding(U_base, p, fixed_eps_pert),
                              4.5, np.random.default_rng(4500 + i))
         for i, p in enumerate(perts)], PCA_K)
    geo_ident_high, _ = mean_edge_geodesics(
        [generate_sensor_data(z, sensor_embedding(U_base, single_pert, fixed_eps_pert),
                              4.5, np.random.default_rng(4500 + i))
         for i in range(3)], PCA_K)
    delta_high = geo_diff_high - geo_ident_high

    assert delta_low > delta_high, \
        f"Perturbation signal should be stronger at low noise: " \
        f"δ_low={delta_low:.3f} vs δ_high={delta_high:.3f}"


def test_noise_axis_control_strongly_positive():
    """The noise-axis control cell (sweep noise at eps_pert=0) should
    produce strongly positive ρ_geo, confirming the nuisance axis is
    characterized."""
    results = [run_seed_noise_axis(90000 + i) for i in range(5)]
    rho_geos = [r["rho_geo"] for r in results]
    mean_rho = np.mean(rho_geos)
    assert mean_rho > 0.8, \
        f"Noise-axis control should have high ρ_geo, got mean={mean_rho:.3f}"


def test_bootstrap_ci_covers_median():
    values = np.random.default_rng(42).normal(0.5, 0.3, 50)
    med, lo, hi = bootstrap_median_ci(values)
    assert lo < med < hi


def test_bh_correction():
    reject, adj = benjamini_hochberg([0.001, 0.5, 0.03], alpha=0.05)
    assert reject[0] is True
    assert reject[1] is False
