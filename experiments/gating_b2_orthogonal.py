"""Experiment B2: orthogonal noise/perturbation axes.

Separates observation noise (eps_noise) from cross-sensor embedding
perturbation (eps_pert) into two independent axes. This resolves the
confound in v3/B1 where a single epsilon drove both effects, making
all metrics track noise scaling rather than cross-sensor inconsistency.

See PREREGISTRATION_V4.md for the full protocol.

Primary endpoint: Spearman rho(mean-edge-geodesic, eps_pert) at each
fixed noise level, with bootstrap CI. Tests whether edge geodesics
track cross-sensor inconsistency when noise is held constant.

Run with:
    uv run --no-project --with numpy --with scipy --with scikit-learn \
        --with matplotlib --with tqdm python experiments/gating_b2_orthogonal.py
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats, linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "biomedical-cohort-transfer" / "src"))

from transportability import top_k_subspace, cocycle_holonomy, geodesic_distance


# ======================================================================
# Frozen parameters — carried from v3 where applicable
# ======================================================================

D = 100
K_LATENT = 3
LATENT_SIGNAL = 5.0
PERTURBATION_SCALE = 1.5
PCA_K = 3
N_SAMPLES_PER_CLASS = 200

OBS_NOISE_BASE = 1.0
OBS_NOISE_RATE = 3.5

N_EPS_STEPS = 20
EPS_PERT_RANGE = np.linspace(0.0, 1.0, N_EPS_STEPS)

FIXED_NOISE_LEVELS = [1.0, 2.0, 3.0, 4.5]

SANITY_SEEDS = list(range(20260850, 20260860))
PRIMARY_SEEDS = list(range(20260860, 20260910))
NEGCTRL_SEEDS = list(range(20260910, 20260960))
NOISE_CTRL_SEEDS = list(range(20260960, 20261010))
N_BOOTSTRAP_CI = 10000
ALPHA = 0.05

NOISE_SWEEP_LEVELS = np.linspace(1.0, 4.5, N_EPS_STEPS)


# ======================================================================
# Data generators — two-axis version
# ======================================================================

def generate_latent(n_per_class, rng):
    n = 2 * n_per_class
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    z = rng.standard_normal((n, K_LATENT))
    mu = np.zeros(K_LATENT)
    mu[0] = LATENT_SIGNAL
    for i in range(n):
        z[i] += mu * y[i]
    return z, y


def generate_base_embedding(rng):
    A = rng.standard_normal((D, K_LATENT))
    Q, _ = linalg.qr(A, mode='economic')
    return Q


def generate_perturbations(n_sensors, rng):
    return [rng.standard_normal((D, K_LATENT)) for _ in range(n_sensors)]


def sensor_embedding(U_base, perturbation, eps_pert):
    W = U_base + eps_pert * PERTURBATION_SCALE * perturbation
    Q, _ = linalg.qr(W, mode='economic')
    return Q


def generate_sensor_data(z, U_s, obs_noise, rng):
    X = z @ U_s.T + rng.normal(0, obs_noise, (len(z), D))
    return X


# ======================================================================
# Metrics
# ======================================================================

def mean_edge_geodesics(Xs, k):
    subspaces = [top_k_subspace(X, k)[0] for X in Xs]
    n = len(subspaces)
    pairwise = []
    for i in range(n):
        j = (i + 1) % n
        pairwise.append(geodesic_distance(subspaces[i], subspaces[j]))
    return float(np.mean(pairwise)), [float(g) for g in pairwise]


def compute_holonomy(Xs, k):
    subspaces = [top_k_subspace(X, k)[0] for X in Xs]
    _, hol_norm = cocycle_holonomy(subspaces)
    return float(hol_norm)


# ======================================================================
# Single-seed run at one fixed noise level
# ======================================================================

def run_seed_at_noise(seed, fixed_noise, identical_perturbation=False):
    """Sweep eps_pert at a fixed observation noise level."""
    data_rng = np.random.default_rng(seed)
    proj_rng = np.random.default_rng(seed + 1000)
    pert_rng = np.random.default_rng(seed + 3000)

    z, y = generate_latent(N_SAMPLES_PER_CLASS, data_rng)
    U_base = generate_base_embedding(proj_rng)

    if identical_perturbation:
        single_pert = pert_rng.standard_normal((D, K_LATENT))
        perturbations = [single_pert, single_pert, single_pert]
    else:
        perturbations = generate_perturbations(3, pert_rng)

    eps_perts = []
    hols = []
    geos = []

    for step, eps_p in enumerate(EPS_PERT_RANGE):
        Us = [sensor_embedding(U_base, p, eps_p) for p in perturbations]
        obs_rng = np.random.default_rng(seed * 100 + step)
        Xs = [generate_sensor_data(z, U_s, fixed_noise, obs_rng) for U_s in Us]

        hol = compute_holonomy(Xs, PCA_K)
        geo, per_edge = mean_edge_geodesics(Xs, PCA_K)

        eps_perts.append(float(eps_p))
        hols.append(hol)
        geos.append(geo)

    rho_geo, p_geo = stats.spearmanr(eps_perts, geos)
    rho_hol, p_hol = stats.spearmanr(eps_perts, hols)

    return {
        "seed": seed,
        "fixed_noise": fixed_noise,
        "eps_perts": eps_perts,
        "holonomies": hols,
        "geodesics": geos,
        "rho_geo": float(rho_geo),
        "p_geo": float(p_geo),
        "rho_hol": float(rho_hol),
        "p_hol": float(p_hol),
    }


# ======================================================================
# Noise-axis control: sweep noise at eps_pert=0
# ======================================================================

def run_seed_noise_axis(seed):
    """Sweep observation noise at eps_pert=0 (no perturbation).

    This is the nuisance axis: geodesics should rise monotonically
    because noise degrades PCA subspace estimation, even with no
    cross-sensor inconsistency. Reported as a characterized control
    cell so reviewers can see the nuisance is documented, not hidden.
    """
    data_rng = np.random.default_rng(seed)
    proj_rng = np.random.default_rng(seed + 1000)
    pert_rng = np.random.default_rng(seed + 3000)

    z, y = generate_latent(N_SAMPLES_PER_CLASS, data_rng)
    U_base = generate_base_embedding(proj_rng)
    perturbations = generate_perturbations(3, pert_rng)

    noises = []
    geos = []
    hols = []

    for step, noise_val in enumerate(NOISE_SWEEP_LEVELS):
        Us = [sensor_embedding(U_base, p, 0.0) for p in perturbations]
        obs_rng = np.random.default_rng(seed * 100 + step)
        Xs = [generate_sensor_data(z, U_s, float(noise_val), obs_rng) for U_s in Us]

        geo, _ = mean_edge_geodesics(Xs, PCA_K)
        hol = compute_holonomy(Xs, PCA_K)

        noises.append(float(noise_val))
        geos.append(geo)
        hols.append(hol)

    rho_geo, p_geo = stats.spearmanr(noises, geos)
    rho_hol, p_hol = stats.spearmanr(noises, hols)

    return {
        "seed": seed,
        "noise_levels": noises,
        "geodesics": geos,
        "holonomies": hols,
        "rho_geo": float(rho_geo),
        "p_geo": float(p_geo),
        "rho_hol": float(rho_hol),
        "p_hol": float(p_hol),
    }


# ======================================================================
# Bootstrap CI
# ======================================================================

def bootstrap_median_ci(values, n_bootstrap=N_BOOTSTRAP_CI, alpha=ALPHA):
    values = np.asarray(values)
    rng = np.random.default_rng(42)
    boot_medians = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(len(values), size=len(values), replace=True)
        boot_medians[b] = np.median(values[idx])
    ci_lo = float(np.percentile(boot_medians, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return float(np.median(values)), ci_lo, ci_hi


def benjamini_hochberg(pvals, alpha=0.05):
    pvals = np.asarray(pvals)
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    adjusted = np.zeros(n)
    adjusted[sorted_idx[-1]] = sorted_pvals[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(
            adjusted[sorted_idx[i + 1]],
            sorted_pvals[i] * n / (i + 1)
        )
    reject = adjusted <= alpha
    return reject.tolist(), adjusted.tolist()


# ======================================================================
# Main experiment
# ======================================================================

def run_experiment():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] B2 Experiment: orthogonal noise/perturbation axes")
    print(f"  Fixed noise levels: {FIXED_NOISE_LEVELS}")
    print(f"  Perturbation sweep: {N_EPS_STEPS} steps in [0, 1]")
    print(f"  Primary seeds: {PRIMARY_SEEDS[0]}–{PRIMARY_SEEDS[-1]} ({len(PRIMARY_SEEDS)} seeds)")
    print(f"  NegCtrl seeds: {NEGCTRL_SEEDS[0]}–{NEGCTRL_SEEDS[-1]} ({len(NEGCTRL_SEEDS)} seeds)")
    print()

    summary = {"noise_levels": {}}

    for noise_level in FIXED_NOISE_LEVELS:
        print(f"\n{'='*65}")
        print(f"NOISE LEVEL: σ = {noise_level:.1f}")
        print(f"{'='*65}")

        # --- Primary: different perturbation ---
        print(f"\n  PRIMARY (different perturbation):")
        primary_results = []
        for seed in tqdm(PRIMARY_SEEDS, desc=f"  σ={noise_level:.1f} primary"):
            r = run_seed_at_noise(seed, noise_level, identical_perturbation=False)
            primary_results.append(r)

        rho_geos = [r["rho_geo"] for r in primary_results]
        p_geos = [r["p_geo"] for r in primary_results]
        rho_hols = [r["rho_hol"] for r in primary_results]

        med_rho_geo, ci_lo_geo, ci_hi_geo = bootstrap_median_ci(rho_geos)
        med_rho_hol, ci_lo_hol, ci_hi_hol = bootstrap_median_ci(rho_hols)

        geo_ci_excludes_zero = (ci_lo_geo > 0) or (ci_hi_geo < 0)
        hol_ci_includes_zero = (ci_lo_hol <= 0 <= ci_hi_hol)

        reject_bh, _ = benjamini_hochberg(p_geos, ALPHA)
        n_sig = sum(reject_bh)

        print(f"    ρ_geo: median={med_rho_geo:+.4f}, "
              f"95% CI=[{ci_lo_geo:+.4f}, {ci_hi_geo:+.4f}], "
              f"excludes 0: {geo_ci_excludes_zero}")
        print(f"    ρ_hol: median={med_rho_hol:+.4f}, "
              f"95% CI=[{ci_lo_hol:+.4f}, {ci_hi_hol:+.4f}], "
              f"includes 0: {hol_ci_includes_zero}")
        print(f"    Per-seed geo significant (BH): {n_sig}/{len(PRIMARY_SEEDS)}")

        # --- Negative control: identical perturbation ---
        print(f"\n  NEGATIVE CONTROL (identical perturbation):")
        negctrl_results = []
        for seed in tqdm(NEGCTRL_SEEDS, desc=f"  σ={noise_level:.1f} negctrl"):
            r = run_seed_at_noise(seed, noise_level, identical_perturbation=True)
            negctrl_results.append(r)

        nc_rho_geos = [r["rho_geo"] for r in negctrl_results]
        nc_med, nc_lo, nc_hi = bootstrap_median_ci(nc_rho_geos)
        nc_ci_includes_zero = (nc_lo <= 0 <= nc_hi)

        print(f"    ρ_geo: median={nc_med:+.4f}, "
              f"95% CI=[{nc_lo:+.4f}, {nc_hi:+.4f}], "
              f"includes 0: {nc_ci_includes_zero}")

        # --- Verdict at this noise level ---
        if not nc_ci_includes_zero:
            verdict = "NEGATIVE_CONTROL_FAILED"
        elif geo_ci_excludes_zero:
            if hol_ci_includes_zero:
                verdict = "B2_CONFIRMED"
            else:
                verdict = "B2_CONFIRMED_HOL_ALSO_MONOTONE"
        else:
            verdict = "B2_FAILED"

        print(f"\n    VERDICT at σ={noise_level:.1f}: {verdict}")

        summary["noise_levels"][str(noise_level)] = {
            "verdict": verdict,
            "primary": {
                "median_rho_geo": med_rho_geo,
                "ci_lo_geo": ci_lo_geo,
                "ci_hi_geo": ci_hi_geo,
                "geo_ci_excludes_zero": geo_ci_excludes_zero,
                "median_rho_hol": med_rho_hol,
                "ci_lo_hol": ci_lo_hol,
                "ci_hi_hol": ci_hi_hol,
                "hol_ci_includes_zero": hol_ci_includes_zero,
                "n_sig_bh": n_sig,
                "per_seed": [{
                    "seed": r["seed"],
                    "rho_geo": r["rho_geo"],
                    "p_geo": r["p_geo"],
                    "rho_hol": r["rho_hol"],
                } for r in primary_results],
            },
            "negative_control": {
                "median_rho_geo": nc_med,
                "ci_lo": nc_lo,
                "ci_hi": nc_hi,
                "ci_includes_zero": nc_ci_includes_zero,
                "per_seed": [{
                    "seed": r["seed"],
                    "rho_geo": r["rho_geo"],
                    "rho_hol": r["rho_hol"],
                } for r in negctrl_results],
            },
        }

    # --- Noise-axis control: sweep noise at eps_pert=0 ---
    print(f"\n\n{'='*65}")
    print("NOISE-AXIS CONTROL (sweep noise at eps_pert=0)")
    print(f"{'='*65}")
    noise_ctrl_results = []
    for seed in tqdm(NOISE_CTRL_SEEDS, desc="  noise-axis control"):
        r = run_seed_noise_axis(seed)
        noise_ctrl_results.append(r)

    nc_noise_rhos = [r["rho_geo"] for r in noise_ctrl_results]
    nc_noise_med, nc_noise_lo, nc_noise_hi = bootstrap_median_ci(nc_noise_rhos)
    print(f"  ρ(geodesic, noise): median={nc_noise_med:+.4f}, "
          f"95% CI=[{nc_noise_lo:+.4f}, {nc_noise_hi:+.4f}]")
    print(f"  (Expected: strongly positive — documents the nuisance axis)")

    summary["noise_axis_control"] = {
        "n_seeds": len(NOISE_CTRL_SEEDS),
        "median_rho_geo": nc_noise_med,
        "ci_lo": nc_noise_lo,
        "ci_hi": nc_noise_hi,
        "per_seed": [{
            "seed": r["seed"],
            "rho_geo": r["rho_geo"],
            "rho_hol": r["rho_hol"],
        } for r in noise_ctrl_results],
    }

    # --- Degradation curve ---
    print(f"\n{'='*65}")
    print("DEGRADATION CURVE: ρ_geo vs noise level")
    print(f"{'='*65}")
    deg_curve = []
    for nl in FIXED_NOISE_LEVELS:
        d = summary["noise_levels"][str(nl)]
        deg_curve.append({
            "noise": nl,
            "median_rho_geo": d["primary"]["median_rho_geo"],
            "ci_lo": d["primary"]["ci_lo_geo"],
            "ci_hi": d["primary"]["ci_hi_geo"],
            "significant": d["primary"]["geo_ci_excludes_zero"],
        })
        print(f"  σ={nl:.1f}: ρ_geo={d['primary']['median_rho_geo']:+.4f} "
              f"[{d['primary']['ci_lo_geo']:+.4f}, {d['primary']['ci_hi_geo']:+.4f}] "
              f"{'*' if d['primary']['geo_ci_excludes_zero'] else ''}")

    rho_degrade, _ = stats.spearmanr(
        [d["noise"] for d in deg_curve],
        [d["median_rho_geo"] for d in deg_curve])
    print(f"  Degradation ρ(median_rho_geo, σ) = {rho_degrade:+.3f}")

    summary["degradation_curve"] = deg_curve
    summary["degradation_rho"] = float(rho_degrade)

    # --- Overall verdict ---
    print(f"\n\n{'='*65}")
    print("OVERALL VERDICTS")
    print(f"{'='*65}")
    for nl in FIXED_NOISE_LEVELS:
        v = summary["noise_levels"][str(nl)]["verdict"]
        print(f"  σ={nl:.1f}: {v}")

    json_path = out_dir / "gating_b2_orthogonal.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # --- Plot ---
    plot_results(summary, out_dir)

    return summary


def plot_results(summary, out_dir):
    n_levels = len(FIXED_NOISE_LEVELS)
    fig, axes = plt.subplots(2, n_levels, figsize=(4 * n_levels, 8), squeeze=False)

    for col, noise_level in enumerate(FIXED_NOISE_LEVELS):
        data = summary["noise_levels"][str(noise_level)]

        primary_rhos = [s["rho_geo"] for s in data["primary"]["per_seed"]]
        negctrl_rhos = [s["rho_geo"] for s in data["negative_control"]["per_seed"]]

        ax_top = axes[0, col]
        ax_top.hist(primary_rhos, bins=15, alpha=0.7, color='steelblue',
                    edgecolor='black', linewidth=0.5)
        ax_top.axvline(0, color='red', linestyle='--', linewidth=1)
        ax_top.axvline(data["primary"]["median_rho_geo"], color='navy',
                       linestyle='-', linewidth=2, label='median')
        ax_top.set_title(f'σ={noise_level:.1f}\n{data["verdict"]}', fontsize=10)
        ax_top.set_xlabel('ρ(geodesic, ε_pert)')
        if col == 0:
            ax_top.set_ylabel('Primary\n(different pert)')
        ax_top.legend(fontsize=8)

        ax_bot = axes[1, col]
        ax_bot.hist(negctrl_rhos, bins=15, alpha=0.7, color='coral',
                    edgecolor='black', linewidth=0.5)
        ax_bot.axvline(0, color='red', linestyle='--', linewidth=1)
        ax_bot.axvline(data["negative_control"]["median_rho_geo"], color='darkred',
                       linestyle='-', linewidth=2, label='median')
        ax_bot.set_xlabel('ρ(geodesic, ε_pert)')
        if col == 0:
            ax_bot.set_ylabel('Negative control\n(identical pert)')
        ax_bot.legend(fontsize=8)

    fig.suptitle('B2: Edge geodesic monotonicity vs perturbation\nat fixed noise levels',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    plot_path = out_dir / "gating_b2_orthogonal.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    run_experiment()
