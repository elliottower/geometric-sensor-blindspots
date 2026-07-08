"""Experiment B1: edge-level vs loop-level monotonicity.

Tests whether pairwise geodesic distance tracks cross-sensor
inconsistency more monotonically than cocycle holonomy around the
sensor loop. See PREREGISTRATION_V3.md for the full protocol.

Primary endpoint: median Δρ = ρ(mean-edge-geodesic, ε) − ρ(loop-holonomy, ε)
with bootstrap 95% CI excluding 0.

Negative control: identical-perturbation sensors → median Δρ CI includes 0.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "biomedical-cohort-transfer" / "src"))

from gating_h7b_v3 import (
    generate_latent,
    generate_base_embedding,
    generate_perturbations,
    sensor_embedding,
    generate_sensor_data,
    compute_holonomy,
    D, K_LATENT, OBS_NOISE_BASE, OBS_NOISE_RATE,
    PERTURBATION_SCALE, PCA_K, EPSILON_RANGE,
)
from transportability import geodesic_distance, top_k_subspace

PRIMARY_SEEDS = list(range(20260750, 20260800))
NEGCTRL_SEEDS = list(range(20260800, 20260850))
N_SAMPLES_PER_CLASS = 200
N_BOOTSTRAP_CI = 10000
ALPHA = 0.05
ONSET_WINDOW_IDX = slice(0, 5)


# ======================================================================
# The three frozen endpoint functions
# ======================================================================

def mean_edge_geodesics(Xs, k):
    """Mean of all three pairwise geodesic distances at one epsilon.

    Pre-committed aggregation: mean of three edges, not best-two or
    any other selection. Returns (mean_geodesic, list_of_three).
    """
    subspaces = [top_k_subspace(X, k)[0] for X in Xs]
    n = len(subspaces)
    pairwise = []
    for i in range(n):
        j = (i + 1) % n
        pairwise.append(geodesic_distance(subspaces[i], subspaces[j]))
    return float(np.mean(pairwise)), [float(g) for g in pairwise]


def compute_delta_rho(epsilons, loop_holonomies, mean_geodesics):
    """Δρ = Spearman(mean-edge-geodesic, ε) − Spearman(loop-holonomy, ε).

    Computed over the full epsilon range. Returns (delta_rho, rho_edge,
    rho_loop, p_edge, p_loop).
    """
    rho_edge, p_edge = stats.spearmanr(epsilons, mean_geodesics)
    rho_loop, p_loop = stats.spearmanr(epsilons, loop_holonomies)
    delta_rho = float(rho_edge) - float(rho_loop)
    return delta_rho, float(rho_edge), float(rho_loop), float(p_edge), float(p_loop)


def bootstrap_median_ci(values, n_bootstrap=N_BOOTSTRAP_CI, alpha=ALPHA):
    """Bootstrap 95% CI for the median of a 1-D array.

    Resamples the array n_bootstrap times, computes the median of each
    resample, returns (median, ci_lo, ci_hi). CI bounds are at
    alpha/2 and 1-alpha/2 percentiles of the bootstrap distribution.
    """
    values = np.asarray(values)
    rng = np.random.default_rng(42)
    boot_medians = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(len(values), size=len(values), replace=True)
        boot_medians[b] = np.median(values[idx])
    ci_lo = float(np.percentile(boot_medians, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return float(np.median(values)), ci_lo, ci_hi


# ======================================================================
# Single-seed run
# ======================================================================

def run_seed(seed, identical_perturbation=False):
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

    epsilons = []
    loop_hols = []
    mean_geos = []
    all_geos = []

    for step, eps in enumerate(EPSILON_RANGE):
        obs_noise = OBS_NOISE_BASE + OBS_NOISE_RATE * eps
        Us = [sensor_embedding(U_base, p, eps) for p in perturbations]
        obs_rng = np.random.default_rng(seed * 100 + step)
        Xs = [generate_sensor_data(z, U_s, obs_noise, obs_rng) for U_s in Us]

        hol_norm, _ = compute_holonomy(Xs, PCA_K)
        mean_geo, per_edge = mean_edge_geodesics(Xs, PCA_K)

        epsilons.append(float(eps))
        loop_hols.append(hol_norm)
        mean_geos.append(mean_geo)
        all_geos.append(per_edge)

    delta_rho, rho_edge, rho_loop, p_edge, p_loop = compute_delta_rho(
        epsilons, loop_hols, mean_geos)

    onset_eps = [epsilons[i] for i in range(*ONSET_WINDOW_IDX.indices(len(epsilons)))]
    onset_geos = [mean_geos[i] for i in range(*ONSET_WINDOW_IDX.indices(len(mean_geos)))]
    if len(onset_eps) >= 2:
        slope_result = stats.linregress(onset_eps, onset_geos)
        onset_slope = float(slope_result.slope)
    else:
        onset_slope = float('nan')

    return {
        "seed": seed,
        "delta_rho": delta_rho,
        "rho_edge": rho_edge,
        "rho_loop": rho_loop,
        "p_edge": p_edge,
        "p_loop": p_loop,
        "onset_slope": onset_slope,
        "epsilons": epsilons,
        "loop_holonomies": loop_hols,
        "mean_geodesics": mean_geos,
        "per_edge_geodesics": all_geos,
    }


# ======================================================================
# Main
# ======================================================================

def run_experiment():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] B1 Experiment: edge vs loop monotonicity")
    print(f"  Primary seeds: {PRIMARY_SEEDS[0]}–{PRIMARY_SEEDS[-1]} ({len(PRIMARY_SEEDS)} seeds)")
    print(f"  NegCtrl seeds: {NEGCTRL_SEEDS[0]}–{NEGCTRL_SEEDS[-1]} ({len(NEGCTRL_SEEDS)} seeds)")
    print()

    # --- Primary ---
    print("=== PRIMARY ===")
    primary_results = []
    for i, seed in enumerate(PRIMARY_SEEDS):
        t0 = time.time()
        result = run_seed(seed, identical_perturbation=False)
        elapsed = time.time() - t0
        primary_results.append(result)
        print(f"  [{i+1:2d}/{len(PRIMARY_SEEDS)}] seed={seed} "
              f"Δρ={result['delta_rho']:+.3f} "
              f"ρ_edge={result['rho_edge']:+.3f} "
              f"ρ_loop={result['rho_loop']:+.3f} "
              f"slope={result['onset_slope']:.3f} "
              f"({elapsed:.1f}s)")

    delta_rhos_primary = [r["delta_rho"] for r in primary_results]
    med_dr, ci_lo_dr, ci_hi_dr = bootstrap_median_ci(delta_rhos_primary)
    primary_ci_excludes_zero = (ci_lo_dr > 0) or (ci_hi_dr < 0)

    rho_loops = [r["rho_loop"] for r in primary_results]
    med_loop, ci_lo_loop, ci_hi_loop = bootstrap_median_ci(rho_loops)
    loop_ci_includes_zero = (ci_lo_loop <= 0 <= ci_hi_loop)

    onset_slopes = [r["onset_slope"] for r in primary_results]
    med_slope, ci_lo_slope, ci_hi_slope = bootstrap_median_ci(onset_slopes)

    rho_edges = [r["rho_edge"] for r in primary_results]
    p_edges = [r["p_edge"] for r in primary_results]
    reject_bh, pvals_corrected = benjamini_hochberg(p_edges, ALPHA)
    n_sig_edge = sum(reject_bh)

    print(f"\n  Primary Δρ: median={med_dr:+.4f}, 95% CI=[{ci_lo_dr:+.4f}, {ci_hi_dr:+.4f}]")
    print(f"  CI excludes 0: {primary_ci_excludes_zero}")
    print(f"  Mechanism check — loop ρ: median={med_loop:+.4f}, "
          f"95% CI=[{ci_lo_loop:+.4f}, {ci_hi_loop:+.4f}], includes 0: {loop_ci_includes_zero}")
    print(f"  Onset slope: median={med_slope:.4f}, 95% CI=[{ci_lo_slope:.4f}, {ci_hi_slope:.4f}]")
    print(f"  Per-seed edge Spearman: {n_sig_edge}/{len(PRIMARY_SEEDS)} significant after BH-FDR")

    # --- Negative control ---
    print("\n=== NEGATIVE CONTROL ===")
    negctrl_results = []
    for i, seed in enumerate(NEGCTRL_SEEDS):
        t0 = time.time()
        result = run_seed(seed, identical_perturbation=True)
        elapsed = time.time() - t0
        negctrl_results.append(result)
        print(f"  [{i+1:2d}/{len(NEGCTRL_SEEDS)}] seed={seed} "
              f"Δρ={result['delta_rho']:+.3f} "
              f"ρ_edge={result['rho_edge']:+.3f} "
              f"ρ_loop={result['rho_loop']:+.3f} "
              f"({elapsed:.1f}s)")

    delta_rhos_negctrl = [r["delta_rho"] for r in negctrl_results]
    med_nc, ci_lo_nc, ci_hi_nc = bootstrap_median_ci(delta_rhos_negctrl)
    negctrl_ci_includes_zero = (ci_lo_nc <= 0 <= ci_hi_nc)

    print(f"\n  NegCtrl Δρ: median={med_nc:+.4f}, 95% CI=[{ci_lo_nc:+.4f}, {ci_hi_nc:+.4f}]")
    print(f"  CI includes 0: {negctrl_ci_includes_zero}")

    # --- Verdict ---
    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)

    if not negctrl_ci_includes_zero:
        verdict = "NEGATIVE_CONTROL_FAILED"
        print("  NEGATIVE CONTROL FAILED — Δρ statistic is miscalibrated.")
        print("  Primary result is uninterpretable. Halt and diagnose.")
    elif primary_ci_excludes_zero:
        if loop_ci_includes_zero:
            verdict = "B1_CONFIRMED"
            print("  B1 CONFIRMED: edge metric is more monotone than loop metric.")
            print("  Mechanism consistency check passed (loop ρ includes 0).")
        else:
            verdict = "B1_CONFIRMED_MECHANISM_UNCLEAR"
            print("  B1 CONFIRMED: edge metric is more monotone than loop metric.")
            print("  BUT mechanism check failed — loop ρ is also significant.")
            print("  Cancellation explanation needs revision.")
    else:
        verdict = "B1_FAILED"
        print("  B1 FAILED: edge metric not reliably more monotone than loop.")
        print("  Multi-sensor inconsistency claim unsupported by either metric.")

    print("=" * 65)

    # --- Save ---
    summary = {
        "verdict": verdict,
        "primary": {
            "n_seeds": len(PRIMARY_SEEDS),
            "median_delta_rho": med_dr,
            "ci_lo": ci_lo_dr,
            "ci_hi": ci_hi_dr,
            "ci_excludes_zero": primary_ci_excludes_zero,
            "median_rho_edge": float(np.median(rho_edges)),
            "median_rho_loop": med_loop,
            "loop_ci_lo": ci_lo_loop,
            "loop_ci_hi": ci_hi_loop,
            "loop_ci_includes_zero": loop_ci_includes_zero,
            "median_onset_slope": med_slope,
            "onset_slope_ci_lo": ci_lo_slope,
            "onset_slope_ci_hi": ci_hi_slope,
            "n_edge_significant_bh": n_sig_edge,
        },
        "negative_control": {
            "n_seeds": len(NEGCTRL_SEEDS),
            "median_delta_rho": med_nc,
            "ci_lo": ci_lo_nc,
            "ci_hi": ci_hi_nc,
            "ci_includes_zero": negctrl_ci_includes_zero,
        },
        "per_seed_primary": [{
            "seed": r["seed"],
            "delta_rho": r["delta_rho"],
            "rho_edge": r["rho_edge"],
            "rho_loop": r["rho_loop"],
            "onset_slope": r["onset_slope"],
        } for r in primary_results],
        "per_seed_negctrl": [{
            "seed": r["seed"],
            "delta_rho": r["delta_rho"],
            "rho_edge": r["rho_edge"],
            "rho_loop": r["rho_loop"],
        } for r in negctrl_results],
    }

    json_path = out_dir / "gating_b1_edge_vs_loop.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {json_path}")

    return summary


def benjamini_hochberg(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (reject_array, adjusted_pvals)."""
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


if __name__ == "__main__":
    run_experiment()
