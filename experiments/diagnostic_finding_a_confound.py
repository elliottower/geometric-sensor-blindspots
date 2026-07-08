"""Diagnostic: does loop holonomy saturate under identical perturbation?

If loop holonomy saturates even when all sensors have the SAME perturbation
(no cross-sensor inconsistency), then the saturation reported in DEV-009
is a noise artifact, not a cross-sensor cancellation phenomenon.

If loop holonomy does NOT saturate under identical perturbation (stays near
zero or rises smoothly), then the cancellation mechanism is confirmed:
saturation requires different perturbation directions to create the
cancellation pattern.

This diagnostic determines whether Finding A from DEV-009 survives the
eps-confound discovered in the B1 negative control analysis.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

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
    N_SAMPLES_PER_CLASS,
)
from gating_b1_edge_vs_loop import mean_edge_geodesics


SEEDS = [20260708, 20260709, 20260710, 20260711, 20260712]


def run_comparison(seed):
    """Run both identical and different perturbation for one seed."""
    data_rng = np.random.default_rng(seed)
    proj_rng = np.random.default_rng(seed + 1000)
    pert_rng = np.random.default_rng(seed + 3000)

    z, y = generate_latent(N_SAMPLES_PER_CLASS, data_rng)
    U_base = generate_base_embedding(proj_rng)
    diff_perts = generate_perturbations(3, pert_rng)

    ident_rng = np.random.default_rng(seed + 4000)
    single_pert = ident_rng.standard_normal((D, K_LATENT))
    ident_perts = [single_pert, single_pert, single_pert]

    results = {"identical": [], "different": []}

    for label, perts in [("identical", ident_perts), ("different", diff_perts)]:
        hols = []
        geos = []
        for step, eps in enumerate(EPSILON_RANGE):
            obs_noise = OBS_NOISE_BASE + OBS_NOISE_RATE * eps
            Us = [sensor_embedding(U_base, p, eps) for p in perts]
            obs_rng = np.random.default_rng(seed * 100 + step)
            Xs = [generate_sensor_data(z, U_s, obs_noise, obs_rng) for U_s in Us]

            hol_norm, _ = compute_holonomy(Xs, PCA_K)
            mean_geo, _ = mean_edge_geodesics(Xs, PCA_K)
            hols.append(hol_norm)
            geos.append(mean_geo)

        rho_hol, p_hol = stats.spearmanr(EPSILON_RANGE, hols)
        rho_geo, p_geo = stats.spearmanr(EPSILON_RANGE, geos)

        results[label] = {
            "holonomies": hols,
            "geodesics": geos,
            "rho_hol": float(rho_hol),
            "p_hol": float(p_hol),
            "rho_geo": float(rho_geo),
            "p_geo": float(p_geo),
            "hol_max": float(max(hols)),
            "hol_min": float(min(hols)),
            "hol_range": float(max(hols) - min(hols)),
        }

    return results


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Finding A confound check")
    print("  Does loop holonomy saturate under identical perturbation?")
    print()

    all_results = {}
    for seed in SEEDS:
        t0 = time.time()
        r = run_comparison(seed)
        elapsed = time.time() - t0

        print(f"  Seed {seed} ({elapsed:.1f}s):")
        print(f"    Identical: ρ_hol={r['identical']['rho_hol']:+.3f} "
              f"(p={r['identical']['p_hol']:.3f})  "
              f"range=[{r['identical']['hol_min']:.2f}, {r['identical']['hol_max']:.2f}]  "
              f"ρ_geo={r['identical']['rho_geo']:+.3f}")
        print(f"    Different: ρ_hol={r['different']['rho_hol']:+.3f} "
              f"(p={r['different']['p_hol']:.3f})  "
              f"range=[{r['different']['hol_min']:.2f}, {r['different']['hol_max']:.2f}]  "
              f"ρ_geo={r['different']['rho_geo']:+.3f}")
        all_results[str(seed)] = r

    print()
    print("=" * 65)
    print("VERDICT")
    print("=" * 65)

    ident_rhos = [all_results[s]["identical"]["rho_hol"] for s in all_results]
    diff_rhos = [all_results[s]["different"]["rho_hol"] for s in all_results]
    ident_ranges = [all_results[s]["identical"]["hol_range"] for s in all_results]
    diff_ranges = [all_results[s]["different"]["hol_range"] for s in all_results]

    print(f"  Identical pert — mean ρ_hol: {np.mean(ident_rhos):+.3f}, "
          f"mean range: {np.mean(ident_ranges):.2f}")
    print(f"  Different pert — mean ρ_hol: {np.mean(diff_rhos):+.3f}, "
          f"mean range: {np.mean(diff_ranges):.2f}")

    if np.mean(ident_ranges) > 1.0:
        print()
        print("  FINDING A CONFOUNDED: Loop holonomy shows large variation")
        print("  even with identical perturbation. Saturation is at least")
        print("  partly a noise artifact, not purely cross-sensor cancellation.")
    else:
        print()
        print("  FINDING A SURVIVES: Loop holonomy stays near zero / low range")
        print("  with identical perturbation. Saturation requires different")
        print("  perturbation directions → cancellation mechanism confirmed.")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "diagnostic_finding_a_confound.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
