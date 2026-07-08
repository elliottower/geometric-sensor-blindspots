"""Diagnostic: World 1 vs World 2 for holonomy(epsilon).

World 1: true holonomy rises monotonically; bouncing is PCA estimation noise.
         → more samples narrows the band and reveals the monotone curve.
World 2: holonomy is genuinely non-monotonic (rotations cancel around the loop).
         → more samples tightens the band but the wiggles persist.

Method: for each (seed, N), compute holonomy at each epsilon with B independent
noise realizations (same latent z and embeddings, different observation noise).
Plot mean ± 95% band. If bands narrow AND curve smooths → World 1.
If bands narrow BUT curve stays wiggly → World 2.

Uses frozen parameters from gating_h7b_v3.py. No parameter changes.
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
    D, K_LATENT, LATENT_SIGNAL, OBS_NOISE_BASE, OBS_NOISE_RATE,
    PERTURBATION_SCALE, PCA_K, N_EPSILON_STEPS, EPSILON_RANGE,
)

DIAGNOSTIC_SEEDS = [20260708, 20260712]
SAMPLE_SIZES = [200, 500, 1000, 2000]
N_NOISE_REPS = 30


def run_diagnostic(seed, n_per_class, epsilon_range, n_reps):
    data_rng = np.random.default_rng(seed)
    proj_rng = np.random.default_rng(seed + 1000)
    pert_rng = np.random.default_rng(seed + 3000)

    z, y = generate_latent(n_per_class, data_rng)
    U_base = generate_base_embedding(proj_rng)
    perturbations = generate_perturbations(3, pert_rng)

    results = []
    for eps in epsilon_range:
        obs_noise = OBS_NOISE_BASE + OBS_NOISE_RATE * eps
        Us = [sensor_embedding(U_base, p, eps) for p in perturbations]

        hols = np.zeros(n_reps)
        for rep in range(n_reps):
            rep_rng = np.random.default_rng(seed * 10000 + int(eps * 1000) * 100 + rep)
            Xs = [generate_sensor_data(z, U_s, obs_noise, rep_rng)
                  for U_s in Us]
            hols[rep], _ = compute_holonomy(Xs, PCA_K)

        results.append({
            "epsilon": float(eps),
            "mean": float(hols.mean()),
            "std": float(hols.std()),
            "ci_lo": float(np.percentile(hols, 2.5)),
            "ci_hi": float(np.percentile(hols, 97.5)),
            "median": float(np.median(hols)),
        })

    return results


def monotonicity_test(results):
    epsilons = [r["epsilon"] for r in results]
    means = [r["mean"] for r in results]
    rho, p = stats.spearmanr(epsilons, means)
    return float(rho), float(p)


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    all_results = {}

    for seed in DIAGNOSTIC_SEEDS:
        seed_label = "FAIL" if seed == 20260708 else "PASS"
        all_results[str(seed)] = {"label": seed_label}

        for n_per_class in SAMPLE_SIZES:
            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Seed {seed} ({seed_label}), "
                  f"N={n_per_class}: computing {N_EPSILON_STEPS} eps x {N_NOISE_REPS} reps...")

            results = run_diagnostic(seed, n_per_class, EPSILON_RANGE, N_NOISE_REPS)
            rho, p_mono = monotonicity_test(results)

            elapsed = time.time() - t0
            print(f"  done in {elapsed:.1f}s. Spearman rho={rho:.3f}, p={p_mono:.4f}")

            all_results[str(seed)][f"N={n_per_class}"] = {
                "n_per_class": n_per_class,
                "n_reps": N_NOISE_REPS,
                "spearman_rho": rho,
                "spearman_p": p_mono,
                "per_epsilon": results,
            }

    json_path = out_dir / "diagnostic_power_holonomy.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    colors = {200: "#999999", 500: "#4477AA", 1000: "#EE6677", 2000: "#228833"}

    for ax, seed in zip(axes, DIAGNOSTIC_SEEDS):
        seed_label = all_results[str(seed)]["label"]
        for n_per_class in SAMPLE_SIZES:
            data = all_results[str(seed)][f"N={n_per_class}"]["per_epsilon"]
            eps = [r["epsilon"] for r in data]
            means = [r["mean"] for r in data]
            ci_lo = [r["ci_lo"] for r in data]
            ci_hi = [r["ci_hi"] for r in data]
            rho = all_results[str(seed)][f"N={n_per_class}"]["spearman_rho"]
            c = colors[n_per_class]
            ax.plot(eps, means, color=c, linewidth=1.5,
                    label=f"N={n_per_class} (ρ={rho:.2f})")
            ax.fill_between(eps, ci_lo, ci_hi, color=c, alpha=0.15)

        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Holonomy norm")
        ax.set_title(f"Seed {seed} [{seed_label}]: holonomy(ε) at increasing N")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("ε (decoherence)")
    plt.tight_layout()
    fig_path = out_dir / "diagnostic_power_holonomy.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")

    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)
    for seed in DIAGNOSTIC_SEEDS:
        seed_label = all_results[str(seed)]["label"]
        print(f"\nSeed {seed} [{seed_label}]:")
        for n_per_class in SAMPLE_SIZES:
            entry = all_results[str(seed)][f"N={n_per_class}"]
            data = entry["per_epsilon"]
            band_widths = [r["ci_hi"] - r["ci_lo"] for r in data]
            print(f"  N={n_per_class:4d}: Spearman ρ={entry['spearman_rho']:+.3f} "
                  f"(p={entry['spearman_p']:.4f}), "
                  f"mean band width={np.mean(band_widths):.3f}")

    print("\nWorld 1 indicators: bands narrow with N, rho increases toward +1")
    print("World 2 indicators: bands narrow but rho stays flat or wiggly")


if __name__ == "__main__":
    main()
