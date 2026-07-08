"""Gating experiment: does multi-sensor sheaf H^1 detect cross-sensor
inconsistency before within-sensor degradation?

Tests H7b and H7c from PREREGISTRATION.md.

Uses transportability.py from biomedical-cohort-transfer (methods 8-11).
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "biomedical-cohort-transfer" / "src"))
from transportability import (
    top_k_subspace,
    cocycle_holonomy,
    geodesic_distance,
    sheaf_q_test,
)

# ======================================================================
# Frozen parameters — do not modify after SHA freeze
# ======================================================================

N_SAMPLES_PER_CLASS = 200
N_EPSILON_STEPS = 20
EPSILON_RANGE = np.linspace(0.0, 1.0, N_EPSILON_STEPS)
PCA_K = 3
N_PERMUTATIONS = 1000
AUC_THRESHOLD = 0.75
HOLONOMY_ALPHA = 0.05

# Sensor feature dimensions
NV_D = 100       # 20 NV centers x 5 features
PHOTON_D = 80    # 8x8 pixels x (coincidence, background) reduced to 80
SPIN_D = 40      # 10 labeled sites x 4 features

# Signal parameters
TEMP_HEALTHY = 310.0
TEMP_TUMOR = 311.5
MU_S_HEALTHY = 6.0
MU_S_TUMOR = 12.0
DELTA_G_BOUND = 0.002


# ======================================================================
# Sensor data generators
# ======================================================================

def generate_nv_diamond(n_per_class, epsilon, rng):
    """NV-diamond intracellular thermometry.

    epsilon controls T2 degradation (0 = clean, 1 = fully degraded).
    Signal: tumor is 1.5 K warmer → frequency shift via dD/dT.
    """
    X = np.zeros((2 * n_per_class, NV_D))
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    for i in range(2 * n_per_class):
        temp = TEMP_HEALTHY + (TEMP_TUMOR - TEMP_HEALTHY) * y[i]
        for s in range(20):
            base = s * 5
            T2_clean = rng.lognormal(mean=np.log(5e-6), sigma=0.3)
            T2 = T2_clean * np.exp(-3 * epsilon)

            freq = -74e3 * temp + rng.normal(0, 1e3 / (T2 * 1e6 + 1e-10))
            linewidth = 1 / (np.pi * T2 + 1e-10) + rng.normal(0, epsilon * 1e5)
            contrast = 0.3 * (1 - 0.5 * epsilon) + rng.normal(0, 0.02)
            T1 = rng.lognormal(mean=np.log(1e-3), sigma=0.2)
            surface = rng.normal(0, epsilon * 0.1)

            X[i, base] = T2 + surface * 1e-7
            X[i, base + 1] = linewidth
            X[i, base + 2] = contrast
            X[i, base + 3] = freq
            X[i, base + 4] = T1

    return X, y


def generate_entangled_photon(n_per_class, epsilon, rng):
    """Entangled-photon microscopy.

    epsilon controls entanglement visibility degradation.
    Signal: tumor has higher scattering coefficient.
    """
    X = np.zeros((2 * n_per_class, PHOTON_D))
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    visibility = 1.0 - 0.5 * epsilon

    for i in range(2 * n_per_class):
        mu_s = MU_S_HEALTHY + (MU_S_TUMOR - MU_S_HEALTHY) * y[i]
        for p in range(40):
            base = p * 2
            signal = mu_s * rng.exponential(0.1) * visibility
            noise = rng.normal(0, 0.3 * epsilon + 0.05)
            X[i, base] = signal + noise
            X[i, base + 1] = rng.poisson(max(1, int(signal * 100 * visibility))) / 100.0

    return X, y


def generate_spin_label(n_per_class, epsilon, rng):
    """Spin-labeled protein ESR detection.

    epsilon controls spin relaxation degradation.
    Signal: biomarker binding shifts g-factor and coupling.
    """
    X = np.zeros((2 * n_per_class, SPIN_D))
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    for i in range(2 * n_per_class):
        bound = y[i]
        for s in range(10):
            base = s * 4
            g = 2.0023 + DELTA_G_BOUND * bound + rng.normal(0, 0.0005 * (1 + 2 * epsilon))
            A = 15.0 + 5.0 * bound + rng.normal(0, 1.0 * (1 + 3 * epsilon))
            linewidth = 0.5 + rng.normal(0, 0.1 * (1 + 2 * epsilon))
            tau_c = 3.0 + 2.0 * bound + rng.normal(0, 0.5 * (1 + 3 * epsilon))

            X[i, base] = g
            X[i, base + 1] = A
            X[i, base + 2] = linewidth
            X[i, base + 3] = tau_c

    return X, y


# ======================================================================
# Metrics
# ======================================================================

def within_sensor_auc(X, y):
    """5-fold CV AUC of logistic regression."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    scores = cross_val_score(clf, X_s, y, cv=5, scoring="roc_auc")
    return float(scores.mean())


def multi_sensor_holonomy(Xs, k, n_perms, rng):
    """Holonomy around the 3-sensor cycle with permutation null."""
    subspaces = []
    for X in Xs:
        U, _ = top_k_subspace(X, k)
        subspaces.append(U)

    _, observed_norm = cocycle_holonomy(subspaces)

    null_norms = np.zeros(n_perms)
    for p in range(n_perms):
        perm_subspaces = []
        for X in Xs:
            X_perm = X[rng.permutation(len(X))]
            U_p, _ = top_k_subspace(X_perm, k)
            perm_subspaces.append(U_p)
        _, null_norms[p] = cocycle_holonomy(perm_subspaces)

    p_value = float(np.mean(null_norms >= observed_norm))

    pairwise_dists = []
    for i in range(len(subspaces)):
        j = (i + 1) % len(subspaces)
        pairwise_dists.append(geodesic_distance(subspaces[i], subspaces[j]))

    return {
        "holonomy_norm": observed_norm,
        "p_value": p_value,
        "null_mean": float(null_norms.mean()),
        "null_std": float(null_norms.std()),
        "null_95": float(np.percentile(null_norms, 95)),
        "pairwise_geodesic": pairwise_dists,
    }


# ======================================================================
# Main experiment
# ======================================================================

def run_gating_experiment(seed=20260708):
    rng = np.random.default_rng(seed)
    results = []

    print(f"[{time.strftime('%H:%M:%S')}] Gating experiment H7b/H7c")
    print(f"  seed={seed}, n_per_class={N_SAMPLES_PER_CLASS}, "
          f"epsilon_steps={N_EPSILON_STEPS}, k={PCA_K}, "
          f"n_perms={N_PERMUTATIONS}")
    print()

    for step, eps in enumerate(EPSILON_RANGE):
        t0 = time.time()

        X_nv, y_nv = generate_nv_diamond(N_SAMPLES_PER_CLASS, eps, rng)
        X_ph, y_ph = generate_entangled_photon(N_SAMPLES_PER_CLASS, eps, rng)
        X_sp, y_sp = generate_spin_label(N_SAMPLES_PER_CLASS, eps, rng)

        auc_nv = within_sensor_auc(X_nv, y_nv)
        auc_ph = within_sensor_auc(X_ph, y_ph)
        auc_sp = within_sensor_auc(X_sp, y_sp)

        hol = multi_sensor_holonomy([X_nv, X_ph, X_sp], PCA_K, N_PERMUTATIONS, rng)

        row = {
            "epsilon": float(eps),
            "auc_nv": auc_nv,
            "auc_photon": auc_ph,
            "auc_spin": auc_sp,
            "min_auc": min(auc_nv, auc_ph, auc_sp),
            "holonomy_norm": hol["holonomy_norm"],
            "holonomy_p": hol["p_value"],
            "null_95": hol["null_95"],
            "pairwise_geodesic": hol["pairwise_geodesic"],
        }
        results.append(row)

        elapsed = time.time() - t0
        sig = "*" if hol["p_value"] < HOLONOMY_ALPHA else " "
        print(f"  [{step+1:2d}/{N_EPSILON_STEPS}] eps={eps:.3f} | "
              f"AUC nv={auc_nv:.3f} ph={auc_ph:.3f} sp={auc_sp:.3f} | "
              f"hol={hol['holonomy_norm']:.4f} p={hol['p_value']:.3f}{sig} | "
              f"{elapsed:.1f}s")

    return analyze_results(results, seed)


def analyze_results(results, seed):
    epsilon_crit_holonomy = None
    epsilon_crit_single = None

    for r in results:
        if epsilon_crit_holonomy is None and r["holonomy_p"] < HOLONOMY_ALPHA:
            epsilon_crit_holonomy = r["epsilon"]
        if epsilon_crit_single is None and r["min_auc"] < AUC_THRESHOLD:
            epsilon_crit_single = r["epsilon"]

    h7b = (epsilon_crit_holonomy is not None
           and any(r["min_auc"] > AUC_THRESHOLD
                   and r["holonomy_p"] < HOLONOMY_ALPHA
                   for r in results))

    h7c = (h7b
           and epsilon_crit_single is not None
           and epsilon_crit_holonomy < epsilon_crit_single)

    summary = {
        "seed": seed,
        "epsilon_crit_holonomy": epsilon_crit_holonomy,
        "epsilon_crit_single_sensor": epsilon_crit_single,
        "H7b_confirmed": h7b,
        "H7c_confirmed": h7c,
        "results": results,
    }

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  epsilon_crit (holonomy):      {epsilon_crit_holonomy}")
    print(f"  epsilon_crit (single-sensor): {epsilon_crit_single}")
    print(f"  H7b (sheaf detects before single-sensor fails): {h7b}")
    print(f"  H7c (cross-sensor more fragile): {h7c}")
    print("=" * 65)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"gating_h7b_seed{seed}.json"

    serializable = dict(summary)
    for r in serializable["results"]:
        r["pairwise_geodesic"] = [float(x) for x in r["pairwise_geodesic"]]

    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    return summary


if __name__ == "__main__":
    run_gating_experiment()
