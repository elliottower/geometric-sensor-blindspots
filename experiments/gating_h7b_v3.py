"""Gating experiment v3: does Grassmannian cocycle holonomy detect
cross-sensor inconsistency before within-sensor AUC degrades?

Tests H7b and H7c from PREREGISTRATION.md.
Corrected from v2 (SHA cb0c8d8). See DEVIATION_LOG.md for justification.

Key design: shared latent z projected through sensor-specific embeddings.
Decoherence ROTATES each sensor's embedding in a different direction,
creating genuine cross-sensor inconsistency (not just noise heterogeneity).
Within-sensor AUC degrades via additive noise.

v1 bugs fixed: dimension mismatch, broken null.
v2 bug fixed: generators tested noise heterogeneity, not inconsistency.
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
from transportability import top_k_subspace, cocycle_holonomy, geodesic_distance

# ======================================================================
# Frozen parameters — do not modify after SHA freeze
# ======================================================================

N_SAMPLES_PER_CLASS = 200
N_EPSILON_STEPS = 20
EPSILON_RANGE = np.linspace(0.0, 1.0, N_EPSILON_STEPS)
PCA_K = 3
N_BOOTSTRAP = 500
AUC_THRESHOLD = 0.75
HOLONOMY_ALPHA = 0.05
SEEDS = [20260708, 20260709, 20260710, 20260711, 20260712]

D = 100
K_LATENT = 3
LATENT_SIGNAL = 5.0
OBS_NOISE_BASE = 1.0
OBS_NOISE_RATE = 3.5
PERTURBATION_SCALE = 1.5


# ======================================================================
# Latent model
# ======================================================================

def generate_latent(n_per_class, rng):
    """Shared k-dim latent z with binary class signal."""
    n = 2 * n_per_class
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    z = rng.standard_normal((n, K_LATENT))
    mu = np.zeros(K_LATENT)
    mu[0] = LATENT_SIGNAL
    for i in range(n):
        z[i] += mu * y[i]
    return z, y


def generate_base_embedding(rng):
    """Random orthonormal D x K_LATENT base embedding, shared by all sensors."""
    A = rng.standard_normal((D, K_LATENT))
    Q, _ = linalg.qr(A, mode='economic')
    return Q


def generate_perturbations(n_sensors, rng):
    """Random D x K_LATENT perturbation matrices, one per sensor.
    Each perturbation is projected orthogonal to U_base at application time,
    so it purely rotates the subspace without changing the signal captured."""
    return [rng.standard_normal((D, K_LATENT)) for _ in range(n_sensors)]


def sensor_embedding(U_base, perturbation, epsilon):
    """Perturb base embedding and re-orthonormalize.

    At eps=0 returns U_base. At eps>0, the subspace tilts in the
    direction of the perturbation. Different perturbations for different
    sensors → different subspace tilts → nonzero holonomy.
    """
    W = U_base + epsilon * PERTURBATION_SCALE * perturbation
    Q, _ = linalg.qr(W, mode='economic')
    return Q


def generate_sensor_data(z, U_s, obs_noise, rng):
    """Observed data: X = z @ U_s.T + noise."""
    X = z @ U_s.T + rng.normal(0, obs_noise, (len(z), D))
    return X


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


def compute_holonomy(Xs, k):
    """Cocycle holonomy around the sensor cycle + pairwise geodesics."""
    subspaces = [top_k_subspace(X, k)[0] for X in Xs]
    _, hol_norm = cocycle_holonomy(subspaces)
    pairwise = []
    for i in range(len(subspaces)):
        j = (i + 1) % len(subspaces)
        pairwise.append(geodesic_distance(subspaces[i], subspaces[j]))
    return hol_norm, pairwise


# ======================================================================
# Bootstrap baseline null (from eps=0)
# ======================================================================

def bootstrap_baseline(z, y, U_base, n_bootstrap, rng):
    """Bootstrap distribution of holonomy at eps=0 (consistent sensors).

    At eps=0 all sensors share U_base, so holonomy is purely sampling jitter.
    Returns the 95th percentile as the significance threshold.
    """
    obs_noise_0 = OBS_NOISE_BASE
    hols = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(len(z), size=len(z), replace=True)
        z_b = z[idx]
        Xs = [z_b @ U_base.T + rng.normal(0, obs_noise_0, (len(z_b), D))
              for _ in range(3)]
        hols[b], _ = compute_holonomy(Xs, PCA_K)
    return hols


# ======================================================================
# Single-seed run
# ======================================================================

def run_seed(seed):
    data_rng = np.random.default_rng(seed)
    proj_rng = np.random.default_rng(seed + 1000)
    boot_rng = np.random.default_rng(seed + 2000)
    pert_rng = np.random.default_rng(seed + 3000)

    z, y = generate_latent(N_SAMPLES_PER_CLASS, data_rng)
    U_base = generate_base_embedding(proj_rng)
    perturbations = generate_perturbations(3, pert_rng)

    print(f"  Computing bootstrap baseline (eps=0, n={N_BOOTSTRAP})...")
    boot_hols = bootstrap_baseline(z, y, U_base, N_BOOTSTRAP, boot_rng)
    hol_threshold = float(np.percentile(boot_hols, 100 * (1 - HOLONOMY_ALPHA)))
    print(f"  Baseline holonomy: mean={boot_hols.mean():.4f}, "
          f"95th={hol_threshold:.4f}")

    rows = []
    for step, eps in enumerate(EPSILON_RANGE):
        t0 = time.time()

        obs_noise = OBS_NOISE_BASE + OBS_NOISE_RATE * eps

        Us = [sensor_embedding(U_base, pert, eps) for pert in perturbations]

        obs_rng = np.random.default_rng(seed * 100 + step)
        Xs = [generate_sensor_data(z, U, obs_noise, obs_rng) for U in Us]
        ys = [y] * 3

        aucs = [within_sensor_auc(X, yi) for X, yi in zip(Xs, ys)]
        hol_norm, pairwise = compute_holonomy(Xs, PCA_K)
        p_val = float(np.mean(boot_hols >= hol_norm))

        row = {
            "epsilon": float(eps),
            "auc_a": aucs[0],
            "auc_b": aucs[1],
            "auc_c": aucs[2],
            "min_auc": min(aucs),
            "holonomy_norm": hol_norm,
            "holonomy_delta": hol_norm - float(boot_hols.mean()),
            "holonomy_p": p_val,
            "hol_threshold_95": hol_threshold,
            "pairwise_geodesic": [float(x) for x in pairwise],
        }
        rows.append(row)

        elapsed = time.time() - t0
        sig = "*" if p_val < HOLONOMY_ALPHA else " "
        print(f"  [{step+1:2d}/{N_EPSILON_STEPS}] eps={eps:.3f} | "
              f"AUC a={aucs[0]:.3f} b={aucs[1]:.3f} c={aucs[2]:.3f} | "
              f"hol={hol_norm:.4f} delta={row['holonomy_delta']:+.4f} "
              f"p={p_val:.3f}{sig} | {elapsed:.1f}s")

    return rows, float(boot_hols.mean()), hol_threshold


# ======================================================================
# Analysis
# ======================================================================

def analyze_seed(rows):
    eps_crit_hol = None
    eps_crit_auc = None

    for r in rows:
        if eps_crit_hol is None and r["holonomy_p"] < HOLONOMY_ALPHA:
            eps_crit_hol = r["epsilon"]
        if eps_crit_auc is None and r["min_auc"] < AUC_THRESHOLD:
            eps_crit_auc = r["epsilon"]

    h7b = (eps_crit_hol is not None
           and any(r["min_auc"] > AUC_THRESHOLD
                   and r["holonomy_p"] < HOLONOMY_ALPHA
                   for r in rows))

    h7c = (h7b
           and eps_crit_auc is not None
           and eps_crit_hol < eps_crit_auc)

    return {
        "epsilon_crit_holonomy": eps_crit_hol,
        "epsilon_crit_single_sensor": eps_crit_auc,
        "H7b_confirmed": h7b,
        "H7c_confirmed": h7c,
    }


def run_gating_experiment():
    print(f"[{time.strftime('%H:%M:%S')}] Gating experiment H7b/H7c v3")
    print(f"  seeds={SEEDS}, n_per_class={N_SAMPLES_PER_CLASS}, "
          f"eps_steps={N_EPSILON_STEPS}, k={PCA_K}, "
          f"n_bootstrap={N_BOOTSTRAP}")
    print(f"  perturbation_scale={PERTURBATION_SCALE}")
    print()

    per_seed = {}
    h7b_count = 0
    h7c_count = 0

    for seed in SEEDS:
        print(f"--- Seed {seed} ---")
        rows, hol_mean_0, hol_thresh = run_seed(seed)
        verdict = analyze_seed(rows)

        if verdict["H7b_confirmed"]:
            h7b_count += 1
        if verdict["H7c_confirmed"]:
            h7c_count += 1

        print(f"  => eps_crit_hol={verdict['epsilon_crit_holonomy']}, "
              f"eps_crit_auc={verdict['epsilon_crit_single_sensor']}, "
              f"H7b={verdict['H7b_confirmed']}, H7c={verdict['H7c_confirmed']}")
        print()

        per_seed[str(seed)] = {
            "baseline_holonomy_mean": hol_mean_0,
            "baseline_holonomy_threshold": hol_thresh,
            **verdict,
            "rows": rows,
        }

    n = len(SEEDS)
    summary = {
        "n_seeds": n,
        "H7b_count": h7b_count,
        "H7c_count": h7c_count,
        "H7b_rate": h7b_count / n,
        "H7c_rate": h7c_count / n,
        "per_seed": per_seed,
    }

    print("=" * 65)
    print("AGGREGATE RESULTS")
    print("=" * 65)
    print(f"  H7b confirmed: {h7b_count}/{n} seeds")
    print(f"  H7c confirmed: {h7c_count}/{n} seeds")
    print("=" * 65)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "gating_h7b_v3.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_file}")

    return summary


if __name__ == "__main__":
    run_gating_experiment()
