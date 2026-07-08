"""Gating experiment v2: does Grassmannian cocycle holonomy detect
cross-sensor inconsistency before within-sensor AUC degrades?

Tests H7b and H7c from PREREGISTRATION.md.
Corrected from v1 (SHA 2a9b023). See DEVIATION_LOG.md for justification.

Fixes from v1:
- All sensors share ambient dimension D=100 (Grassmannian geometry
  requires same ambient space; v1 had D=100/80/40 which crashes)
- Pool-and-resplit null replaces row-permutation (row-perm is a no-op
  for PCA — verified empirically: null std = 0.0)
- Reports holonomy increase from eps=0 baseline
- 5 seeds instead of 1
- Correct naming: "cocycle holonomy" throughout (not "sheaf H^1")
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
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
N_NULL_RESAMPLES = 1000
AUC_THRESHOLD = 0.75
HOLONOMY_ALPHA = 0.05
SEEDS = [20260708, 20260709, 20260710, 20260711, 20260712]

D = 100
SIGNAL_DIM = 10
SIGNAL_STRENGTH = 0.7


# ======================================================================
# Three sensors: same ambient space, shared signal, different decoherence
#
# Signal: class 1 has +SIGNAL_STRENGTH in features 0..SIGNAL_DIM-1.
# At eps=0, all sensors see the same structure (d' ≈ 1.6, AUC ≈ 0.87).
# Decoherence adds sensor-specific noise that degrades AUC and rotates
# the PCA subspace in different directions per sensor.
# ======================================================================

def _shared_base(n_per_class, rng):
    """Shared signal in standard-normal background."""
    n = 2 * n_per_class
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    X = rng.standard_normal((n, D))
    for i in range(n):
        X[i, :SIGNAL_DIM] += SIGNAL_STRENGTH * y[i]
    return X, y


def generate_sensor_a(n_per_class, epsilon, rng):
    """Sensor A: isotropic noise increase across all signal features."""
    X, y = _shared_base(n_per_class, rng)
    X[:, :SIGNAL_DIM] += rng.normal(0, 3 * epsilon, (len(X), SIGNAL_DIM))
    return X, y


def generate_sensor_b(n_per_class, epsilon, rng):
    """Sensor B: asymmetric degradation (features 0-4 much noisier)."""
    X, y = _shared_base(n_per_class, rng)
    X[:, :5] += rng.normal(0, 5 * epsilon, (len(X), 5))
    X[:, 5:SIGNAL_DIM] += rng.normal(0, 0.5 * epsilon, (len(X), 5))
    return X, y


def generate_sensor_c(n_per_class, epsilon, rng):
    """Sensor C: systematic bias (rotates subspace) + partial noise."""
    X, y = _shared_base(n_per_class, rng)
    bias = rng.standard_normal(SIGNAL_DIM) * 3 * epsilon
    X[:, :SIGNAL_DIM] += bias
    X[:, 5:SIGNAL_DIM] += rng.normal(0, 3 * epsilon, (len(X), 5))
    return X, y


GENERATORS = [generate_sensor_a, generate_sensor_b, generate_sensor_c]


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


def pool_resplit_null(Xs, k, n_resamples, rng):
    """Pool all sensors, randomly split into 3 groups of same sizes,
    compute holonomy on each split. This destroys sensor identity while
    preserving the overall data distribution.

    Unlike row-permutation (which is a no-op for PCA), this produces
    a genuine null distribution because different random subsets of
    pooled data yield different PCA subspaces.
    """
    pooled = np.vstack(Xs)
    sizes = [len(X) for X in Xs]
    null_norms = np.zeros(n_resamples)
    for p in range(n_resamples):
        idx = rng.permutation(len(pooled))
        splits = []
        start = 0
        for sz in sizes:
            splits.append(pooled[idx[start:start + sz]])
            start += sz
        subspaces = [top_k_subspace(sp, k)[0] for sp in splits]
        _, null_norms[p] = cocycle_holonomy(subspaces)
    return null_norms


# ======================================================================
# Single-seed run
# ======================================================================

def run_seed(seed):
    rng = np.random.default_rng(seed)
    rows = []
    hol_baseline = None

    for step, eps in enumerate(EPSILON_RANGE):
        t0 = time.time()

        sensor_data = [gen(N_SAMPLES_PER_CLASS, eps, rng) for gen in GENERATORS]
        Xs = [d[0] for d in sensor_data]
        ys = [d[1] for d in sensor_data]

        aucs = [within_sensor_auc(X, y) for X, y in zip(Xs, ys)]
        hol_norm, pairwise = compute_holonomy(Xs, PCA_K)
        null_norms = pool_resplit_null(Xs, PCA_K, N_NULL_RESAMPLES, rng)
        p_val = float(np.mean(null_norms >= hol_norm))

        if hol_baseline is None:
            hol_baseline = hol_norm

        row = {
            "epsilon": float(eps),
            "auc_a": aucs[0],
            "auc_b": aucs[1],
            "auc_c": aucs[2],
            "min_auc": min(aucs),
            "holonomy_norm": hol_norm,
            "holonomy_delta": hol_norm - hol_baseline,
            "holonomy_p": p_val,
            "null_mean": float(null_norms.mean()),
            "null_std": float(null_norms.std()),
            "null_95": float(np.percentile(null_norms, 95)),
            "pairwise_geodesic": [float(x) for x in pairwise],
        }
        rows.append(row)

        elapsed = time.time() - t0
        sig = "*" if p_val < HOLONOMY_ALPHA else " "
        print(f"  [{step+1:2d}/{N_EPSILON_STEPS}] eps={eps:.3f} | "
              f"AUC a={aucs[0]:.3f} b={aucs[1]:.3f} c={aucs[2]:.3f} | "
              f"hol={hol_norm:.4f} delta={hol_norm - hol_baseline:+.4f} "
              f"p={p_val:.3f}{sig} | {elapsed:.1f}s")

    return rows, hol_baseline


# ======================================================================
# Aggregate and analyze
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
    print(f"[{time.strftime('%H:%M:%S')}] Gating experiment H7b/H7c v2")
    print(f"  seeds={SEEDS}, n_per_class={N_SAMPLES_PER_CLASS}, "
          f"eps_steps={N_EPSILON_STEPS}, k={PCA_K}, "
          f"n_null={N_NULL_RESAMPLES}")
    print()

    per_seed = {}
    h7b_count = 0
    h7c_count = 0

    for seed in SEEDS:
        print(f"--- Seed {seed} ---")
        rows, hol_baseline = run_seed(seed)
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
            "holonomy_at_zero": hol_baseline,
            **verdict,
            "rows": rows,
        }

    summary = {
        "n_seeds": len(SEEDS),
        "H7b_count": h7b_count,
        "H7c_count": h7c_count,
        "H7b_rate": h7b_count / len(SEEDS),
        "H7c_rate": h7c_count / len(SEEDS),
        "per_seed": per_seed,
    }

    print("=" * 65)
    print("AGGREGATE RESULTS")
    print("=" * 65)
    print(f"  H7b confirmed: {h7b_count}/{len(SEEDS)} seeds")
    print(f"  H7c confirmed: {h7c_count}/{len(SEEDS)} seeds")
    print("=" * 65)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "gating_h7b_v2.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_file}")

    return summary


if __name__ == "__main__":
    run_gating_experiment()
