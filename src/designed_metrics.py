"""Designed metrics: invariances CHOSEN, not inherited.

L1 — Invariance-audited composite (closed-form, no training).
     Three components with provably different invariance properties:
       T_loc:   breaks translation invariance (sees device offsets)
       T_scale: breaks scale invariance (sees device gain)
       T_geo:   preserves both (the geodesic control-within-the-metric)

L2 — Tunable-invariance dial.
     One parameter lambda in [0,1] interpolating between:
       lambda=0: fully translation/scale-invariant (reproduces geodesic)
       lambda=1: fully sensitive (reproduces L1)

Same interface as methods.py: each takes (X, y, device_ids) and
returns {"score": float, "name": str, ...}.

Pre-registration: docs/PREREGISTRATION_DESIGNED_METRICS.md (SHA 740a1de).
"""

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA


PCA_K = 3
N_FEATURES_PER_CENTER = 5


def _amplitude_mask(d_features):
    """Mask selecting amplitude-dependent features (excludes resonance frequency).

    Feature layout per NV center: 0=T2, 1=linewidth, 2=contrast, 3=frequency, 4=T1.
    Index 3 (frequency) is set by crystal field D(T) and is gain-independent.
    """
    n_centers = d_features // N_FEATURES_PER_CENTER
    mask = np.ones(d_features, dtype=bool)
    for c in range(n_centers):
        mask[c * N_FEATURES_PER_CENTER + 3] = False
    return mask


def _t_loc(X, device_ids):
    """Location term: mean pairwise distance between per-device feature means.

    Breaks translation invariance — sensitive to additive per-device offsets.
    Grand-mean-centered so it measures device SPREAD, not global shift.
    """
    devices = np.unique(device_ids)
    if len(devices) < 2:
        return 0.0
    device_means = []
    for dev in devices:
        device_means.append(X[device_ids == dev].mean(axis=0))
    device_means = np.array(device_means)
    grand_mean = device_means.mean(axis=0)
    centered = device_means - grand_mean

    dists = []
    for i in range(len(devices)):
        for j in range(i + 1, len(devices)):
            dists.append(np.linalg.norm(centered[i] - centered[j]))
    return np.mean(dists)


def _t_scale(X, device_ids):
    """Scale term: coefficient of variation of per-device amplitude spread.

    Breaks scale invariance — sensitive to multiplicative per-device gain.
    Operates on amplitude features only (excludes resonance frequency).
    """
    devices = np.unique(device_ids)
    if len(devices) < 2:
        return 0.0
    amp_mask = _amplitude_mask(X.shape[1])
    device_spreads = []
    for dev in devices:
        X_dev_amp = X[device_ids == dev][:, amp_mask]
        device_spreads.append(X_dev_amp.std())
    spreads = np.array(device_spreads)
    mean_spread = spreads.mean()
    if mean_spread < 1e-12:
        return 0.0
    return spreads.std() / mean_spread


def _geodesic_distance(U1, U2):
    """Grassmannian geodesic distance between two (d, k) orthonormal bases."""
    M = U1.T @ U2
    svals = np.clip(linalg.svdvals(M), -1.0, 1.0)
    angles = np.arccos(svals)
    return np.sqrt(np.sum(angles ** 2))


def _t_geo(X, device_ids, k=PCA_K):
    """Subspace-angle term: mean pairwise Grassmannian geodesic across devices.

    Translation- AND scale-invariant. This is the existing geodesic, included
    as the control-within-the-metric (should stay flat on sigma_device/sigma_gain).
    """
    devices = np.unique(device_ids)
    if len(devices) < 2:
        return 0.0
    subspaces = {}
    for dev in devices:
        mask = device_ids == dev
        pca = PCA(n_components=k)
        pca.fit(X[mask])
        subspaces[int(dev)] = pca.components_.T

    devs = sorted(subspaces.keys())
    dists = []
    for i in range(len(devs)):
        for j in range(i + 1, len(devs)):
            dists.append(_geodesic_distance(subspaces[devs[i]], subspaces[devs[j]]))
    return np.mean(dists) if dists else 0.0


def designed_l1_composite(X, y, device_ids):
    """L1: Invariance-audited composite metric (closed-form, no training).

    Returns scalar score (T_loc + T_scale) plus individual components.
    T_geo is reported separately as the invariant control.
    """
    t_loc = _t_loc(X, device_ids)
    t_scale = _t_scale(X, device_ids)
    t_geo = _t_geo(X, device_ids)
    return {
        "score": t_loc + t_scale,
        "name": "designed_l1_composite",
        "t_loc": t_loc,
        "t_scale": t_scale,
        "t_geo": t_geo,
    }


def designed_l1_loc(X, y, device_ids):
    """L1 location component only (for per-component rho analysis)."""
    return {"score": _t_loc(X, device_ids), "name": "designed_l1_loc"}


def designed_l1_scale(X, y, device_ids):
    """L1 scale component only (for per-component rho analysis)."""
    return {"score": _t_scale(X, device_ids), "name": "designed_l1_scale"}


def designed_l2_dial(X, y, device_ids, lam=1.0):
    """L2: Tunable-invariance dial.

    lambda=0: subtracts full device mean -> geodesic-like (blind to offsets).
    lambda=1: keeps device mean -> fully sensitive (reproduces L1 location term).

    For the sweep, call designed_l2_sweep which returns rho(lambda).
    This function evaluates at a single lambda for the method registry.
    """
    devices = np.unique(device_ids)
    if len(devices) < 2:
        return {"score": 0.0, "name": f"designed_l2_lam{lam:.1f}"}

    device_means = {}
    for dev in devices:
        device_means[dev] = X[device_ids == dev].mean(axis=0)

    X_adjusted = X.copy()
    for dev in devices:
        mask = device_ids == dev
        X_adjusted[mask] = X[mask] - (1.0 - lam) * device_means[dev]

    t_loc_adj = _t_loc(X_adjusted, device_ids)

    amp_mask = _amplitude_mask(X.shape[1])
    device_spreads = []
    for dev in devices:
        X_dev_amp = X_adjusted[device_ids == dev][:, amp_mask]
        raw_std = X_dev_amp.std()
        device_spreads.append(raw_std)
    spreads = np.array(device_spreads)
    mean_spread = spreads.mean()
    t_scale_adj = spreads.std() / mean_spread if mean_spread > 1e-12 else 0.0

    return {
        "score": t_loc_adj + t_scale_adj,
        "name": f"designed_l2_lam{lam:.1f}",
        "lambda": lam,
        "t_loc_adj": t_loc_adj,
        "t_scale_adj": t_scale_adj,
    }


def designed_l2_sweep(X, y, device_ids, n_lambda=11):
    """L2 sweep: evaluate at n_lambda values from 0 to 1.

    Returns the full lambda-score curve for mechanism demonstration.
    """
    lambdas = np.linspace(0.0, 1.0, n_lambda)
    scores = []
    for lam in lambdas:
        result = designed_l2_dial(X, y, device_ids, lam=lam)
        scores.append(result["score"])
    return {
        "name": "designed_l2_sweep",
        "lambdas": lambdas.tolist(),
        "scores": scores,
    }


DESIGNED_METHODS = {
    "l1_composite": designed_l1_composite,
    "l1_loc": designed_l1_loc,
    "l1_scale": designed_l1_scale,
    "l2_lam0.0": lambda X, y, d: designed_l2_dial(X, y, d, lam=0.0),
    "l2_lam0.5": lambda X, y, d: designed_l2_dial(X, y, d, lam=0.5),
    "l2_lam1.0": lambda X, y, d: designed_l2_dial(X, y, d, lam=1.0),
}
