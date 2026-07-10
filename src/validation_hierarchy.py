"""Validation hierarchy for designed metrics (4 levels).

Level 1: Intervention effectiveness (IIA)
Level 2: Faithfulness (diversity + reconstruction)
Level 3: Distributional quality (KL + logit difference)
Level 4: Equivariance (primary for sigma_device and sigma_gain)

Pre-registration: docs/PREREGISTRATION_DESIGNED_METRICS.md
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def equivariance_accuracy(metric_fn, X, y, device_ids, expect_sensitive,
                          n_offsets=50, offset_scale=0.05):
    """Level 4: equivariance test for a single metric.

    Applies known additive device offsets and checks whether the metric
    responds as predicted by its claimed invariance properties.

    Args:
        metric_fn: callable(X, y, device_ids) -> {"score": float, ...}
        X: feature matrix
        y: labels
        device_ids: device indices
        expect_sensitive: if True, the metric CLAIMS to see device offsets
            (delta should be large). If False, the metric CLAIMS to be
            blind (delta should be near zero).
        n_offsets: number of random offsets to test
        offset_scale: std of the random offset vectors

    Returns:
        accuracy: fraction of offsets where behavior matches the claim.
            For sensitive metrics: delta > threshold counts as correct.
            For blind metrics: delta < threshold counts as correct.
    """
    devices = np.unique(device_ids)
    rng = np.random.default_rng(42)

    feature_scales = np.std(X, axis=0)
    feature_scales = np.where(feature_scales < 1e-12, 1.0, feature_scales)

    base_result = metric_fn(X, y, device_ids)
    base_score = base_result["score"]
    score_scale = max(abs(base_score), 1e-6)

    correct = 0
    for _ in range(n_offsets):
        dev = rng.choice(devices)
        offset = rng.normal(0, 1.0, size=X.shape[1]) * feature_scales * offset_scale
        X_shifted = X.copy()
        X_shifted[device_ids == dev] += offset

        shifted_result = metric_fn(X_shifted, y, device_ids)
        shifted_score = shifted_result["score"]
        relative_delta = abs(shifted_score - base_score) / score_scale

        if expect_sensitive:
            correct += 1 if relative_delta > 0.05 else 0
        else:
            correct += 1 if relative_delta < 0.05 else 0

    return correct / n_offsets


def equivariance_accuracy_gain(metric_fn, X, y, device_ids, expect_sensitive,
                               n_trials=50, gain_scale=0.1):
    """Level 4 for multiplicative gain axis.

    Same logic as equivariance_accuracy but applies multiplicative scaling
    to amplitude features only (excludes resonance frequency).
    """
    from designed_metrics import _amplitude_mask

    devices = np.unique(device_ids)
    rng = np.random.default_rng(42)
    amp_mask = _amplitude_mask(X.shape[1])

    base_result = metric_fn(X, y, device_ids)
    base_score = base_result["score"]
    score_scale = max(abs(base_score), 1e-6)

    correct = 0
    for _ in range(n_trials):
        dev = rng.choice(devices)
        gain = rng.normal(1.0, gain_scale)
        X_scaled = X.copy()
        X_scaled[np.ix_(device_ids == dev, amp_mask)] *= gain

        scaled_result = metric_fn(X_scaled, y, device_ids)
        scaled_score = scaled_result["score"]
        relative_delta = abs(scaled_score - base_score) / score_scale

        if expect_sensitive:
            correct += 1 if relative_delta > 0.05 else 0
        else:
            correct += 1 if relative_delta < 0.05 else 0

    return correct / n_trials


def interchange_iia(X, y, device_ids, classifier=None):
    """Level 1: intervention effectiveness (IIA).

    Swap device-attributable component between cross-label pairs,
    check if classifier prediction flips as predicted.
    """
    devices = np.unique(device_ids)
    device_means = {d: X[device_ids == d].mean(axis=0) for d in devices}

    if classifier is None:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        classifier = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
        classifier.fit(X_sc, y)
        X_for_pred = X_sc
    else:
        X_for_pred = X

    hits, total = 0, 0
    for di in devices:
        for dj in devices:
            if di == dj:
                continue
            idx_a_pool = np.where((device_ids == di) & (y == 0))[0]
            idx_b_pool = np.where((device_ids == dj) & (y == 1))[0]
            for idx_a in idx_a_pool[:10]:
                for idx_b in idx_b_pool[:5]:
                    x_intervened = X[idx_a].copy()
                    x_intervened += (device_means[dj] - device_means[di])
                    if classifier is not None:
                        pred_orig = classifier.predict(X_for_pred[idx_a:idx_a+1])[0]
                        x_int_scaled = scaler.transform(x_intervened.reshape(1, -1))
                        pred_new = classifier.predict(x_int_scaled)[0]
                    if pred_orig != y[idx_b]:
                        total += 1
                        if pred_new == y[idx_b]:
                            hits += 1
    return hits / max(total, 1)


def diversity_ratio(X, device_ids, X_intervened, device_ids_new):
    """Level 2: faithfulness — diversity preservation after intervention."""
    devices_orig = np.unique(device_ids)
    devices_new = np.unique(device_ids_new)
    var_orig = np.mean([X[device_ids == d].var() for d in devices_orig])
    var_new = np.mean([X_intervened[device_ids_new == d].var() for d in devices_new])
    return var_new / max(var_orig, 1e-12)
