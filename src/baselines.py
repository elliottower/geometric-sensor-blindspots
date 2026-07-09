"""Scalar baselines for comparison against geometric methods.

Same interface as methods.py: each takes (X, y, device_ids) and
returns {"score": float, "name": str}.

These are the "when does geometry beat standard statistics?" controls.
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def baseline_ttest(X, y, device_ids):
    """Two-sample t-test: max |t-statistic| across features."""
    X0 = X[y == 0]
    X1 = X[y == 1]
    t_stats, _ = stats.ttest_ind(X0, X1, axis=0)
    return {"score": np.max(np.abs(t_stats)), "name": "ttest_max"}


def baseline_cohens_d(X, y, device_ids):
    """Mean Cohen's d across features."""
    X0 = X[y == 0]
    X1 = X[y == 1]
    pooled_std = np.sqrt((X0.var(axis=0) + X1.var(axis=0)) / 2)
    pooled_std = np.clip(pooled_std, 1e-12, None)
    d = np.abs(X1.mean(axis=0) - X0.mean(axis=0)) / pooled_std
    return {"score": d.mean(), "name": "cohens_d_mean"}


def baseline_lasso(X, y, device_ids):
    """LASSO classification accuracy (cross-validated)."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    clf = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=500)
    scores = cross_val_score(clf, X_sc, y, cv=3, scoring="roc_auc")
    return {"score": scores.mean(), "name": "lasso_auc"}


def baseline_random_forest(X, y, device_ids):
    """Random forest classification accuracy (cross-validated)."""
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
    scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
    return {"score": scores.mean(), "name": "random_forest_auc"}


def baseline_logistic(X, y, device_ids):
    """Plain logistic regression AUC (no regularization)."""
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
    scores = cross_val_score(clf, X_sc, y, cv=3, scoring="roc_auc")
    return {"score": scores.mean(), "name": "logistic_auc"}


def baseline_irm_penalty(X, y, device_ids):
    """IRM (Invariant Risk Minimization) penalty proxy.

    Measures how much the optimal linear predictor varies across devices.
    Lower penalty = more invariant across environments.
    Score = 1 / (1 + penalty) so higher = better.
    """
    devices = np.unique(device_ids)
    if len(devices) < 2:
        return {"score": 1.0, "name": "irm_penalty"}
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    per_device_coef = []
    for dev in devices:
        mask = device_ids == dev
        if mask.sum() < 10:
            continue
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
        clf.fit(X_sc[mask], y[mask])
        per_device_coef.append(clf.coef_.ravel())
    if len(per_device_coef) < 2:
        return {"score": 1.0, "name": "irm_penalty"}
    coefs = np.array(per_device_coef)
    penalty = np.mean(np.var(coefs, axis=0))
    return {"score": 1.0 / (1.0 + penalty), "name": "irm_penalty"}


def baseline_dro_worst_device(X, y, device_ids):
    """DRO (Distributionally Robust Optimization) proxy.

    Train on all data, report worst-case per-device AUC.
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
    clf.fit(X_sc, y)
    devices = np.unique(device_ids)
    worst_auc = 1.0
    for dev in devices:
        mask = device_ids == dev
        y_dev = y[mask]
        if len(np.unique(y_dev)) < 2:
            continue
        proba = clf.predict_proba(X_sc[mask])[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_dev, proba)
        worst_auc = min(worst_auc, auc)
    return {"score": worst_auc, "name": "dro_worst_device_auc"}


BASELINES = {
    "ttest": baseline_ttest,
    "cohens_d": baseline_cohens_d,
    "lasso": baseline_lasso,
    "random_forest": baseline_random_forest,
    "logistic": baseline_logistic,
    "irm": baseline_irm_penalty,
    "dro": baseline_dro_worst_device,
}
