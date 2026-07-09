"""Unified method interface for NV-diamond experiments.

Every method takes the same input:
    X:          (n_samples, d_features) feature matrix
    y:          (n_samples,) binary labels (0=healthy, 1=tumor)
    device_ids: (n_samples,) integer device index per sample

And returns a dict with at minimum:
    "score": float  — the primary metric value (higher = more signal/robustness)
    "name":  str    — method name

Methods are registered in METHODS dict keyed by integer ID (1-24).
"""

import numpy as np
from scipy import stats, linalg
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


PCA_K = 3


def _pca_subspace(X, k=PCA_K):
    """Extract top-k PCA subspace as (d, k) orthonormal basis."""
    pca = PCA(n_components=k)
    pca.fit(X)
    return pca.components_.T


def _per_device_subspaces(X, device_ids, k=PCA_K):
    """Extract PCA subspaces per device."""
    devices = np.unique(device_ids)
    subspaces = {}
    for dev in devices:
        mask = device_ids == dev
        subspaces[int(dev)] = _pca_subspace(X[mask], k)
    return subspaces


def _geodesic_distance(U1, U2):
    """Grassmannian geodesic distance between two (d, k) orthonormal bases."""
    M = U1.T @ U2
    svals = np.clip(linalg.svdvals(M), -1.0, 1.0)
    angles = np.arccos(svals)
    return np.sqrt(np.sum(angles ** 2))


def _compose_holonomy(subspaces_list):
    """Holonomy around a loop of subspaces. Returns ||I - T||_F."""
    k = subspaces_list[0].shape[1]
    T = np.eye(k)
    n = len(subspaces_list)
    for i in range(n):
        U_from = subspaces_list[i]
        U_to = subspaces_list[(i + 1) % n]
        T = T @ (U_from.T @ U_to)
    return np.linalg.norm(np.eye(k) - T, "fro")


# =====================================================================
# Methods 1-5: Bracket norm family
# =====================================================================

def _bracket_norm_raw(X, y):
    """Bracket norm: ||[dX/dy, dX/dt]|| where t is a continuous axis.

    Simplified for two-class data: measure how much the class-difference
    direction rotates across the feature space. Uses SVD of the cross-
    covariance between class-conditioned means and within-class variation.
    """
    X0, X1 = X[y == 0], X[y == 1]
    delta_mu = X1.mean(axis=0) - X0.mean(axis=0)
    delta_mu_norm = np.linalg.norm(delta_mu)
    if delta_mu_norm < 1e-12:
        return 0.0
    delta_hat = delta_mu / delta_mu_norm
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    cov_delta = cov @ delta_hat
    cross_component = cov_delta - (cov_delta @ delta_hat) * delta_hat
    return np.linalg.norm(cross_component)


def method_01_bracket_norm(X, y, device_ids):
    return {"score": _bracket_norm_raw(X, y), "name": "bracket_norm_raw"}


def method_02_bracket_norm_corrected(X, y, device_ids):
    n_features = X.shape[1]
    raw = _bracket_norm_raw(X, y)
    return {"score": raw / np.sqrt(n_features), "name": "bracket_norm_corrected"}


def method_03_transport_stable_bracket(X, y, device_ids):
    """Frechet-variance-penalized bracket norm across devices."""
    devices = np.unique(device_ids)
    per_device_bn = []
    for dev in devices:
        mask = device_ids == dev
        per_device_bn.append(_bracket_norm_raw(X[mask], y[mask]))
    bn_array = np.array(per_device_bn)
    return {
        "score": bn_array.mean() - bn_array.std(),
        "name": "transport_stable_bracket",
    }


def method_04_phenotype_projected_bracket(X, y, device_ids):
    """Bracket norm projected onto the diagnostic (class-difference) axis."""
    X0, X1 = X[y == 0], X[y == 1]
    delta_mu = X1.mean(axis=0) - X0.mean(axis=0)
    norm = np.linalg.norm(delta_mu)
    if norm < 1e-12:
        return {"score": 0.0, "name": "phenotype_projected_bracket"}
    delta_hat = delta_mu / norm
    proj_scores = X @ delta_hat
    proj_bn = np.abs(proj_scores[y == 1].mean() - proj_scores[y == 0].mean())
    return {"score": proj_bn, "name": "phenotype_projected_bracket"}


def method_05_localized_bracket(X, y, device_ids):
    """Ratio of per-device bracket norm to global bracket norm."""
    global_bn = _bracket_norm_raw(X, y)
    if global_bn < 1e-12:
        return {"score": 0.0, "name": "localized_bracket"}
    devices = np.unique(device_ids)
    per_device_bn = []
    for dev in devices:
        mask = device_ids == dev
        per_device_bn.append(_bracket_norm_raw(X[mask], y[mask]))
    return {
        "score": np.mean(per_device_bn) / global_bn,
        "name": "localized_bracket",
    }


# =====================================================================
# Methods 6-7: Curvature
# =====================================================================

def _build_knn_graph(X, k=10):
    """Build k-NN graph with Euclidean weights."""
    from scipy.spatial import KDTree
    tree = KDTree(X)
    dists, indices = tree.query(X, k=k + 1)
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            W[i, j] = dists[i, j_idx]
            W[j, i] = dists[i, j_idx]
    return W


def _ollivier_ricci_curvature_edge(W, i, j, alpha=0.5):
    """ORC for a single edge (i, j) in a weighted adjacency matrix."""
    from scipy.optimize import linear_sum_assignment
    n = W.shape[0]

    def _measure(node):
        neighbors = np.where(W[node] > 0)[0]
        if len(neighbors) == 0:
            m = np.zeros(n)
            m[node] = 1.0
            return m
        weights = W[node, neighbors]
        weights = weights / weights.sum()
        m = np.zeros(n)
        m[node] = alpha
        m[neighbors] += (1 - alpha) * weights
        return m

    mu_i = _measure(i)
    mu_j = _measure(j)

    supp_i = np.where(mu_i > 1e-15)[0]
    supp_j = np.where(mu_j > 1e-15)[0]

    if len(supp_i) == 0 or len(supp_j) == 0:
        return 0.0

    cost = np.zeros((len(supp_i), len(supp_j)))
    for a, si in enumerate(supp_i):
        for b, sj in enumerate(supp_j):
            cost[a, b] = np.linalg.norm(np.array([si]) - np.array([sj]))

    supply = mu_i[supp_i]
    demand = mu_j[supp_j]

    min_len = min(len(supply), len(demand))
    supply_trunc = supply[:min_len]
    demand_trunc = demand[:min_len]
    cost_trunc = cost[:min_len, :min_len]

    row_ind, col_ind = linear_sum_assignment(cost_trunc)
    transport_dist = cost_trunc[row_ind, col_ind].sum()

    d_ij = W[i, j] if W[i, j] > 0 else 1.0
    return 1.0 - transport_dist / d_ij


def method_06_ollivier_ricci(X, y, device_ids):
    """Mean ORC across edges of k-NN graph, comparing class-boundary edges."""
    n_sub = min(200, X.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], n_sub, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]
    W = _build_knn_graph(X_sub, k=8)
    edges = list(zip(*np.where(np.triu(W) > 0)))
    if not edges:
        return {"score": 0.0, "name": "ollivier_ricci_curvature"}
    cross_curv = []
    within_curv = []
    for i, j in edges[:100]:
        kappa = _ollivier_ricci_curvature_edge(W, i, j)
        if y_sub[i] != y_sub[j]:
            cross_curv.append(kappa)
        else:
            within_curv.append(kappa)
    cross_mean = np.mean(cross_curv) if cross_curv else 0.0
    within_mean = np.mean(within_curv) if within_curv else 0.0
    return {"score": within_mean - cross_mean, "name": "ollivier_ricci_curvature"}


def method_07_forman_ricci(X, y, device_ids):
    """Forman-Ricci curvature: combinatorial approximation."""
    n_sub = min(200, X.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], n_sub, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]
    W = _build_knn_graph(X_sub, k=8)
    degree = (W > 0).sum(axis=1)
    edges = list(zip(*np.where(np.triu(W) > 0)))
    if not edges:
        return {"score": 0.0, "name": "forman_ricci_curvature"}
    cross_curv = []
    within_curv = []
    for i, j in edges[:100]:
        shared = np.sum((W[i] > 0) & (W[j] > 0))
        frc = 4 - degree[i] - degree[j] + 3 * shared
        if y_sub[i] != y_sub[j]:
            cross_curv.append(frc)
        else:
            within_curv.append(frc)
    cross_mean = np.mean(cross_curv) if cross_curv else 0.0
    within_mean = np.mean(within_curv) if within_curv else 0.0
    return {"score": within_mean - cross_mean, "name": "forman_ricci_curvature"}


# =====================================================================
# Methods 8-9: Grassmannian
# =====================================================================

def method_08_grassmannian_geodesic(X, y, device_ids):
    """Mean pairwise Grassmannian geodesic distance across devices."""
    subspaces = _per_device_subspaces(X, device_ids)
    devs = sorted(subspaces.keys())
    dists = []
    for i in range(len(devs)):
        for j in range(i + 1, len(devs)):
            dists.append(_geodesic_distance(subspaces[devs[i]], subspaces[devs[j]]))
    return {"score": np.mean(dists) if dists else 0.0, "name": "grassmannian_geodesic"}


def method_09_grassmannian_holonomy(X, y, device_ids):
    """Holonomy around the device loop."""
    subspaces = _per_device_subspaces(X, device_ids)
    devs = sorted(subspaces.keys())
    if len(devs) < 3:
        return {"score": 0.0, "name": "grassmannian_holonomy"}
    sub_list = [subspaces[d] for d in devs]
    return {"score": _compose_holonomy(sub_list), "name": "grassmannian_holonomy"}


# =====================================================================
# Methods 10-11: Sheaf cohomology
# =====================================================================

def _per_device_effect_estimates(X, y, device_ids):
    """Compute per-device effect size (Cohen's d) and SE for the diagnostic signal."""
    devices = np.unique(device_ids)
    estimates = {}
    for dev in devices:
        mask = device_ids == dev
        X_dev = X[mask]
        y_dev = y[mask]
        X0 = X_dev[y_dev == 0]
        X1 = X_dev[y_dev == 1]
        pooled_std = np.sqrt((X0.var(axis=0).mean() + X1.var(axis=0).mean()) / 2)
        if pooled_std < 1e-12:
            estimates[int(dev)] = {"beta": 0.0, "se": 1.0}
            continue
        d = (X1.mean() - X0.mean()) / pooled_std
        se = np.sqrt(2.0 / X0.shape[0] + d ** 2 / (2 * X0.shape[0]))
        estimates[int(dev)] = {"beta": d, "se": max(se, 1e-8)}
    return estimates


def method_10_sheaf_h1(X, y, device_ids):
    """Sheaf H1 obstruction: Cochran's Q on per-device effect estimates."""
    estimates = _per_device_effect_estimates(X, y, device_ids)
    betas = np.array([e["beta"] for e in estimates.values()])
    ses = np.array([e["se"] for e in estimates.values()])
    weights = 1.0 / (ses ** 2)
    beta_pooled = np.sum(weights * betas) / np.sum(weights)
    Q = np.sum(weights * (betas - beta_pooled) ** 2)
    df = len(betas) - 1
    p_value = 1.0 - stats.chi2.cdf(Q, df) if df > 0 else 1.0
    return {"score": Q, "p_value": p_value, "name": "sheaf_h1"}


def method_11_sheaf_q_per_edge(X, y, device_ids):
    """Per-edge sheaf Q: which device pair is most inconsistent."""
    estimates = _per_device_effect_estimates(X, y, device_ids)
    devs = sorted(estimates.keys())
    max_q = 0.0
    for i in range(len(devs)):
        for j in range(i + 1, len(devs)):
            bi, bj = estimates[devs[i]]["beta"], estimates[devs[j]]["beta"]
            si, sj = estimates[devs[i]]["se"], estimates[devs[j]]["se"]
            q = (bi - bj) ** 2 / (si ** 2 + sj ** 2)
            max_q = max(max_q, q)
    return {"score": max_q, "name": "sheaf_q_per_edge"}


# =====================================================================
# Method 12: TransportKit fusion (simplified)
# =====================================================================

def method_12_transportkit(X, y, device_ids):
    """Simplified TransportKit: subspace alignment + effect consistency."""
    geo = method_08_grassmannian_geodesic(X, y, device_ids)["score"]
    sheaf = method_10_sheaf_h1(X, y, device_ids)["score"]
    return {"score": geo + 0.1 * sheaf, "name": "transportkit_fusion"}


# =====================================================================
# Method 13: CKA
# =====================================================================

def method_13_cka(X, y, device_ids):
    """Linear CKA between device representation matrices."""
    devices = np.unique(device_ids)
    if len(devices) < 2:
        return {"score": 0.0, "name": "cka"}
    cka_vals = []
    for i in range(len(devices)):
        for j in range(i + 1, len(devices)):
            Xi = X[device_ids == devices[i]]
            Xj = X[device_ids == devices[j]]
            n = min(Xi.shape[0], Xj.shape[0])
            Xi, Xj = Xi[:n], Xj[:n]
            Xi = Xi - Xi.mean(axis=0)
            Xj = Xj - Xj.mean(axis=0)
            hsic_xy = np.linalg.norm(Xi.T @ Xj, "fro") ** 2
            hsic_xx = np.linalg.norm(Xi.T @ Xi, "fro") ** 2
            hsic_yy = np.linalg.norm(Xj.T @ Xj, "fro") ** 2
            denom = np.sqrt(hsic_xx * hsic_yy)
            cka_vals.append(hsic_xy / denom if denom > 1e-12 else 0.0)
    return {"score": np.mean(cka_vals), "name": "cka"}


# =====================================================================
# Method 14: Procrustes
# =====================================================================

def method_14_procrustes(X, y, device_ids):
    """Procrustes distance between per-device PCA embeddings."""
    devices = np.unique(device_ids)
    if len(devices) < 2:
        return {"score": 0.0, "name": "procrustes"}
    embeddings = {}
    for dev in devices:
        mask = device_ids == dev
        pca = PCA(n_components=PCA_K)
        embeddings[int(dev)] = pca.fit_transform(X[mask])
    devs = sorted(embeddings.keys())
    dists = []
    for i in range(len(devs)):
        for j in range(i + 1, len(devs)):
            Ei = embeddings[devs[i]]
            Ej = embeddings[devs[j]]
            n = min(Ei.shape[0], Ej.shape[0])
            Ei, Ej = Ei[:n], Ej[:n]
            Ei = Ei - Ei.mean(axis=0)
            Ej = Ej - Ej.mean(axis=0)
            Ei = Ei / np.linalg.norm(Ei, "fro")
            Ej = Ej / np.linalg.norm(Ej, "fro")
            U, _, Vt = np.linalg.svd(Ei.T @ Ej)
            R = U @ Vt
            dists.append(np.linalg.norm(Ei - Ej @ R.T, "fro"))
    return {"score": np.mean(dists), "name": "procrustes"}


# =====================================================================
# Registry
# =====================================================================

METHODS = {
    1: method_01_bracket_norm,
    2: method_02_bracket_norm_corrected,
    3: method_03_transport_stable_bracket,
    4: method_04_phenotype_projected_bracket,
    5: method_05_localized_bracket,
    6: method_06_ollivier_ricci,
    7: method_07_forman_ricci,
    8: method_08_grassmannian_geodesic,
    9: method_09_grassmannian_holonomy,
    10: method_10_sheaf_h1,
    11: method_11_sheaf_q_per_edge,
    12: method_12_transportkit,
    13: method_13_cka,
    14: method_14_procrustes,
}

# =====================================================================
# Methods 15-16: Persistent homology (H0 only — union-find is correct
# for H0; hand-rolled H1 was a broken heuristic, so we restrict to
# connected-component persistence which we can implement correctly)
# =====================================================================

def _persistence_h0(X, n_sub=150):
    """Compute H0 persistence diagram via Kruskal-style union-find.

    Returns list of (birth, death) pairs for connected components.
    Birth is always 0 (all points born at filtration start); death
    is the edge weight at which two components merge.
    """
    rng = np.random.default_rng(0)
    if X.shape[0] > n_sub:
        idx = rng.choice(X.shape[0], n_sub, replace=False)
        X = X[idx]
    D = squareform(pdist(X))
    n = D.shape[0]

    triu_i, triu_j = np.triu_indices(n, k=1)
    edge_weights = D[triu_i, triu_j]
    order = np.argsort(edge_weights)

    parent = np.arange(n)
    rank = np.zeros(n, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    diagram = []
    for idx in order:
        i, j = triu_i[idx], triu_j[idx]
        w = edge_weights[idx]
        ri, rj = find(i), find(j)
        if ri != rj:
            diagram.append((0.0, w))
            if rank[ri] < rank[rj]:
                parent[ri] = rj
            elif rank[ri] > rank[rj]:
                parent[rj] = ri
            else:
                parent[rj] = ri
                rank[ri] += 1

    roots = set()
    for i in range(n):
        roots.add(find(i))
    for _ in roots:
        diagram.append((0.0, D.max()))

    return diagram


def _wasserstein_persistence_distance(dgm1, dgm2):
    """Wasserstein-1 distance between two H0 persistence diagrams.

    Matches diagram points via optimal assignment, with unmatched points
    projected to the diagonal.
    """
    from scipy.optimize import linear_sum_assignment

    if not dgm1 and not dgm2:
        return 0.0
    pts1 = np.array(dgm1) if dgm1 else np.zeros((0, 2))
    pts2 = np.array(dgm2) if dgm2 else np.zeros((0, 2))

    n1, n2 = len(pts1), len(pts2)
    n = n1 + n2

    cost = np.zeros((n, n))
    for i in range(n1):
        for j in range(n2):
            cost[i, j] = np.sum(np.abs(pts1[i] - pts2[j]))
    for i in range(n1):
        for j in range(n2, n):
            pers = pts1[i, 1] - pts1[i, 0]
            cost[i, j] = pers / 2.0
    for i in range(n1, n):
        for j in range(n2):
            pers = pts2[j, 1] - pts2[j, 0]
            cost[i, j] = pers / 2.0

    row_ind, col_ind = linear_sum_assignment(cost)
    return cost[row_ind, col_ind].sum()


def method_15_persistent_h0(X, y, device_ids):
    """Total H0 persistence (sum of all component lifetimes)."""
    dgm = _persistence_h0(X)
    return {"score": sum(d - b for b, d in dgm), "name": "persistent_h0"}


def method_16_wasserstein_persistence(X, y, device_ids):
    """Wasserstein-1 distance between H0 persistence diagrams of class 0 vs class 1."""
    dgm0 = _persistence_h0(X[y == 0])
    dgm1 = _persistence_h0(X[y == 1])
    dist = _wasserstein_persistence_distance(dgm0, dgm1)
    return {"score": dist, "name": "wasserstein_persistence"}


# =====================================================================
# Method 17: Quantum Fisher information flatness
# =====================================================================

def method_17_qfi_flatness(X, y, device_ids):
    """QFI flatness: features whose Fisher information is constant across devices.

    For classical feature data, QFI reduces to the inverse variance of
    the per-device Fisher information estimates. High score = flat = robust.
    """
    devices = np.unique(device_ids)
    per_device_fi = []
    for dev in devices:
        mask = device_ids == dev
        X_dev = X[mask]
        y_dev = y[mask]
        X0 = X_dev[y_dev == 0]
        X1 = X_dev[y_dev == 1]
        mu_diff = X1.mean(axis=0) - X0.mean(axis=0)
        pooled_var = (X0.var(axis=0) + X1.var(axis=0)) / 2
        pooled_var = np.clip(pooled_var, 1e-12, None)
        fi = np.sum(mu_diff ** 2 / pooled_var)
        per_device_fi.append(fi)
    fi_arr = np.array(per_device_fi)
    mean_fi = fi_arr.mean()
    if mean_fi < 1e-12:
        return {"score": 0.0, "name": "qfi_flatness"}
    cv = fi_arr.std() / mean_fi
    return {"score": 1.0 / (1.0 + cv), "name": "qfi_flatness"}


# =====================================================================
# Method 18: Berry phase (geometric phase)
# =====================================================================

def method_18_berry_phase(X, y, device_ids):
    """Berry phase: geometric phase accumulated by the diagnostic direction
    as we traverse the device loop.

    The diagnostic axis (class-difference direction) rotates as we move
    across devices. The accumulated angle measures decoherence anisotropy.
    """
    devices = sorted(np.unique(device_ids))
    if len(devices) < 3:
        return {"score": 0.0, "name": "berry_phase"}
    directions = []
    for dev in devices:
        mask = device_ids == dev
        X_dev = X[mask]
        y_dev = y[mask]
        X0 = X_dev[y_dev == 0]
        X1 = X_dev[y_dev == 1]
        delta = X1.mean(axis=0) - X0.mean(axis=0)
        norm = np.linalg.norm(delta)
        if norm < 1e-12:
            return {"score": 0.0, "name": "berry_phase"}
        directions.append(delta / norm)

    phase = 0.0
    n = len(directions)
    for i in range(n):
        j = (i + 1) % n
        dot = np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0)
        phase += np.arccos(np.abs(dot))
    return {"score": phase, "name": "berry_phase"}


# =====================================================================
# Method 19: Spectral gap stability
# =====================================================================

def method_19_spectral_gap(X, y, device_ids):
    """Spectral gap of the k-NN graph Laplacian, compared across devices.

    Stable spectral gap = robust cluster structure under decoherence.
    """
    devices = np.unique(device_ids)
    gaps = []
    for dev in devices:
        mask = device_ids == dev
        X_dev = X[mask]
        n_sub = min(150, X_dev.shape[0])
        if n_sub < 10:
            continue
        rng = np.random.default_rng(int(dev))
        idx = rng.choice(X_dev.shape[0], n_sub, replace=False)
        X_sub = X_dev[idx]
        W = _build_knn_graph(X_sub, k=min(8, n_sub - 1))
        W = (W + W.T) / 2
        degree = W.sum(axis=1)
        degree = np.clip(degree, 1e-12, None)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        L_norm = np.eye(n_sub) - D_inv_sqrt @ W @ D_inv_sqrt
        eigvals = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
        if len(eigvals) >= 2:
            gaps.append(eigvals[1])
    if not gaps:
        return {"score": 0.0, "name": "spectral_gap_stability"}
    gap_arr = np.array(gaps)
    mean_gap = gap_arr.mean()
    stability = mean_gap / (1.0 + gap_arr.std())
    return {"score": stability, "name": "spectral_gap_stability"}


# =====================================================================
# Method 20: Wasserstein distance
# =====================================================================

def method_20_wasserstein(X, y, device_ids):
    """1D Wasserstein distance between class-conditional distributions,
    averaged across devices.

    Projects onto top PCA component for a 1D distribution comparison.
    """
    devices = np.unique(device_ids)
    w_dists = []
    for dev in devices:
        mask = device_ids == dev
        X_dev = X[mask]
        y_dev = y[mask]
        pca = PCA(n_components=1)
        proj = pca.fit_transform(X_dev).ravel()
        p0 = np.sort(proj[y_dev == 0])
        p1 = np.sort(proj[y_dev == 1])
        n = min(len(p0), len(p1))
        if n < 2:
            continue
        p0_q = np.quantile(p0, np.linspace(0, 1, 100))
        p1_q = np.quantile(p1, np.linspace(0, 1, 100))
        w_dists.append(np.mean(np.abs(p0_q - p1_q)))
    if not w_dists:
        return {"score": 0.0, "name": "wasserstein_1d"}
    return {"score": np.mean(w_dists), "name": "wasserstein_1d"}


# =====================================================================
# Method 21: Von Neumann entropy stability
# =====================================================================

def method_21_vn_entropy(X, y, device_ids):
    """Von Neumann entropy of the normalized covariance (density matrix proxy).

    S = -Tr(rho log rho) where rho = C / Tr(C), C = covariance.
    Stability = 1 / (1 + CV of per-device entropies).
    """
    devices = np.unique(device_ids)
    entropies = []
    for dev in devices:
        mask = device_ids == dev
        X_dev = X[mask]
        X_centered = X_dev - X_dev.mean(axis=0)
        C = X_centered.T @ X_centered / X_dev.shape[0]
        eigvals = np.real(np.linalg.eigvalsh(C))
        eigvals = eigvals[eigvals > 1e-12]
        if len(eigvals) == 0:
            entropies.append(0.0)
            continue
        rho = eigvals / eigvals.sum()
        S = -np.sum(rho * np.log(rho))
        entropies.append(S)
    ent_arr = np.array(entropies)
    mean_ent = ent_arr.mean()
    if mean_ent < 1e-12:
        return {"score": 0.0, "name": "vn_entropy_stability"}
    cv = ent_arr.std() / mean_ent
    return {"score": 1.0 / (1.0 + cv), "name": "vn_entropy_stability"}


# =====================================================================
# Method 22: Fidelity / diamond norm proxy
# =====================================================================

def method_22_fidelity(X, y, device_ids):
    """Fidelity between per-device covariance matrices (density matrix proxy).

    F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.
    Mean pairwise fidelity across devices.
    """
    devices = np.unique(device_ids)
    if len(devices) < 2:
        return {"score": 1.0, "name": "fidelity_proxy"}

    def _density_matrix(X_dev):
        X_c = X_dev - X_dev.mean(axis=0)
        C = X_c.T @ X_c / X_dev.shape[0]
        eigvals = np.real(np.linalg.eigvalsh(C))
        eigvals = np.clip(eigvals, 0, None)
        tr = eigvals.sum()
        if tr < 1e-12:
            return np.eye(C.shape[0]) / C.shape[0]
        return C / tr

    rhos = {}
    for dev in devices:
        mask = device_ids == dev
        rhos[int(dev)] = _density_matrix(X[mask])

    fidelities = []
    devs = sorted(rhos.keys())
    for i in range(len(devs)):
        for j in range(i + 1, len(devs)):
            rho = rhos[devs[i]]
            sigma = rhos[devs[j]]
            sqrt_rho = linalg.sqrtm(rho)
            if np.any(np.isnan(sqrt_rho)):
                fidelities.append(0.0)
                continue
            M = sqrt_rho @ sigma @ sqrt_rho
            eigvals = np.real(np.linalg.eigvalsh(M))
            eigvals = np.clip(eigvals, 0, None)
            F = np.sum(np.sqrt(eigvals)) ** 2
            fidelities.append(min(F, 1.0))
    return {"score": np.mean(fidelities) if fidelities else 0.0, "name": "fidelity_proxy"}


# =====================================================================
# Method 23: Persistent sheaf cohomology
# =====================================================================

def method_23_persistent_sheaf(X, y, device_ids):
    """Persistent sheaf cohomology: track sheaf Q across decoherence proxy.

    Uses subsampled dataset at increasing noise levels to simulate a
    filtration. Reports total area under the Q(epsilon) curve.
    """
    base_estimates = _per_device_effect_estimates(X, y, device_ids)
    noise_levels = np.linspace(0, 0.5, 10)
    rng = np.random.default_rng(42)
    q_values = []
    for noise in noise_levels:
        perturbed_betas = []
        perturbed_ses = []
        for dev, est in base_estimates.items():
            b = est["beta"] + rng.normal(0, noise * abs(est["beta"]) + 1e-6)
            s = est["se"] * (1.0 + noise)
            perturbed_betas.append(b)
            perturbed_ses.append(s)
        betas = np.array(perturbed_betas)
        ses = np.array(perturbed_ses)
        weights = 1.0 / (ses ** 2)
        beta_pooled = np.sum(weights * betas) / np.sum(weights)
        Q = np.sum(weights * (betas - beta_pooled) ** 2)
        q_values.append(Q)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    auc = trapz_fn(q_values, noise_levels)
    return {"score": auc, "name": "persistent_sheaf_cohomology"}


# Method 24 (Chern number) CUT: the naive plaquette-phase approach is not
# gauge-invariant (needs Fukui-Hatsugai-Suzuki lattice gauge method). A
# broken estimator that returns 0 would spuriously "confirm" H6c. Slot
# reserved for a correct implementation if FHS is added later.


# =====================================================================
# Registry (23 methods — slot 24 intentionally empty)
# =====================================================================

METHODS = {
    1: method_01_bracket_norm,
    2: method_02_bracket_norm_corrected,
    3: method_03_transport_stable_bracket,
    4: method_04_phenotype_projected_bracket,
    5: method_05_localized_bracket,
    6: method_06_ollivier_ricci,
    7: method_07_forman_ricci,
    8: method_08_grassmannian_geodesic,
    9: method_09_grassmannian_holonomy,
    10: method_10_sheaf_h1,
    11: method_11_sheaf_q_per_edge,
    12: method_12_transportkit,
    13: method_13_cka,
    14: method_14_procrustes,
    15: method_15_persistent_h0,
    16: method_16_wasserstein_persistence,
    17: method_17_qfi_flatness,
    18: method_18_berry_phase,
    19: method_19_spectral_gap,
    20: method_20_wasserstein,
    21: method_21_vn_entropy,
    22: method_22_fidelity,
    23: method_23_persistent_sheaf,
}
