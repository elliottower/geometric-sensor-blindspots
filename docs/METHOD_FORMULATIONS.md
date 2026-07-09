# Mathematical Formulations for All 24 Methods

Notation:
- X^(r) = readout matrix under decoherence regime r, shape (n_samples, d_features)
- x_i^(r) = row i of X^(r) (one sample's features)
- y_i in {0, 1} = disease label (healthy, tumor)
- R = set of decoherence regimes
- S = set of sensor modalities (for multi-sensor)

---

## 1. Bracket norm (raw)

**From**: bracket-norm repo
**Existing code**: Yes

For feature j, the bracket norm measures direction instability of
the perturbation response when moving from regime r to regime r':

    BN_j = || hat{v}_j^(r) - hat{v}_j^(r') ||_2

where hat{v}_j^(r) = normalized mean-difference vector between classes
in the subspace spanned by feature j and its k nearest correlated features:

    v_j^(r) = mean(X^(r)[y=1, N(j)]) - mean(X^(r)[y=0, N(j)])
    hat{v}_j^(r) = v_j^(r) / ||v_j^(r)||

N(j) = indices of the k features most correlated with j.

**Robustness score**: BN_j averaged over all regime pairs (r, r').
Lower = more robust.

**Reduces to**: absolute mean difference when k=1 and only 2 regimes.

---

## 2. Bracket norm / sqrt(n)

**From**: bracket-norm repo
**Existing code**: Yes

    BN_corrected_j = BN_j / sqrt(|N(j)|)

Corrects for the fact that higher-dimensional neighborhoods have more
degrees of freedom for spurious direction change.

**Reduces to**: BN_j when k=1.

---

## 3. Transport-stable bracket norm

**From**: direction-instability-drug-validity repo
**Existing code**: Yes

    BN_transport_j = BN_j * exp(-lambda * Var_Frechet(hat{v}_j))

where Var_Frechet is the Frechet variance of the directions {hat{v}_j^(r)}
on the unit sphere across regimes:

    Var_Frechet = (1/|R|) sum_r d_sphere(hat{v}_j^(r), mu_Frechet)^2

mu_Frechet = Frechet mean on S^(k-1).

Penalizes features whose instability is itself unstable (noisy noise).

---

## 4. Phenotype-projected bracket norm

**From**: direction-instability-drug-validity repo
**Existing code**: Yes

Let w = classifier weight vector (e.g., from logistic regression on
pooled data). The diagnostic-projected bracket norm:

    BN_diag_j = |<hat{v}_j^(r) - hat{v}_j^(r'), w/||w||>|

Only counts direction change that aligns with the diagnostic axis.
A feature whose direction changes orthogonally to w is still
diagnostically stable.

---

## 5. Localized bracket norm

**From**: direction-instability-drug-validity repo
**Existing code**: Yes

    BN_local_j = BN_j(within sensor s) / BN_j(across all sensors)

For multi-sensor: ratio of within-modality instability to total
instability. If a feature is unstable only within NV data but not
when you include photon data, it may be a sensor-specific artifact.

---

## 6. Ollivier-Ricci curvature (ORC)

**From**: psychiatric-comorbidity-curvature repo
**Existing code**: Yes (NetworkX + POT)

Construct k-NN graph G on samples X^(r). For edge (i,j):

    kappa_ORC(i,j) = 1 - W_1(mu_i, mu_j) / d(i,j)

where W_1 is the 1-Wasserstein distance between the probability
measures mu_i = uniform on neighbors of i, mu_j = uniform on neighbors
of j, and d(i,j) is the graph distance.

**Robustness metric**: For each decoherence regime, compute
mean curvature of the graph. Curvature stability across regimes
= robustness of the sample-space geometry.

    ORC_stability = Var_r[ mean_{(i,j) in E} kappa_ORC^(r)(i,j) ]

Lower variance = more robust geometry.

**Reduces to**: On a 1D graph, ORC relates to discrete second derivative
of the degree sequence (Forman curvature).

---

## 7. Forman-Ricci curvature

**From**: epidemiology-boundary-conditions repo
**Existing code**: Yes

Combinatorial curvature on the k-NN graph:

    kappa_F(i,j) = 4 - deg(i) - deg(j) + 3 * |triangles through (i,j)|

Purely combinatorial (no optimal transport solve). O(|E|) vs O(|E| * k^3)
for ORC. Same stability metric as ORC.

**Reduces to**: degree deficit 4 - deg(i) - deg(j) on triangle-free graphs.

---

## 8. Grassmannian geodesic distance

**From**: genetic-perturbation-holonomy repo
**Existing code**: Yes

For regimes r, r': compute top-k principal subspaces U^(r), U^(r')
of X^(r), X^(r') (each in R^(d x k)). The geodesic distance on
Gr(k, d):

    d_Gr(U^(r), U^(r')) = sqrt(sum_i theta_i^2)

where theta_i = arccos(sigma_i(U^(r)^T U^(r'))) are the principal angles.

**Robustness metric**: Mean geodesic distance across all regime pairs.
Small = subspace structure preserved under decoherence.

**Boundary condition (from epi paper)**: Requires n_samples >> d_features / k
for stable subspace estimation. At n < 500, k > 3, detection breaks down
on Gr(3, 34) — similar boundary expected here.

---

## 9. Grassmannian holonomy

**From**: genetic-perturbation-holonomy repo
**Existing code**: Yes

Given a LOOP of conditions (r_1, r_2, ..., r_m, r_1), parallel-transport
a subspace around the loop. The holonomy:

    Hol = U^(r_1)^T (prod_{i=1}^{m} P_i) U^(r_1)

where P_i = orthogonal projection from Gr(k,d) step r_i to r_{i+1}
(via Procrustes alignment of successive subspaces).

Hol = I means the loop is trivial (flat connection).
||Hol - I||_F > 0 means non-trivial curvature in the space of conditions.

**For multi-sensor**: loop over sensor modalities
NV -> photon -> spin -> NV. Non-zero holonomy means the biomarker's
subspace representation is modality-dependent in a path-dependent way.

---

## 10. Sheaf H^1 obstruction

**From**: epidemiology-boundary-conditions repo
**Existing code**: Yes (transportkit)

Build a cellular sheaf F on a graph where:
- Vertices = regimes (or sensors)
- Edges = pairs of regimes that should be comparable
- Stalk F(v) = feature space at vertex v
- Restriction maps rho_{v->e} = linear map from vertex stalk to edge stalk

The 0th coboundary operator delta_0: C^0 -> C^1 maps local sections
(one per vertex) to edge disagreements:

    (delta_0 s)(e = (u,v)) = rho_{v->e} s(v) - rho_{u->e} s(u)

H^1(F) = ker(delta_1) / im(delta_0).

dim H^1 > 0 means there is NO global section consistent with all
local measurements. In our context: the sensors or regimes give
INCONSISTENT readings that cannot be reconciled by any linear
transformation.

**Robustness metric**: dim H^1 as a function of decoherence.
At what decoherence level does inconsistency first appear?

**Reduces to**: Cochran's Q test when stalks are R^1 (scalar features)
and restriction maps are identity. This is proven in the epi paper.

---

## 11. Per-edge sheaf Q

**From**: epidemiology-boundary-conditions repo
**Existing code**: Yes

For each edge e = (u,v) in the sheaf:

    Q_e = || rho_{v->e} s(v) - rho_{u->e} s(u) ||^2

Localizes the H^1 obstruction to specific sensor-regime pairs.
The edges with largest Q_e are where the inconsistency lives.

**Diagnostic**: If Q_{NV-photon} >> Q_{NV-spin}, the NV-photon
comparison is the problematic one.

---

## 12. TransportKit fusion

**From**: transport-wrapper repo
**Existing code**: Yes

Two engines:
- **Engine E** (sheaf): H^1 obstruction score (method 10)
- **Engine C** (Grassmannian): geodesic distance (method 8)

Fusion rule:
    transport_score = Engine_E * Engine_C (multiplicative)

A feature transports only if BOTH the sheaf is consistent AND
the subspace is preserved.

This is the practical tool — methods 8-11 are diagnostic,
method 12 gives the final binary verdict.

---

## 13. CKA (Centered Kernel Alignment)

**From**: genetic-perturbation-holonomy repo
**Existing code**: Yes

    CKA(X^(r), X^(r')) = ||K^(r) K^(r')||_F^2 /
                          (||K^(r) K^(r)||_F * ||K^(r') K^(r')||_F)

where K^(r) = centered kernel matrix of X^(r) (linear kernel:
K = X X^T, or RBF).

CKA in [0, 1]. CKA = 1 means identical representation geometry.

**Robustness**: mean CKA across regime pairs. Higher = more robust.

---

## 14. Procrustes distance

**From**: genetic-perturbation-holonomy repo
**Existing code**: Yes

    d_Proc(X^(r), X^(r')) = min_Q ||X^(r) Q - X^(r')||_F

where Q is orthogonal. Solved by SVD of X^(r)^T X^(r').

Measures the rotation distance between configurations after
optimal alignment. Invariant to global rotation but sensitive to
shape change.

---

## 15. Persistent homology (Vietoris-Rips)

**From**: New (use ripser / giotto-tda)
**Existing code**: ripser package

Build the Vietoris-Rips filtration on X^(r). Compute persistence
diagrams PD_k^(r) for homology dimensions k = 0, 1, 2.

Persistence diagram PD = {(b_i, d_i)} where b_i = birth, d_i = death
of each topological feature.

**Robustness metric**: Bottleneck distance between persistence diagrams:

    d_B(PD^(r), PD^(r')) = inf_gamma sup_p ||p - gamma(p)||_inf

where gamma ranges over bijections between diagrams (augmented
with diagonal projections).

By the stability theorem (Cohen-Steiner, Edelsbrunner, Harer 2007):

    d_B(PD(X), PD(Y)) <= d_H(X, Y)

where d_H = Hausdorff distance. So topological features are
at least as stable as the point cloud geometry.

**What we're looking for**: topological features (loops, voids)
in the readout space that persist across decoherence regimes.
These are decoherence-robust structural biomarkers.

---

## 16. Wasserstein distance on persistence diagrams

**From**: New (giotto-tda)
**Existing code**: giotto-tda package

    d_W^p(PD^(r), PD^(r')) = (inf_gamma sum_p ||p - gamma(p)||_inf^p)^(1/p)

More sensitive than bottleneck to the distribution of feature lifetimes.
Use p=2 (standard).

**Interpretation**: Total shift in topological structure between regimes.

---

## 17. Quantum Fisher information (QFI) flatness

**From**: New
**Existing code**: None (implement from scratch)

For a parametric family of quantum states rho(theta), the QFI:

    F_Q(theta) = 2 sum_{m,n: p_m+p_n>0}
                 (p_m - p_n)^2 / (p_m + p_n) * |<m|d_theta rho|n>|^2

where rho = sum p_m |m><m|.

In our CLASSICAL simulation, we approximate: treat the sensor
readout distribution P_theta(x) as a classical statistical model
indexed by the biological parameter theta (temperature, binding state).
Then QFI reduces to the classical Fisher information:

    F_C(theta) = E[ (d/d_theta log P_theta(x))^2 ]

**Robustness metric**: Flatness of F_C as a function of decoherence:

    QFI_stability = Var_r[ F_C^(r)(theta_diagnostic) ]

A flat QFI means the diagnostic information is preserved regardless
of decoherence level.

**When quantum**: For actual quantum sensor states, QFI >= F_C
(quantum Cramer-Rao bound). The gap QFI - F_C quantifies the
advantage of quantum measurement strategies.

---

## 18. Berry phase (geometric phase)

**From**: New
**Existing code**: None (implement from scratch)

Consider a loop in parameter space: (T2, sigma_surface, temp, device)
returns to its starting point. The quantum state |psi(lambda)>
acquires a geometric phase:

    gamma = i oint <psi| d|psi> d_lambda

In our classical approximation: define |psi^(r)> as the normalized
first principal component of X^(r). Then the Berry phase around a
loop of decoherence conditions:

    gamma = arg det[ <psi^(r_1)|psi^(r_2)> <psi^(r_2)|psi^(r_3)>
                     ... <psi^(r_m)|psi^(r_1)> ]

gamma = 0 means flat parameter space (no anisotropy).
gamma != 0 means the decoherence landscape has directional structure
that a simple scalar measure would miss.

**Relation to holonomy (method 9)**: Berry phase is the Abelian
(scalar) case of holonomy. Method 9 is the non-Abelian (matrix) case.
Berry phase detects anisotropy; holonomy detects subspace rotation.

---

## 19. Spectral gap stability

**From**: New
**Existing code**: None (use scipy.sparse.linalg)

Build the graph Laplacian L = D - A on the k-NN graph of X^(r).
Eigenvalues 0 = lambda_0 <= lambda_1 <= ... <= lambda_n.

The spectral gap = lambda_1 (Fiedler value) measures graph connectivity.

**Robustness metric**: How does the spectral gap change under decoherence?

    gap_stability = |lambda_1^(r) - lambda_1^(r')| / lambda_1^(baseline)

By Weyl's perturbation theorem:
    |lambda_k(L) - lambda_k(L')| <= ||L - L'||_2

So spectral gaps are at least as stable as the Laplacian perturbation.

**What we're looking for**: Decoherence regimes where the spectral gap
collapses (graph becomes disconnected) vs. persists (connectivity robust).

---

## 20. Wasserstein distance (distributions)

**From**: New (POT package, already have for ORC)
**Existing code**: POT (Python Optimal Transport)

    W_p(P^(r), P^(r')) = (inf_{pi in Gamma(P^(r), P^(r'))}
                          E_{(x,y)~pi}[||x-y||^p])^(1/p)

Full-distribution comparison, not just moments.

W_1 captures mean shift. W_2 also captures spread changes.
Use W_2 as the primary metric.

**Advantage over t-test**: captures nonlinear distribution changes
(bimodality, tail behavior) that moment-based tests miss.

---

## 21. Von Neumann entropy stability

**From**: New
**Existing code**: None (implement with numpy)

For a density matrix rho (in quantum case) or its classical analog:

    S(rho) = -Tr(rho log rho)

Classical analog: use the covariance matrix Sigma^(r) of X^(r),
normalized to trace 1:

    Sigma_norm = Sigma / Tr(Sigma)
    S_classical = -sum_i lambda_i log lambda_i

where lambda_i are eigenvalues of Sigma_norm (these are the
variance fractions along each PC).

**Robustness metric**: Entropy stability across regimes.

    S_stability = Var_r[ S_classical(X^(r)) ]

If the effective dimensionality of the readout (measured by entropy)
is stable, the information content is preserved.

---

## 22. Fidelity / diamond norm proxy

**From**: New
**Existing code**: None (implement from scratch)

For two quantum channels E, E' (decoherence maps), the diamond norm:

    ||E - E'||_diamond = max_{rho} ||E(rho) - E'(rho)||_1

Classical proxy: for two regime-specific readout maps (trained linear
classifiers), the maximum performance gap:

    fidelity_proxy = max_theta |AUC^(r)(theta) - AUC^(r')(theta)|

where theta ranges over hyperparameters of a linear classifier.

This gives an upper bound on how different the information content is
between regimes.

---

## 23. Persistent sheaf cohomology

**From**: New (combine methods 10 + 15)
**Existing code**: Partial (need to combine)

Build the sheaf (method 10) but now FILTER it by a decoherence
parameter epsilon. At each epsilon level, compute H^1.

The result: a persistence diagram for H^1 as a function of epsilon.

    PSC = {(epsilon_birth, epsilon_death)} for each cohomology class

Cohomology classes that PERSIST across a wide range of epsilon are
structurally robust inconsistencies (real multi-sensor disagreements).
Classes that die quickly are noise.

**Unique to this paper**: This is the novel method. It combines
sheaf cohomology (which detects multi-sensor inconsistency) with
persistent homology (which separates signal from noise). No existing
paper does this on quantum sensor data.

---

## 24. Chern number (integer topological invariant)

**From**: New
**Existing code**: None

Over a 2D parameter space (e.g., T2 x sigma_surface), define a
state |psi(lambda_1, lambda_2)> at each point. The first Chern number:

    c_1 = (1/2pi) int_M F_{12} d_lambda_1 d_lambda_2

where F_{12} = partial_1 A_2 - partial_2 A_1 is the Berry curvature
(A_i = -i <psi|partial_i psi> is the Berry connection).

c_1 is an INTEGER. It cannot drift continuously. Either the topology
changes (phase transition) or it doesn't.

**Classical computation**: discretize the parameter space into a grid,
compute Berry phases around each plaquette, sum up.

**What this gives us**: A BINARY certificate. If c_1 != 0 for a
biomarker feature, the feature lives in a topologically nontrivial
region of parameter space and CANNOT be continuously deformed away
by small decoherence changes.

If c_1 = 0 for all features at a given decoherence level, no
topological protection exists — features are vulnerable.

---

## Summary: method taxonomy

### By what they measure

**Feature-level robustness** (per-feature scores):
1-5 (bracket norms), 17 (QFI), 22 (fidelity proxy)

**Sample-space geometry** (whole-dataset properties):
6-7 (curvature), 15-16 (persistent homology), 19 (spectral gap),
20 (Wasserstein), 21 (entropy)

**Cross-regime/cross-sensor consistency** (relational):
8-14 (Grassmannian, sheaf, CKA, Procrustes, TransportKit)

**Topological invariants** (discrete certificates):
18 (Berry phase), 23 (persistent sheaf cohomology), 24 (Chern number)

### By computational cost

**O(n) or O(n log n)**: 1-5 (bracket), 7 (Forman), 13 (CKA), 17 (QFI)
**O(n^2)**: 6 (ORC), 14 (Procrustes), 19 (spectral), 20 (Wasserstein), 21 (entropy)
**O(n^2 log n) to O(n^3)**: 8-9 (Grassmannian), 15-16 (persistent homology)
**O(grid^2)**: 18 (Berry), 24 (Chern)
**Depends on sheaf size**: 10-12 (sheaf), 23 (persistent sheaf)

### By existing implementation

**Ready to run** (code exists in Elliot's repos): 1-14
**Need implementation** (use existing packages): 15-16 (ripser/giotto-tda),
19 (scipy), 20 (POT)
**Need implementation from scratch**: 17, 18, 21, 22, 23, 24
