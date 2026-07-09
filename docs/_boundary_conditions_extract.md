# Complete Extract: All Prior Papers → Quantum Sensor Paper

What to import, what to quote, what section numbers map where.

---

## Paper 1: Boundary Conditions (epidemiology-boundary-conditions)

**Title**: "When Does Geometry Help Causal Inference? Boundary Conditions for Sheaf, Curvature, and Subspace Methods in Clinical Epidemiology"
**Status**: Unsubmitted (flagship, v4c, 1191 lines)
**This is the TEMPLATE paper. Same structure, same question, new domain.**

### Section map

| Section | Title | Quantum analog |
|---------|-------|----------------|
| §1 (intro) | Intro + contributions | Same framing: "repackage standard stats or genuine advantage?" |
| §2.1 | Sheaf consistency testing | §Methods: Sheaf on sensor-regime graph |
| §2.2 | Confound detection | §Methods: Sensor calibration drift detection |
| §2.3 | Curvature-based edge validation | §Methods: ORC/Forman on sensor readout graphs |
| §2.4 | Treatment heterogeneity detection | Not directly relevant |
| §3.1 | Condition 1: Scalar vs subspace | §Results: Same boundary on quantum sensor data |
| §3.2 | Condition 2: Global vs edge-specific | §Results: Per-sensor vs global decoherence |
| §3.3 | Condition 3: Method failures | §Results: Negative results on quantum data |
| §3.4 | Confound detection results | §Results: Calibration drift detection |
| §4 | Cross-domain (AD) | Already done (quantum = third domain) |
| §5 | Real-data validation | §Real-data: published NV-diamond datasets |
| §6 | Discussion | Cite boundary conditions paper directly |
| App A | Simulation parameters | Quantum sim params |
| App B | Sheaf Q = Cochran's Q proof | Cite directly, same proof holds |

### Key equations to reuse

**Eq 1 — Sheaf Q test (scalar stalks)**:
```
Q = o^T Σ^{-1} o, where o = D_0 β̂, Σ = D_0 diag(se²) D_0^T
```
Quote: "the sheaf Laplacian test statistic is [Eq 1], where D_0 is the signed incidence matrix and Q ~ χ²_{rank(D_0)} under the null."

**For quantum paper**: Same formula. Vertices = decoherence regimes or sensor modalities. Stalks = biomarker estimates per regime. Restriction maps = identity (same biomarker measured differently).

**Eq 2 — Berry phase holonomy norm**:
```
||Φ - I_k||_F = 2√2 |sin(π sin²r)|
```
Quote: "For the linked-column construction (two columns of V_0 rotating toward shared perpendicular directions with π/2 phase offset at radius r), the holonomy norm is [Eq 2]."

**For quantum paper**: Use this as the PLANTED signal in the multi-sensor simulation. The measured holonomy (1.851) matched the prediction (1.869) to within 1%. Same construction for quantum sensor modality loops.

**Eq in Appendix — Sheaf Q = Cochran's Q proof**:
```
Q_S = β̂^T D_0^T (D_0 W^{-1} D_0^T)^{-1} D_0 β̂
    = β̂^T (W - W 1(1^T W 1)^{-1} 1^T W) β̂
    = Σ_k w_k (β̂_k - β̄)² = Q_C
```
Quote: "On scalar stalks, the sheaf Laplacian test statistic is exactly Cochran's Q — not approximately, but algebraically via the W-weighted Schur complement."

**For quantum paper**: Cite this directly. Same reduction applies to quantum sensor data when features are scalar. This is a sanity check: if our sheaf test on scalar NV-diamond features doesn't match Cochran's Q, something is wrong.

### Key numerical results to cite

| Result | Value | Where | Quantum relevance |
|--------|-------|-------|-------------------|
| Sheaf Q = Cochran's Q | Numerically identical at every replicate | §3.1 | Sanity check |
| Berry phase detected | 1.851 (predicted 1.869), p < 0.001 | §3.1, Table 1 | Same construction for sensor loops |
| Max pairwise distance below threshold | 0.730 < 0.767 | Table 1 | Pairwise tests miss it; holonomy catches it |
| Detection boundary | 80% power at ≥500 subjects/site on Gr(3,34) | Table 2, §3.1 | Predict: ≥N samples/regime needed for quantum |
| ABIDE confirms boundary | p=0.16 with n<175/site | §3.1 | Confirms boundary holds on real data |
| Per-edge Q detects DAG | Q>1500, p<10^{-300} for mech edges | Table 3 | Per-sensor Q will detect inconsistent sensors |
| Global test misses | p=0.659 (non-significant) | §3.2 | Don't use global tests on mixed sensors |
| H1 classifier | 85.2% accuracy, 0 false positives, 61 pairs | Table 6 | Template for quantum biomarker classifier |
| ORC below chance | AUROC=0.466 | Table 4 | Predict ORC may fail on quantum sensor graphs too |
| Forman-Ricci = degree deficit | AUROC=0.677 | Table 4 | Don't overclaim curvature; it's degree structure |
| PCA destroys CATE | ARI=-0.011 to -0.016 | Table 5 | Dimensionality reduction may destroy quantum signal too |
| Signal accumulates as √m | √m ≈ 5 for m=24 steps | §3.1 | Same accumulation in sensor modality loops |

### Three boundary conditions (exact formulations)

**Condition 1: Subspace-valued data**
- Grassmannian holonomy detects global inconsistency invisible to pairwise tests
- Composed parallel transport accumulates signal as √m over m loop steps
- On scalar data, sheaf Q reduces algebraically to Cochran's Q
- Quote: "The transition from scalar to subspace-valued data is the primary determinant of whether geometric structure carries information."

**Condition 2: Edge-specific heterogeneity**
- Per-edge sheaf Q recovers planted DAG structure that global tests miss
- H1 effect-modifier classifies pairs with a three-order-of-magnitude gap
- Quote: "When a DAG contains both heterogeneous and homogeneous edges, global tests dilute the signal."

**Condition 3: Method-specific failures**
- ORC below chance (0.466); Forman-Ricci = degree-deficit (0.677)
- PCA destroys treatment effect signal regardless of CATE estimator
- Quote: "Absent the boundary conditions, geometric methods reduce to standard tools or fail outright."

### Reusable code

**File**: `experiments/ms_heterogeneity/experiments.py`

| Function | What it does | Quantum use |
|----------|-------------|-------------|
| `_principal_angles(U1, U2)` | Principal angles between subspaces | Method 8 |
| `_geodesic_distance(U1, U2)` | Grassmannian geodesic distance | Method 8 |
| `_transport_matrix(U_from, U_to)` | Parallel transport via SVD | Methods 8-9 |
| `_cocycle_holonomy(subspaces)` | Composed holonomy around loop | Method 9 |
| `_generate_cocycle_sections(V0, r, m, noise_level, rng)` | Berry phase construction | Simulation oracle |
| `_sheaf_obstruction_subspace(local_sections, adjacency)` | Sheaf H^1 on subspace-valued data | Method 10 |
| `_sheaf_q_test(estimates)` | Sheaf Q test on scalar estimates | Method 11 |
| `run_cocycle_obstruction(seed)` | Full cocycle experiment | Template for quantum loop experiments |
| `run_sheaf_dag_adjudication(seed)` | Per-edge DAG test | Template for per-sensor tests |
| `run_h1_effect_modifier_suite(seed)` | H1 classification | Template for quantum biomarker classification |

**File**: `experiments/batch3_expansion/02_enigma_holonomy/pipeline.py`

| Function | What it does | Quantum use |
|----------|-------------|-------------|
| `principal_angles()` | Same as above (cleaner API) | Method 8 |
| `geodesic_distance()` | Same | Method 8 |
| `transport_matrix()` | Same | Method 9 |
| `compose_holonomy(subspaces)` | Returns (holonomy_matrix, norm) | Method 9 |
| `holonomy_permutation_test()` | Permutation null for holonomy | Statistical testing |
| `berry_phase_boundary_analysis()` | Full boundary analysis with power | Detection boundary for quantum |
| `pca_subspace(X, k)` | PCA subspace extraction | Preprocessing |
| `generate_enigma_simulation()` | Generate simulated multi-site data | Template for quantum simulation |
| `generate_berry_phase_simulation()` | Generate Berry phase data | Template for quantum Berry phase |
| `build_cycles()` | Build loops from sites | Build sensor modality loops |

**File**: `experiments/batch3_expansion/01_systematic_mr/pipeline.py`

| Function | What it does | Quantum use |
|----------|-------------|-------------|
| `compute_cochran_q(betas, ses)` | Cochran's Q from estimates | Sanity check (should match sheaf Q) |
| `compute_power(Q, df, alpha)` | Statistical power from noncentral χ² | Power analysis for quantum experiments |
| `classify_pair(p_value, alpha)` | Binary classification | Classify biomarker transportability |
| `bootstrap_ci(results, n_boot)` | Bootstrap confidence interval | CI for quantum results |

**File**: `experiments/batch3_expansion/03_bnlearn_curvature/pipeline.py`

| Function | What it does | Quantum use |
|----------|-------------|-------------|
| `forman_ricci_curvature(G, u, v)` | Forman-Ricci on single edge | Method 7 |
| `augmented_forman_curvature(G, u, v)` | Forman + triangles | Method 7 variant |
| `compute_edge_features()` | All 7 edge features | Methods 6-7 |
| `learn_dag()` | PC algorithm DAG learning | Not directly needed |
| `compute_aurocs()` | AUROC for edge feature discrimination | Evaluate methods |

**File**: `experiments/clinical_epi/experiments.py`

| Function | What it does | Quantum use |
|----------|-------------|-------------|
| `_sheaf_test(estimates)` | Scalar sheaf test | Method 10 (scalar case) |
| `_cochran_q(estimates)` | Cochran's Q directly | Sanity check |
| `_ollivier_ricci_graph(G, alpha)` | ORC computation | Method 6 |
| `run_sheaf_federated_ehr(seed)` | Federated consistency experiment | Template for multi-device consistency |

---

## Paper 2: Bracket Norm (bracket-norm)

**Title**: "Bracket Norm Identifies Causally Important Brain Regions From Population Geometry"
**Venue**: eLife (submitted)

### What to cite

Quote: "All 19 geometric metrics we test predict optogenetic silencing importance only because they track the number of recorded neurons. After controlling for recording yield, all 19 collapse."

**For quantum paper**: Same risk. Quantum sensor features may predict disease only because they track ensemble size (number of NV centers). MUST correct for n_sensors_per_sample. BN/√n correction is exactly what we need.

Quote: "The corrected quantity BN/√n is invariant to population size (CV 2.0% across a 25× range)"

**For quantum paper**: Test that our BN/√(n_sensors) is invariant to ensemble size.

### Key sections

| Section | Quantum relevance |
|---------|-------------------|
| Fig 1: 19 metrics vs silencing | Template for "24 methods vs ground truth" figure |
| §Results: all 19 track n | Cautionary tale — must correct for ensemble size |
| §Results: BN/√n correction | Method 2 formulation |
| §Results: photoinhibition increases BN | Causal perturbation validation — decoherence is our "perturbation" |

### Reusable code

**File**: `geometry/distances.py` (IDENTICAL copy in drug-perturbation-geometry)

| Function | Quantum use |
|----------|-------------|
| `principal_angles(U, V)` | Method 8 |
| `grassmannian_distance(U, V)` | Method 8 |
| `subspace_overlap(U, V)` | Method 8 variant |
| `cka(X, Y)` | Method 13 |
| `debiased_cka(X, Y)` | Method 13 (better) |
| `chordal_distance(U, V)` | Method 8 variant |
| `all_subspace_distances(U, V)` | Compute all distances at once |

**File**: `geometry/subspace.py`

| Function | Quantum use |
|----------|-------------|
| `fit_pca_subspace(X, k)` | Preprocessing for Grassmannian methods |
| `fit_lda_subspace(X, y, k)` | Supervised subspace extraction |
| `fit_das_subspace(X, y, k)` | Distributed Alignment Search |

---

## Paper 3: Drug Perturbation Geometry (drug-perturbation-geometry)

**Title**: "Direction Instability Predicts Cross-Cell-Line Drug Mechanism Transport in LINCS L1000"
**Venue**: PLOS Comp Bio (submitted)

### What to cite

Quote: Direction instability "quantifies how much a drug's perturbation signature rotates (vs. merely rescales) across cell lines."

**For quantum paper**: Same metric. Replace "cell lines" with "decoherence regimes." A biomarker feature whose direction rotates under decoherence is unstable.

### Key sections

| Section | Quantum relevance |
|---------|-------------------|
| Core question: "Can geometry predict which mechanisms transport?" | EXACTLY our question for quantum biomarkers |
| §Methods: direction_stability | Method 1 (bracket norm raw) |
| §Methods: frechet_variance | Method 3 (transport-stable bracket) |
| §Methods: subspace_transport | Methods 8-9 (Grassmannian) |
| §Results: invariance analysis | Template for decoherence invariance |
| §Results: core defenses | Template for our robustness defenses |

### Reusable code

**File**: `geometry/drug_transport.py`

| Function | Quantum use |
|----------|-------------|
| `direction_stability(signatures)` | Method 1: bracket norm across regimes |
| `magnitude_stability(signatures)` | Scalar baseline comparison |
| `frechet_variance(signatures)` | Method 3: transport-stable bracket |
| `subspace_transport(X_source, X_target, k)` | Method 8: Grassmannian transport |
| `gene_level_consistency(signatures, top_k)` | Feature-level consistency (per-biomarker) |
| `compute_all_transport_metrics(signatures)` | Run everything at once |

**File**: `geometry/bracket_norm.py`

| Function | Quantum use |
|----------|-------------|
| `compute_bracket_norm(activity, choice_binary, evidence)` | Methods 1-2: bracket norm |
| `aggregate_region_metrics(region_metrics)` | Aggregate across sensor regions |

---

## Paper 4: Psychiatric Comorbidity Curvature (psychiatric-comorbidity-curvature)

**Title**: "Edge-Level Discrete Curvature Identifies Structurally Anomalous Connections in a Psychiatric Mechanism Network"
**Venue**: GigaScience (submitted)

### What to cite

Quote: "Within-cluster edges consistently have higher curvature than cross-cluster edges (4/6 data-driven significant, effect sizes d = 0.46-1.55)."

**For quantum paper**: Construct k-NN graph on sensor readouts. Within-class (tumor/healthy) edges should have higher ORC than cross-class edges. Test whether this holds under decoherence.

Quote: "ORC is the only edge-level metric that discriminates epistemically disconfirmed claims from higher-verdict claims (d = 0.73, p = 0.011)."

**For quantum paper**: ORC may discriminate robust from fragile biomarker features on the readout graph.

### Reusable code

**File**: `experiments/curvature_core.py`

| Function | Quantum use |
|----------|-------------|
| `forman_ricci_curvature(G)` | Method 7: all edges at once |
| `curvature_profile(curvatures)` | Summary statistics of curvature distribution |
| `find_core_subgraph(G, curvatures)` | Extract high-curvature subgraph |

**File**: `experiments/data_driven/weighted_orc.py`

| Function | Quantum use |
|----------|-------------|
| `ollivier_ricci_curvature(G, alpha)` | Method 6: ORC on full graph |
| `degree_preserving_rewire(G)` | Null model for curvature |
| `run_edge_null(G, n_perms)` | Permutation test for ORC significance |
| `compute_all_edge_features(G)` | All edge features (ORC, Forman, Jaccard, betweenness, clustering) |
| `method_comparison(G, curvatures, claims, verdicts)` | Compare multiple edge metrics |

---

## Paper 5: Cohort Transfer (biomedical-cohort-transfer)

**Title**: "Grassmannian geodesic distance predicts cross-cohort classifier degradation under analytical heterogeneity"
**Venue**: PLOS ONE (under review)

### What to cite

Quote: "After partialing out source internal AUC, geodesic distance becomes a strong predictor on datasets with genuine analytical heterogeneity."

**For quantum paper**: After partialing out sensor quality, geodesic distance should predict cross-regime classifier degradation.

**Critical negative result**: The method returns NULL when cohorts share standardized platforms (TCGA, SPIROMICS, Affymetrix).

**For quantum paper**: PREDICT that if quantum sensors are properly calibrated (same device, same protocol), Grassmannian distance will return null. The method only adds value when there IS heterogeneity. This is a testable boundary condition.

### Results table to reference

| Dataset | Partial ρ | Note |
|---------|-----------|------|
| CRC microbiome (heterogeneous) | +0.61 | Works when platforms differ |
| QMDiab metabolomics (heterogeneous) | +0.40 | Works across biofluids |
| TCGA-BRCA (centralized) | -0.12 | Null — shared platform |
| SPIROMICS (centralized) | +0.08 | Null — centralized |
| Affymetrix (shared chip) | -0.004 | Null — same chip across sites |

**Quantum prediction**: heterogeneous decoherence → geodesic distance works. Uniform decoherence → returns null (which is correct — no heterogeneity to detect).

---

## Paper 6: MS/MS Subspace Collapse (msms-subspace-collapse)

**Title**: "MS/MS Benchmark Embeddings Collapse Across Instruments"
**Venue**: GigaByte (submitted)

### What to cite

Quote: "Binary fingerprint embeddings produce PCA subspaces that are near-orthogonal across all MS/MS instrument types. Geodesic distance cannot predict cross-instrument performance degradation because every instrument pair lands between 85% and 97% of maximum distance — a dynamic range of 11.8%."

**For quantum paper**: This is a WARNING. If quantum sensor readout PCA subspaces also collapse to near-orthogonal, Grassmannian distance will saturate and be uninformative. Must check dynamic range of subspace distances across decoherence regimes.

Preregistered result: ρ = -0.211, 95% CI [-0.523, +0.145] — null.

---

## Paper 7: Cross-Design Evidence Discordance (cross-design-evidence-discordance)

**Title**: "Cross-Design Evidence Discordance Diagnoses Phase III Drug Failure Modes Across Neurodegenerative and Cardiometabolic Disease"
**Venue**: BMC Medical Research Methodology (submitted)

### Quantum relevance

Different study designs (RCT, observational, MR) disagree on the same causal question. Parallel: different sensor modalities disagree on the same biological signal. The multi-sensor sheaf H^1 test is structurally identical to cross-design discordance testing.

---

## Paper 8: Ecological Bias COVID (ecological-bias-covid)

**Title**: "Ecological bias distorts effect sizes in federated COVID-19 analysis"
**Venue**: BMC Medical Research Methodology (submitted)

### Quantum relevance

Site-level averages hide within-site heterogeneity. Parallel: per-device averages may hide within-device NV-center heterogeneity. If we average across 20 NV centers per sample, we may lose the diagnostic signal that lives in the VARIANCE across NV centers.

**Experiment to add**: Compare per-NV-center analysis vs. device-averaged analysis. Does averaging destroy information?

---

## Paper 9: Direction Instability Atlas (direction-instability-atlas)

**Title**: "A Pre-Registered Direction Instability Atlas Across Three Perturbation Modalities"
**Venue**: GigaScience (submitted)

### Quantum relevance

The atlas tests direction instability across drug, genetic, and disease perturbations. The quantum paper adds a FOURTH modality: decoherence perturbation. Same metric (direction instability), new perturbation type.

---

## Paper 10: Bracket Norm Validity (bracket-norm-validity, unsubmitted)

**Title**: "Bracket Norm Is Not Mechanism: Validity Conditions for Noncommutative Perturbation Scores"

### Quantum relevance

Tests five failure modes of bracket norm:
1. Toxicity-corrected BN → decoherence-corrected BN (Method 3)
2. Essential-gene correction → essential-sensor correction
3. Cell-health correction → tissue-health correction
4. Phenotype-projected BN → diagnostic-projected BN (Method 4)
5. Localized BN → sensor-specific BN (Method 5)

All five bracket norm variants in our quantum method list come from this paper.

---

## Summary: Code import plan

### Core geometry (shared across repos — use ONE copy)

From `bracket-norm/geometry/distances.py` (or drug-perturbation-geometry, identical):
- `principal_angles`, `grassmannian_distance`, `subspace_overlap`
- `cka`, `debiased_cka`
- `chordal_distance`, `all_subspace_distances`

From `bracket-norm/geometry/subspace.py`:
- `fit_pca_subspace`, `fit_lda_subspace`

### Bracket norm family (Methods 1-5)

From `drug-perturbation-geometry/geometry/drug_transport.py`:
- `direction_stability` → Method 1
- `frechet_variance` → Method 3
- `subspace_transport` → Method 8

From `bracket-norm/geometry/bracket_norm.py`:
- `compute_bracket_norm` → Methods 1-2

### Curvature (Methods 6-7)

From `psychiatric-comorbidity-curvature/experiments/data_driven/weighted_orc.py`:
- `ollivier_ricci_curvature` → Method 6
- `compute_all_edge_features` → Methods 6-7 + baselines

From `psychiatric-comorbidity-curvature/experiments/curvature_core.py`:
- `forman_ricci_curvature` → Method 7

### Grassmannian + holonomy (Methods 8-9)

From `epidemiology-boundary-conditions/experiments/batch3_expansion/02_enigma_holonomy/pipeline.py`:
- `compose_holonomy` → Method 9
- `holonomy_permutation_test` → Statistical testing
- `berry_phase_boundary_analysis` → Detection boundary

### Sheaf (Methods 10-12)

From `epidemiology-boundary-conditions/experiments/ms_heterogeneity/experiments.py`:
- `_sheaf_q_test` → Method 11
- `_sheaf_obstruction_subspace` → Method 10

From `epidemiology-boundary-conditions/experiments/clinical_epi/experiments.py`:
- `_sheaf_test` → Method 10 (scalar)
- `_cochran_q` → Sanity check

From `epidemiology-boundary-conditions/experiments/batch3_expansion/01_systematic_mr/pipeline.py`:
- `compute_cochran_q`, `compute_power`, `bootstrap_ci` → Analysis pipeline

### CKA + Procrustes (Methods 13-14)

From `bracket-norm/geometry/distances.py`:
- `cka`, `debiased_cka` → Method 13
- (Procrustes is scipy.spatial.procrustes — no custom code needed) → Method 14

### New methods (15-24)

Need implementation from scratch or packages:
- 15-16: `ripser` or `giotto-tda` (persistent homology)
- 17: Implement Fisher information from scratch
- 18: Implement Berry phase from scratch (use holonomy code as template)
- 19: `scipy.sparse.linalg.eigsh` (spectral gap)
- 20: `pot` (Wasserstein, already used for ORC)
- 21: Implement entropy from scratch (eigenvalues of normalized covariance)
- 22: Implement fidelity proxy from scratch
- 23: Combine methods 10 + 15 (persistent sheaf cohomology)
- 24: Implement Chern number from scratch (discretized Berry curvature sum)
