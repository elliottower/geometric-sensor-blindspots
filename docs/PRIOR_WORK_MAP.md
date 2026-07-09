# Master Map: All Prior Papers → Quantum Sensor Paper

Everything from Elliot Tower's 9 submitted papers + 3 unsubmitted papers
that ports to the quantum sensing paper. Section numbers, equations, results,
quotes, importable code, and predicted quantum analogs.

---

## Table of Contents

1. [Boundary Conditions (flagship)](#1-boundary-conditions-flagship)
2. [Bracket Norm (eLife)](#2-bracket-norm-elife)
3. [Drug Perturbation Geometry (PLOS Comp Bio)](#3-drug-perturbation-geometry-plos-comp-bio)
4. [Bracket Norm Validity (unsubmitted)](#4-bracket-norm-validity-unsubmitted)
5. [Psychiatric Comorbidity Curvature (GigaScience)](#5-psychiatric-comorbidity-curvature-gigascience)
6. [Cohort Transfer (PLOS ONE)](#6-cohort-transfer-plos-one)
7. [MS/MS Subspace Collapse (GigaByte)](#7-msms-subspace-collapse-gigabyte)
8. [Cross-Design Evidence Discordance (BMC MRM)](#8-cross-design-evidence-discordance-bmc-mrm)
9. [Ecological Bias COVID (BMC MRM)](#9-ecological-bias-covid-bmc-mrm)
10. [Direction Instability Atlas (GigaScience)](#10-direction-instability-atlas-gigascience)
11. [TransportKit (transport-wrapper)](#11-transportkit-transport-wrapper)
12. [Mechanistic Validity + Reference](#12-mechanistic-validity--reference)
13. [Master Code Import Table](#13-master-code-import-table)
14. [Master Quote Bank](#14-master-quote-bank)
15. [Predicted Negative Results](#15-predicted-negative-results)

---

## 1. Boundary Conditions (flagship)

**Title**: "When Does Geometry Help Causal Inference? Boundary Conditions for
Sheaf, Curvature, and Subspace Methods in Clinical Epidemiology"
**Repo**: `epidemiology-boundary-conditions`
**Paper**: `paper/paper_v4c.tex`
**Status**: Unsubmitted (v4c, 1191 lines)

**THIS IS THE TEMPLATE PAPER. The quantum paper is its sequel.**

### Section map → quantum paper sections

| Boundary §  | Title | Quantum paper analog |
|---|---|---|
| §1 | Introduction + contributions | Intro: "repackage standard stats in quantum language, or genuine advantage?" |
| §2.1 | Sheaf consistency testing | Methods: sheaf on sensor-regime graph |
| §2.2 | Confound detection | Methods: sensor calibration drift detection |
| §2.3 | Curvature-based edge validation | Methods: ORC/Forman on sensor readout graphs |
| §2.4 | Treatment heterogeneity detection | (Not directly relevant) |
| §3.1 | Condition 1: scalar vs subspace | Results: same boundary on quantum sensor data |
| §3.2 | Condition 2: global vs edge-specific | Results: per-sensor vs global decoherence |
| §3.3 | Condition 3: method failures | Results: negative results on quantum data |
| §5 | Real-data validation | Real-data: published NV-diamond datasets |
| App B | Sheaf Q = Cochran's Q proof | Cite directly. Same proof holds for quantum stalks. |

### Key equations (reuse verbatim)

**Eq 1 — Sheaf Q test (scalar stalks)**:
```latex
Q = \mathbf{o}^\top \Sigma^{-1} \mathbf{o}, \quad
\mathbf{o} = D_0 \boldsymbol{\hat\beta}, \quad
\Sigma = D_0 \operatorname{diag}(\mathrm{se}^2) D_0^\top
```
Q ~ chi²_{rank(D_0)} under the null.

**Quantum use**: Vertices = decoherence regimes or sensor modalities.
Stalks = biomarker estimates per regime. Same formula, new domain.

**Eq 2 — Berry phase holonomy norm** (linked-column construction):
```latex
\|\Phi - I_k\|_F = 2\sqrt{2}\,|\sin(\pi \sin^2 r)|
```

**Quantum use**: Plant this signal in the multi-sensor simulation.
The measured holonomy (1.851) matched the prediction (1.869) to
within 1%. Same construction for quantum sensor modality loops.

**Appendix B — Sheaf Q = Cochran's Q algebraic proof**:
```latex
Q_S = \hat\beta^\top D_0^\top (D_0 W^{-1} D_0^\top)^{-1} D_0 \hat\beta
    = \hat\beta^\top (W - W \mathbf{1}(\mathbf{1}^\top W \mathbf{1})^{-1} \mathbf{1}^\top W) \hat\beta
    = \sum_k w_k (\hat\beta_k - \bar\beta)^2 = Q_C
```

**Quantum use**: Sanity check. On scalar NV-diamond features, our
sheaf test MUST match Cochran's Q. If it doesn't, implementation bug.

### Three boundary conditions (exact formulation)

**Condition 1: Subspace-valued data**
> "The transition from scalar to subspace-valued data is the primary
> determinant of whether geometric structure carries information."

- Grassmannian holonomy detects global inconsistency invisible to pairwise tests
- Signal accumulates as √m over m loop steps; noise as random walk
- On scalar data, sheaf Q = Cochran's Q (no advantage)

**Condition 2: Edge-specific heterogeneity**
> "When a DAG contains both heterogeneous and homogeneous edges,
> global tests dilute the signal."

- Per-edge sheaf Q recovers planted DAG structure (p < 10⁻³⁰⁰)
- Global test misses (p = 0.659)

**Condition 3: Method-specific failures**
> "Absent the boundary conditions, geometric methods reduce to
> standard tools or fail outright."

- ORC below chance (AUROC = 0.466)
- Forman-Ricci = degree-deficit feature (AUROC = 0.677)
- PCA destroys treatment effect signal (ARI = -0.011 to -0.016)

### Key numerical results

| Result | Value | Section |
|---|---|---|
| Berry phase detected | 1.851 (predicted 1.869), p < 0.001 | §3.1 |
| Pairwise distance below threshold | 0.730 < 0.767 | §3.1 |
| Detection boundary | ≥500 subjects/site on Gr(3,34) | §3.1 |
| ABIDE confirms boundary | p = 0.16, n < 175/site | §3.1 |
| Per-edge Q | Q > 1500, p < 10⁻³⁰⁰ | §3.2 |
| Global test misses | p = 0.659 | §3.2 |
| H1 classifier | 85.2% accuracy, 0 FP, 61 pairs | §5 |
| ORC below chance | AUROC = 0.466 | §3.3 |
| Forman = degree deficit | AUROC = 0.677 | §3.3 |
| Signal accumulates as √m | √24 ≈ 5 for m=24 steps | §3.1 |

### Importable code

**`experiments/batch3_expansion/02_enigma_holonomy/pipeline.py`**:

| Function | Quantum method |
|---|---|
| `principal_angles()` | Method 8 |
| `geodesic_distance()` | Method 8 |
| `transport_matrix()` | Method 9 |
| `compose_holonomy(subspaces)` | Method 9 |
| `holonomy_permutation_test()` | Statistical testing |
| `berry_phase_boundary_analysis()` | Detection boundary |
| `build_cycles()` | Build sensor modality loops |

**`experiments/ms_heterogeneity/experiments.py`**:

| Function | Quantum method |
|---|---|
| `_sheaf_q_test(estimates)` | Method 11 |
| `_sheaf_obstruction_subspace(sections, adj)` | Method 10 |
| `_cocycle_holonomy(subspaces)` | Method 9 |
| `_generate_cocycle_sections(V0, r, m, noise, rng)` | Simulation oracle |

**`experiments/batch3_expansion/01_systematic_mr/pipeline.py`**:

| Function | Quantum method |
|---|---|
| `compute_cochran_q(betas, ses)` | Sanity check |
| `compute_power(Q, df, alpha)` | Power analysis |
| `bootstrap_ci(results, n_boot)` | Confidence intervals |

**`experiments/batch3_expansion/03_bnlearn_curvature/pipeline.py`**:

| Function | Quantum method |
|---|---|
| `forman_ricci_curvature(G, u, v)` | Method 7 |
| `augmented_forman_curvature(G, u, v)` | Method 7 variant |
| `compute_edge_features()` | Methods 6-7 + baselines |

---

## 2. Bracket Norm (eLife)

**Title**: "Bracket Norm Tracks Causally Important Brain Regions From
Population Geometry"
**Repo**: `bracket-norm`
**Paper**: `paper/paper_v20.tex`

### Section map

| Section | Title | Quantum relevance |
|---|---|---|
| §2.1 | All 19 metrics confounded by neuron count | **CAUTIONARY TALE**: quantum features may track ensemble size |
| §2.2 | BN/√n predicts silencing importance | **Method 2**: BN/√(n_sensors) |
| §2.3 | Photoinhibition increases BN | **VALIDATION**: decoherence = perturbation that increases BN |
| §2.4 | Human ECoG cross-species | Cross-modality generalization precedent |
| §2.5 | Task-specific rankings | Modality-specific rankings expected |
| §4.3 | Bracket norm definition | Formula for Methods section |
| §4.4 | Dimensional correction | √n correction formula |

### Key equations

**BN definition** (§4.3):
```
BN = || ξ(high evidence) - ξ(low evidence) ||
```
where ξ = mean(x_right) - mean(x_left) within evidence quartiles.

**Dimensional correction** (§4.4):
```
BN/√n invariant: mean 0.394 ± 0.008, CV 2.0%, ρ with neuron count = +0.085
```

### Key results

| Result | Value | Quantum prediction |
|---|---|---|
| All 19 metrics confounded | ρ > 0.8 with neuron count | Features may correlate with n_sensors |
| BN partial ρ survives | +0.753 after controlling n | BN should survive after controlling sensor count |
| BN/√n predicts silencing | ρ = 0.833, p = 0.007 | BN/√n should predict robustness |
| Top-3/bottom-3 match | p = 0.0006 | Top/bottom features should match ground truth |
| ALM photoinhibition +68% | 32/32 sessions, p < 10⁻⁶ | Decoherence increases feature instability |
| ECoG confound eliminated | ρ = 0.0 after √n | √n_sensor correction eliminates confound |
| Task-specific rankings | cross-task ρ = -0.079 | Modality-specific rankings |

### Importable code

**`geometry/distances.py`** (identical copy in drug-perturbation-geometry):

| Function | Quantum method |
|---|---|
| `principal_angles(U, V)` | Method 8 |
| `grassmannian_distance(U, V)` | Method 8 |
| `subspace_overlap(U, V)` | Method 8 variant |
| `cka(X1, X2)` | Method 13 |
| `debiased_cka(X1, X2)` | Method 13 (better) |
| `chordal_distance(U, V)` | Method 8 variant |
| `all_subspace_distances(U, V)` | Compute all at once |

**`geometry/subspace.py`**:

| Function | Use |
|---|---|
| `fit_pca_subspace(activity, labels, k)` | Preprocessing for Grassmannian methods |
| `fit_lda_subspace(activity, labels, k)` | Supervised subspace extraction |

---

## 3. Drug Perturbation Geometry (PLOS Comp Bio)

**Title**: "Direction Instability: A Magnitude-Invariant Metric for
Cross-Cell-Line Drug Mechanism Transport"
**Repo**: `drug-perturbation-geometry`
**Paper**: `paper/drug_transport_plos_v3.tex`

### Section map

| Section | Title | Quantum relevance |
|---|---|---|
| §2.2 | Direction instability definition | **Method 1** formula |
| §2.3 | Magnitude correction | Must correct for signal strength |
| §3.1 | DI tracks mechanism conservation | Template for decoherence invariance |
| §3.2 | Target breadth determines transport | Broadband vs narrow features |
| §3.3 | Predictive validation (LOO) | LOO prediction template |

### Key equation

**Direction instability** (§2.2, Eq 1):
```
D = 1 - (2 / K(K-1)) Σ_{i<j} ŝᵢᵀ ŝⱼ
```
D ∈ [0, 2]. D=0 collinear, D=1 orthogonal, D>1 anticorrelated.

### Key results

| Result | Value | Quantum prediction |
|---|---|---|
| DI anticorrelates with Jaccard | ρ = -0.79, p < 10⁻³⁰⁰ | DI anticorrelates with feature overlap across regimes |
| Pan-HDAC vs selective | DI 0.62 vs 0.96 | Broadband features low DI, narrow features high DI |
| LOO AUROC | 0.986 (null 0.500 ± 0.012) | LOO prediction of robustness |
| Magnitude correction needed | DI correlates with norm at ρ = -0.41 | Must correct for signal strength |
| Cytotoxicity defense | removing stress genes preserves gradient (ρ = 0.83) | Removing decoherence features preserves diagnostic gradient |

### Importable code

**`geometry/drug_transport.py`**:

| Function | Quantum method |
|---|---|
| `direction_stability(signatures: dict)` | Method 1 (dict API: {regime: vector}) |
| `magnitude_stability(signatures: dict)` | Scalar baseline |
| `frechet_variance(signatures)` | Method 3 component |
| `subspace_transport(X_source, X_target, k)` | Method 8 |
| `compute_all_transport_metrics(signatures)` | Everything at once |

---

## 4. Bracket Norm Validity (unsubmitted)

**Title**: "Bracket Norm Is Not Mechanism: Validity Conditions for
Noncommutative Perturbation Scores"
**Repo**: `direction-instability-drug-validity`
**Paper**: `paper/bracket_norm_validity_v8.tex`

### The 7-rung validity ladder (§2.5)

| Rung | Clinical | Quantum analog |
|---|---|---|
| 1. Process view declared | What is a "context"? | What is a "decoherence regime"? |
| 2. Reliable vector field | Adequate SNR | Sensor readout has adequate SNR |
| 3. Perturbation estimable | Effect isolated from background | Bio signal isolated from decoherence |
| 4. Localized to biology | Instability in pathway features | Robust features in diagnostic channels |
| 5. Phenotype-aligned | Matches known target | Stability aligns with diagnostic axis |
| 6. Transports across contexts | Low Fréchet variance | Replicates across decoherence regimes |
| 7. Predicts holdout | External outcome | Predicts clinical outcome |

### The 5 bracket norm variants

All from `geometry/bracket_norm.py`:

| Variant | Function | Quantum method # |
|---|---|---|
| Raw DI | `direction_instability(sigs)` | 1 |
| Phenotype-projected | `phenotype_projected_bracket(sigs, w)` | 4 |
| Toxicity-corrected | `toxicity_corrected_bracket(sigs, stress_genes)` | (decoherence-corrected analog) |
| Transport-stable | `transport_stable_bracket(sigs, penalty)` | 3 |
| Localized | `localization_score(sigs, mask)` | 5 |

### Key results

| Result | Value | Quantum prediction |
|---|---|---|
| Corrections preserve ranking | ρ > 0.99 (LINCS), 0.91 (Perturb-seq), 0.99 (JUMP) | Corrections should preserve feature rankings |
| Phenotype-projected ρ | 0.38, p < 10⁻²⁷ with on-target connectivity | Diagnostic-projected BN should correlate with ground truth |
| Raw bracket misses | ρ = -0.04 | Raw bracket without projection misses diagnostic axis |
| 3 domains tested | LINCS, Perturb-seq, JUMP | Quantum = 4th domain |

### Additional importable code

**`geometry/grassmannian.py`**:

| Function | Use |
|---|---|
| `frechet_mean_subspace(subspaces, n_iter=50)` | Fréchet mean on Gr(k,d) for method 3 |
| `project_out_subspace(U, confound)` | Remove decoherence subspace before measuring geometry |

---

## 5. Psychiatric Comorbidity Curvature (GigaScience)

**Title**: "Edge-Level Discrete Curvature Decomposes Bridge Structure
Across Disease Networks"
**Repo**: `psychiatric-comorbidity-curvature`
**Paper**: `paper/paper_v4.tex`

### Section map

| Section | Title | Quantum relevance |
|---|---|---|
| §2 | The Degree Confound | **WARNING**: ORC r = -0.71 with degree |
| §3 | Edge-Level with Null Models | Must use permutation null |
| §4 | Cross-Domain Replication (6 graphs) | Template for 3-sensor replication |
| §5.3 | Method Benchmarking | Only ORC discriminates (d=0.73) |

### Key equation

**ORC** (§2, Eq 1):
```
κ(u,v) = 1 - W₁(μ_u, μ_v) / d(u,v)
```
Lazy random walk (α=0.5), W₁ via POT.

### Key results

| Result | Value | Quantum prediction |
|---|---|---|
| ORC vs degree confound | r = -0.71 | Must control for degree in sensor graphs |
| Within vs cross (LDSC) | d = 1.44, p = 0.001 | Within-class edges higher curvature |
| Within vs cross (comorbidity) | d = 1.55, p < 0.001 | Same pattern on readout graph |
| ORC uniquely discriminates | d = 0.73, p = 0.011 | Only ORC, not Jaccard/betweenness/spectral |
| Betweenness | d = 0.40, p = 0.294 | Fails |
| Jaccard | d = 0.14, p = 0.641 | Fails |
| Spectral gap | d = 0.00 | Fails completely |

### Importable code

**`experiments/data_driven/weighted_orc.py`**:

| Function | Quantum method |
|---|---|
| `ollivier_ricci_curvature(G, alpha)` | Method 6 |
| `degree_preserving_rewire(G)` | Null model |
| `run_edge_null(G, n_perms)` | Significance testing |
| `compute_all_edge_features(G)` | All 5 edge metrics at once |

**`experiments/curvature_core.py`**:

| Function | Quantum method |
|---|---|
| `forman_ricci_curvature(G)` | Method 7 |

---

## 6. Cohort Transfer (PLOS ONE)

**Title**: "Grassmannian geodesic distance predicts cross-cohort classifier
degradation under analytical heterogeneity"
**Repo**: `biomedical-cohort-transfer`
**Paper**: `paper_v5_plos.tex`

### Section map

| Section | Title | Quantum relevance |
|---|---|---|
| §3 | Methods: geodesic, partial correlation | Method 8 formula |
| §4.1 | Geometric batch detection | All z > 25 → decoherence regimes will be detectable |
| §4.2 | CRC: partial ρ = +0.61 | Heterogeneous sensors → geodesic works |
| §4.6 | Boundary: 5 null results | Centralized platform → geodesic null |

### Key equation

**Grassmannian geodesic** (§1, Eq 1):
```
d_Gr(V₁, V₂) = (Σᵢ θᵢ²)^{1/2}
```
θᵢ = arccos(σᵢ) for σᵢ singular values of U₁ᵀU₂.

### Key results

| Dataset | Partial ρ | Quantum prediction |
|---|---|---|
| CRC microbiome (heterogeneous) | +0.61 | Different decoherence → works |
| QMDiab metabolomics (heterogeneous) | +0.40 | Different biofluids → works |
| SPIROMICS (centralized) | +0.08 (null) | Same calibration → null |
| TCGA (centralized) | -0.12 (null) | Same vendor → null |
| Affymetrix (shared chip) | -0.004 (null) | Same platform → null |

### MOTHER LODE: `src/transportability.py`

This single file implements methods 1, 8, 9, 10, 11:

| Function | Quantum method |
|---|---|
| `principal_angles()` | Method 8 core |
| `geodesic_distance()` | Method 8 |
| `transport_matrix()` | Method 9 core |
| `cocycle_holonomy()` | Method 9 |
| `sheaf_h1_two_cohort()` | Method 10 (2 regimes) |
| `sheaf_h1_multi_cohort()` | Method 10 (N regimes) |
| `sheaf_q_test()` | Method 11 |
| `bracket_norm_score()` | Method 1 |
| `top_k_subspace()` | Preprocessing |
| `centroid_distance()` | Baseline |
| `domain_classifier_auc()` | Baseline |

---

## 7. MS/MS Subspace Collapse (GigaByte)

**Title**: "MS/MS Benchmark Embeddings Collapse Across Instruments"
**Repo**: `msms-subspace-collapse`
**Paper**: `paper/paper.tex`

### What to cite (WARNING result)

| Result | Value |
|---|---|
| Geodesic distance span | 85-97% of theoretical maximum |
| Dynamic range | 11.8% |
| Primary result | ρ = -0.211, CI [-0.523, +0.145] (null) |
| Minimum detectable | ρ = 0.33 at 80% power |

**Quantum prediction**: If quantum readout PCA subspaces collapse to
near-orthogonal (like MS/MS fingerprints), Grassmannian distance will
saturate and be uninformative. MUST check dynamic range. Binary features
(peak presence/absence) have higher collapse risk than continuous features
(fluorescence intensity).

---

## 8. Cross-Design Evidence Discordance (BMC MRM)

**Title**: "Cross-Design Evidence Discordance Diagnoses Phase III Drug
Failure Modes"
**Repo**: `cross-design-evidence-discordance`

### Quantum analog

Different study designs disagree on same question = different sensor
modalities disagree on same biology. Sheaf H¹ is structurally identical
to cross-design discordance.

### Failure mode taxonomy (ports directly)

| Drug failure mode | Quantum analog |
|---|---|
| Zombie mechanism | Decoherence artifact (looks robust, is sensor artifact) |
| Translation gap | Bio-real feature that doesn't transport across devices |
| Exposure mismatch | Sensor measures wrong quantity under decoherence |

### Key insight

> "MR-null status alone accounted for all predictive power... The
> observational leg added no discriminative information."

**Quantum analog**: Under clean conditions, ALL features associate with
disease. The discriminator is which survive decoherence. The clean-condition
association adds nothing.

---

## 9. Ecological Bias COVID (BMC MRM)

**Title**: "Ecological bias distorts effect sizes in federated COVID-19
analysis"
**Repo**: `ecological-bias-covid`

### Key result

| Result | Value |
|---|---|
| Ecological β vs individual OR | +0.55 vs 9.9 (350-fold divergence) |
| I² | 99.8% |
| Consistency test z | 25, p < 0.0001 |

### Quantum analog

Per-device averaged readouts vs per-NV-center readouts may give
qualitatively different conclusions. The ecological regression
coefficient is "a different quantity, not a noisy estimate."

**Experiment to add**: Compare per-NV-center analysis vs device-averaged
analysis. Does averaging destroy diagnostic information?

---

## 10. Direction Instability Atlas (GigaScience)

**Title**: "A Pre-Registered Direction Instability Atlas Across Three
Perturbation Modalities"
**Repo**: `direction-instability-atlas`
**Paper**: `paper/atlas_paper_v2.tex`

### Three modalities → quantum is the 4th

| Atlas modality | N | Quantum analog |
|---|---|---|
| LINCS L1000 (drugs) | 8,949 | NV-diamond thermometry |
| Tahoe single-cell (drugs) | 379 | Entangled-photon microscopy |
| JUMP Cell Painting (CRISPR) | 7,946 + 12,590 | Spin-labeled ESR |

### Key results

| Result | Value | Quantum prediction |
|---|---|---|
| Cross-modal concordance | partial ρ = 0.19, p = 2.1 × 10⁻⁴ | Cross-sensor concordance expected |
| DI predicts transport | AUROC = 0.945 | DI should predict robustness |
| Essential genes lower DI | ρ = -0.33 | "Essential" features (diagnostic-core) lower DI |
| Knockout vs OE uncorrelated | ρ = 0.01 | Different perturbation types → different rankings |

### Pre-registration structure to replicate

Batch 1 SHA `0d07a01`, Batch 2A SHA `a17c125`. Same protocol for quantum.

---

## 11. TransportKit (transport-wrapper)

**Repo**: `transport-wrapper`

### Two-engine architecture

**Engine E** (sheaf H¹): effect transportability across strata.
Validated: 85% accuracy / 0 FP on 61 MR pairs.

**Engine C** (Grassmannian geodesic): classifier transportability.
Validated: partial ρ = 0.61, 22% LOO-MAE gain.

**Fusion rule** (fusion.py):
```
Both valid + agree      → verdict "high" confidence
Both valid + disagree   → "DISAGREE" (diagnostic)
One valid               → single-channel, "medium"
Neither valid           → "UNKNOWN" (honest)
NEVER averages across an invalid channel
```

### Importable code

**`transportkit/engines.py`**:

| Function | Quantum method |
|---|---|
| `effect_transportability(estimates, ses)` | Methods 10-11 |
| `classifier_transportability(X_source, X_target, k)` | Method 8 |
| `pca_subspace(X, k)` | Preprocessing |
| `geodesic_distance(U1, U2)` | Method 8 |
| `permutation_null_z(Xs, k)` | Heterogeneity gate |

**`transportkit/fusion.py`**:

| Function | Quantum method |
|---|---|
| `fuse(effect_res, clf_res)` | Method 12 |
| `transportability_report(...)` | End-to-end verdict |

**Dataclasses**: `EffectResult`, `ClassifierResult`, `FusedVerdict`
— all directly reusable.

### Quantum adaptation

1. Replace "strata" with "decoherence regimes"
2. Replace "cohorts" with "sensor configurations"
3. Add third engine for quantum-specific metrics (Berry phase, QFI)
4. Boundary-condition gate pattern reusable as-is

---

## 12. Mechanistic Validity + Reference

### Validity framework (5 lenses for quantum biomarkers)

| Lens | Quantum question |
|---|---|
| Construct | "Decoherence-robust biomarker" — falsifiable? |
| Internal | Does biomarker causally track disease (not decoherence)? |
| External | Generalizes across devices, tissues, regimes? |
| Measurement | Are robustness metrics themselves reliable? |
| Interpretive | Right level? (single-sensor vs multi-sensor) |

### 5 verdict tiers

| Tier | Name | Quantum meaning |
|---|---|---|
| 1 | Proposed | Feature identified, no robustness test |
| 2 | Causally suggestive | Survives single-sensor decoherence perturbation |
| 3 | Mechanistically supported | Survives + cross-regime + direction preserved |
| 4 | Triangulated | Multiple independent methods converge |
| 5 | Validated | All 5 lenses pass across devices and tissues |

### Transport hierarchy (reference paper)

| Level | Quantum meaning |
|---|---|
| Object | Same features selected across devices |
| Role | Same diagnostic function across regimes |
| Subspace | Same readout subspace (Grassmannian proximity) |
| Structural | Same geometry under calibration freedom |
| Process | Same dynamics under decoherence trajectory |

### 5 failure modes (reference paper)

| Mode | Quantum prediction |
|---|---|
| Evidence misfire | Different methods disagree on which features are robust |
| Claim laundering | Single-sensor robustness used to claim multi-sensor robustness |
| Mimic mechanism | Method always reports "robust" (e.g., CKA on low-d data) |
| Zombie biomarker | Feature called robust but tracks device drift |
| Reference debt | "Decoherence-robust" label applied before cross-device validation |

---

## 13. Master Code Import Table

Priority-ordered. Import ONE copy of each function.

### Tier 1: Use directly (no adaptation needed)

| Method # | Function | Best source file |
|---|---|---|
| 1 | `direction_instability(sigs)` | `direction-instability-drug-validity/geometry/bracket_norm.py` |
| 2 | same / √n_sensors | same |
| 3 | `transport_stable_bracket(sigs, penalty)` | same |
| 4 | `phenotype_projected_bracket(sigs, w)` | same |
| 5 | `localization_score(sigs, mask)` | same |
| 6 | `ollivier_ricci_curvature(G, alpha)` | `psychiatric-comorbidity-curvature/experiments/data_driven/weighted_orc.py` |
| 7 | `forman_ricci_curvature(G)` | `psychiatric-comorbidity-curvature/experiments/curvature_core.py` |
| 8 | `geodesic_distance(U1, U2)` | `genetic-perturbation-holonomy/geometry/grassmannian.py` |
| 9 | `compose_holonomy(subspaces)` | same |
| 10 | `sheaf_h1_multi_cohort(...)` | `biomedical-cohort-transfer/src/transportability.py` |
| 11 | `sheaf_q_test(...)` | same |
| 12 | `fuse(effect_res, clf_res)` | `transport-wrapper/transportkit/fusion.py` |
| 13 | `cka(X, Y)` / `debiased_cka(X, Y)` | `genetic-perturbation-holonomy/geometry/distances.py` |
| 14 | `scipy.spatial.procrustes` | stdlib |

### Tier 2: Use existing packages

| Method # | Package | Function |
|---|---|---|
| 15 | `ripser` or `giotto-tda` | Persistent homology |
| 16 | `giotto-tda` | Wasserstein/bottleneck on persistence diagrams |
| 19 | `scipy.sparse.linalg` | `eigsh` for spectral gap |
| 20 | `pot` (already used for ORC) | `ot.emd2` for Wasserstein |

### Tier 3: Implement from scratch

| Method # | What | Template |
|---|---|---|
| 17 | QFI flatness | Classical Fisher info via score function |
| 18 | Berry phase | Use `compose_holonomy` as starting point (Abelian case) |
| 21 | Von Neumann entropy | `np.linalg.eigvalsh` on normalized covariance |
| 22 | Fidelity proxy | Max AUC gap across classifiers |
| 23 | Persistent sheaf cohomology | Combine methods 10 + 15 |
| 24 | Chern number | Discretized Berry curvature plaquette sum |

### Utility functions (shared)

| Function | Source | Use |
|---|---|---|
| `fit_pca_subspace(X, k)` | `bracket-norm/geometry/subspace.py` | Preprocessing |
| `frechet_mean_subspace(subspaces)` | `direction-instability-drug-validity/geometry/grassmannian.py` | Method 3 |
| `project_out_subspace(U, confound)` | same | Decoherence correction |
| `holonomy_permutation_test(...)` | `genetic-perturbation-holonomy/geometry/grassmannian.py` | Statistical testing |
| `compute_power(Q, df, alpha)` | `epidemiology-boundary-conditions/.../pipeline.py` | Power analysis |
| `bootstrap_ci(results, n_boot)` | same | Confidence intervals |
| `degree_preserving_rewire(G)` | `psychiatric-comorbidity-curvature/.../weighted_orc.py` | Null model |

---

## 14. Master Quote Bank

Quotes from submitted/in-progress papers, reframed for quantum paper introduction.

### The confound warning (bracket-norm, §2.1)
> "All 19 geometric metrics we test predict optogenetic silencing importance
> only because they track the number of recorded neurons. After controlling
> for recording yield, all 19 collapse."

**Quantum reframe**: All quantum sensor feature metrics may track ensemble
size (number of NV centers). After controlling for sensor count, standard
metrics may collapse.

### The direction/magnitude distinction (drug-perturbation, §1)
> "Existing approaches... conflate two independent properties: how *strong*
> a drug's effect is (magnitude) and how *consistent* its direction is
> across contexts."

**Quantum reframe**: Existing quantum biomarker approaches conflate signal
strength with directional consistency across decoherence regimes.

### The algebraic reduction (boundary-conditions, Abstract)
> "sheaf consistency testing on scalar data reduces algebraically to
> Cochran's Q"

**Quantum use**: When stalks are scalar, sheaf adds nothing over standard
heterogeneity tests. Value comes from subspace-valued stalks.

### The boundary conditions (boundary-conditions, §1)
> "The question is whether these connections produce practical advantages
> or repackage standard statistics in topological language."

**Quantum reframe**: Same question. Do geometric methods for quantum sensor
robustness produce practical advantages, or repackage standard sensor
calibration in topological language?

### The ecological fallacy (ecological-bias, Abstract)
> "Ecological regression produces coefficients that are incommensurable with
> patient-level effects — a different quantity, not a noisy estimate."

**Quantum reframe**: Per-device averaged readouts vs per-sensor-element
readouts may give qualitatively different conclusions.

### The failure mode insight (cross-design, Abstract)
> "MR-null status alone accounted for all predictive power... The
> observational leg added no discriminative information."

**Quantum reframe**: Under clean conditions, all features associate with
disease. The discriminator is which survive decoherence.

### The subspace collapse warning (MS/MS, Abstract)
> "Binary fingerprint embeddings produce PCA subspaces that are
> near-orthogonal across all instrument types... dynamic range of 11.8%."

**Quantum use**: Must check that quantum readout subspaces don't collapse
the same way.

### The validity problem (bracket-norm-validity, §2.4)
> "Consider two drugs with identical direction instability: Drug A transports
> a therapeutic mechanism; Drug B transports cellular violence. The raw metric
> cannot distinguish these cases."

**Quantum reframe**: Two features with identical robustness scores: one
diagnostically meaningful, one saturated under decoherence. Raw metric
cannot distinguish. The validity ladder can.

### The negative result (boundary-conditions, §3.3)
> "Absent the boundary conditions, geometric methods reduce to standard
> tools or fail outright."

**Quantum use**: Cite directly. Same expectation for quantum data.

---

## 15. Predicted Negative Results

These are the honest negatives we should pre-register and report.

| Prediction | Source paper | Reasoning |
|---|---|---|
| Chern number = 0 for all features | New | Near-term sensors don't explore enough parameter space for topological transitions |
| ORC below chance on quantum readout graphs | Boundary conditions §3.3 | ORC = 0.466 on clinical graphs; degree confound likely worse on sensor graphs |
| Sheaf Q = Cochran's Q on scalar features | Boundary conditions App B | Proven algebraically; must hold on quantum data too |
| Grassmannian distance null under uniform decoherence | Cohort transfer §4.6 | Same as centralized platform null (SPIROMICS, TCGA) |
| Subspace collapse risk for binary features | MS/MS paper | 85-97% saturation observed on binary fingerprints |
| PCA may destroy diagnostic signal | Boundary conditions §3.3 | ARI = -0.011 to -0.016 observed |
| LASSO beats all geometric methods at low decoherence | General | When signal is strong and noise is low, geometry is overhead |
| Von Neumann entropy tracks decoherence linearly | New | Summary statistic, not structural — provides no boundaries |
| Forman-Ricci = degree deficit on sensor graphs | Boundary conditions §3.3 | Confirmed on 4 benchmark DAGs (AUROC = 0.677) |
| Averaging across NV centers destroys information | Ecological bias | Ecological fallacy: device-level ≠ sensor-level |
