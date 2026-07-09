# Extraction: Bracket Norm + Drug Transport + Validity Papers

## Paper 1: Bracket Norm (eLife submission)

**Full title**: "Bracket Norm Tracks Causally Important Brain Regions From Population Geometry"
**Repo**: `bracket-norm`
**Latest tex**: `paper/paper_v20.tex`

### Section structure

| Section | Title | Lines |
|---|---|---|
| §1 | Introduction | ~L33-38 |
| §2 | Results | L42 onward |
| §2.1 | All 19 geometric metrics are confounded by neuron count | L45 |
| §2.2 | Bracket norm tracks silencing importance | L101 |
| §2.3 | Photoinhibition increases bracket norm while degrading behavior | L144 |
| §2.4 | Human ECoG generalizes across species and recording modality | L185 |
| §2.5 | Region rankings are task-specific | L216 |
| §3 | Discussion | L314 |
| §4 | Methods | L347 |
| §4.1 | Datasets | L350 |
| §4.2 | Silencing effect size | L374 |
| §4.3 | Bracket norm definition | L379 |
| §4.4 | Dimensional correction | L389 |

### Key equations

**Bracket norm definition** (§4.3, Eq. unnumbered ~L384):
```
BN = || ξ(high evidence) - ξ(low evidence) ||
```
where ξ = mean(x_right) - mean(x_left) computed within evidence quartiles.

**Dimensional correction** (§4.4, ~L395):
```
BN/√n = || ξ_high - ξ_low || / √n
```
Invariant: mean 0.394 ± 0.008, CV 2.0%, ρ with neuron count = +0.085 (p=0.07).

### Key results for quantum paper

| Result | Value | Section | Quantum analog |
|---|---|---|---|
| All 19 metrics confounded by n | ρ > 0.8 with neuron count | §2.1, Table 1 | All sensor features may be confounded by ensemble size (number of NV centers) |
| BN partial ρ survives | +0.753 after controlling neuron count | §2.2 | BN should survive after controlling sensor count |
| BN/√n predicts silencing | ρ = +0.833, p = 0.007 | §2.2 | BN/√n_sensors should predict feature robustness |
| Top-3 match exact | p = 0.012, exact combinatorial | §2.2, Table 3 | Top-3 robust features should match ground truth |
| Non-overlapping tails | permutation p = 0.0008 | §2.2 | Clean separation between robust and fragile features |
| ALM photoinhibition +68% | 32/32 sessions, p < 10^-6 | §2.3 | Perturbation increases BN = decoherence increases feature instability |
| ECoG confound eliminated | ρ = 0.0 with electrode count after √n correction | §2.4, Table 5 | √n_sensor correction should eliminate ensemble-size confound |
| Task-specific rankings | cross-task ρ = -0.079, n=18 | §2.5 | Sensor-modality-specific rankings expected |

### Importable code

**`bracket-norm/geometry/subspace.py`**:
- `fit_pca_subspace(activity, labels, k)` → (n_neurons, k) basis
- `fit_lda_subspace(activity, labels, k)` → (n_neurons, k) basis

**`bracket-norm/geometry/distances.py`**:
- `principal_angles(U, V)` → array of angles in radians
- `grassmannian_distance(U, V)` → float (geodesic on Gr(k,n))
- `subspace_overlap(U, V)` → float [0,1]
- `gauge_normalized_distance(U, V, X1, X2)` → float (effective-rank normalized)
- `cka(X1, X2)` → float (centered kernel alignment, in same file below L80)

**Usage for quantum paper**: Import `grassmannian_distance` and `principal_angles` directly for method 8 (Grassmannian geodesic distance). Import `cka` for method 13.

---

## Paper 2: Drug Perturbation Geometry (PLOS Comp Bio submission)

**Full title**: "Direction Instability: A Magnitude-Invariant Metric for Cross-Cell-Line Drug Mechanism Transport"
**Repo**: `drug-perturbation-geometry`
**Latest tex**: `paper/drug_transport_plos_v3.tex`

### Section structure

| Section | Title | Lines |
|---|---|---|
| Author Summary | | L50-66 |
| §1 | Introduction | L68 |
| §2 | Methods | L119 |
| §2.1 | Data | L121 |
| §2.2 | Direction instability | L144 |
| §2.3 | Magnitude correction | L159 |
| §2.4 | Complementary metrics | L173 |
| §2.5 | Population null model | L187 |
| §2.6 | Pre-registration | L202 |
| §3 | Results | L213 |
| §3.1 | Direction instability tracks biological mechanism conservation | L215 |
| §3.2 | Target breadth determines transport | L274 |
| §3.3 | Predictive validation | L515 |

### Key equations

**Direction instability** (§2.2, Eq. 1, L150):
```
D = 1 - (2 / K(K-1)) Σ_{i<j} ŝᵢᵀ ŝⱼ
```
where ŝᵢ = sᵢ / ||sᵢ||₂ are unit-normalized signature vectors.

D ∈ [0, 2]. D=0 collinear. D=1 orthogonal. D>1 anticorrelated.

### Key results for quantum paper

| Result | Value | Section | Quantum analog |
|---|---|---|---|
| DI anticorrelates with Jaccard | ρ = -0.79, p < 10^-300 | §3.1 | DI should anticorrelate with feature-overlap across decoherence regimes |
| Pan-HDAC vs selective | DI 0.62 vs 0.96 | §3.2 | Broadband features (like temperature) should have low DI; narrow features high DI |
| LOO AUROC | 0.986 (permutation null 0.500 ± 0.012) | §3.3 | LOO prediction of feature robustness in held-out decoherence regime |
| Magnitude correction | DI correlates with norm at ρ = -0.41 | §2.3 | Must correct for signal strength: weaker quantum signals → noisier → inflated DI |
| 8,949 drugs, 978 genes | full dataset size | §2.1 | Scale reference: our simulation is smaller but controlled |
| Cytotoxicity defense | stress-gene removal preserves HDAC gradient (ρ = 0.83) | §3.2 | Must test: does removing decoherence-dominated features preserve diagnostic gradient? |
| Genetic triangulation | drug-shRNA cosine validates DI | §3.2 | Need independent validation channel for quantum features (not same sensor) |
| Imatinib case study | DI = 0.97, signatures rotate 88° avg | §3.2 | Example of context-dependent biomarker that looks different in each regime |

### Importable code

**`drug-perturbation-geometry/geometry/drug_transport.py`**:
- `direction_stability(signatures: dict)` → dict with direction_instability, mean_pairwise_cosine, mean_rotation_angle_deg, n_cell_lines
- `magnitude_stability(signatures: dict)` → dict with magnitude_cv, mean_norm

**`drug-perturbation-geometry/geometry/bracket_norm.py`** (if exists):
- Check for additional bracket norm variants

**`drug-perturbation-geometry/geometry/distances.py`**:
- Same Grassmannian functions as bracket-norm repo

**`drug-perturbation-geometry/geometry/subspace.py`**:
- Same PCA/LDA subspace functions

**Usage for quantum paper**: Import `direction_stability` for methods 1-2 (raw bracket norm). The dict-based API takes {regime_name: feature_vector} which maps perfectly to {decoherence_regime: sensor_readout}.

---

## Paper 3: Bracket Norm Validity (unsubmitted, in progress)

**Full title**: "Bracket Norm Is Not Mechanism: Validity Conditions for Noncommutative Perturbation Scores"
**Repo**: `direction-instability-drug-validity`
**Latest tex**: `paper/bracket_norm_validity_v8.tex`

### Section structure

| Section | Title | Lines |
|---|---|---|
| §1 | Introduction | L55 |
| §2 | Background | L115 |
| §2.1 | Direction instability | L117 |
| §2.2 | Grassmannian geodesic distance in Perturb-seq | L138 |
| §2.3 | Direction instability in Cell Painting morphological profiles | L150 |
| §2.4 | The validity problem | L165 |
| §2.5 | The validity ladder | L185 |
| §3 | Related work | L216 |
| §4 | Methods | L262 |

### The 7-rung validity ladder (§2.5)

1. **Process view declared** — what counts as a "context"?
2. **Reliable vector field** — adequate SNR for direction to be meaningful
3. **Perturbation estimable** — effect isolated from background
4. **Localized to biology** — instability in pathway-relevant features, not global
5. **Phenotype-aligned** — direction of conservation matches known target
6. **Transports across contexts** — score replicates in held-out contexts (low Fréchet variance)
7. **Predicts holdout** — metric predicts external outcome in new data

**Quantum analog**: Same ladder for quantum sensor features:
1. What is a "decoherence regime"? (T2 level, surface noise, device, temperature)
2. Sensor readout has adequate SNR
3. Biological signal isolated from decoherence
4. Robust features are in diagnostic channels, not noise channels
5. Feature stability aligns with diagnostic axis (tumor vs healthy)
6. Feature replicates across held-out decoherence regimes
7. Feature predicts clinical outcome in new data

### Key results

| Result | Value | Section | Quantum analog |
|---|---|---|---|
| Corrections preserve ranking | LINCS ρ > 0.99, Perturb-seq ρ = 0.91, JUMP-CP ρ = 0.99 | Abstract | Validity corrections should preserve feature rankings |
| Phenotype-projected bracket | ρ = 0.38, p < 10^-27 with on-target genetic connectivity | Abstract | Diagnostic-projected bracket should correlate with ground truth |
| Raw bracket does NOT | ρ = -0.04 | Abstract | Raw bracket without projection misses the diagnostic axis |
| Transport-stable beats raw | all 5 CV folds | Abstract | Fréchet variance penalty should improve LOO prediction |
| Localization informative null | helps enzyme inhibitors, not receptor drugs | Abstract | Localization may help for some sensor types but not others |
| Proteasome shift 660-900 positions | after essential-gene correction | Abstract | Some features may shift dramatically after decoherence correction |
| 3 domains tested | LINCS (8,949), Perturb-seq (1,676), JUMP-CP (25,254) | Abstract | Our quantum paper is a 4th domain |

### The 5 bracket norm variants (importable)

All from `direction-instability-drug-validity/geometry/bracket_norm.py`:

1. **`direction_instability(signatures)`** — Raw DI = 1 - mean(pairwise cosine). Method 1.

2. **`phenotype_projected_bracket(signatures, phenotype_direction)`** — Projects pairwise differences onto a target phenotype axis. Method 4 in quantum paper. For quantum: phenotype_direction = classifier weight vector for tumor vs healthy.

3. **`toxicity_corrected_bracket(signatures, stress_genes)`** — Removes stress/toxicity genes before computing DI. Quantum analog: remove decoherence-dominated features (e.g., T2-derived features) before computing.

4. **`transport_stable_bracket(signatures, frechet_penalty=1.0)`** — Raw DI minus Fréchet variance of unit directions on the sphere. Method 3. Penalizes features whose instability is itself unstable.

5. **`localization_score(signatures, region_mask)`** — Ratio of within-pathway to global instability. Method 5. For quantum: ratio of within-modality to cross-modality instability.

6. **`validity_ladder_score(raw_bracket, ...)`** — Evaluates against all 7 rungs. Could adapt for quantum validity assessment.

### Additional importable code

**`direction-instability-drug-validity/geometry/grassmannian.py`**:
- `principal_angles(U1, U2)` → angles
- `geodesic_distance(U1, U2)` → float
- `subspace_overlap(U, V)` → float
- `frechet_mean_subspace(subspaces, n_iter=50)` → (d, k) basis — Fréchet mean on Gr(k,d) via iterative projection
- `project_out_subspace(U, confound)` → residual (d, k) basis — projects confounding subspace out on the Grassmannian

**Usage for quantum paper**: `frechet_mean_subspace` is needed for method 3 (transport-stable bracket). `project_out_subspace` is needed for decoherence-corrected variants (remove the decoherence subspace before measuring feature geometry).

---

## Cross-paper code reuse summary

| Quantum method # | Function to import | Source repo | Source file |
|---|---|---|---|
| 1 (raw BN) | `direction_instability(sigs)` | direction-instability-drug-validity | geometry/bracket_norm.py |
| 2 (BN/√n) | `direction_instability(sigs) / sqrt(n_sensors)` | direction-instability-drug-validity | geometry/bracket_norm.py |
| 3 (transport-stable) | `transport_stable_bracket(sigs, penalty)` | direction-instability-drug-validity | geometry/bracket_norm.py |
| 4 (phenotype-projected) | `phenotype_projected_bracket(sigs, w)` | direction-instability-drug-validity | geometry/bracket_norm.py |
| 5 (localized) | `localization_score(sigs, mask)` | direction-instability-drug-validity | geometry/bracket_norm.py |
| 8 (Grassmannian geodesic) | `grassmannian_distance(U, V)` | bracket-norm | geometry/distances.py |
| 9 (holonomy) | Need to compose `principal_angles` in a loop | bracket-norm | geometry/distances.py |
| 13 (CKA) | `cka(X1, X2)` | bracket-norm | geometry/distances.py |
| 14 (Procrustes) | Implement from `scipy.spatial.procrustes` | — | scipy |
| Subspace fitting | `fit_pca_subspace(X, labels, k)` | bracket-norm | geometry/subspace.py |
| Fréchet mean | `frechet_mean_subspace(subspaces)` | direction-instability-drug-validity | geometry/grassmannian.py |
| Confound removal | `project_out_subspace(U, confound)` | direction-instability-drug-validity | geometry/grassmannian.py |
| Direction stability (dict API) | `direction_stability(sigs_dict)` | drug-perturbation-geometry | geometry/drug_transport.py |
| Magnitude stability | `magnitude_stability(sigs_dict)` | drug-perturbation-geometry | geometry/drug_transport.py |
| Validity ladder | `validity_ladder_score(...)` | direction-instability-drug-validity | geometry/bracket_norm.py |

---

## Key quotes for the quantum paper introduction

From bracket-norm paper (§1, L35-36):
> "All 19 geometric metrics we test predict silencing importance only because they track the number of recorded neurons; after controlling for recording yield, all 19 collapse."

**Quantum analog**: "All quantum sensor feature metrics may predict diagnostic importance only because they track the number of NV centers per measurement; after controlling for ensemble size, standard metrics may collapse."

From drug-perturbation paper (§1, L73-76):
> "Existing approaches... conflate two independent properties: how *strong* a drug's effect is (magnitude) and how *consistent* its direction is across contexts."

**Quantum analog**: "Existing approaches to quantum sensor biomarker validation conflate two independent properties: how strong the sensor's raw signal is and how consistent the readout direction is across decoherence regimes."

From validity paper (§2.4, L167-183):
> "Consider two drugs with identical direction instability, D = 0.55: Drug A is a pan-HDAC inhibitor [transports a therapeutic mechanism]; Drug B is a mitochondrial toxin [transports cellular violence]. The raw metric cannot distinguish these cases."

**Quantum analog**: "Consider two sensor features with identical robustness scores: Feature A is temperature-sensitive and diagnostically meaningful; Feature B is robust only because it saturates under decoherence and carries no biological information. The raw metric cannot distinguish these cases. The validity ladder can."

From boundary conditions paper abstract:
> "sheaf consistency testing on scalar data reduces algebraically to Cochran's Q"

**Quantum analog**: Must verify this reduction holds for quantum sensor readouts with scalar features. If stalks are scalar (single sensor reading), sheaf adds nothing. Value comes from subspace-valued stalks (multi-feature readouts).
