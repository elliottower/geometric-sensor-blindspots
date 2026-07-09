# Extract: Transport-Wrapper, Holonomy, Atlas, Mechanistic Validity/Reference

## 1. TransportKit (transport-wrapper repo)

### Architecture

Two independent engines fused under a shared boundary-condition gate:

**Engine E (effect transportability)** — sheaf H^1 obstruction on scalar
effect estimates across strata. On scalar stalks this is algebraically
Cochran's Q (proven in boundary_conditions_geometry paper, Appendix C).

```
Significant Q => H^1 != 0 => effect does NOT transport
```

Validated: 85% accuracy / zero false positives on 61 MR pairs.
Boundary gate: transport verdict only trusted when test is powered
(all 9 misclassifications were underpowered false negatives).

**Engine C (classifier transportability)** — Grassmannian geodesic distance
between cohort PCA subspaces, predicting cross-cohort AUC gap after
partialing out source quality.

Validated: partial rho = 0.61 (CRC), 22% LOO-MAE gain.
Boundary gate: valid only under genuine analytical heterogeneity
(permutation-null z high); null on centralized/shared platforms.

**Fusion rule** (fusion.py):
```
fuse() -> TRANSPORTS / DOES_NOT_TRANSPORT / UNKNOWN / DISAGREE
```
- Both valid + agree → verdict with "high" confidence
- Both valid + disagree → "DISAGREE" (diagnostic: localizes WHERE)
- One valid → single-channel verdict, "medium" confidence
- Neither valid → "UNKNOWN" (honest: insufficient information)
- NEVER averages across an invalid channel

### Importable Code

```
transportkit/
  engines.py      # effect_transportability(), classifier_transportability()
                  # pca_subspace(), geodesic_distance(), permutation_null_z()
  fusion.py       # fuse(), transportability_report() (one-call convenience)
  adapters.py     # load_effect_families(), load_cohort_matrices(), se_from_row()
```

**Key functions for quantum paper:**

| Function | File | What it does | Quantum use |
|---|---|---|---|
| `effect_transportability(estimates, ses)` | engines.py:38 | Cochran's Q + power gate | Per-feature robustness across decoherence regimes |
| `classifier_transportability(X_source, X_target, k)` | engines.py:122 | Grassmannian geodesic + heterogeneity gate | Cross-regime subspace drift |
| `pca_subspace(X, k)` | engines.py:77 | Top-k PCA subspace | Extract readout subspaces |
| `geodesic_distance(U1, U2)` | engines.py:84 | L2 norm of principal angles | Method 8 |
| `permutation_null_z(Xs, k)` | engines.py:92 | Is distance distinguishable from shuffled? | Heterogeneity gate for quantum sensors |
| `fuse(effect_res, clf_res)` | fusion.py:52 | Two-channel fusion with gates | Method 12 |
| `transportability_report(...)` | fusion.py:94 | One-call convenience | End-to-end transportability verdict |

**Dataclasses:**
- `EffectResult(Q, df, p, I2, transports, powered, valid, note)`
- `ClassifierResult(geodesic, z_vs_null, predicted_auc_gap, analytic_heterogeneity, valid, note)`
- `FusedVerdict(verdict, confidence, channels_used, agreement, effect, classifier, rationale)`

### Quantum adaptation needed

1. Replace "strata" (study designs) with "decoherence regimes"
2. Replace "cohorts" with "sensor configurations / devices"
3. Add a third engine for quantum-specific metrics (Berry phase, QFI)
4. The boundary-condition gate pattern is directly reusable

---

## 2. Genetic Perturbation Holonomy

### Core Question (README)

> "A gene knockout in K562 leukemia cells produces a transcriptomic response.
> The same knockout in RPE1 retinal cells produces a different response. Are
> these 'the same mechanism' or not?"

Applied to quantum: "An NV-diamond sensor in decoherence regime A produces a
readout. The same sensor in regime B produces a different readout. Is the
biomarker 'the same feature' or not?"

### Geometry Module — DIRECTLY IMPORTABLE

**geometry/grassmannian.py** — Full parallel transport + holonomy:

| Function | Line | Signature | Quantum use |
|---|---|---|---|
| `principal_angles(U1, U2)` | 9 | `(d,k), (d,k) -> (k,)` | Core primitive |
| `geodesic_distance(U1, U2)` | 17 | `(d,k), (d,k) -> float` | Method 8 |
| `subspace_overlap(U, V)` | 22 | `(d,k), (d,k) -> float` | Quick alignment check |
| `transport_matrix(U_from, U_to)` | 28 | `(d,k), (d,k) -> (k,k)` | Parallel transport step |
| `compose_holonomy(subspaces)` | 39 | `list[(d,k)] -> ((k,k), float)` | Method 9: loop holonomy |
| `pca_subspace(X, k)` | 69 | `(n,d), int -> (d,k)` | Subspace extraction |
| `pairwise_distance_matrix(subspaces)` | 57 | `list[(d,k)] -> (m,m)` | Full distance matrix |
| `holonomy_permutation_test(data, cycle, k, n_perms, rng)` | 76 | `-> dict` | Significance test for holonomy |

**geometry/distances.py** — CKA, Procrustes, and more:

| Function | Line | Quantum use |
|---|---|---|
| `cka(X, Y, kernel='linear')` | 81 | Method 13 |
| `debiased_cka(X, Y)` | 119 | Finite-sample corrected CKA |
| `gauge_normalized_distance(U, V, X1, X2)` | 43 | Gauge-invariant geodesic (handles amplitude scaling) |
| `chordal_distance(U, V)` | 172 | Alternative subspace distance |
| `all_subspace_distances(U, V)` | 182 | All 5 metrics at once |

**geometry/subspace.py** — PCA/LDA extraction utilities (not read but exists).

### Key Connection to Boundary Conditions Paper

From README:
> "Sheaf holonomy detected global inconsistency in MR evidence networks.
> Here, the 'evidence network' is a cell-type graph with genetic perturbation
> subspaces on each node."

For quantum: the "evidence network" is a sensor-modality graph with readout
subspaces on each node.

### Data Sources

- Replogle et al. 2022 — genome-scale Perturb-seq, ~2.5M cells, ~10,000 knockdowns
- K562 + RPE1 cell types
- Pipeline status: data pipeline being set up

---

## 3. Direction Instability Atlas

### Paper (atlas_paper_v2.tex)

**Title**: "A Pre-Registered Direction Instability Atlas Across Three Perturbation Modalities"

**Three modalities:**
1. **LINCS L1000 transcriptomics** — 8,949 drugs
2. **Tahoe-100M single-cell transcriptomics** — 379 drugs
3. **JUMP Cell Painting morphology** — 7,946 CRISPR knockouts + 12,590 ORF overexpressions

**DI formula** (§1):
```
DI = 1 - mean(cos(s_i, s_j))
```
Mean pairwise cosine dissimilarity among a perturbation's signatures across contexts.

**Key results:**
- Cross-modal concordance: partial rho = 0.19, n = 378, p = 2.1e-4
- DI predicts mechanism transport: AUROC = 0.945 (pre-registered, 8,949 LINCS drugs)
- Essential genes show LOWER DI (rho = -0.33) — inverts prediction
- Knockout vs overexpression DI uncorrelated (rho = 0.01, n = 5,220)
- Expression variance, paralog count each explain < 0.4% of DI variance

**Pre-registration:** Batch 1 SHA `0d07a01`, Batch 2A SHA `a17c125`

### Quantum mapping

The atlas has 3 perturbation modalities (drugs, knockouts, overexpressions).
The quantum paper adds a 4th: **decoherence perturbation**.

| Atlas modality | Quantum analog |
|---|---|
| Drug perturbation signature | Sensor readout under decoherence perturbation |
| Cell line context | Decoherence regime / device |
| Cross-cell-line DI | Cross-regime DI |
| Mechanism transport | Biomarker robustness |

The DI formula is directly applicable: compute DI for each candidate biomarker
feature across decoherence regimes. High DI = feature direction changes with
decoherence = non-robust.

### Importable code

```
geometry/
  direction_instability.py  # DI computation
  magnitude_correction.py   # Second-order confound removal
experiments/
  batch1_lincs/             # LINCS L1000 pipeline
  batch2_cross_modal/       # Cross-modal concordance
```

---

## 4. Mechanistic Validity Framework

### Five Validity Lenses (README)

| Lens | Tradition | Core question |
|---|---|---|
| Construct | Philosophy of science | Is the claim falsifiable and well-defined? |
| Internal | Neuroscience | Is the causal evidence sound? |
| External | Pharmacology | Does it generalize beyond test conditions? |
| Measurement | Measurement theory | Are the metrics reliable and calibrated? |
| Interpretive | MI | Is the description level declared and consistent? |

### Six Evidence Families

| Family | What it asks |
|---|---|
| Causal | Does X causally produce Y? |
| Structural | Do the weights encode the claimed computation? |
| Information-theoretic | What information flows where? |
| Behavioral | Does the circuit reproduce model behavior? |
| Representational | What geometric structure do activations have? |
| Measurement-theoretic | Are the metrics themselves reliable? |

### Verdict Tiers

| Tier | Name | What it means |
|---|---|---|
| 1 | Proposed | Structural alignment only, no causal evidence |
| 2 | Causally suggestive | Necessity established (ablation degrades behavior) |
| 3 | Mechanistically supported | Necessity + sufficiency |
| 4 | Triangulated | Multiple independent metrics converge |
| 5 | Validated | All five lenses pass |

### Quantum biomarker application

Map each quantum biomarker claim through the 5 lenses:

| Lens | Quantum biomarker question |
|---|---|
| Construct | "Decoherence-robust biomarker" — falsifiable? What would refute it? |
| Internal | Does the biomarker causally track disease state (not decoherence)? |
| External | Does it generalize across devices, tissues, decoherence regimes? |
| Measurement | Are the geometric robustness metrics themselves reliable? |
| Interpretive | Is the "robust feature" claim at the right level (single sensor vs. multi-sensor)? |

The 24 geometric methods become evidence families:
- Bracket norm → Representational family
- Sheaf H^1 → Structural family
- Holonomy → Causal family (transport is causal invariance)
- TransportKit fusion → Triangulation (two independent engines converging)

A biomarker that passes all 24 methods under pre-registered conditions
reaches Tier 4 (Triangulated). Passing the validity lenses elevates to Tier 5.

---

## 5. Mechanistic Reference (PAPER_PLAN.md)

### Core Contributions

1. **Transport hierarchy across views** — what "same mechanism" means depends on
   the view:
   - Object = component overlap
   - Role = functional equivalence
   - Subspace = Grassmannian proximity
   - Structural = gauge-invariant transport
   - Process = trajectory equivalence

2. **Five reference failure modes:**
   - **Evidence misfire** — different methods find different circuits (view mismatch)
   - **Claim laundering** — evidence at one level used for claims at another
   - **Mimic mechanism** — method always "finds" referent (e.g., unconstrained DAS)
   - **Zombie mechanism** — refuted referent that persists in terminology
   - **Reference debt** — conjecture treated as established before validation

3. **Scaffold-path decomposition:**
   - Weight = structural capacity
   - Activation = runtime recruitment
   - No-go theorem for gauge-symmetric systems (triangulation provably necessary)

### Quantum biomarker mapping

The transport hierarchy applies directly to quantum biomarker claims:

| Transport level | Quantum meaning |
|---|---|
| Object | Same features selected across devices |
| Role | Same diagnostic function across regimes |
| Subspace | Same readout subspace (Grassmannian proximity) |
| Structural | Same geometric structure under gauge freedom (calibration invariance) |
| Process | Same temporal dynamics under decoherence trajectory |

The five failure modes predict quantum biomarker failures:

| Failure mode | Quantum prediction |
|---|---|
| Evidence misfire | Different robustness methods disagree on which features are robust |
| Claim laundering | Single-sensor robustness used to claim multi-sensor robustness |
| Mimic mechanism | A method that always reports "robust" (e.g., CKA on low-d data) |
| Zombie biomarker | A feature reported as robust that was later shown to track device drift |
| Reference debt | "Decoherence-robust" label applied before cross-device validation |

### Worked examples relevant to quantum

- **12. Sheaf cohomology obstruction (negative certificate):**
  "H^1 ≠ 0 proving no global linear label exists. Formal reference failure
  with topological (not empirical) certificate."
  → This is exactly what method 10 does for multi-sensor quantum data.

- **14. mTOR/rapamycin — reference failure with clinical consequences:**
  "In vitro validation (Tier 3+) but in vivo transport failure."
  → Analog: lab-validated quantum biomarker that fails in clinical tissue.

---

## Summary: What to import for the quantum paper

### Code that works out of the box

| Source repo | Module | Functions | Methods covered |
|---|---|---|---|
| genetic-perturbation-holonomy | geometry/grassmannian.py | geodesic_distance, compose_holonomy, holonomy_permutation_test, pca_subspace | 8, 9 |
| genetic-perturbation-holonomy | geometry/distances.py | cka, debiased_cka, gauge_normalized_distance, chordal_distance, all_subspace_distances | 13, 14 |
| transport-wrapper | transportkit/engines.py | effect_transportability, classifier_transportability, permutation_null_z | 10, 11, 12 |
| transport-wrapper | transportkit/fusion.py | fuse, transportability_report | 12 |

### Code that needs adaptation

| Source repo | Module | What to adapt |
|---|---|---|
| direction-instability-atlas | geometry/direction_instability.py | Replace cell-line context with decoherence regime |
| transport-wrapper | transportkit/adapters.py | Write quantum sensor data adapter |

### Frameworks to apply

| Source | What | How |
|---|---|---|
| mechanistic-validity | 5 lenses + 6 evidence families + verdict tiers | Score each quantum biomarker claim |
| mechanistic-reference | Transport hierarchy + 5 failure modes | Predict and diagnose biomarker failures |
| direction-instability-atlas | Pre-registration structure | SHA-freeze methods before experiments |
