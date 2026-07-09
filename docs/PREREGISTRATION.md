# Pre-Registration: Geometric Methods for Quantum Sensor Biomarker Robustness

**Status**: CODE FROZEN, REVIEWED AND APPROVED BY PERPLEXITY

## Frozen Code

All analysis code, simulation oracle, methods, baselines, runner pipeline,
and analysis module are frozen at the commit SHA below. No modifications to
any `src/` file are permitted after this SHA until experiments complete and
results are reported.

**Freeze SHA**: `8401d543ecb6de720bd2a0fdc41c3ec4dc660f13`

**Frozen files**:
- `src/nv_diamond.py` — NV-diamond intracellular thermometry simulation oracle
- `src/methods.py` — 23 geometric methods (methods 1-23; slot 24 intentionally empty)
- `src/baselines.py` — 7 scalar baselines (t-test, Cohen's d, LASSO, RF, logistic, IRM, DRO)
- `src/runner.py` — multi-seed experiment runner pipeline
- `src/analysis.py` — bootstrap CI, Spearman rho, confirmatory hypothesis testing
- `tests/test_nv_diamond.py` — 10 simulation tests
- `tests/test_methods.py` — 35 method tests (including known-answer H0 persistence)
- `tests/test_baselines.py` — 9 baseline tests

**All 54 tests pass at freeze time.**

### Methods Cut and Why

**Method 24 (Chern number) — CUT.** The naive plaquette-phase computation is not
gauge-invariant; a correct implementation requires the Fukui-Hatsugai-Suzuki
lattice gauge method. A broken estimator that returns 0 would spuriously
"confirm" H6c. Slot reserved for a correct FHS implementation.

**Method 15 (persistent homology) — RESTRICTED to H0 only.** The hand-rolled H1
loop was a triangle-counting heuristic, not correct Vietoris-Rips persistent
H1. H0 (connected components via union-find) is provably correct and has a
known-answer test (two-cluster gap detection). H5a's prediction about H1
adding nothing is WITHDRAWN — it cannot be tested without a correct H1.

**Method 16 — RENAMED from "bottleneck persistence" to "wasserstein persistence".**
The implementation uses optimal assignment (Wasserstein-1 matching cost),
not the L-infinity bottleneck metric.

## Simulation Oracle

NV-diamond intracellular thermometry with 5 decoherence axes:

| Axis | Parameter | Range | Physics | Composite? |
|------|-----------|-------|---------|------------|
| T2 degradation | `alpha_T2` | [0, 1] | Spin coherence loss | **YES** — simultaneously affects T2, linewidth (1/πT2), and contrast (1-0.5·α) |
| Surface noise | `sigma_surface` | [0, 0.5] | Diamond surface defects | YES — affects T2 noise and linewidth variance |
| Device variation | `sigma_device` | [0, 0.1] | Calibration offsets | No — additive temperature offset only |
| Temperature drift | `sigma_temp` | [0, 2.0] K | Environmental fluctuation | No — additive per-sample noise |
| Ensemble size | `n_centers` | {5..50} | NV centers per measurement | No — changes dimensionality only |

**Composite axis note**: `alpha_T2` is a *composite physical axis* — increasing
it simultaneously degrades T2 coherence time, broadens the ODMR linewidth
(linewidth = 1/πT2), and reduces fluorescence contrast. This is physically
correct (real T2 loss does cause these coupled effects), but it means "method
detects alpha_T2 degradation" tests detection of the *composite physical
effect*, not of isolated coherence loss. All results involving `alpha_T2`
must be interpreted with this coupling in mind. Similarly, `sigma_surface`
affects both T2 perturbation and linewidth variance.

Ground truth: tumor tissue (311.5 K) vs healthy tissue (310.0 K), 1.5 K difference.
Signal mechanism: NV zero-field splitting dD/dT = -74 kHz/K.

## Experiment Design

### Replication Protocol

Each condition (axis × level) is run with **N=20 independent seeds** (base_seed
1000, seeds 1000-1019). This produces 20 independent realizations per condition,
enabling bootstrap confidence intervals on all statistics.

**Two-axis protocol**: sweep one decoherence axis (10 levels) while holding all
others at baseline (zero decoherence).

**Parameters**:
- `n_per_class = 200` (samples per class per device)
- `n_devices = 3`
- `n_seeds = 20` (independent realizations per condition)
- `base_seed = 1000`
- 10 levels × 5 axes = 50 conditions × 20 seeds = 1,000 datasets
- 23 methods + 7 baselines = 30 evaluations per dataset
- **Total: 30,000 method evaluations**

**Execution command**:
```bash
uv run --with scipy --with scikit-learn --with tqdm \
    python src/runner.py --output results/sweep_results.json

uv run --with scipy --with scikit-learn \
    python src/analysis.py --input results/sweep_results.json \
    --output results/analysis.json
```

## Statistical Analysis Plan

### Primary Metric: Spearman ρ(score, decoherence_level)

All cross-method comparisons use **Spearman rank correlation between a method's
score and the decoherence level**, computed per seed, then summarized as
**median ρ across seeds with bootstrap 95% CI** (10,000 bootstrap resamples).

This normalizes across methods' wildly different score scales (a bracket norm
vs. an AUC vs. a curvature value) by reducing everything to a monotonic
association strength on [-1, +1]. A method that "tracks" decoherence well
has |ρ| near 1; a method insensitive to decoherence has ρ near 0.

### Directional Comparisons

When hypothesis H says "method A detects X better than method B," the test is:
compute ρ_A and ρ_B for each of the 20 seeds (paired), take the per-seed
difference (ρ_A − ρ_B), and report the bootstrap 95% CI on the median
difference. If the CI excludes 0, the comparison is significant.

### Confirmatory vs. Exploratory Split

**1 Correctness Gate** (pass/fail analytic check, no α needed):

| ID | Hypothesis | Test |
|----|-----------|------|
| **H4d** | Sheaf H^1 reduces to Cochran's Q on scalar stalks | Run both on identical scalar input; must match to floating-point tolerance |

H4d is an analytic identity, not a stochastic comparison. It either matches
or it doesn't — bootstrap CIs and Bonferroni correction are inapplicable.
If H4d fails, the sheaf implementation is wrong and all sheaf-based results
are suspect. It runs first as a correctness gate.

**3 Confirmatory Tests** (2 hypotheses, H3a tested on 2 axes; Bonferroni-corrected α = 0.05/3 = 0.0167):

| ID | Hypothesis | Test |
|----|-----------|------|
| **H3a** | Grassmannian geodesic detects subspace drift at lower decoherence than feature-level methods | CI on median(ρ_geodesic − ρ_bracket) excludes 0 on alpha_T2 **and** sigma_device |
| **H6d** | Von Neumann entropy tracks decoherence monotonically | |ρ| > 0.8 on alpha_T2 |

**1 Voided Hypothesis** (untestable under adopted statistical framework):

| ID | Hypothesis | Status |
|----|-----------|--------|
| **H8a** | LASSO beats all geometric methods at low decoherence | UNTESTABLE — requires direct cross-scale score comparison incompatible with the Spearman-rho normalization. See DEVIATION_LOG.md. |

**H3a companion test**: Because alpha_T2 is a composite axis (moves T2,
linewidth, and contrast together), a subspace method lighting up early
could track the composite rather than "subspace drift" per se. To
disentangle this, H3a is tested on **both** alpha_T2 (composite) and
sigma_device (a clean, non-composite axis that only adds a temperature
offset). The sigma_device result is the cleaner test of the subspace-drift
mechanism; the alpha_T2 result is the composite-axis companion.

These test the headline claims: (1) geometry adds value over baselines
at high decoherence (H3a), (2) information-theoretic methods track noise
monotonically (H6d). H8a (baselines dominate at low decoherence) was voided
post-execution as untestable under the adopted Spearman-rho framework.

**All other hypotheses (H1a-c, H2a-c, H3b-c, H4a-c, H5a-c, H6a-b, H7a-c,
H8b-c, H9a-c, H10) are EXPLORATORY.** Their results are reported descriptively
(median ρ with bootstrap 95% CI per method per axis) but are NOT tested against
a significance threshold. Any exploratory finding that looks promising is
explicitly flagged as requiring independent replication.

### Multiple Testing

The 3 confirmatory tests use **Bonferroni correction** (α_per_test = 0.0167).
H8a was voided post-execution (untestable under the Spearman-rho framework;
see DEVIATION_LOG.md), reducing the confirmatory pool from 4 to 3 and adjusting
α from 0.05/4 = 0.0125 to 0.05/3 = 0.0167. No correction is applied to
exploratory hypotheses because they carry no significance claim.

## Methods

### Geometric (23 — slot 24 empty)

| ID | Method | Origin | Family |
|----|--------|--------|--------|
| 1 | Bracket norm (raw) | bracket-norm repo | Bracket |
| 2 | Bracket norm / sqrt(n) | bracket-norm repo | Bracket |
| 3 | Transport-stable bracket | direction-instability repo | Bracket |
| 4 | Phenotype-projected bracket | direction-instability repo | Bracket |
| 5 | Localized bracket | direction-instability repo | Bracket |
| 6 | Ollivier-Ricci curvature | psychiatric-comorbidity repo | Curvature |
| 7 | Forman-Ricci curvature | epidemiology repo | Curvature |
| 8 | Grassmannian geodesic | genetic-perturbation repo | Grassmannian |
| 9 | Grassmannian holonomy | genetic-perturbation repo | Grassmannian |
| 10 | Sheaf H^1 (Cochran's Q) | epidemiology repo | Sheaf |
| 11 | Per-edge sheaf Q | epidemiology repo | Sheaf |
| 12 | TransportKit fusion | transport-wrapper repo | Fusion |
| 13 | Linear CKA | genetic-perturbation repo | Alignment |
| 14 | Procrustes distance | genetic-perturbation repo | Alignment |
| 15 | Persistent H0 (union-find) | New (TDA, H0 only) | Topology |
| 16 | Wasserstein persistence distance | New (TDA, H0 diagrams) | Topology |
| 17 | QFI flatness | New (quantum info) | Quantum |
| 18 | Berry phase | New (quantum info) | Quantum |
| 19 | Spectral gap stability | New (spectral) | Spectral |
| 20 | Wasserstein distance (1D) | New (OT) | Transport |
| 21 | Von Neumann entropy stability | New (quantum info) | Quantum |
| 22 | Fidelity proxy | New (quantum info) | Quantum |
| 23 | Persistent sheaf cohomology | New (sheaf + TDA) | Sheaf |

### Baselines (7)

| Name | Method | What it tests |
|------|--------|---------------|
| ttest | Max t-statistic | Feature-level signal strength |
| cohens_d | Mean Cohen's d | Effect size |
| lasso | L1-regularized logistic (3-fold CV AUC) | Sparse classification |
| random_forest | Random forest (3-fold CV AUC) | Nonlinear classification |
| logistic | Logistic regression (3-fold CV AUC, no penalty) | Linear classification |
| irm | IRM invariance penalty proxy | Cross-device invariance |
| dro | DRO worst-device AUC | Worst-case robustness |

**Fairness note**: Classification baselines (LASSO, RF, logistic) are evaluated
via 3-fold cross-validated AUC on the same (X, y, device_ids) triple that
geometric methods receive. The geometric methods' scores are NOT AUCs —
cross-method comparison is done exclusively via Spearman ρ (which method's
score tracks decoherence level more monotonically), not by raw score ranking.

## Pre-Registered Hypotheses

Full hypothesis text in `docs/HYPOTHESES.md`. Summary below.

**(C) = Confirmatory, (E) = Exploratory**

### H1 (Bracket family) — all (E)
- H1a: Raw bracket > t-test under anisotropic decoherence; reduces under isotropic
- H1b: Transport-stable > raw when sigma_device > 0.05
- H1c: Phenotype-projected best bracket when signal is small relative to noise

### H2 (Curvature) — all (E)
- H2a: ORC detects cluster merging before AUC drops
- H2b: Forman agrees with ORC for k >= 10, disagrees for k < 5
- H2c: ORC fails when n < 50 per class

### H3 (Grassmannian) — H3a (C), rest (E)
- **H3a (C)**: Geodesic detects subspace drift at lower decoherence than feature-level methods — tested on alpha_T2 (composite) AND sigma_device (clean axis)
- H3b: Holonomy non-zero even at low decoherence
- H3c: Both fail when n_samples < 500 and d_features > 50

### H4 (Sheaf) — H4d (correctness gate), rest (E)
- H4a: H^1 = 0 for single-device data at all decoherence levels
- H4b: H^1 > 0 above a specific multi-device threshold
- H4c: Sheaf Q identifies which device is most inconsistent
- **H4d (correctness gate)**: On scalar stalks, sheaf H^1 = Cochran's Q (known analytic reduction — pass/fail, no α needed)

### H5 (Persistent homology) — all (E), restricted to H0
- H5a: **WITHDRAWN** (requires correct H1, which we cut)
- H5b: Wasserstein persistence distance between class diagrams correlates with decoherence
- H5c: **WITHDRAWN** (refers to imaging data not in NV-diamond simulation)

### H6 (Quantum-specific) — H6d (C), rest (E)
- H6a: QFI identifies frequency (temperature) as most robust feature
- H6b: Berry phase unique detector of anisotropic decoherence
- H6c: **WITHDRAWN** (Chern number method 24 cut due to correctness concerns)
- **H6d (C)**: Von Neumann entropy tracks decoherence monotonically (|Spearman ρ| > 0.8)

### H7 (Multi-sensor integration) — all (E)
- H7a: TransportKit fusion best single-number multi-sensor score
- H7b: Persistent sheaf discovers thresholds invisible to single-sensor methods
- H7c: Cross-sensor consistency more fragile than within-sensor robustness

### H8 (Baselines) — H8a (C), rest (E)
- **H8a (C)**: LASSO beats all geometric methods at low decoherence
- H8b: IRM beats bracket norms when many regimes in training set
- H8c: No scalar baseline matches sheaf H^1 for multi-sensor inconsistency

### H9 (Computational) — all (E)
- H9a: Bracket family < 1s per feature on CPU
- H9b: Persistent homology is the computational bottleneck
- H9c: Persistent sheaf cohomology most expensive overall

### H10 (Meta) — (E)
- Boundary conditions predictable from method mathematical structure

## Reporting Commitment

1. All results reported regardless of whether hypotheses are confirmed or refuted
2. No post-hoc addition of methods or modification of hypotheses
3. **Confirmatory results** reported with bootstrap CI and Bonferroni-corrected α
4. **Exploratory results** reported descriptively with CI, explicitly labeled exploratory
5. Any exploratory finding flagged as requiring independent replication
6. Null results reported with equal prominence
7. Withdrawn hypotheses (H5a, H5c, H6c) documented with reasons
8. Raw JSON results (all 30,000 evaluations) committed alongside paper

## Review Gate

**This document and all frozen code must be reviewed by Perplexity (or equivalent
external review) BEFORE the runner is executed with production parameters.**

Reviewer should check:
- Simulation physics plausibility (NV-diamond parameters)
- Method implementations match mathematical descriptions
- Baseline implementations are fair comparisons (same data, comparable evaluation)
- Hypotheses are falsifiable and specific
- Statistical analysis plan is sound (replication, CIs, confirmatory/exploratory split)
- No data leakage or circular reasoning in the pipeline
- Composite axes (alpha_T2, sigma_surface) correctly documented
