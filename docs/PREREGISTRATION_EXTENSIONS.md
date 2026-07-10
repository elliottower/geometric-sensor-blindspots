# Pre-Registration Extensions: Three New Axes

**Status**: FROZEN before any extension experiments run.
**Parent pre-registration**: `docs/PREREGISTRATION.md` (SHA `8401d54`)
**Extension freeze SHA (v1)**: `567c3bc` (2026-07-09)
**Amended freeze SHA (v2)**: `e31a495`
**Amendment date**: 2026-07-09

These extensions add three new experimental axes to the original 5-axis
design. The original pre-registration, methods, baselines, and analysis
pipeline are unchanged. New code is added only to the simulator
(`nv_diamond.py`) and runner; no existing frozen methods are modified.

## Amendment log

| # | Date | What changed | Why |
|---|------|------|-----|
| A1 | 2026-07-09 | H_gain prediction flipped; pilot disclosure added | Smoke test (2 seeds, n_per_class=30) revealed geodesic stays flat under gain — consistent with mechanism (scalar gain preserves subspace angle). Original prediction was backwards. |
| A2 | 2026-07-09 | Alpha bookkeeping: extensions declared as separate confirmatory family | Parent used 0.05/3; extensions are independent experiments, not additions to the original family. |
| A3 | 2026-07-09 | H_dT criterion tightened from rho-vs-AUC to paired effect-size comparison | Original criterion was trivially satisfiable (rho is scale-free, so bracket always tracks). |
| A4 | 2026-07-09 | Gain applied only to amplitude features, not resonance frequency | Resonance frequency is set by crystal field D(T), not collection optics. |
| A5 | 2026-07-09 | H_gain downgraded from confirmatory to exploratory | Pilot data informed prediction direction; exploratory status removes ambiguity. |

---

## Extension 2: Multiplicative device gain (`gain_device`)

### Rationale

The original `sigma_device` axis models calibration differences as additive
temperature offsets. Real device-to-device variation also includes
multiplicative effects (microwave delivery efficiency, fluorescence
collection, NV orientation distribution) that scale signal amplitudes.

Per-device scalar gain g_d rescales feature vectors without changing their
direction in feature space. Subspace-angle methods (Grassmannian geodesic,
Procrustes, CKA) are therefore invariant to this transformation for the
same reason they are invariant to additive offsets — both preserve the
geometric structure these methods measure. This extension tests whether the
invariance-blindness principle extends beyond translation to include scale
transformations.

### Pilot disclosure

A smoke test (2 seeds, n_per_class=30, n_devices=2, gain applied to ALL
features including frequency) was run before the hypothesis was finalized.
Results: geodesic rho = -0.01, CKA = -0.33, Procrustes = -0.12,
Berry = 0.00. These pilot results informed the direction of H_gain below.
The full experiment differs from the pilot in: (a) 100 seeds vs 2,
(b) full-size data (n_per_class=200, n_devices=3), (c) gain applied only
to amplitude features (frequency excluded). The pilot data are discarded
and not included in any analysis.

### Design

- New axis `gain_device`: per-device multiplicative gain factor
  g_d ~ N(1, sigma_gain), applied to amplitude-dependent features only
  (T2, linewidth, contrast, T1 — NOT resonance frequency, which is set
  by crystal field splitting and is independent of collection optics).
- sigma_gain in [0, 0.15], 10 linearly spaced levels.
- 100 seeds (base_seed 1100, seeds 1100-1199).
- All other decoherence parameters at baseline (zero).
- 30 methods x 10 levels x 100 seeds = 30,000 evaluations.

### Hypothesis H_gain (EXPLORATORY)

Subspace-angle methods (Grassmannian geodesic, CKA, Procrustes, Berry
phase) achieve |rho| < 0.15 on `gain_device`, demonstrating that
the invariance-blindness principle extends from translation invariance
to scale invariance: these methods are blind to per-device transformations
that preserve feature-vector direction.

**Falsifier**: If Grassmannian geodesic achieves |rho| > 0.5 on
`gain_device` (responsive, unlike on additive `sigma_device`), then
scale transformations break subspace geometry in ways additive offsets
do not, and the invariance-blindness principle is specific to translation.

**Why exploratory**: A 2-seed pilot was run before this hypothesis was
finalized (see pilot disclosure above). Although the pilot data are
discarded and the full experiment differs in three ways, the prediction
direction was informed by pilot observation. Exploratory status removes
any ambiguity about confirmatory integrity. The mechanism argument
(scalar gain preserves subspace angle) carries the scientific claim
independently of the statistical label.

## Extension 3: Smaller temperature difference (`delta_T` sweep)

### Rationale

The current simulation uses Delta_T = 1.5 K, at the upper end of
reported intracellular temperature gradients. At this signal level,
all classifiers achieve AUC = 1.0 on all levels of sigma_device and
sigma_surface, making it impossible to identify regimes where geometric
methods offer earlier detection than classification. A Delta_T sweep
finds the crossover.

### Design

- Sweep Delta_T in {0.2, 0.3, 0.5, 0.75, 1.0, 1.5} K.
- For each Delta_T, sweep alpha_T2 only (10 levels), 100 seeds.
- 30 methods x 10 levels x 6 Delta_T values x 100 seeds = 180,000 evals.
- base_seed 1200, seeds 1200-1299.

### Hypothesis H_dT (CONFIRMATORY, alpha = 0.025)

There exists Delta_T* < 0.5 K at which the following two conditions
hold simultaneously at maximum alpha_T2 = 1.0:

1. **Classifier degradation**: Median LASSO AUC across seeds drops
   below 0.6 (near chance), confirming the classification signal is
   effectively destroyed.

2. **Geometric discrimination**: The paired difference in bracket norm
   raw scores between alpha_T2 = 0.0 and alpha_T2 = 1.0
   (Delta_bracket = score(0.0) - score(1.0)) remains positive with
   Cohen's d > 0.5 (medium effect) across seeds, confirming geometric
   methods still discriminate between intact and degraded structure
   with non-trivial effect size.

This tests whether geometric methods produce actionable signal (not
just monotonic tracking) at signal levels where classifiers fail.

**Falsifier**: If no such Delta_T* exists — either classifiers never
degrade below 0.6 even at Delta_T = 0.2 K, or bracket norm's paired
effect size drops below d = 0.5 whenever classifiers degrade — then
geometric methods offer no discriminative advantage below the
classification threshold in this simulation.

**Why alpha = 0.025**: Same family as H_gain (see above).

## Extension 4: Axis interactions (alpha_T2 x sigma_device)

### Rationale

The original one-axis-at-a-time design does not test whether the
sensitivity map composes linearly. In real NV measurements, T2
degradation and device variation co-occur. This extension tests whether
bracket norm's sigma_device signal is suppressed when alpha_T2
collapses cluster structure.

### Design

- 2D grid: alpha_T2 x sigma_device, 6 x 6 levels.
- alpha_T2 in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- sigma_device in {0.0, 0.02, 0.04, 0.06, 0.08, 0.1}
- 50 seeds per cell (base_seed 1300, seeds 1300-1349).
- 30 methods x 36 cells x 50 seeds = 54,000 evaluations.

### Hypothesis H_int (EXPLORATORY)

The joint response is NOT additive: bracket norm raw's sigma_device
sensitivity (partial rho) decreases as alpha_T2 increases beyond the
curvature collapse threshold (~0.4). The interaction is expected because
alpha_T2 destroys cluster structure, leaving less structure for
device offsets to perturb.

This hypothesis is exploratory (no alpha correction) because the
expected direction of interaction is uncertain — alpha_T2 might
increase rather than decrease device sensitivity by spreading out
class centroids.

## Analysis

All extensions use the same Spearman rho / bootstrap CI / median
framework from the parent pre-registration. Extension 2 and 3 results
are reported in the same sensitivity-map format. Extension 4 reports
partial correlations and interaction residuals.

## Confirmatory family structure

| Family | Tests | Alpha per test | Source |
|--------|-------|---------------|--------|
| Parent (original 5-axis) | H3a (alpha_T2), H3a (sigma_device), H6d | 0.05/3 = 0.0167 | PREREGISTRATION.md |
| Extensions | H_dT | 0.05/1 = 0.05 (conservatively kept at 0.025) | This document |

H_gain and H_int are exploratory and excluded from confirmatory
families. H_dT is the sole confirmatory extension test; alpha is
conservatively kept at 0.025 rather than relaxed to 0.05.
