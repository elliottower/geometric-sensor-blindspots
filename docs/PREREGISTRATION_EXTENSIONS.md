# Pre-Registration Extensions: Three New Axes

**Status**: FROZEN before any extension experiments run.
**Parent pre-registration**: `docs/PREREGISTRATION.md` (SHA `8401d54`)
**Extension freeze date**: 2026-07-09

These extensions add three new experimental axes to the original 5-axis
design. The original pre-registration, methods, baselines, and analysis
pipeline are unchanged. New code is added only to the simulator
(`nv_diamond.py`) and runner; no existing frozen methods are modified.

## Extension 2: Multiplicative device gain (`gain_device`)

### Rationale

The original `sigma_device` axis models calibration differences as additive
temperature offsets. Real device-to-device variation also includes
multiplicative effects (microwave delivery efficiency, fluorescence
collection, NV orientation distribution) that scale signal amplitudes.
The paper's invariance-blindness principle predicts that
translation-invariant-but-not-scale-invariant methods should "wake up"
under multiplicative noise — verifying this converts a Limitations
hand-wave into a confirmed second prediction.

### Design

- New axis `gain_device`: per-device multiplicative gain factor
  g_d ~ N(1, sigma_gain), applied to all features for that device.
- sigma_gain in [0, 0.15], 10 linearly spaced levels.
- 100 seeds (base_seed 1100, seeds 1100-1199).
- All other decoherence parameters at baseline (zero).
- 30 methods x 10 levels x 100 seeds = 30,000 evaluations.

### Hypothesis H_gain (CONFIRMATORY, alpha = 0.025)

Translation-invariant-but-not-scale-invariant methods (Grassmannian
geodesic, CKA, Procrustes, Berry phase) achieve |rho| > 0.5 on
`gain_device`, demonstrating that the invariance-blindness principle
is *specific* to translation invariance, not a general "device-blind"
property.

**Falsifier**: If Grassmannian geodesic achieves |rho| < 0.15 on
`gain_device` (flat, as on additive `sigma_device`), the
invariance-blindness principle is narrower than claimed — these methods
are device-blind in general, not just translation-blind. This is still
a clean result, but changes the scope of the claim.

**Why alpha = 0.025**: Two new confirmatory tests (H_gain, H_dT) at
Bonferroni 0.05/2 = 0.025.

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

There exists Delta_T* < 0.5 K at which LASSO AUC drops below 0.8
(median across seeds) at maximum alpha_T2 = 1.0, while bracket norm
raw maintains |rho| > 0.8 on alpha_T2 — demonstrating geometric
methods detect structural degradation below the classification threshold.

**Falsifier**: If no such Delta_T* exists (either classifiers never
degrade or bracket norm degrades first), geometric methods offer no
advantage over classification in any regime tested.

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
