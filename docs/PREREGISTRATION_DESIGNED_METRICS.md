# Pre-Registration: Designed Metrics for the Device-Variation Blind Spot

**Status**: FROZEN before any designed-metric experiments run.
**Parent pre-registration**: docs/PREREGISTRATION.md (SHA 8401d54)
**Extensions doc**: docs/PREREGISTRATION_EXTENSIONS.md (SHA e31a495)
**This freeze SHA (v1)**: <FILL AFTER COMMIT>
**Date**: 2026-07-10

## Motivation

The parent study established the invariance-blindness law: a geometric
method's blind spot equals its invariance group. All 30 catalogued methods
inherited their invariances accidentally from their math, and ALL achieve
|rho| < 0.15 on sigma_device (device-calibration variation) as a result.

This document pre-registers a LADDER of four *designed* metrics whose
invariances are CHOSEN, not inherited, ordered by complexity:

  L1  Invariance-audited composite (closed-form vector)
  L2  Tunable-invariance dial (analytic, one parameter lambda)
  L3  Adaptive metric-selection procedure
  DC  Device-conditional structured autoencoder metric (DC-SAM, learned)

QFI-grounded metrics are explicitly EXCLUDED from this family and
deferred to future work, because the current QFI/Berry implementations
are classical proxies applied to feature-space statistics, not direct
quantum-state measurements (see paper_v7.tex, Limitations).

## Primary hypothesis (CONFIRMATORY, one family)

H_designed: At least one designed metric in {L1, L2, DC} achieves
|rho| > 0.8 between its score and sigma_device, on data where all 30
inherited-invariance methods achieve |rho| < 0.15.

- Alpha: this is ONE confirmatory family. Single primary test at
  alpha = 0.05. The ladder members L1/L2/DC are reported as an ABLATION
  under this single hypothesis, NOT as separate headline claims.
- L3 (selection procedure) is EXPLORATORY.

**Falsifier**: If no designed metric exceeds |rho| = 0.5 on sigma_device,
the "chosen-invariance" claim fails: breaking the invariance is
insufficient to close the blind spot in this simulation.

## Specificity requirement (anti-"detect-everything" guard)

A metric sensitive to everything is useless (cf. persistent-H0
non-specificity). Every designed metric that PASSES H_designed on
sigma_device MUST ALSO satisfy the specificity criterion:

  |rho| < 0.3 on the n_centers (ensemble-size) axis, declared a priori
  as the nuisance axis the metrics should remain BLIND to.

A metric that is sensitive to sigma_device AND n_centers has universal,
not chosen, sensitivity and does NOT count as closing the blind spot.

## Interpretation rule (pre-committed)

- If L1 (closed-form) PASSES H_designed and specificity: the headline is
  "a principled closed-form decomposition closes the blind spot; learning
  is unnecessary." L1 is then the recommended method.
- If L1 FAILS but DC-SAM PASSES: the headline is "the blind spot requires
  a learned, device-conditional representation." DC-SAM is a constructive
  existence proof, not a replacement for interpretable methods.
- L2's role is confirmatory of the MECHANISM regardless of L1/DC outcome:
  the score must vary monotonically from |rho|~0 at lambda=0 (geodesic
  reproduction) to |rho|>0.8 at lambda=1.

## Validation hierarchy (pre-committed, 4 levels)

Every designed metric that passes H_designed on sigma_device is
validated through a 4-level hierarchy. A metric must pass ALL FOUR
levels to count as "closing the blind spot." This hierarchy is ported
from the pi-SAE grokking paper's evaluation framework, where IIA
alone was shown to be wildly insufficient (memorized lookup tables
scored IIA=1.0; equivariance testing separated real causal variables
at >95% from artifacts at <51%).

| Level | Name | Test | Threshold | What it catches |
|-------|------|------|-----------|-----------------|
| 1 (weakest) | Intervention effectiveness | Swap device-attributable component between samples, check if classifier prediction flips | IIA > 0.5 | Necessary but not sufficient — lookup tables pass this |
| 2 | Faithfulness | Diversity ratio rho + reconstruction MSE | rho > 0.8, MSE < tau | Exposes template-overwrite interventions (rho~0 = collapse) and hallucinated structure (high MSE + high score) |
| 3 | Distributional quality | KL divergence + normalized logit difference of classifier post-intervention | KL < 2.0, logit_diff in [0, 2] | Continuous signal where binary IIA saturates; flags distortion even when IIA=1.0 |
| 4 (strongest) | Equivariance | Apply known device-offset group action, test whether metric's structure transforms as predicted | Equivariance accuracy > 0.8 | The single most discriminating validator — directly tests whether the invariance-blindness law holds per-method |

Level 4 (equivariance) is the PRIMARY validator for sigma_device and
sigma_gain, because both are clean group actions (additive translation,
multiplicative scaling). For axes without a natural group action
(alpha_T2, sigma_surface), validation falls back to Levels 1-3.

IIA is Level 1 only — necessary but explicitly insufficient per the
pi-SAE paper's own findings.

## Design (shared)

- Axis under test: sigma_device, 10 levels in [0, 0.1], 100 seeds.
- Specificity axis: n_centers in {5,8,10,15,20,25,30,35,40,50}, 100 seeds.
- Validation hierarchy: all 4 levels applied to every designed metric
  AND to the 30 existing methods (the existing methods should fail
  Level 4 on sigma_device — this is the invariance-blindness law
  tested causally, not just correlationally).
- DC-SAM ONLY: regenerate with n_devices=8 (rank condition on per-device
  prior means; n_devices=3 is too thin). L1/L2/L3 use existing n_devices=3
  data AND the n_devices=8 data for parity.
- Analysis: Spearman rho + bootstrap CI + median, identical to parent.

## Build / evaluation order

1. L1 (no training, ~30 lines) — the "do you even need learning" control.
2. L2 (L1 + scalar interpolation knob).
3. DC-SAM (needs n_devices=8 regen + training).
4. L3 (thin wrapper over per-axis results).

## Amendment log

| # | Date | Change | Why |
|---|------|--------|-----|
| — | 2026-07-10 | Initial freeze | — |
