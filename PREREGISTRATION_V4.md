# Pre-Registration v4: Orthogonal Noise/Perturbation Axes

Supersedes PREREGISTRATION_V3.md. Motivated by the discovery that the
v3 data model confounds observation noise and cross-sensor perturbation
through a single epsilon parameter, making all downstream metrics
(geodesic, holonomy) primarily track noise scaling rather than
cross-sensor inconsistency. The B1 negative control correctly caught
this: identical-perturbation sensors had Δρ ≈ 0.6 (larger than
different-perturbation Δρ ≈ 0.35), because the edge metric tracks
noise scaling more monotonically than the loop metric regardless of
cross-sensor inconsistency.

A follow-up diagnostic (diagnostic_finding_a_confound.py) confirmed
that Finding A from DEV-009 (loop holonomy saturation attributed to
cross-sensor cancellation) is also partly confounded: loop holonomy
shows comparable saturation range (~2.7) under identical perturbation
(no cancellation possible) as under different perturbation. The
cancellation mechanism claim requires re-examination under noise/
perturbation separation.

## Design change

The single `eps` parameter is split into two orthogonal axes:

- **`eps_pert`** controls cross-sensor embedding divergence only:
  `U_s(eps_pert) = QR(U_base + eps_pert * PERTURBATION_SCALE * P_s)`
- **`obs_noise`** controls observation noise only:
  `X = z @ U_s.T + N(0, obs_noise)`

The primary endpoint sweeps `eps_pert` at each of several **fixed**
noise levels, so noise-scaling effects are constant within each sweep
and cannot produce spurious monotonicity.

**Measurement-level interaction (pre-registered):** The generative
model axes are independent (verified:
test_generative_model_axes_independent). However, the geodesic
*measurement* has a known interaction: noise degrades PCA subspace
estimation, which attenuates the perturbation signal. At fixed
eps_pert=0.5, the pert-attributable geodesic component (geo_diff −
geo_ident) drops from ~1.0 at σ=1.0 to ~0.0 at σ=4.5 (verified:
test_noise_attenuates_pert_signal). This is expected physics (noise
masks signal), not a generative-model confound. The noise grid below
characterizes this attenuation.

## Hypothesis

**B2 (perturbation-axis geodesic monotonicity):** At fixed observation
noise, pairwise geodesic distance between sensor subspaces increases
monotonically with cross-sensor perturbation magnitude (eps_pert).
The Spearman correlation ρ(mean-edge-geodesic, eps_pert) is positive
in the median across held-out seeds, at each tested noise level.

**Noise-degradation prediction:** Detection strength (median ρ_geo)
decreases with noise level. The degradation curve — median ρ_geo as
a function of σ — characterizes the operating regime boundary.

## Data model

Generators are defined in `gating_b2_orthogonal.py`. Shared parameters
with v3:

    D = 100
    K_LATENT = 3
    LATENT_SIGNAL = 5.0
    PERTURBATION_SCALE = 1.5
    PCA_K = 3
    N_SAMPLES_PER_CLASS = 200

New parameters:

    FIXED_NOISE_LEVELS = [1.0, 2.0, 3.0, 4.5]
    N_EPS_STEPS = 20
    EPS_PERT_RANGE = linspace(0.0, 1.0, 20)
    NOISE_SWEEP_LEVELS = linspace(1.0, 4.5, 20)

The noise levels span the range that the v3 single-epsilon model
traversed (OBS_NOISE_BASE=1.0 to OBS_NOISE_BASE+OBS_NOISE_RATE=4.5).
All four noise levels are frozen from the sanity seeds (20260850–
20260859, 10 seeds) and tested on fresh confirmatory seeds.

**PERTURBATION_SCALE = 1.5 provenance:** Carried forward unchanged
from v3 (SHA 884b462), where it was tuned under the confounded
single-epsilon model. The scale sets the units of the perturbation
axis but does not interact with the noise axis in the generative
model. It was not re-tuned for B2. The degradation curve is reported
in these units.

## What re-randomizes per seed

Each seed draws fresh:
- Latent data z, y (via data_rng = seed)
- Base embedding U_base (via proj_rng = seed + 1000)
- Sensor perturbation matrices P_0, P_1, P_2 (via pert_rng = seed + 3000)
- Observation noise at each step (via obs_rng = seed * 100 + step)

## Primary endpoint

**Statistic:** For each seed, at each fixed noise level, compute over
the full eps_pert range [0, 1] (20 steps):
- ρ_geo = Spearman(mean-of-three-pairwise-geodesics, eps_pert)

**Edge aggregation (pre-committed):** Mean of all three pairwise
geodesic distances at each eps_pert. Frozen as in V3.

**Decision rule (confirmatory):** The single confirmatory test is at
**σ = 1.0** (pre-committed as the most favorable operating point):
1. Compute ρ_geo for each of 50 primary seeds at σ = 1.0
2. Report the median ρ_geo and its bootstrap 95% CI (10000 resamples,
   bootstrap RNG seed = 42 for reproducibility)
3. B2 is confirmed if the CI excludes 0

Significance level α = 0.05 for this single test. No multiplicity
correction is needed because σ = 1.0 is the only confirmatory level.

**Degradation curve (characterization):** At σ ∈ {2.0, 3.0, 4.5},
compute the same statistic (median ρ_geo with bootstrap 95% CI,
bootstrap RNG seed = 42) and report whether the CI excludes 0 at each
level. These are descriptive, not confirmatory — they characterize
the noise regime boundary. Report the full degradation curve (median
ρ_geo vs σ across all four levels). The scientific claim is: "edge
geodesic detects cross-sensor perturbation at fixed noise, with
detection strength decaying as noise rises, crossing non-significance
around σ ≈ X." This is a boundary-condition characterization; σ = 1.0
is the confirmatory anchor and the higher-noise levels map the
operating regime.

## Negative control (perturbation axis)

**Design:** Three sensors with identical perturbation (P_0 = P_1 =
P_2). At all eps_pert values, all sensors share the same embedding,
so there is no cross-sensor inconsistency. The only variation in
geodesics comes from observation noise jitter.

**Statistic:** ρ_geo per seed, computed identically to the primary.

**Criterion:** At each noise level, the median ρ_geo bootstrap 95%
CI must *include* 0. If the negative control is significant at any
noise level, the test is miscalibrated at that noise level.

The negative control is run on 50 separate seeds (20260910–20260959).

**Pre-test validation:** Verified to pass in the test suite
(test_negctrl_rho_geo_near_zero: 5 seeds at σ=2.0, mean ρ_geo near
0) and in the 10-seed sanity check (σ=1.0: median −0.002 [−0.19,
+0.12]; σ=3.0: median −0.08 [−0.24, +0.19]).

## Noise-axis positive control

**Design:** Sweep observation noise from σ=1.0 to σ=4.5 (20 steps)
at eps_pert=0 (no cross-sensor perturbation). All sensors share the
same embedding at every noise level. Geodesics measure only PCA
subspace estimation noise, which increases monotonically with σ.

**Statistic:** ρ_geo = Spearman(mean-edge-geodesic, σ) per seed.

**Prediction:** ρ_geo should be strongly positive (median > 0.8),
confirming that the nuisance axis (noise-driven geodesic increase)
is characterized and accounted for.

**Purpose:** This cell documents the nuisance rather than hiding it.
A reviewer reading only the primary result might ask "doesn't noise
also increase geodesics?" This control cell shows: yes, it does,
and that is precisely why the primary holds noise fixed. The noise
axis is reported side-by-side with the perturbation axis so the
two effects are visible and their separation is explicit.

Run on 50 seeds (20260960–20261009).

This is a reported characterization, not a gate. The noise axis is
expected to produce high ρ_geo; failure (low ρ_geo) would indicate a
problem with the geodesic metric's noise sensitivity, which would
require investigation but would not invalidate the primary endpoint.

## Mechanism re-examination (Finding A)

**Purpose:** Determine whether loop holonomy saturation (DEV-009) is
a cross-sensor cancellation phenomenon or a noise artifact.

**Design:** At each fixed noise level, report ρ_hol (Spearman of
loop-holonomy vs eps_pert) alongside ρ_geo for both the primary and
negative control conditions.

**Predictions:**
- If cancellation is real: ρ_hol should be lower for primary seeds
  (different perturbation creates cancellation) than for negative
  control seeds (identical perturbation, no cancellation). Both
  should have low ρ_hol, but the primary should be lower.
- If saturation was purely a noise artifact: ρ_hol should be
  near-zero in both conditions at fixed noise (no noise ramp to
  drive either).

The diagnostic_finding_a_confound.py results (under the old v3
single-epsilon model) showed ρ_hol ≈ 0.53 for identical perturbation
vs ρ_hol ≈ 0.27 for different perturbation — but those were
confounded by simultaneous noise scaling. Under the two-axis design
(fixed noise), the noise component is constant, so any remaining
ρ_hol difference between conditions isolates the cancellation effect.

This is exploratory, not a gating criterion. Finding A's
interpretation will be updated based on the results regardless of
whether B2 passes.

## Loop holonomy secondary comparison

At each noise level, report the median ρ_hol for primary seeds
alongside the median ρ_geo. If ρ_hol remains near zero while ρ_geo
is significantly positive, this confirms the edge metric's advantage
is specific to the perturbation axis, not a generic property of the
metric's noise behavior. This is the properly un-confounded version
of the B1 claim, and it is now defensible because the noise-axis
confound has been removed.

## Seeds

**Sanity (already used for parameter selection):** 10 seeds,
20260850–20260859. Used to verify the signal exists and to select
the noise grid values. NOT used in the confirmatory run.

**Primary:** 50 seeds, 20260860–20260909.
**Negative control:** 50 seeds, 20260910–20260959.
**Noise-axis control:** 50 seeds, 20260960–20261009.

None overlap with v3 seeds (20260708–20260712), B1 seeds
(20260750–20260849), or the Finding A diagnostic seeds. The sanity
seeds (20260850–59) are excluded from the confirmatory analysis.

## Secondary endpoints

**Per-seed Spearman significance:** Report p_geo per seed at each
noise level. Apply Benjamini-Hochberg FDR at α = 0.05 across the
50 tests within each noise level. Report the fraction significant.
This characterizes individual-seed reliability.

**Degradation curve Spearman:** Report the Spearman correlation
between noise level and median ρ_geo across the four noise levels.
Expected to be strongly negative (detection degrades with noise).

## Outcomes

- **B2 confirmed (σ=1.0 CI excludes 0, negative control passes):**
  Per-edge geodesic detects cross-sensor perturbation at the most
  favorable noise level. The degradation curve across σ ∈ {2.0, 3.0,
  4.5} characterizes where detection strength fades, mapping the
  operating regime boundary. This is the expected outcome based on
  sanity data (ρ_geo = +0.53 at σ=1.0, +0.33 at σ=3.0).

- **B2 fails (σ=1.0 CI includes 0):** The geodesic metric does not
  reliably track cross-sensor perturbation even under the most
  favorable noise conditions. The multi-sensor inconsistency claim
  is unsupported. Pivot to single-sensor boundary conditions.

- **Negative control fails:** The test is miscalibrated. Halt and
  diagnose. Not expected given the two-axis design and pre-test
  validation.

## Frozen parameters

All parameters are defined in gating_b2_orthogonal.py. The data
generators are a direct two-axis extension of v3, with the single
`obs_noise = OBS_NOISE_BASE + OBS_NOISE_RATE * eps` replaced by
an independent `obs_noise` parameter.

## Provenance

- **v3 data model (SHA 884b462):** single-epsilon confound identified
- **B1 negative control failure:** Δρ ≈ 0.6 for identical perturbation
  (documented in session, not yet in a results JSON)
- **Finding A confound diagnostic:**
  experiments/diagnostic_finding_a_confound.py and
  experiments/results/diagnostic_finding_a_confound.json
- **Sanity data (not used in confirmatory run):**
  10-seed checks at σ=1.0 and σ=3.0 on seeds 20260850–20260859
  (primary ρ_geo = +0.53 at σ=1.0, +0.33 at σ=3.0;
  negctrl ρ_geo ≈ 0 at both levels)
- **Measurement interaction documented:**
  test_noise_attenuates_pert_signal (pert-attributable delta drops
  from ~1.0 at σ=1.0 to ~0.0 at σ=4.5)
- **B2 test suite validation:** tests/test_gating_b2.py (14/14 pass)
- **Prior deviation log:** DEV-009 + DEV-010 in DEVIATION_LOG.md
