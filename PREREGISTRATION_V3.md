# Pre-Registration v3: Edge-Level vs Loop-Level Monotonicity

Supersedes PREREGISTRATION_V2.md (SHA 884b462). Motivated by
DEV-009: loop cocycle holonomy is genuinely non-monotone (0/5 seeds
passed Spearman monotonicity test), while per-edge pairwise geodesic
distances are significantly monotone in most seeds. The v3 H7b/H7c
framework is retired. This preregistration tests whether the edge-level
metric is a reliably better inconsistency tracker than the loop metric.

## Hypothesis

**B1 (edge-over-loop monotonicity):** Pairwise geodesic distance
between sensor subspaces tracks cross-sensor inconsistency more
monotonically than cocycle holonomy around the sensor loop. The
differential Spearman correlation Δρ = ρ(mean-edge-geodesic, ε) −
ρ(loop-holonomy, ε) is positive in the median across held-out seeds.

## Data model

Identical to v3 (gating_h7b_v3.py, SHA 884b462): shared latent z in
R^k (k=3, LATENT_SIGNAL=5.0), sensor-specific embeddings
U_s(eps) = QR(U_base + eps * PERTURBATION_SCALE * P_s), observation
noise sigma(eps) = OBS_NOISE_BASE + OBS_NOISE_RATE * eps. D=100,
PCA_K=3. All frozen parameters carried forward unchanged.

## What re-randomizes per seed

Each of the 50 seeds draws fresh:
- Latent data z, y (via data_rng = seed)
- Base embedding U_base (via proj_rng = seed + 1000)
- Sensor perturbation matrices P_0, P_1, P_2 (via pert_rng = seed + 3000)
- Observation noise at each epsilon step (via obs_rng = seed * 100 + step)

The sensor geometry (U_base and all P_s) varies across seeds. This
is critical because the diagnostic (DEV-009) showed edge 2→0
monotonicity is geometry-dependent (significant in 2/5 seeds, not
in 3/5). The 50-seed CI must reflect this geometry variability.

## Primary endpoint

**Statistic:** For each seed, compute over the full epsilon range
[0, 1] (20 steps):
- ρ_edge = Spearman(mean-of-three-pairwise-geodesics, ε)
- ρ_loop = Spearman(loop-holonomy-norm, ε)
- Δρ = ρ_edge − ρ_loop

**Edge aggregation (pre-committed):** Mean of all three pairwise
geodesic distances (edge 0→1, edge 1→2, edge 2→0) at each epsilon.
Not "best two" or "most monotone" — the mean of three is frozen.

**Decision rule:** Compute Δρ for each of 50 seeds. Report the
median Δρ and its bootstrap 95% CI (10000 resamples of the 50-seed
set). B1 is confirmed if the CI excludes 0. Significance level
α = 0.025 (Bonferroni-corrected, see Multiple Testing below).

**Interpretation:** A positive median Δρ with CI excluding 0 means
per-edge geodesic distance tracks decoherence-induced inconsistency
more monotonically than the loop holonomy, confirming that loop
cancellation degrades the loop invariant's sensitivity.

## Secondary endpoint

**Onset-regime severity slope:** For each seed, compute OLS slope of
mean-of-three-geodesics vs ε over the first 5 epsilon steps
(EPSILON_RANGE[:5], i.e. indices 0–4: 0.000, 0.053, 0.105, 0.158,
0.211). Selection is by index, not by float threshold, to avoid
off-by-one from floating-point comparison. Report the median slope
and its bootstrap 95% CI across 50 seeds.

This characterizes the magnitude of edge-level sensitivity in the
onset regime where the diagnostic showed the clearest signal. The
window ε ∈ [0, 0.21] was learned from the diagnostic seeds (08, 12)
and is frozen here — it is not a test of whether this window is
"correct," but a descriptor of how large the effect is where it was
observed.

**Per-seed full-range Spearman:** Report ρ_edge per seed over
ε ∈ [0, 1]. Apply Benjamini-Hochberg FDR correction at α = 0.05
across the 50 tests. This characterizes how often the edge metric
is individually significant, but it is exploratory and does not
gate any decision.

## Negative control

**Design:** Three sensors with identical perturbation (P_0 = P_1 =
P_2 = same random matrix). At all epsilon values, the three sensors
have identical embeddings U_s(eps), so there is no cross-sensor
inconsistency — only shared decoherence and observation noise.

**Statistic:** Compute Δρ per seed exactly as the primary endpoint.

**Criterion:** The median Δρ bootstrap 95% CI must *include* 0.
If the negative control produces a significant Δρ, the test is
miscalibrated and B1 cannot be interpreted. This is a hard gate.

The negative control is run on 50 seeds (20260800–20260849) separate
from the primary seeds.

## Mechanism-consistency check

**Not a positive control.** This check verifies that the diagnostic
finding (loop cancellation) replicates on held-out data.

**Prediction:** Across the 50 primary seeds, the per-seed ρ_loop
distribution should have median not significantly different from 0
(bootstrap CI includes 0), while ρ_edge should have median > 0.
If ρ_loop turns out significantly positive (the loop IS monotone
on fresh seeds), the cancellation mechanism is not confirmed and
the motivation for B is undermined. Report honestly either way.

## Seeds

**Primary:** 50 seeds, 20260750–20260799.
**Negative control:** 50 seeds, 20260800–20260849.

None of these seeds have been generated. The diagnostic analysis
used only seeds 20260708 and 20260712 (from the v3 experiment).
Seeds 20260709, 20260710, 20260711 were used in v3 but not in the
diagnostic that motivated this preregistration.

## Multiple testing

The primary endpoint (median Δρ CI excludes 0) is tested at α = 0.05.
It is a single confirmatory test — no correction needed.

The negative control (median Δρ CI includes 0) is a calibration gate,
not a co-equal hypothesis. It is evaluated at its own α = 0.05. We do
not Bonferroni-correct the primary against the gate because the gate
is a pass/fail check on the test's calibration, not a finding that
competes for false-positive budget. If the gate fails, the primary is
uninterpretable regardless of its p-value.

The secondary per-seed Spearman tests (50 tests) are FDR-corrected
separately via Benjamini-Hochberg at α = 0.05.

## Outcomes

- **B1 confirmed** (primary CI excludes 0, negative control CI
  includes 0, mechanism check consistent): Per-edge geodesic is a
  validated monotone severity metric where loop holonomy is not.
  Proceed with edge-level metric in the multi-sensor method grid.
  Report the onset slope (secondary) as the effect size descriptor.

- **B1 confirmed but mechanism check fails** (primary significant
  but loop ρ is also significant on fresh seeds): The edge metric
  works but the cancellation explanation is wrong. Revise the
  mechanism story; the edge metric may be better for a different
  reason than hypothesized.

- **B1 fails** (primary CI includes 0): Edge-level geodesic is not
  reliably more monotone than loop holonomy across geometries. The
  multi-sensor inconsistency claim cannot be supported by either
  metric. Pivot to single-sensor boundary conditions.

- **Negative control fails** (negative control CI excludes 0):
  The Δρ statistic is miscalibrated. Halt and diagnose before
  interpreting the primary result.

## Frozen parameters

All data-generation parameters are inherited from gating_h7b_v3.py
(SHA 884b462) without modification. The only changes are the test
statistic (Δρ replacing first-crossing), the seed range, and the
sample size (50 seeds replacing 5).

N_SAMPLES_PER_CLASS = 200 is carried forward. The diagnostic power
analysis showed that increasing N does not resolve the loop
non-monotonicity (World 2), so the issue is the metric, not the
sample size. N=200 is sufficient for computing per-seed Spearman
correlations over 20 epsilon steps.

## Provenance

- **Diagnostic seeds used to design this preregistration:**
  20260708, 20260712 (power analysis), plus 20260709–20260711
  (pairwise-vs-loop plot on existing v3 rows only).
- **Diagnostic artifacts:**
  experiments/diagnostic_power_holonomy.py,
  experiments/results/diagnostic_power_holonomy.json,
  experiments/results/diagnostic_power_holonomy.png,
  experiments/results/diagnostic_pairwise_vs_loop.png
- **Deviation log:** DEV-009 in DEVIATION_LOG.md documents the
  0/5 monotonicity finding and the retracted seed-20260712 claim.
