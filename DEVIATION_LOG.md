# Deviation Log

Pre-run corrections to gating experiment H7b/H7c. All changes made
before any data was generated.

## DEV-001: Fix broken permutation null (2026-07-08)

**Original SHA**: 2a9b023
**Severity**: Correctness — experiment cannot produce valid results

**Bug**: `multi_sensor_holonomy` permuted rows within each sensor,
then recomputed PCA subspaces. Row permutation does not change the
covariance matrix, so PCA returns identical subspaces every time.
The null distribution collapses to a single point (empirically
verified: null std = 0.000000). The p-value is always ~1.0, making
H7b impossible to confirm — the opposite of what a correct test
would show.

**Fix**: Replace with pool-and-resplit null. Pool all sensors' data,
randomly split into 3 groups of the same sizes, compute holonomy on
each split. This destroys sensor identity while preserving the overall
data distribution, producing a genuine null with nonzero variance.

## DEV-002: Fix dimension mismatch crash (2026-07-08)

**Original SHA**: 2a9b023
**Severity**: Crash — experiment cannot run at all

**Bug**: The three sensors had different feature dimensions
(NV_D=100, PHOTON_D=80, SPIN_D=40). `cocycle_holonomy` requires
`U_from.T @ U_to` which needs same ambient dimension. Verified:
numpy raises `ValueError: matmul: Input operand 1 has a mismatch`.

**Fix**: Redesign as three NV-diamond setups in the same D=100
feature space, each with a different decoherence mechanism
(T2 decay, surface noise, temperature drift). This preserves the
cross-sensor inconsistency hypothesis while making the geometry
well-defined. The multi-modality claim moves to a later experiment.

## DEV-003: Add epsilon=0 baseline reporting (2026-07-08)

**Original SHA**: 2a9b023
**Severity**: Design — result may be uninterpretable

**Issue**: Holonomy may be nonzero at epsilon=0 simply because the
three sensors have different noise characteristics. Reporting only
absolute holonomy conflates "sensors are different" with
"decoherence broke consistency."

**Fix**: Report `holonomy_delta = H(eps) - H(0)` alongside absolute
holonomy. Both the absolute p-value and the delta trajectory are
recorded.

## DEV-004: Multi-seed (2026-07-08)

**Original SHA**: 2a9b023
**Severity**: Design — single-seed binary decision is fragile

**Fix**: Run 5 seeds (20260708-20260712). Report per-seed results
and aggregate confirmation rate.

## DEV-005: Naming correction (2026-07-08)

**Original SHA**: 2a9b023
**Severity**: Terminology

**Issue**: Docstrings said "sheaf H^1" but the code computes
Grassmannian cocycle holonomy (method 9 in the survey, not method 10).
These are related but distinct mathematical objects.

**Fix**: All references now say "cocycle holonomy."

## DEV-006: Redesign generators — noise heterogeneity vs genuine inconsistency (2026-07-08)

**Original SHA**: cb0c8d8 (v2)
**Severity**: Construct validity — experiment measures wrong thing

**Issue**: v2 generators all called `_shared_base` with identical
signal structure, differing only in additive noise scaling. At eps=0
all three sensors were the same distribution, so the "cross-sensor
inconsistency" detected by holonomy was purely noise heterogeneity
(different variance patterns), not genuine disagreement about the
biology. The pool-and-resplit null also keys on distributional
differences, so it fires on noise-level divergence rather than signal
inconsistency. H7b would be "confirmed" for a baked-in reason.

**Fix**: v3 uses a latent-projection model. A shared k-dim latent z
(with class signal) is projected through sensor-specific embeddings
U_s(eps) = QR(U_base + eps * scale * P_s), where P_s is a random
perturbation matrix unique to each sensor. At eps=0, all sensors
share U_base (no inconsistency). At eps>0, each sensor's measurement
basis tilts in a different direction, creating genuine geometric
inconsistency that holonomy can detect.

The null is now a bootstrap of the eps=0 (consistent) case, testing
"does holonomy exceed the sampling-jitter baseline?"

Negative control test added: three sensors with the SAME perturbation
(P_s identical) must return H7b=False.

## DEV-007: Signal calibration (2026-07-08)

**Original SHA**: cb0c8d8 (v2)
**Severity**: Calibration — AUC=1.0 everywhere

**Issue**: v1 and v2 NV-diamond physics model had frequency signal
111 kHz vs noise ~200 Hz (SNR > 500). AUC was 1.0 at all
decoherence levels. The gating experiment cannot function if AUC
never degrades.

**Fix**: v3 uses a controlled signal model (LATENT_SIGNAL=5.0,
OBS_NOISE_BASE=1.0, OBS_NOISE_RATE=3.5) calibrated so AUC starts
~0.97 at eps=0 and crosses 0.75 around eps=0.55. This gives a clear
window where holonomy can fire before AUC drops.

## DEV-008: Givens rotation too weak in high D (2026-07-08)

**Issue**: Givens rotation in a single 2D plane out of D=100
dimensions moves the k-dim subspace by ~theta/sqrt(D), requiring
impractically large angles to create detectable holonomy.

**Fix**: Replace with additive perturbation + QR re-orthonormalization.
Random D x k perturbation matrices move the subspace in many
dimensions simultaneously, creating detectable holonomy at reasonable
perturbation scales.

## DEV-009: v3 result reinterpretation — 0/5 seeds had monotone loop holonomy (2026-07-08)

**Original SHA**: 884b462 (v3)
**Severity**: Interpretation — the reported 3/5 H7c confirmation rate
was substantively 0/5

**Finding**: Post-hoc diagnostic analysis (pure analysis of existing
v3 rows, no parameter changes) revealed that cocycle holonomy around
the 3-sensor loop is genuinely non-monotonic in epsilon. Spearman
rank correlation of loop holonomy vs epsilon across all 5 seeds:

    Seed 20260708 [v3 FAIL]: rho = +0.30, p = 0.207
    Seed 20260709 [v3 FAIL]: rho = +0.36, p = 0.123
    Seed 20260710 [v3 PASS]: rho = +0.30, p = 0.205
    Seed 20260711 [v3 PASS]: rho = +0.40, p = 0.083
    Seed 20260712 [v3 PASS]: rho = +0.01, p = 0.960

No seed achieves p < 0.05 for monotonicity. The v3 "3/5 pass" and
"2/5 fail" distinction was entirely determined by where stochastic
plateau-noise spikes happened to land relative to the AUC degradation
threshold — not by genuine early detection of cross-sensor
inconsistency.

**Retraction of prior interpretation**: The claim made during initial
v3 analysis that "seed 20260712 genuinely caught the initial rise at
eps=0.105" is explicitly retracted. Seed 20260712 has loop holonomy
rho = 0.01 (zero monotonicity, p = 0.96), meaning its eps_crit_hol =
0.105 was a noise spike, not signal detection. This was the seed
previously identified as the strongest H7c confirmation.

**Mechanism — loop cancellation originally claimed, now qualified**:
Pairwise geodesic distances between individual sensor pairs show
significantly higher monotonicity than the loop holonomy in the same
seeds:

    Edge 0→1: rho = 0.38, 0.57, 0.58, 0.48, 0.63 (significant in 4/5)
    Edge 1→2: rho = 0.28, 0.52, 0.49, 0.47, 0.58 (significant in 4/5)
    Edge 2→0: rho = 0.44, 0.58, 0.15, 0.21, 0.20 (significant in 2/5)

The per-edge subspace distances do increase with epsilon (subspaces
genuinely move apart), but the composed transport around the 3-sensor
loop introduces cancellations that destroy the monotone signal.
This is confirmed by the divergence: edges are monotone (rho ~ 0.5),
the loop built from those same edges is not (rho ~ 0.0-0.4).

**UPDATE (see DEV-010)**: This cancellation mechanism claim is
confounded. A diagnostic (diagnostic_finding_a_confound.py) showed
that loop holonomy saturates with comparable range (~2.7) under
identical perturbation (no cancellation possible), and is actually
MORE monotone (mean rho=0.53) than under different perturbation
(mean rho=0.27). The edge monotonicity in both conditions is driven
primarily by the noise ramp (obs_noise = 1.0 + 3.5*eps), not by
cross-sensor inconsistency. The cancellation mechanism requires
re-examination under noise/perturbation separation (B2, V4).

A power analysis (diagnostic_power_holonomy.py) over N in {200, 500,
1000, 2000} confirmed this is World 2 (genuinely non-monotone metric),
not World 1 (noisy estimator of a monotone signal): band widths did
not narrow with increasing N (1.75 → 1.58 for seed 08, 1.72 → 1.67
for seed 12), and Spearman rho plateaued well below +1.

**Consequence**: Loop cocycle holonomy (method 9) is a saturating
onset detector, not a monotone severity tracker. It cannot support
H7c ("holonomy fires before AUC drops") as a robust claim because
the first-crossing statistic on a non-monotone, saturating signal is
noise-dominated. The original v3 H7b/H7c framework is retired.

**Corrective action**: PREREGISTRATION_V3.md reframed the multi-sensor
claim using per-edge geodesic Δρ as the primary endpoint. However, the
B1 negative control revealed the single-epsilon confound (DEV-010),
requiring a further redesign to PREREGISTRATION_V4.md with orthogonal
noise/perturbation axes.

**Diagnostic artifacts** (not part of the frozen experiment):
- experiments/diagnostic_power_holonomy.py — World 1 vs World 2 analysis
- experiments/results/diagnostic_power_holonomy.json
- experiments/results/diagnostic_power_holonomy.png
- experiments/results/diagnostic_pairwise_vs_loop.png


## DEV-010: Single-epsilon confound — noise and perturbation are entangled (2026-07-08)

**Original SHA**: 884b462 (v3 data model, inherited by B1)
**Severity**: Design — all v3-derived metrics track noise scaling,
not cross-sensor inconsistency

**Finding**: The v3 data model uses a single parameter `eps` to drive
both observation noise (`obs_noise = OBS_NOISE_BASE + OBS_NOISE_RATE *
eps`) and cross-sensor embedding perturbation (`U_s = QR(U_base +
eps * PERTURBATION_SCALE * P_s)`). This creates a confound: as `eps`
increases, noise and perturbation increase together, making it
impossible to attribute metric changes to either cause independently.

**Evidence from B1 negative control**: The B1 experiment
(PREREGISTRATION_V3.md) tested Δρ = ρ_edge − ρ_loop as the primary
endpoint. The negative control (identical perturbation, no
inconsistency) was expected to produce Δρ ≈ 0. Instead:

    Identical perturbation: Δρ ≈ +0.60 (ρ_edge ≈ 0.96, ρ_loop ≈ 0.36)
    Different perturbation: Δρ ≈ +0.35 (ρ_edge ≈ 0.41, ρ_loop ≈ 0.06)

Δρ is LARGER without inconsistency because the edge metric tracks
noise scaling (σ = 1.0 + 3.5ε) almost perfectly (ρ ≈ 0.96), while
cross-sensor inconsistency disrupts this clean noise-scaling
monotonicity (ρ drops to ≈ 0.41).

**Evidence from Finding A re-examination**: A diagnostic
(diagnostic_finding_a_confound.py) on the 5 original v3 seeds showed:

    Identical perturbation: mean ρ_hol = +0.527, range 2.74
    Different perturbation: mean ρ_hol = +0.271, range 2.62

Loop holonomy saturates with similar amplitude in both conditions.
Under identical perturbation, 3/5 seeds had individually significant
ρ_hol (p < 0.05), compared to 0/5 under different perturbation. The
loop saturation reported in DEV-009 is at least partly a noise
artifact, not purely cross-sensor cancellation.

**Root cause**: The noise ramp dominates both the edge and loop
metrics. At eps_pert=0 (no perturbation), observation noise alone
produces perfectly monotone geodesics (ρ = 1.0, verified in
test_gating_b2.py::test_noise_axis_produces_monotone_geodesic_confound).
The perturbation signal is secondary to the noise signal.

**Consequence**: All three versions of the gating experiment (v3,
B1 Δρ, B1 slope) are confounded. The v3 H7c first-crossing, the B1
Δρ differential, and the B1 onset slope all measure noise scaling
sensitivity, not inconsistency detection. No endpoint derived from
the single-epsilon data model can cleanly test the multi-sensor claim.

**Fix**: PREREGISTRATION_V4.md splits the single `eps` into two
orthogonal axes: `eps_pert` (perturbation only) and `obs_noise`
(noise only). The B2 experiment sweeps `eps_pert` at each of several
fixed noise levels, making the noise component constant within each
sweep. The negative control (identical perturbation at fixed noise)
is validated to produce ρ_geo ≈ 0 in the test suite.

**Diagnostic artifacts**:
- experiments/diagnostic_finding_a_confound.py
- experiments/results/diagnostic_finding_a_confound.json
- experiments/gating_b2_orthogonal.py
- tests/test_gating_b2.py (11/11 pass)
