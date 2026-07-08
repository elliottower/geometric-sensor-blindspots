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
