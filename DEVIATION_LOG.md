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
