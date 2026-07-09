# Deviation Log

All pre-freeze modifications to tests, methods, or hypotheses are recorded here
with the reason for the change. This log is part of the frozen pre-registration.

## Pre-Freeze Deviations

### 2026-07-09: H0 persistence known-answer test corrected

**What changed**: The initial known-answer test for `_persistence_h0` asserted
that two well-separated Gaussian clusters (σ=0.1, separation=10) would produce
exactly one long H0 bar (lifetime > 5) and all other bars below 1.0. That
assertion was incorrect: it did not account for the survivor bar.

**Why the original was wrong**: H0 persistence diagrams include "survivor"
bars — the final connected component never dies (its death is set to the max
pairwise distance in the filtration). With 100 points across two clusters,
the survivor bar has death ≈ max(D) ≈ 17, creating a second long bar. This
is correct H0 behavior, not a bug in the implementation.

**Corrected test**: Asserts that (1) at least one inter-cluster merge death
exceeds 5.0 (the gap between clusters), and (2) all within-cluster merge
deaths are below 2.0. This still locks the two-cluster gap behavior that
the test exists to verify, while correctly handling the survivor convention.

**Impact on hypotheses**: None — this change affects only the test's assertion
logic, not the method implementation or any pre-registered prediction.

### 2026-07-09: Method 24 (Chern number) cut

**What changed**: Method 24 removed from the registry; slot intentionally left
empty.

**Why**: The implementation used a naive plaquette-phase sum that is not
gauge-invariant. A correct Chern number requires the Fukui-Hatsugai-Suzuki
lattice gauge method. The broken estimator typically returned 0, which would
have spuriously "confirmed" hypothesis H6c (Chern = 0 at realistic decoherence).

**Impact on hypotheses**: H6c (Chern = 0 null result) WITHDRAWN. Cannot be
tested without a correct estimator.

### 2026-07-09: Method 15 restricted to H0 only

**What changed**: `_persistence_diagrams` replaced with `_persistence_h0`
(Kruskal-style union-find for connected components only). The previous H1
computation was a triangle-counting heuristic, not correct Vietoris-Rips
persistent H1.

**Impact on hypotheses**: H5a (H1 adds nothing for two-cluster data) WITHDRAWN.
H5c (H1 most valuable for imaging data) WITHDRAWN.

### 2026-07-09: Method 16 renamed from "bottleneck" to "wasserstein persistence"

**What changed**: The distance computation between persistence diagrams uses
optimal assignment (Wasserstein-1 matching cost), not the L-infinity bottleneck
metric.

**Impact on hypotheses**: H5b reworded from "bottleneck distance" to
"Wasserstein persistence distance."

### 2026-07-09: H4d separated as correctness gate

**What changed**: H4d (sheaf H^1 = Cochran's Q on scalar stalks) reclassified
from "confirmatory hypothesis" to "correctness gate." It is a pass/fail analytic
identity check, not a stochastic directional comparison — an analytic identity
cannot false-positive by chance, so it does not consume an α slot.

**Impact on hypotheses**: Bonferroni correction now applies to 4 statistical
tests (H3a on alpha_T2, H3a on sigma_device, H8a, H6d) at α = 0.05/4 = 0.0125.

### 2026-07-09: H6d reworded from "linearly" to "monotonically"

**What changed**: The test uses Spearman ρ (monotonicity), not Pearson r or
R² (linearity). The claim now matches the test.

### 2026-07-09: H3a extended to sigma_device companion axis

**What changed**: H3a is now tested on both alpha_T2 (composite axis) and
sigma_device (clean, non-composite axis). The sigma_device result is the
cleaner test of subspace-drift detection.

**Reason**: alpha_T2 simultaneously affects T2, linewidth, and contrast. A
subspace method lighting up early on alpha_T2 could track the composite
rather than subspace drift per se.

## Post-Execution Deviations

### 2026-07-09: H8a reclassified as untestable under Spearman framework

**What changed**: H8a as pre-registered ("At alpha_T2 in [0.0, 0.22], median
LASSO AUC > best geometric method") requires a direct cross-scale score
comparison. The adopted Spearman-rho framework normalizes all methods to
monotonic association strength on [-1, +1], making raw score comparisons
inapplicable. LASSO AUC is on [0, 1] while bracket norms are on [10^7, 10^10]
and curvatures are on [-30, 1]. No meaningful "greater than" comparison exists
across these scales.

**Underlying observation**: At alpha_T2 < 0.22, all classification baselines
(LASSO, logistic, RF, DRO) achieve AUC = 1.0000 on all 20 seeds. The
classification task is trivially solved. Geometric methods show measurable
variation at these levels (bracket norms, persistent H0, Berry phase all
change monotonically). This is consistent with the spirit of H8a: baselines
dominate at low decoherence because the classification problem has not yet
degraded, while geometric methods detect structural changes that
classification cannot resolve (the structure changes before the classification
boundary does).

**Impact**: H8a reported as untestable under the adopted statistical
framework. The relevant observation (baseline ceiling saturation at low
decoherence) is reported descriptively.

**Alpha denominator change**: The pre-registered Bonferroni denominator was 4
(H3a on alpha_T2, H3a on sigma_device, H8a, H6d) at alpha = 0.05/4 = 0.0125.
Voiding H8a post-execution reduces the denominator to 3, yielding alpha =
0.05/3 = 0.0167. This is a post-hoc alpha change: the confirmatory threshold
became less stringent because one test was removed. We report the corrected
alpha (0.0167) throughout the paper and note that H6d (|rho| = 0.90, p < 0.001)
would have passed under the stricter pre-registered threshold (0.0125) as well,
and both H3a tests are refutations (wrong direction), so the substantive
conclusions are identical under either threshold.

### 2026-07-09: Classification baselines return constant AUC on sigma_surface and sigma_device

**What changed**: LASSO, logistic, RF, and DRO achieve AUC = 1.0000 at all
decoherence levels on the sigma_surface and sigma_device axes across all 20
seeds. Spearman rho on a constant vector is undefined (NaN).

**Why this happens**: The 1.5K tumor-healthy temperature difference produces
a frequency shift of ~111 kHz (via dD/dT = -74 kHz/K), which remains easily
classifiable even at the maximum surface noise (sigma_surface = 0.5) or
device variation (sigma_device = 0.1) in the sweep range. These axes perturb
features without degrading the classification boundary.

**Impact**: These axes are informative for geometric methods (which detect
structural changes below the classification threshold) but uninformative for
classification baselines. Reported as "baselines saturated at AUC = 1.0;
Spearman rho undefined due to constant scores."
