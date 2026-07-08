# Pre-Registration v2: Gating Experiment for H7b/H7c

Supersedes original PREREGISTRATION.md (SHA 2a9b023). See DEVIATION_LOG.md
for the full correction history.

## Hypothesis

**H7b**: Grassmannian cocycle holonomy detects cross-sensor
inconsistency at a decoherence threshold epsilon_crit where
individual sensors still classify well (AUC > 0.75).

**H7c**: The holonomy detection threshold epsilon_crit is lower
than the single-sensor AUC degradation threshold. Cross-sensor
geometric consistency is more fragile than within-sensor discriminability.

## Method

Cocycle holonomy (method 9 in the survey): parallel transport around
a 3-sensor cycle on the Grassmannian Gr(k, D). Holonomy norm = 0
implies consistent sensors; > 0 implies the cycle of subspaces
cannot be reconciled by a single global rotation.

## Data model

Shared latent z in R^k (k=3) with binary class signal (mu=5.0 in
the first component). Each sensor projects z through a
sensor-specific embedding U_s(eps) in R^{D x k} (D=100).

At eps=0: all sensors share a common orthonormal embedding U_base.
At eps>0: each sensor's embedding is perturbed by a sensor-specific
random matrix P_s, creating genuine inconsistency:
U_s(eps) = QR(U_base + eps * scale * P_s).

Observation noise sigma(eps) = 1.0 + 3.5 * eps adds independently.

## Operationalization

**"Internally consistent"**: Within-sensor 5-fold CV logistic
regression AUC > 0.75.

**"Cross-sensor inconsistent"**: Holonomy norm exceeds the 95th
percentile of a bootstrap distribution computed at eps=0 (where
sensors ARE consistent by construction).

**H7b confirmed if**: there exists an epsilon where holonomy exceeds
the bootstrap threshold AND all three within-sensor AUCs > 0.75.

**H7c confirmed if**: epsilon_crit (first epsilon where holonomy
fires) < epsilon where any single sensor AUC first drops below 0.75.

## Controls

**Negative control**: three sensors with identical perturbation
(same P_s for all sensors) must return H7b = False. If this fails,
the test is miscalibrated. This is tested in the pytest suite.

**Positive control (implicit)**: at eps=0 all sensors share U_base,
so holonomy should be ~0 and NOT significant vs the bootstrap null.
This is tested in the pytest suite.

## Experimental design

5 seeds (20260708-20260712). 20 epsilon steps from 0 to 1.
200 samples per class (400 total). PCA subspace k=3.
500 bootstrap resamples for the baseline null.

See `experiments/gating_h7b_v3.py` for full implementation.

## Outcomes

- H7b+H7c confirmed (>=4/5 seeds): proceed with full 24-method grid
- H7b only (holonomy fires but not before AUC drops): weaker claim,
  revise paper scope
- H7b fails (<2/5 seeds): cross-sensor geometric consistency claim
  is not supported. Pivot to single-sensor boundary conditions.

## Frozen parameters

All parameters are in `experiments/gating_h7b_v3.py`. No parameter
may be changed after the SHA freeze without logging a deviation in
DEVIATION_LOG.md.
