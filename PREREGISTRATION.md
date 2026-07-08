# Pre-Registration: Gating Experiment for H7b/H7c

## Hypothesis

**H7b**: Sheaf H^1 (multi-sensor subspace holonomy) detects cross-sensor
inconsistency at a decoherence threshold epsilon_crit where individual
sensors still appear internally consistent.

**H7c**: The multi-sensor inconsistency threshold epsilon_crit is LOWER
than the single-sensor degradation threshold for each individual sensor.
Cross-sensor consistency is more fragile than within-sensor robustness.

## Operationalization

**"Internally consistent"**: Within-sensor classification AUC > 0.75
(better than chance for a 2-class problem with noise).

**"Cross-sensor inconsistent"**: Holonomy norm > permutation null 95th
percentile (p < 0.05 from 1000 permutations).

**H7b confirmed if**: There exists an epsilon where holonomy_p < 0.05
AND all three within-sensor AUCs > 0.75.

**H7c confirmed if**: epsilon_crit (first epsilon where holonomy_p < 0.05)
< min(epsilon where any single sensor AUC < 0.75).

## Experimental design

Three simulated sensors, one decoherence axis each, swept from 0 to 1
in 20 steps. 200 samples per class (400 total). PCA subspace k=3.
1000 permutations for holonomy null.

See `experiments/gating_h7b.py` for full implementation.

## Analysis plan

1. For each epsilon level, compute:
   - Within-sensor AUC (logistic regression, 5-fold CV) for each sensor
   - Multi-sensor holonomy norm (3-sensor cycle)
   - Holonomy p-value (permutation test)
2. Identify epsilon_crit and single-sensor degradation thresholds
3. Report whether H7b and H7c hold

## Outcomes

- If H7b+H7c hold: proceed with full 24-method experimental grid
- If H7b holds but H7c fails: sheaf detects inconsistency, but not
  earlier than single-sensor methods. Still useful, but weaker claim.
- If H7b fails: the multi-sensor sheaf claim is dead. Pivot to
  single-sensor boundary conditions only (still a valid paper, just
  less novel).

## Frozen parameters

All parameters are in `experiments/gating_h7b.py`. No parameter may
be changed after the SHA freeze without logging a deviation.
