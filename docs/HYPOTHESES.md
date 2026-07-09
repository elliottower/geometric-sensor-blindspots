# Pre-Registration Hypotheses

These hypotheses must be frozen under a commit SHA BEFORE any
experiment runs. They are predictions, not conclusions.

## Structural hypotheses (which methods beat scalar baselines, and when)

### H1: Bracket norm family (methods 1-5)

**H1a**: Raw bracket norm (method 1) will outperform t-test feature
selection when decoherence is ANISOTROPIC (different axes degrade at
different rates). When decoherence is isotropic (all axes degrade
equally), bracket norm reduces to mean-difference ranking and provides
no advantage.

**Reasoning**: Bracket norm measures direction instability, which only
matters when there IS a direction to be unstable in. Under isotropic
noise, all directions are equally affected.

**H1b**: Transport-stable bracket (method 3) will outperform raw bracket
when device-to-device variation is high (sigma_device > 0.05). Below
that threshold, the Frechet variance penalty is wasted computation.

**H1c**: Phenotype-projected bracket (method 4) will be the BEST bracket
variant when the diagnostic signal (temperature difference) is small
relative to the decoherence noise. It focuses instability measurement
on the one axis that matters.

### H2: Curvature (methods 6-7)

**H2a**: ORC stability (method 6) will detect decoherence-induced
cluster merging BEFORE classification AUC drops. The curvature of the
sample graph changes (becomes more negative / less positive) as classes
blur together, but the class centroids may still be separable.

**H2b**: Forman-Ricci (method 7) will agree with ORC for k-NN graphs
with k >= 10 (enough triangles for Forman to approximate ORC) but
disagree for sparse graphs (k < 5).

**H2c**: ORC will fail (provide no additional information over scalar
baselines) when the sample size is small (n < 50 per class) because
the k-NN graph becomes unstable.

### H3: Grassmannian (methods 8-9)

**H3a**: Grassmannian geodesic distance (method 8) will detect subspace
drift at lower decoherence levels than any feature-level method.
The subspace captures collective structure that individual features miss.

**H3b**: Grassmannian holonomy (method 9) will be NON-ZERO on the
multi-sensor loop NV -> photon -> spin -> NV even at LOW decoherence,
because the three sensor modalities impose genuinely different
geometric structure on the readout space.

**H3c**: Both Grassmannian methods will fail when n_samples < 500 and
d_features > 50 (insufficient samples for stable PCA in the subspace).
This is the Gr(3,34) boundary from the epi paper, adapted.

### H4: Sheaf cohomology (methods 10-11)

**H4a**: dim H^1 will be 0 (consistent) for single-sensor, single-device
data at all decoherence levels. The sheaf only adds value when there
are MULTIPLE sensors or MULTIPLE devices to be inconsistent across.

**H4b**: dim H^1 will become > 0 (inconsistent) for multi-sensor data
at a SPECIFIC decoherence threshold. Below this threshold, all sensors
agree. Above it, the sensors disagree about the biology. Identifying
this threshold is the main practical output for clinical deployment.

**H4c**: Sheaf Q (method 11) will correctly identify WHICH sensor
is responsible for inconsistency (the one with the highest decoherence).

**H4d**: On scalar features (d=1 stalks), sheaf H^1 will give identical
results to Cochran's Q heterogeneity test. This is a KNOWN reduction
(proven in the epi paper) and serves as a sanity check.

### H5: Persistent homology (methods 15-16)

**H5a**: Persistent H_1 features (loops in the readout space) will
persist across decoherence regimes when the underlying biology creates
ring-like structures in feature space (e.g., cyclical metabolic states).
When the biology is simply two clusters (tumor vs healthy), H_1 adds
nothing over H_0 (connected components).

**H5b**: Bottleneck distance (method 16) between persistence diagrams
will be a TIGHTER bound on decoherence severity than any scalar
statistic, because the stability theorem guarantees it tracks Hausdorff
distance.

**H5c**: Persistent homology will be MOST valuable for the entangled-photon
sensor (sensor 2) because imaging data naturally has spatial topology
(holes, connected regions) that spectroscopic data (sensors 1, 3) does not.

### H6: Quantum-specific methods (methods 17-18, 21-22, 24)

**H6a**: QFI flatness (method 17) will identify the diagnostic feature
(NV resonance frequency = temperature) as the MOST informationally
robust feature, even when other features have higher raw signal-to-noise.
QFI measures information per unit parameter change, not raw magnitude.

**H6b**: Berry phase (method 18) will be the ONLY method that detects
ANISOTROPIC decoherence (T2 degradation in one direction, surface noise
in another) at the single-feature level. Other methods see aggregate
degradation; Berry phase sees the directional structure.

**H6c**: Chern number (method 24) will be 0 for all features at
decoherence levels relevant to current NV-diamond technology (T2 > 1 μs).
Topological phase transitions in the biomarker space require more extreme
parameter changes than current sensors encounter. If true, this is a
NEGATIVE result: topological protection is not relevant for near-term
quantum biosensors. (This is the kind of null result that justifies
pre-registration.)

**H6d**: Von Neumann entropy (method 21) will track decoherence almost
linearly (entropy increases as information is lost). It will NOT provide
sharper boundaries than other methods — it's a summary statistic, not
a structural one.

### H7: Multi-sensor integration (methods 9, 10, 12, 23)

**H7a**: TransportKit fusion (method 12) will produce the BEST
single-number robustness score for multi-sensor biomarkers, because
it combines subspace alignment (structural) with sheaf consistency
(relational).

**H7b**: Persistent sheaf cohomology (method 23) will discover
decoherence thresholds that no single-sensor method can detect.
At decoherence level epsilon < epsilon_crit, all sensors agree.
At epsilon > epsilon_crit, they disagree — but individual sensors
still look fine internally. The sheaf finds it; single-sensor methods
miss it.

**H7c**: The multi-sensor inconsistency threshold (epsilon_crit from H7b)
will be LOWER than the single-sensor degradation threshold for each
individual sensor. Cross-sensor consistency is MORE FRAGILE than
within-sensor robustness.

### H8: Baselines

**H8a**: LASSO feature selection will beat ALL geometric methods at
low decoherence (alpha_T2 < 0.2, sigma_surface < 0.1). When the signal
is strong and noise is low, fancy geometry is overhead.

**H8b**: IRM (Invariant Risk Minimization) will beat bracket norms
(methods 1-5) when there are many decoherence regimes in the training
set (|R| > 10) because IRM is trained to find invariances while
bracket norms detect them post-hoc.

**H8c**: No scalar baseline will match sheaf H^1 (method 10) for
detecting multi-sensor inconsistency, because scalar methods cannot
represent the RELATIONAL structure between sensors.

## Computational hypotheses

**H9a**: Methods 1-5 (bracket family) will be fast enough to run on
every candidate biomarker in real time (< 1 second per feature on CPU).

**H9b**: Methods 15-16 (persistent homology) will be the computational
bottleneck, requiring GPU or subsampling for n_samples > 1000.

**H9c**: Method 23 (persistent sheaf cohomology) will be the most
expensive overall because it combines two expensive operations.

## Meta-hypothesis

**H10**: The "boundary condition" — the decoherence level at which
a geometric method stops helping — will be PREDICTABLE from the
method's mathematical structure:

- Feature-level methods (1-5, 17): fail when SNR < 1 per feature
- Sample-space methods (6-7, 15-16, 19-21): fail when n_samples < 100
- Relational methods (8-14): fail when n_regimes < 3
- Topological methods (18, 23, 24): fail when parameter space is
  effectively 1D (only one decoherence axis matters)

If H10 is true, a practitioner can choose the right method by
inspecting their experimental setup — no trial-and-error needed.
