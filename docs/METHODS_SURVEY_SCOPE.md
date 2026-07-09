# Geometric Methods for Quantum Sensor Biomarker Robustness: Full Survey Scope

## The question

When does geometric/topological post-processing improve quantum sensor
biomarker robustness? Under what conditions does each method help, and
under what conditions does it reduce to standard statistics?

Same structure as `epidemiology-boundary-conditions` but for quantum
sensor decoherence instead of causal inference.

## The 24 methods

### A. From existing papers (directly portable)

| # | Method | Origin repo | What it tests | Quantum analog |
|---|--------|-------------|---------------|----------------|
| 1 | Bracket norm (raw) | bracket-norm | Direction instability of perturbation response | Feature direction instability under decoherence perturbation |
| 2 | Bracket norm / sqrt(n) | bracket-norm | Size-corrected direction instability | Corrected for number of sensor channels |
| 3 | Transport-stable bracket | direction-instability-drug-validity | Penalizes unreplicable instability (Frechet variance) | Penalizes features unstable across decoherence regimes |
| 4 | Phenotype-projected bracket | direction-instability-drug-validity | Instability projected onto target direction | Instability projected onto diagnostic axis (healthy vs disease) |
| 5 | Localized bracket | direction-instability-drug-validity | Ratio pathway-specific to global instability | Ratio sensor-modality-specific to total instability |
| 6 | Ollivier-Ricci curvature | psychiatric-comorbidity-curvature | Edge-level graph curvature (optimal transport) | ORC on sensor readout nearest-neighbor graph across regimes |
| 7 | Forman-Ricci curvature | epidemiology-boundary-conditions | Combinatorial curvature (degree deficit) | Fast proxy for ORC on large sensor graphs |
| 8 | Grassmannian geodesic distance | genetic-perturbation-holonomy | Principal angles between subspaces | Distance between readout subspaces across decoherence regimes |
| 9 | Grassmannian holonomy | genetic-perturbation-holonomy | Parallel transport around loops of conditions | Holonomy around loops of (device, temperature, tissue) conditions |
| 10 | Sheaf H^1 obstruction | epidemiology-boundary-conditions | Global inconsistency of local measurements | Global inconsistency of multi-sensor local readouts |
| 11 | Per-edge sheaf Q | epidemiology-boundary-conditions | Which specific edges are inconsistent | Which specific sensor-regime pairs disagree |
| 12 | TransportKit fusion | transport-wrapper | Engine E + Engine C combined verdict | Fused sensor transportability verdict |
| 13 | CKA | genetic-perturbation-holonomy | Centered kernel alignment between representations | Alignment between readout representations across regimes |
| 14 | Procrustes distance | genetic-perturbation-holonomy | Rotation distance between matched configurations | Rotation between sensor readout configurations |

### B. New methods (quantum-specific or from TDA/info geometry)

| # | Method | Mathematical basis | What it tests |
|---|--------|--------------------|---------------|
| 15 | Persistent homology (Vietoris-Rips) | Persistent H_0, H_1, H_2 | Topological features of readout point cloud that persist across decoherence |
| 16 | Bottleneck/Wasserstein on persistence diagrams | Stability theorem for persistence | How much the topology shifts between regimes (stable under small noise) |
| 17 | Quantum Fisher information flatness | Fisher info metric on statistical manifold | Features with constant QFI across decoherence = metrologically invariant |
| 18 | Berry phase (geometric phase) | Connection curvature on state-space fiber bundle | Path-dependence of sensor state under parameter loops = decoherence anisotropy |
| 19 | Spectral gap stability | Weyl perturbation bounds on Laplacian | Eigenvalue gaps of readout graph Laplacian robust under decoherence |
| 20 | Wasserstein distance (distributions) | Optimal transport between readout distributions | Full-distribution shift between regimes (not just moments) |
| 21 | Von Neumann entropy stability | Quantum entropy of density matrix | How much the information content of sensor state changes under decoherence |
| 22 | Fidelity / diamond norm proxy | Quantum channel fidelity | Upper bound on how distinguishable two decoherence channels are from readout |
| 23 | Persistent sheaf cohomology | Sheaf cohomology filtered by decoherence parameter | Track obstruction classes as decoherence increases — which persist? |
| 24 | Chern number (integer topological invariant) | Characteristic class of state-space bundle | Discrete (integer) invariant that CANNOT drift continuously — binary certificate |

## The simulated oracle

### Single-sensor simulation (NV-diamond intracellular thermometry)

Ground truth: two tissue classes (tumor vs healthy) differ by 1-2°C.

Sensor model:
- NV-center fluorescence time-traces
- Parameters: T2 relaxation time (1-10 μs), surface noise amplitude,
  temperature, magnetic field
- Readout: time-resolved photoluminescence → extracted features
- Decoherence axes: T2 degradation, surface noise increase, temperature
  drift, device-to-device variation

For each of 24 methods, compute robustness score across decoherence regimes.
Ground truth known → we can measure which methods correctly identify
robust features vs artifacts.

### Multi-sensor simulation (the exciting part)

Three quantum sensor modalities on the same biological system:
1. **NV-diamond magnetometry** — local magnetic field (cellular iron, neural currents)
2. **Entangled-photon microscopy** — sub-diffraction-limit imaging
3. **Spin-labeled protein detection** — specific protein binding events

Each sensor has its own decoherence profile. The biological signal is shared.

Questions the multi-modal setup answers:
- **Sheaf cohomology**: can we stitch local readouts from different sensors
  into a globally consistent biological picture? H^1 ≠ 0 means NO →
  at least one sensor is giving an inconsistent signal under these conditions
- **Grassmannian holonomy**: transport a biomarker around the loop
  NV → photon → spin → NV. If holonomy ≠ 0, the biomarker doesn't
  survive the full multi-sensor pipeline
- **Transport fusion**: for each candidate biomarker, ask "does it transport
  across decoherence regimes AND across sensor modalities?"
- **Bracket norm**: which biomarkers have stable direction across BOTH
  decoherence perturbation AND modality switching?

This is where the paper becomes genuinely novel:
- Single-sensor robustness → your basic boundary-conditions result
- Multi-sensor consistency → sheaf/holonomy results unique to this paper
- Cross-modal transportability → the transportkit fusion on quantum data

## Connection back to prior work

| Prior paper | What this paper adds |
|---|---|
| bracket-norm | Same metric, new domain (quantum decoherence instead of neural silencing) |
| direction-instability-drug-validity | Same failure-mode analysis (decoherence-corrected bracket = toxicity-corrected bracket analog) |
| psychiatric-comorbidity-curvature | ORC on sensor readout graphs instead of disease networks |
| genetic-perturbation-holonomy | Holonomy on sensor modality loops instead of cell-type loops |
| epidemiology-boundary-conditions | SAME PAPER STRUCTURE — "when does method X beat scalar alternatives?" |
| transport-wrapper | TransportKit applied to quantum sensor data |
| mechanistic-validity | Validity lenses applied to quantum biomarker claims |

## Boundary conditions to discover

For each of the 24 methods, characterize:
1. **When it helps** — under what decoherence/noise/dimensionality regime
   does it outperform scalar alternatives?
2. **When it reduces** — under what conditions does it collapse to a
   standard test? (e.g., sheaf Q → Cochran's Q on scalar stalks)
3. **When it fails** — false confidence regime, or conditions where it
   gives wrong answers
4. **Required sample size** — detection power boundary (like the Gr(3,34)
   boundary in the epi paper)

## Pre-registration structure

Same protocol as all other papers:
- Freeze all 24 method implementations + simulation parameters under SHA
- Pre-register hypotheses about which methods will succeed in which regimes
- Run experiments
- Report honestly, including nulls

## Deliverables

1. **24-method comparison table** — like Table 1 of epi-boundary-conditions
   but for quantum sensor decoherence
2. **Boundary condition map** — for each method, the structural conditions
   under which it adds value
3. **Multi-sensor sheaf/holonomy result** — the novel finding
4. **TransportKit quantum adapter** — the practical tool
5. **Pre-registered, SHA-frozen, deterministic** — the credibility
