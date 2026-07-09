# Extract: Curvature, Cohort Transfer, MS/MS, Cross-Design, Ecological Bias

Comprehensive extraction for quantum sensing paper reuse.

---

## 1. Psychiatric Comorbidity Curvature (GigaScience, submitted)

**Paper**: `psychiatric-comorbidity-curvature/paper/paper_v4.tex`
**Title**: "Edge-Level Discrete Curvature Decomposes Bridge Structure Across Disease Networks"

### Section structure

| Section | Label | Content |
|---|---|---|
| §1 | Introduction | ORC literature review, degree confound identification |
| §2 (sec:degree) | The Degree Confound | Node-level ORC correlates r=-0.71 with degree |
| §3 (sec:method) | Method: Edge-Level Curvature with Null Models | Degree-preserving and weight-permutation nulls |
| §4 (sec:replication) | Cross-Domain Replication | 6 data-driven graphs, 3 disease domains |
| §5 (sec:casestudy) | Case Study: Psychiatric Mechanism Network | 130-node graph |
| §5.1 (sec:edge) | Edge-Level Decomposition | 32/103 edges significant |
| §5.2 (sec:verdict) | Verdict-Curvature Relationship | Disconfirmed claims lower curvature |
| §5.3 (sec:benchmarking) | Method Benchmarking | 5 edge-level methods compared |
| §5.4 (sec:prediction) | Cross-Validated Verdict Prediction | LOOCV AUC = 0.61 |
| §5.5 (sec:mr) | Cross-Disorder MR | 49/56 pairs significant |
| §6 (sec:robustness) | Robustness | Catalog perturbation, leave-one-family-out |
| §7 | Related Work | |
| §8 (sec:limitations) | Limitations | |
| §9 | Conclusion | |

### Key equations

**ORC formula (§2, Eq. 1)**:
```
κ(u,v) = 1 - W₁(μ_u, μ_v) / d(u,v)
```
where μ_u is a lazy random walk (idleness α=0.5), W₁ is Wasserstein-1 distance under graph shortest-path metric, solved via POT library.

**Edge-level z-score (§3)**:
```
z = (κ_obs - κ̄_null) / σ_null
```
from N=200 degree-preserving double-edge swaps.

### Key results to cite

| Result | Value | Section |
|---|---|---|
| Node-level ORC vs degree | r = -0.71, p < 0.001 | §2 |
| Within vs cross-cluster (psychiatric LDSC) | d = 1.44, p = 0.001 | §4 |
| Within vs cross-cluster (gene-sharing) | d = 1.52, p < 0.001 | §4 |
| Within vs cross-cluster (comorbidity OR) | d = 1.55, p < 0.001 | §4 |
| Within vs cross-cluster (WHO WMH) | d = 0.46, p < 0.001 | §4 |
| Within vs cross-cluster (autoimmune) | d = 0.47, p = 0.81 (NS) | §4 |
| Within vs cross-cluster (cardiometabolic) | d = 0.77, p = 0.12 (NS) | §4 |
| Strongest bottleneck edge | depression-convergence hub: z = -5.46, κ = -0.87 | §5.1 |
| Disconfirmed vs rest | p = 0.007, d = 0.89 | §5.2 |
| ORC discrimination (unique among 5 methods) | d = 0.73, p = 0.011 | §5.3 |
| Edge clustering (next best) | d = 0.40, p = 0.294 | §5.3 |
| Degree | d = 0.32, p = 0.302 | §5.3 |
| Betweenness | d = 0.19, p = 0.841 | §5.3 |
| Jaccard overlap | d = 0.14, p = 0.641 | §5.3 |
| Spectral gap | d = 0.00 | §5.3 |
| LOOCV AUC | 0.61 | §5.4 |
| ORC vs betweenness correlation | r = -0.674 | §6 |
| Depression-convergence perturbation survival | 100% | §6 |

### Quantum paper relevance

**Direct port**: ORC on sensor readout k-NN graphs (method 6 in our survey). The degree confound finding is a WARNING: if quantum sensor readout graphs have degree heterogeneity, node-level ORC will be a degree artifact. Must use edge-level ORC with permutation null.

**Key quote for quantum paper** (§2):
> "Node-level ORC is a degree artifact ($r = -0.71$); the standard practice of reporting it without a null model... produces misleading bottleneck narratives."

**Negative result to port** (§5.3): Among 5 edge-level methods, ONLY ORC discriminated. Betweenness, Jaccard, clustering, spectral gap all failed. Prediction: same pattern on quantum sensor graphs.

### Importable code

| File | Key functions | Use in quantum paper |
|---|---|---|
| `experiments/data_driven/weighted_orc.py` | `ollivier_ricci_curvature()`, `degree_preserving_rewire()`, `run_edge_null()` | Core ORC computation with null model |
| `experiments/data_driven/weighted_orc.py` | `jaccard_edge_overlap()`, `edge_betweenness()`, `edge_clustering_coefficient()`, `spectral_edge_gap()` | Comparison methods |
| `experiments/data_driven/weighted_orc.py` | `compute_all_edge_features()` | One-call to get all 5 edge metrics |
| `experiments/data_driven/weighted_orc.py` | `method_comparison()` | Runs full Disconfirmed-vs-rest comparison across all methods |
| `experiments/curvature_core.py` | `forman_ricci_curvature()` | Forman-Ricci (method 7) |
| `experiments/curvature_core.py` | `build_comorbidity_graph()` | Graph construction pattern |
| Dependencies | `networkx`, `numpy`, `scipy`, `pot` (Python Optimal Transport) | All needed for quantum version |

---

## 2. Cohort Transfer (PLOS ONE, under review)

**Paper**: `biomedical-cohort-transfer/paper_v5_plos.tex`
**Title**: "Grassmannian geodesic distance predicts cross-cohort classifier degradation under analytical heterogeneity, after controlling for source classifier quality"

### Section structure

| Section | Label | Content |
|---|---|---|
| §1 | Introduction | Domain adaptation theory, Grassmannian proposal |
| §2 | Data | 7 datasets described |
| §2.1 (sec:crc_data) | CRC microbiome | 9 studies, 824 samples, primary |
| §2.2 (sec:qmdiab_data) | QMDiab | 3 biofluids × 3 ethnicities |
| §2.3 (sec:metab_data) | MTBLS7260 | 15 plates, null result |
| §2.4 (sec:ibd_data) | IBD | 5 studies, underpowered |
| §2.5 (sec:copd_data) | SPIROMICS COPD | Centralized platform null |
| §2.6 (sec:brca_geo_data) | Breast cancer GEO | Shared microarray null |
| §2.7 (sec:tcga_data) | TCGA-BRCA | Centralized sequencing null |
| §3 (sec:methods) | Methods | Geodesic, partial correlation, permutation null, clustered bootstrap |
| §4 | Results | |
| §4.1 (sec:batch) | Geometric batch detection | z > 25 on all datasets |
| §4.2 (sec:crc) | CRC: partial correlation | partial ρ = +0.61 |
| §4.3 (sec:qmdiab) | QMDiab | partial ρ = +0.40 |
| §4.4 (sec:metabolomics) | MTBLS7260 | Null (noise floor) |
| §4.5 (sec:loo) | LOO prospective validation | MAE reduced 42% |
| §4.6 (sec:boundary) | Boundary conditions | 5 null results explained |
| §5 (sec:discussion) | Discussion | Boundary conditions synthesis |

### Key equations

**Grassmannian geodesic distance (§1, Eq. 1)**:
```
d_Gr(V₁, V₂) = (Σᵢ θᵢ²)^(1/2)
```
where θᵢ = arccos(σᵢ) for σᵢ the singular values of U₁ᵀU₂.

**Chordal distance (§3)**:
```
d_ch(V₁, V₂) = ||U₁U₁ᵀ - U₂U₂ᵀ||_F
```

### Key results to cite

| Result | Value | Section |
|---|---|---|
| CRC partial ρ (geodesic) | +0.61, CI [-0.02, +0.80], 97% positive | §4.2 |
| CRC ΔR² (adding geodesic) | +0.226 | §4.2 |
| QMDiab partial ρ | +0.40, CI [+0.04, +0.73] | §4.3 |
| LOO MAE (full model) | 0.057, 42% better than baseline | §4.5 |
| LOO MAE (22% better than AUC-only) | 0.057 vs 0.074 | §4.5 |
| LOO ρ | 0.80 | §4.5 |
| IBD partial ρ | -0.11 (null, underpowered, 5 studies) | §4.6 |
| SPIROMICS partial ρ | +0.08 (null, centralized platform) | §4.6 |
| TCGA partial ρ | -0.12 (null, centralized sequencing) | §4.6 |
| Breast cancer GEO partial ρ | -0.004 (null, shared microarray) | §4.6 |
| k-sensitivity range | Stable ρ = +0.56 to +0.63 across k=10-50 | §4.2 |
| Permutation null z-scores | >25 on all datasets | §4.1 |
| Source AUC confound | ρ = +0.677 with gap | §4.2 |

### Three boundary conditions for method to work (§5, Practitioner Checklist)

1. **Analytical heterogeneity** — Cohorts must differ in measurement platform/protocol/matrix
2. **Subspace separability** — Cohort PCA subspaces distinguishable from label-permutation null (z > 3)
3. **Between-cohort dominance** — Between-cohort shift must exceed within-cohort disease-state variance

### Quantum paper relevance

**Direct port**: Grassmannian geodesic distance = method 8 in our survey. The EXACT same code computes distance between sensor readout subspaces across decoherence regimes.

**Critical insight for quantum paper**: The boundary conditions predict when the quantum method will work:
- Different decoherence regimes = analytical heterogeneity ✓ (should work)
- Same device, same calibration = centralized platform (will return null)
- High within-device noise exceeding between-regime shift = IBD failure mode

**Key quote** (§4.6 synthesis):
> "Geodesic distance... predicts transportability when three conditions hold: (1) substantial analytical heterogeneity exists... (2) the resulting shift is captured by the top-k PCA subspace... (3) between-cohort variance exceeds within-cohort disease-state variance."

**Null prediction for quantum**: If all NV-diamond sensors use the same vendor's diamond crystals and same ODMR protocol, geodesic distance will return null (same as SPIROMICS/TCGA centralized result).

### Importable code

| File | Key functions | Use in quantum paper |
|---|---|---|
| `src/transportability.py` | `principal_angles()` | Core angle computation |
| `src/transportability.py` | `geodesic_distance()` | Grassmannian distance (method 8) |
| `src/transportability.py` | `transport_matrix()` | For holonomy computation |
| `src/transportability.py` | `cocycle_holonomy()` | Grassmannian holonomy (method 9) |
| `src/transportability.py` | `top_k_subspace()` | PCA subspace extraction |
| `src/transportability.py` | `sheaf_h1_two_cohort()` | Sheaf H¹ for 2 cohorts (method 10) |
| `src/transportability.py` | `sheaf_h1_multi_cohort()` | Sheaf H¹ for N cohorts (method 10) |
| `src/transportability.py` | `sheaf_q_test()` | Per-edge sheaf Q (method 11) |
| `src/transportability.py` | `bracket_norm_score()` | Bracket norm score (method 1) |
| `src/transportability.py` | `centroid_distance()` | Centroid baseline |
| `src/transportability.py` | `domain_classifier_auc()` | Domain classifier baseline |
| `src/transportability.py` | `directional_transport_score()` | Directional transport |
| `src/transportability.py` | `fisher_rao_confound_score()` | Confound detection |

**THIS IS THE MOTHER LODE** — `transportability.py` already implements methods 1, 8, 9, 10, 11 from the quantum survey in a single importable module.

---

## 3. MS/MS Subspace Collapse (GigaByte, submitted)

**Paper**: `msms-subspace-collapse/paper/paper.tex`
**Title**: "MS/MS Benchmark Embeddings Collapse Across Instruments"

### Section structure

| Section | Content |
|---|---|
| Introduction | Cross-instrument transferability concern |
| Data Description | 2140 spectra, 10 instrument types, 3 ionization families |
| Methods | 6 distance metrics, degradation metrics, bootstrap, confound test |
| (Confirmatory analyses C1-C5 in PREREGISTRATION.md) | |

### Key results to cite

| Result | Value |
|---|---|
| Geodesic distances span | 85-97% of theoretical maximum |
| Dynamic range | 11.8% |
| Primary preregistered result | ρ = -0.211, CI [-0.523, +0.145] (null) |
| Confound: EI-B pairs | Zero shared compounds → degradation analytically fixed at 1.0 |
| Minimum detectable effect at 80% power | ρ = 0.33 |
| Domain classifier vs geodesic | 17× more group separation |

### Quantum paper relevance

**WARNING result**: If quantum sensor readout embeddings are as collapsed as MS/MS fingerprint embeddings, geodesic distance between instrument pairs will be compressed to a narrow range and uninformative. The 85-97% near-orthogonality means ALL instrument subspaces look equally different — no gradation.

**Key quote** (abstract):
> "Geodesic distances span 85--97\% of the theoretical maximum, with a dynamic range of 11.8\%."

**Prediction for quantum paper**: If NV-diamond readout features are binary (presence/absence of spectral peaks), the subspace collapse risk is HIGH. If features are continuous (fluorescence intensity, T2 decay curves), this failure mode is less likely.

**Confound lesson**: EI-B pairs had zero compound overlap, creating a trivial confound. In the quantum setting, ensure decoherence regimes share overlapping biological signal — if regime A destroys ALL signal, degradation is analytically 1.0 and any distance metric trivially correlates.

### Importable code

| File | Key functions | Use in quantum paper |
|---|---|---|
| `src/pipeline.py` | `run_pairwise_analysis()` | Full pairwise distance computation across 6 metrics |
| `src/pipeline.py` | `spectral_similarity_task()` | Cross-instrument matching degradation |
| `src/pipeline.py` | `cross_site_classification()` | Classifier-based degradation |

---

## 4. Cross-Design Evidence Discordance (BMC MRM, submitted)

**Paper**: `cross-design-evidence-discordance/paper/paper_v4_bmc.tex`
**Title**: "Cross-Design Evidence Discordance Diagnoses Phase III Drug Failure Modes"

### Key results to cite

| Result | Value |
|---|---|
| Correct classifications | 18/22 scored families |
| Ablation: MR-only vs cross-design rule | Identical (zero McNemar disagreements) |
| Three failure modes | Zombie mechanisms, translation gaps, exposure mismatches |
| Effector-neutralization boundary | Anti-TNF, anti-IL-17, abatacept show null MR because targets are consequences not causes |

### Quantum paper relevance

**Structural analog**: Different study designs disagree → different sensor modalities disagree. The cross-design concordance rule is exactly what sheaf H¹ detects:
- Concordant (H¹ ≈ 0): sensors agree on the biology
- Discordant (H¹ ≠ 0): sensors disagree — diagnose which one is wrong

**Failure-mode taxonomy ports directly**:
- "Zombie mechanisms" → Decoherence artifacts (feature appears robust but is actually a sensor artifact, not biology)
- "Translation gaps" → Features that are biologically real but don't transport across devices
- "Exposure mismatches" → Sensor measures different physical quantity than intended under decoherence

**Key quote** (abstract):
> "MR-null status alone accounted for all predictive power... The observational leg was non-trivial for every scored family and added no discriminative information."

Quantum analog: If ALL sensor features look "associated" with disease under clean conditions, the key discriminator is which ones survive decoherence (the "MR" equivalent). The clean-condition association (the "observational" equivalent) adds nothing.

---

## 5. Ecological Bias COVID (BMC MRM, submitted)

**Paper**: `ecological-bias-covid/paper/ecological_bias_v8.tex`
**Title**: "Ecological bias distorts effect sizes in federated COVID-19 analysis"

### Key results to cite

| Result | Value |
|---|---|
| CDC ecological β vs individual OR | β = +0.55 vs OR = 9.9 |
| Mexico ecological β vs individual OR | β = +1.31 vs OR = 11.3 |
| Meta-analytic divergence | 350-fold on pooled effect |
| I² | 99.8% |
| Consistency test z-score | z = 25, p < 0.0001 |
| Elderly proportion explains variation | β = +14.2, p < 10⁻¹⁵ |

### Quantum paper relevance

**Aggregation bias warning**: If you aggregate across NV centers within a device, or across devices within a lab, you lose individual-sensor information. The ecological regression coefficient is "a different quantity, not a noisy estimate" — same risk when averaging quantum sensor readouts.

**Key quote** (abstract):
> "Ecological regression produces coefficients that are incommensurable with patient-level effects---a different quantity, not a noisy estimate."

**Quantum analog**: Per-device averaged readouts vs per-NV-center readouts may give qualitatively different conclusions about which biomarkers are robust. The quantum paper should test both levels of aggregation.

**Consistency test**: The multivariate consistency test (z = 25) detecting "coordinated heterogeneity invisible to standard tests" is exactly what sheaf H¹ does. Could cite this as prior evidence that multi-site coordinated heterogeneity is real and detectable.

### Importable code

| File | Key functions |
|---|---|
| `data/geometric_results/advanced_geometry.py` | Geometric heterogeneity detection |
| `data/geometric_results/compute_geometric_chunked.py` | Chunked computation for large datasets |
| `paper/analyses/s10_consistency_calibration.py` | Multivariate consistency test |

---

## Summary: Code import priority

### Tier 1 — Import directly for quantum paper (methods already implemented)

1. **`biomedical-cohort-transfer/src/transportability.py`** — THE key file. Has geodesic_distance, cocycle_holonomy, sheaf_h1_two_cohort, sheaf_h1_multi_cohort, sheaf_q_test, bracket_norm_score, principal_angles, transport_matrix. This is methods 1, 8, 9, 10, 11 in one file.

2. **`psychiatric-comorbidity-curvature/experiments/data_driven/weighted_orc.py`** — ORC computation with edge-level null model. Methods 6 (ORC) and all comparison metrics.

3. **`psychiatric-comorbidity-curvature/experiments/curvature_core.py`** — Forman-Ricci curvature. Method 7.

### Tier 2 — Adapt for quantum paper

4. **`msms-subspace-collapse/src/pipeline.py`** — Pairwise analysis pipeline. Adapt the multi-metric comparison framework for quantum sensor pairs.

5. **`ecological-bias-covid/paper/analyses/s10_consistency_calibration.py`** — Multivariate consistency test. Adapt for multi-sensor consistency.

### Tier 3 — Reference for framing

6. Cross-design concordance rule → frame multi-sensor disagreement as "cross-design discordance" for quantum sensors.
