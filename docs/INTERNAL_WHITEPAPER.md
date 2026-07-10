# Internal Whitepaper: Geometric Invariance Predicts Sensor Blind Spots

**Status**: Working document for cross-session context transfer.
**Date**: 2026-07-10
**Author**: Elliot Tower

---

## 1. What this project is

A pre-registered comparison of **23 geometric methods + 7 scalar baselines** on simulated NV-diamond intracellular thermometry across controlled decoherence axes. The central finding is the **invariance-blindness principle**: a geometric method's mathematical invariance properties predict which sensor noise sources it detects and which it is blind to.

**Simulation**: NV-diamond temperature sensing (healthy tissue 310.0 K vs tumor 311.5 K). Per sample: 5 features per NV center (T2, linewidth, contrast, resonance frequency, T1) x n_centers (default 20) = 100 features. Datasets: n_per_class=200 samples per class, k=3 devices, binary classification.

**Six decoherence axes** (each swept independently, 10 levels, baseline=0):
1. `alpha_T2` [0,1] — T2 coherence degradation (composite: affects T2 + linewidth + contrast)
2. `sigma_surface` [0,0.5] — surface noise
3. `sigma_device` [0,0.1] — additive per-device calibration offset (THE discriminating axis)
4. `sigma_temp` [0,2.0] K — per-sample temperature drift
5. `n_centers` {5..50} — ensemble size (dimensionality change)
6. `sigma_gain` [0,0.15] — multiplicative per-device gain (Extension 1, new)

**Universal metric**: Spearman rho(method_score, decoherence_level), computed per seed, reported as median with bootstrap 95% CI.

---

## 2. The central finding (from paper v7, 150k evaluations)

**The invariance-blindness principle**: Any method invariant to translations in feature space — f(X + 1c^T) = f(X) — is blind to additive device calibration variation (sigma_device). This is falsifiable and confirmed without exception across all 23 methods.

**The discriminating axis is sigma_device** (additive offset, simplest physics):
- 25/30 methods achieve |rho| > 0.5 on alpha_T2 (composite axis)
- Only 6/30 achieve |rho| > 0.5 on sigma_device
- The 24 methods blind to sigma_device are ALL translation-invariant by construction

**Methods that SEE sigma_device** (|rho| > 0.5):
- Cohen's d (-0.70), localized bracket (-0.66), bracket norm raw (+0.65), bracket norm corrected (+0.65), t-test (-0.56), IRM penalty (-0.55)

**Methods BLIND to sigma_device** (|rho| < 0.15):
- Grassmannian geodesic (-0.05), holonomy, Berry phase (+0.006), CKA (+0.01), Procrustes (+0.01), spectral gap, QFI flatness, fidelity proxy, persistent sheaf, VN entropy, all curvatures

**Key confirmatory result (H3a, REFUTED)**: Pre-registered prediction that geodesic outperforms bracket norms was wrong. Geodesic is flat on sigma_device (rho=-0.05) while bracket norm achieves +0.65. The refutation IS the finding: geodesic is "correct" in reporting no subspace change, but this invariance makes it blind.

---

## 3. Extension experiments (3 new axes)

Pre-registration: `docs/PREREGISTRATION_EXTENSIONS.md` (SHA e31a495).

### Extension 1: Multiplicative device gain (H_gain) — COMPLETED

**Design**: sigma_gain in [0,0.15], per-device gain g_d ~ N(1, sigma_gain) applied to amplitude features only (NOT resonance frequency — set by crystal field D(T)). 100 seeds, 30k evaluations.

**Hypothesis (EXPLORATORY)**: Subspace-angle methods (geodesic, CKA, Procrustes, Berry) achieve |rho| < 0.15 on gain, extending invariance-blindness from translation to scale.

**Results** (30,000 evaluations, 0 errors, 54 minutes):

| Method | median rho | Predicted blind? | Outcome |
|--------|-----------|-----------------|---------|
| grassmannian_geodesic | +0.048 | Yes | CONFIRMED (flat) |
| cka | -0.115 | Yes | CONFIRMED (flat) |
| berry_phase | +0.073 | Yes | CONFIRMED (flat) |
| procrustes | +0.230 | Yes | BORDERLINE (above 0.15, std=0.29) |

3 of 4 predicted methods confirmed flat. Procrustes at +0.230 is above the 0.15 threshold but with std=0.29 the CI straddles it — **underpowered at 100 seeds**.

**Surprise findings** (not predicted, scale-sensitive):
- sheaf_h1: +0.818 — local consistency disrupted by per-device gain
- sheaf_q_per_edge: +0.806
- vn_entropy_stability: -0.806 — eigenvalue spread directly scaled by gain
- fidelity_proxy: -0.655
- IRM penalty: +0.564
- transportkit_fusion: +0.503

**Critical validation**: bracket_norm_raw went from +0.85 in 2-seed smoke test to +0.10 in the full run. The smoke test artifact was caused by (a) 2 seeds and (b) resonance frequency being scaled (before the physics fix). This validates the pre-registration discipline.

All classifiers (LASSO, logistic, RF, DRO) score exactly 0.000 — gain at sigma_g ≤ 0.15 doesn't affect classification.

### Extension 2: Smaller Delta_T (H_dT) — RUNNING

**Design**: Sweep alpha_T2 at Delta_T in {0.2, 0.3, 0.5, 0.75, 1.0, 1.5} K (default is 1.5 K). 100 seeds per Delta_T value, 6 x 100 x 30 x 10 = 180k evaluations.

**Hypothesis (CONFIRMATORY, alpha=0.025)**: At reduced signal (Delta_T < 0.5 K), LASSO AUC drops below 0.6 AND paired Cohen's d > 0.5 between bracket-norm rho in the maintained-signal bracket vs the collapsed bracket.

**Status**: COMPLETE. All 6 Delta_T values x 100 seeds = 180k evaluations, 0 errors.

**Results — LASSO AUC at maximum decoherence (alpha_T2=1.0)**:

| Delta_T (K) | LASSO AUC at alpha_T2=0 | LASSO AUC at alpha_T2=1.0 | Below 0.6? |
|-------------|------------------------|--------------------------|------------|
| 0.20 | 0.977 | 0.510 | YES (near chance) |
| 0.30 | 0.999 | 0.521 | YES |
| 0.50 | 1.000 | 0.553 | YES |
| 0.75 | 1.000 | 0.603 | NO (barely above) |
| 1.00 | 1.000 | 0.656 | NO |
| 1.50 | 1.000 | 0.755 | NO |

At Delta_T=0.2K (the smallest signal), LASSO starts at 0.977 and drops to 0.510 — essentially chance. Even at Delta_T=0.5K, LASSO degrades to 0.553. The threshold crossing (AUC < 0.6) occurs between 0.5K and 0.75K.

Meanwhile bracket norm raw achieves rho=+1.000 at ALL Delta_T values — perfectly monotonic tracking regardless of signal strength. Geometric methods detect structural changes even when classification has completely failed.

**Method rho stability across Delta_T** (all measured on alpha_T2 axis):

| Method | rho at 0.2K | rho at 0.5K | rho at 1.5K | Stable? |
|--------|-----------|-----------|-----------|---------|
| bracket_norm_raw | +1.000 | +1.000 | +1.000 | YES |
| persistent_h0 | +1.000 | +1.000 | +1.000 | YES |
| berry_phase | +0.988 | +1.000 | +1.000 | YES |
| geodesic | +0.770 | +0.891 | +0.636 | NO (degrades at large dT) |
| VN entropy | +0.261 | +0.685 | +0.855 | NO (improves with larger dT) |
| sheaf_h1 | +0.012 | -0.055 | -0.127 | FLAT (null everywhere) |

### Extension 3: Axis interactions (H_int) — RUNNING

**Design**: 2D grid sweep: alpha_T2 x sigma_device (6x6 grid, 50 seeds per cell). 6x6x50x30=54k evaluations.

**Hypothesis (EXPLORATORY)**: Tests whether single-axis sensitivity map composes linearly under joint perturbation.

**Status**: COMPLETE. 54,000 results, 0 errors, 81 minutes.

**Key finding — the interaction is REAL and asymmetric**:

Direction 1: rho(score vs alpha_T2) at each sigma_device level → **no interaction**. Every method's T2-sensitivity is identical whether sigma_device=0 or 0.1. Device variation does not modulate T2 response.

Direction 2: rho(score vs sigma_device) at each alpha_T2 level → **strong interaction for bracket norms**:

| alpha_T2 | bracket_norm rho vs dev | localized_bracket rho vs dev | geodesic rho vs dev |
|----------|------------------------|------------------------------|---------------------|
| 0.00 | +1.000 | -1.000 | +0.029 |
| 0.20 | +1.000 | -1.000 | +0.086 |
| 0.40 | +1.000 | -1.000 | -0.143 |
| 0.60 | +0.971 | -0.971 | +0.029 |
| 0.80 | +0.886 | -0.886 | -0.116 |
| 1.00 | +0.543 | -0.543 | -0.029 |

Bracket norm's device-sensitivity degrades from rho=1.0 to 0.54 as T2 decoherence increases. At high decoherence (alpha_T2=1.0), the T2 effect swamps the device offset signal. Geodesic stays flat at all T2 levels (still blind). Cohen's d is perfectly robust (-1.000 at all T2 levels). LASSO goes from AUC=1.0 (flat, rho=0) at low T2 to responsive (rho=-0.78) at high T2, where classification degrades enough that device variation matters.

**Interpretation**: The sensitivity map composes ASYMMETRICALLY. T2 decoherence degrades device-sensitive methods but device variation never degrades T2-sensitive methods. The practical implication: in high-decoherence regimes, even device-sensitive methods lose resolution on the calibration axis.

---

## 4. Extension 4: Designed metrics (L1, L2, L3, DC-SAM)

Pre-registration: `docs/PREREGISTRATION_DESIGNED_METRICS.md` (SHA 740a1de, amended A1-A2).

### Motivation

ALL 30 catalogued methods achieve |rho| < 0.15 on sigma_device. The invariance-blindness law says this is because their invariances were inherited accidentally from their math. Extension 4 builds metrics with CHOSEN invariances to close the blind spot.

### The ladder

**L1 — Invariance-audited composite** (closed-form, ~30 lines, no training):
- T_loc: mean pairwise distance between per-device feature means. Breaks translation invariance → sees device offsets.
- T_scale: coefficient of variation of per-device amplitude spread. Breaks scale invariance → sees gain.
- T_geo: Grassmannian geodesic (unchanged). Invariant control-within-the-metric.
- Scalar score: T_loc + T_scale. T_geo reported separately.

**L2 — Tunable-invariance dial** (one parameter lambda in [0,1]):
- lambda=0: subtracts full device mean → geodesic-like (blind)
- lambda=1: keeps device mean → L1-like (sensitive)
- The mechanism proof: rho(lambda) should rise monotonically from ~0 to >0.8.

**L3 — Adaptive metric selection** (routing procedure, EXPLORATORY):
- Estimates dominant noise geometry, routes to appropriate metric.
- Expected to FAIL if restricted to existing methods only (none see sigma_device).

**DC-SAM — Device-conditional structured autoencoder** (~120 lines, needs training):
- MLP encoder, latent split z_dev + z_sig, device-conditional prior.
- "If simple doesn't work, learning can" existence proof.

### Confirmatory hypothesis

H_designed: At least one designed metric in {L1, L2, DC} achieves |rho| > 0.8 on sigma_device (alpha=0.05, single family).

**Specificity guard**: must also achieve |rho| < 0.3 on n_centers (anti-detect-everything).

**Pre-committed interpretation**:
- L1 passes → "closed-form decomposition closes the blind spot; learning unnecessary"
- Only DC-SAM passes → "blind spot requires learned device-conditional representation"

### Validation hierarchy (4 levels, all metrics must pass all 4)

| Level | Name | Test | Threshold |
|-------|------|------|-----------|
| 1 (weakest) | IIA | Swap device component, check prediction flip | IIA > 0.5 |
| 2 | Faithfulness | Diversity ratio + reconstruction MSE | rho > 0.8, MSE < tau |
| 3 | Distributional | KL divergence + logit difference post-intervention | KL < 2.0, logit_diff in [0,2] |
| 4 (strongest) | Equivariance | Apply known group action, test predicted transformation | accuracy > 0.8 |

Level 4 (equivariance) is primary for sigma_device and sigma_gain because both are clean group actions (additive, multiplicative).

### Scale-sensitive comparators (from H_gain)

VN entropy (rho=-0.806 on gain) and sheaf H1 (rho=+0.818 on gain) are registered as EXPLORATORY comparators on sigma_device. Key question: does VN entropy already serve as T_scale?

### Current implementation status

**DONE** (code written and unit-tested):
- `src/designed_metrics.py` — L1 composite, L1 components, L2 dial, L2 sweep. 10/10 tests pass.
- `src/validation_hierarchy.py` — Level 4 equivariance (additive + multiplicative). 5/5 tests pass.
- Pre-registration amended (A1: 200 seeds, A2: VN/sheaf comparators).

**Unit test results confirming correctness**:
- T_loc responds to device offset (>2x increase) ✓
- T_scale responds to device gain (>2x increase) ✓
- T_geo flat under both offset and gain (ratio 0.5-2.0) ✓
- L2 at lambda=0 is blind, lambda=1 is sensitive ✓
- L2 sweep is monotonic from blind to sensitive ✓
- L1 specificity: score does NOT scale linearly with n_centers ✓
- Equivariance distinguishes geodesic (blind, acc>0.7) from T_loc (sensitive, acc>0.7) ✓

**NOT DONE** (blocked on Perplexity review of amended pre-registration):
- Confirmatory sigma_device sweep with L1/L2 (200 seeds)
- L3 implementation (thin wrapper)
- DC-SAM implementation (~120 lines, needs training)
- Levels 1-3 of validation hierarchy on real sweep data

### Amendment A1-A2 details

A1: Bumped seed count from 100 to 200. Procrustes at std=0.29 on 100 seeds was underpowered against the 0.15 threshold on H_gain.

A2: Added VN entropy and sheaf H1 as exploratory scale-sensitive comparators on sigma_device. Emerged from H_gain; registered before sigma_device run to keep them non-post-hoc.

---

## 5. Pre-registration SHAs and process

| Document | SHA | Status |
|----------|-----|--------|
| Original pre-registration (5 axes, 23+7 methods) | 8401d54 | FROZEN, all experiments run |
| Extensions v1 (gain, dT, grid) | 567c3bc | Superseded by v2 |
| Extensions v2 (amended: pilot disclosure, alpha, dT criterion, gain physics, exploratory downgrade) | e31a495 | FROZEN, H_gain done, H_dT/H_int running |
| Designed metrics v1 (L1/L2/L3/DC-SAM + hierarchy) | 740a1de | FROZEN, amended A1-A2, needs re-stamp |

**Process rule**: NO experiments run without pre-registration frozen AND Perplexity confirmation. This was enforced when the author nearly ran experiments before the pre-registration issues were caught and fixed.

---

## 6. Code structure

```
geometric-sensor-robustness/
├── src/
│   ├── nv_diamond.py          # Simulation oracle (6 axes, generate_dataset)
│   ├── methods.py             # 23 geometric methods (METHODS dict, IDs 1-23)
│   ├── baselines.py           # 7 scalar baselines (BASELINES dict)
│   ├── designed_metrics.py    # L1, L2 designed metrics (NEW)
│   ├── validation_hierarchy.py # Level 4 equivariance (NEW)
│   ├── runner.py              # Experiment runner (1D sweep + 2D grid)
│   ├── analysis.py            # Bootstrap CI, rho computation
│   └── generate_paper_figures.py
├── tests/
│   ├── test_methods.py
│   ├── test_baselines.py
│   ├── test_nv_diamond.py
│   ├── test_designed_metrics.py    # 10 tests, all pass (NEW)
│   └── test_validation_hierarchy.py # 5 tests, all pass (NEW)
├── results/
│   ├── sweep_results_100seed.json  # Main 150k experiment (5 axes)
│   ├── sweep_results_gain.json     # H_gain (30k, COMPLETE)
│   ├── sweep_results_dT_0.2.json   # H_dT partial (RUNNING)
│   └── sweep_results_grid.jsonl    # H_int partial (RUNNING)
├── docs/
│   ├── paper_v7.tex                # Current paper draft
│   ├── PREREGISTRATION.md          # Original (SHA 8401d54)
│   ├── PREREGISTRATION_EXTENSIONS.md   # Extensions (SHA e31a495)
│   ├── PREREGISTRATION_DESIGNED_METRICS.md # Designed metrics (SHA 740a1de)
│   └── DESIGNED_METRICS_SPEC.md    # Buildable specs for L1/L2/L3/DC-SAM
└── ...
```

**Run commands**:
```bash
# Main sweep (one axis, 100 seeds)
uv run --with scipy --with scikit-learn --with tqdm python src/runner.py \
  --axes sigma_gain --n-seeds 100 --base-seed 1100 \
  --output results/sweep_results_gain.json

# Delta_T sweep
uv run --with scipy --with scikit-learn --with tqdm python src/runner.py \
  --axes alpha_T2 --n-seeds 100 --base-seed 1200 --delta-t 0.3 \
  --output results/sweep_results_dT_0.3.json

# 2D grid
uv run --with scipy --with scikit-learn --with tqdm python src/runner.py \
  --grid alpha_T2 sigma_device --grid-levels 6 6 --n-seeds 50 --base-seed 1300 \
  --output results/sweep_results_grid.json
```

---

## 7. What's next (priority order)

1. ~~Wait for H_dT and H_int to finish~~ — **BOTH COMPLETE** (2026-07-10)
2. **Get Perplexity review of amended designed-metrics pre-registration** (200 seeds + VN/sheaf comparators)
3. **Re-stamp SHA** after review confirms amendments
4. **Run confirmatory L1/L2 sweep on sigma_device** (200 seeds, ~2 hours)
5. **Full analysis of H_dT and H_int results** — raw numbers computed, need paper-ready interpretation
6. **Implement L3 and DC-SAM** if L1 alone doesn't close the blind spot
7. **Update paper_v8.tex** with all extension results + designed metrics

### Paper structure update needed

paper_v7.tex covers the original 5-axis experiment. Needs:
- New axis table entry for sigma_gain
- H_gain results section (scale-invariance blindness extends the principle)
- Sheaf H1/VN entropy surprise finding on gain
- Bracket-norm reversal as pre-registration validation story
- H_dT and H_int results (when available)
- Designed metrics results (when confirmatory run completes)
- Updated abstract and conclusion

### Venue

Not yet decided. Candidates include F1000Research (registered reports), PLOS ONE, or a sensor-focused journal. The pre-registration discipline is the selling point.

---

## 8. Key numbers reference table

### Original experiment (150k evaluations, 100 seeds, SHA 8401d54)

| Family | Best method | alpha_T2 | sigma_surf | sigma_dev | sigma_temp | n_cent | >0.8 |
|--------|------------|----------|------------|-----------|------------|--------|------|
| Bracket | bracket_norm_raw | **+1.00** | **+0.90** | +0.65 | **+0.99** | **+0.95** | 4/5 |
| Bracket | transport_stable | **+1.00** | **+0.93** | -0.01 | **+1.00** | **+0.99** | 4/5 |
| Topology | persistent_h0 | **+1.00** | **+1.00** | +0.44 | **+0.98** | **+1.00** | 4/5 |
| Alignment | CKA | **-1.00** | **-0.99** | +0.01 | **-1.00** | **+1.00** | 4/5 |
| Quantum | fidelity_proxy | **-0.96** | **-0.99** | +0.01 | **+1.00** | **-1.00** | 4/5 |
| Alignment | Procrustes | **+1.00** | **+1.00** | +0.01 | **+1.00** | **-1.00** | 4/5 |
| Grassmannian | geodesic | +0.61 | -0.52 | -0.05 | -0.07 | **+0.88** | 1/5 |
| Sheaf | sheaf_h1 | -0.15 | -0.17 | -0.06 | +0.43 | -0.59 | 0/5 |
| Classification | LASSO | **-0.97** | --- | --- | **-1.00** | --- | 2/2 |

### H_gain extension (30k evaluations, 100 seeds)

| Method | rho on sigma_gain | Invariance |
|--------|------------------|------------|
| geodesic | +0.048 | Scale-blind (confirmed) |
| CKA | -0.115 | Scale-blind (confirmed) |
| Berry | +0.073 | Scale-blind (confirmed) |
| Procrustes | +0.230 | Borderline (underpowered) |
| sheaf_h1 | +0.818 | **Scale-sensitive** (surprise) |
| VN entropy | -0.806 | **Scale-sensitive** (surprise) |
| fidelity_proxy | -0.655 | Scale-sensitive |
| IRM penalty | +0.564 | Scale-sensitive |
| All classifiers | 0.000 | Unaffected |
