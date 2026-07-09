# Simulated Quantum Sensor Oracle: Design Document

## Philosophy

Every method gets tested on data where we KNOW the ground truth.
The simulation must:
1. Have a known biological signal that is recoverable in principle
2. Have decoherence/noise that degrades that signal in controlled ways
3. Let us vary the degradation continuously so we can find boundary conditions
4. Support multi-sensor configurations where the same biology is observed
   by different modalities with different noise profiles

## Sensor 1: NV-diamond intracellular thermometry

### Physics model

NV-center spin-state fluorescence measures local temperature via
the zero-field splitting parameter D(T). The readout is an optically
detected magnetic resonance (ODMR) spectrum — a dip in fluorescence
intensity as microwave frequency sweeps through the spin transition.

**Signal**: Temperature shift dD/dT ~ -74 kHz/K near 300 K.
Tumor tissue is 1-2 K warmer than healthy tissue (Kucsko et al. 2013,
Nature 500:54-58).

**Readout vector**: For each NV center, we observe a fluorescence
time-trace I(t) under pulsed microwave excitation. From this:
- T2 (spin decoherence time) from Ramsey/Hahn echo decay
- ODMR linewidth (inversely proportional to T2*)
- Peak fluorescence contrast
- Resonance frequency (encodes temperature)
- Relaxation rates T1, T2, T2*

So for N sensors per sample, we get a feature vector in R^(5N).
Typical: N = 10-50 NV centers per cell.

### Simulation parameters

```
n_samples = 200 per class (tumor, healthy)
n_sensors_per_sample = 20  (NV centers)
d_features_per_sensor = 5  (T2, linewidth, contrast, freq, T1)
d_total = 100  (20 * 5)

# Ground truth signal
temp_healthy = 310.0  # K (37°C)
temp_tumor = 311.5    # K (38.5°C)

# Each feature is a function of temperature + noise
# freq = D(T) + noise  (this is the diagnostic feature)
# T2 = T2_intrinsic * exp(-gamma * surface_noise)
# contrast = contrast_base * (1 - beta * T2_degradation)
# etc.
```

### Decoherence axes (continuously variable)

1. **T2 degradation**: T2 ranges from 10 μs (clean diamond) to 0.5 μs
   (biologically degraded). Controlled by `alpha_T2 in [0, 1]`.
   At alpha_T2 = 0, all features are clean. At alpha_T2 = 1,
   T2-derived features are washed out.

2. **Surface noise**: Paramagnetic surface spins add broadband noise.
   `sigma_surface in [0, 0.5]`. Adds correlated noise across all
   features from the same NV center.

3. **Device-to-device variation**: Systematic offset in calibration.
   `delta_device ~ N(0, sigma_device)` with `sigma_device in [0, 0.1]`.
   This is a per-device shift, not per-sample.

4. **Temperature drift**: Ambient temperature fluctuation.
   `delta_temp ~ N(0, sigma_temp)` with `sigma_temp in [0, 2.0]` K.
   This directly interferes with the diagnostic signal.

5. **Ensemble size**: Number of NV centers per measurement.
   `n_sensors_per_sample in {5, 10, 20, 50}`.
   Fewer sensors = noisier per-sample estimates.

### Feature generation

```python
def generate_nv_sample(temp, alpha_T2, sigma_surface, delta_device,
                       delta_temp, n_sensors, rng):
    features = []
    actual_temp = temp + delta_temp + delta_device
    for i in range(n_sensors):
        T2_clean = rng.lognormal(mean=np.log(5e-6), sigma=0.3)
        T2 = T2_clean * np.exp(-3 * alpha_T2)
        
        freq = -74e3 * actual_temp + rng.normal(0, 1e3 / T2)
        linewidth = 1 / (np.pi * T2) + rng.normal(0, sigma_surface * 1e5)
        contrast = 0.3 * (1 - 0.5 * alpha_T2) + rng.normal(0, 0.02)
        T1 = rng.lognormal(mean=np.log(1e-3), sigma=0.2)
        
        surface_noise = rng.normal(0, sigma_surface)
        features.extend([T2 + surface_noise * 1e-7,
                         linewidth,
                         contrast,
                         freq,
                         T1])
    return np.array(features)
```

## Sensor 2: Entangled-photon microscopy

### Physics model

Entangled photon pairs (signal + idler) achieve sub-shot-noise imaging.
The biological signal is optical density / scattering from tissue
structure. In our simulation, tumor tissue has higher scattering
coefficient (mu_s ~ 10-15 cm^-1) than healthy (mu_s ~ 5-8 cm^-1).

**Readout**: For each pixel in a 2D field of view, we get:
- Coincidence count rate (signal)
- Single-photon count rate (background)
- Visibility of Hong-Ou-Mandel dip (entanglement quality)
- Spatial frequency content (from Fourier analysis of image)

For a p x p image, features are in R^(4p^2). We downsample to
k principal components.

### Simulation parameters

```
image_size = 8  # 8x8 pixel simplified field of view
n_features = 4  # per pixel
d_total_raw = 256
d_pca = 20  # keep top 20 PCs

# Ground truth
mu_s_healthy = 6.0  # cm^-1
mu_s_tumor = 12.0   # cm^-1
```

### Decoherence axes

1. **Entanglement degradation**: Visibility V goes from 1.0 (perfect
   entanglement) to 0.5 (classical limit). `V in [0.5, 1.0]`.
   Below V=0.5, quantum advantage vanishes.

2. **Detector dark counts**: Background noise proportional to
   `dark_rate in [0, 0.3]`.

3. **Photon loss**: Channel transmission `eta in [0.1, 1.0]`.
   Signal-to-noise degrades as eta^2 for coincidence detection.

4. **Spatial resolution degradation**: Point spread function width
   `sigma_psf in [0.5, 3.0]` pixels.

## Sensor 3: Spin-labeled protein detection

### Physics model

Electron spin resonance (ESR) of site-directed spin labels (SDSL)
on proteins. Detects protein conformational changes and binding events.
The biological signal: a target protein's spin label environment changes
upon ligand binding (tumor biomarker binding vs no binding).

**Readout**: ESR spectrum features per labeled site:
- g-factor (sensitive to local polarity)
- Hyperfine coupling A (sensitive to label mobility)
- Linewidth (sensitive to spin-spin interactions / aggregation)
- Rotational correlation time tau_c

For M labeled sites, features in R^(4M).

### Simulation parameters

```
n_sites = 10  # labeled protein sites
d_features_per_site = 4
d_total = 40

# Bound vs unbound conformational shift
delta_g_bound = 0.002
delta_A_bound = 5.0  # MHz
delta_tau_bound = 2.0  # ns
```

### Decoherence axes

1. **Spin relaxation**: T1e degradation, `alpha_relax in [0, 1]`.
2. **Concentration noise**: Label occupancy varies, `sigma_conc in [0, 0.5]`.
3. **Background spins**: Free radical contaminants, `n_background in [0, 20]`.

## Multi-sensor integration

All three sensors observe the SAME biological system (same tissue sample,
same disease state). The shared latent variable is the disease label
(tumor vs healthy). Each sensor provides a different VIEW of the same
underlying biology.

The multi-sensor simulation generates:
```python
def generate_multi_sensor_sample(disease_label, decoherence_params, rng):
    nv_features = generate_nv_sample(
        temp=310 + 1.5 * disease_label,  # tumor is warmer
        **decoherence_params['nv'], rng=rng)
    
    photon_features = generate_photon_sample(
        mu_s=6 + 6 * disease_label,  # tumor scatters more
        **decoherence_params['photon'], rng=rng)
    
    spin_features = generate_spin_sample(
        bound=disease_label,  # tumor has biomarker bound
        **decoherence_params['spin'], rng=rng)
    
    return nv_features, photon_features, spin_features
```

## Experimental matrix

### Single-sensor boundary conditions

For each of the 24 methods, sweep each decoherence axis while holding
others at baseline. Measure:
- Classification AUC (tumor vs healthy) using features selected by method
- Rank correlation with ground truth feature importance
- Computational cost (wall time)

Sweep grid per sensor:
- NV: 5 axes x 10 levels = 50 conditions
- Photon: 4 axes x 10 levels = 40 conditions
- Spin: 3 axes x 10 levels = 30 conditions

Total single-sensor: 120 conditions x 24 methods x 3 sensors = 8,640 experiments

Each experiment: generate 400 samples (200/class), extract features,
run method, evaluate. All deterministic given RNG seed.

### Multi-sensor experiments

For the sheaf/holonomy/transport methods (methods 8-14, 23):
- Generate matched multi-sensor data
- Vary ONE sensor's decoherence while keeping others clean
- Vary ALL sensors' decoherence simultaneously
- Cross-sensor consistency under degradation

This is where methods 10 (sheaf H^1), 9 (holonomy), 12 (TransportKit),
and 23 (persistent sheaf cohomology) get their unique test.

### Scalar baselines

For every geometric method, compare against:
- Raw t-test on each feature independently
- LASSO feature selection + logistic regression
- Random forest feature importance
- IRM (Invariant Risk Minimization) — the standard robustness baseline
- DRO (Distributionally Robust Optimization) — the other standard

A geometric method is only interesting if it beats these under SOME
conditions. The boundary conditions paper maps WHICH conditions.

## What pre-registration looks like

Before running ANY experiment:
1. Freeze all 24 method implementations
2. Freeze the simulation code
3. Freeze the experimental grid
4. Pre-register hypotheses:
   - "Persistent homology will outperform scalar baselines when
     T2 degradation > 0.5 because the topological structure of the
     readout manifold is more stable than point estimates"
   - "Sheaf H^1 will detect multi-sensor inconsistency before any
     single-sensor method shows degradation"
   - "Berry phase will be the first method to detect anisotropic
     decoherence (different degradation in different directions)"
   - etc. for all 24 methods
5. Commit SHA
6. THEN run experiments
7. Report ALL results, including nulls
