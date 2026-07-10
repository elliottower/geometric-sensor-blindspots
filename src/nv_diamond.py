"""NV-diamond intracellular thermometry simulation oracle.

Physics model: NV-center spin-state fluorescence measures local temperature
via the zero-field splitting parameter D(T). Readout is optically detected
magnetic resonance (ODMR) spectrum features.

Signal: Temperature shift dD/dT ~ -74 kHz/K near 300 K.
Tumor tissue is ~1.5 K warmer than healthy tissue (Kucsko et al. 2013).

Per NV center, 5 features: T2, ODMR linewidth, fluorescence contrast,
resonance frequency, T1 relaxation. For N centers per sample, feature
vector is in R^(5N).

Six decoherence axes (each independently controllable):
  1. T2 degradation:       alpha_T2 in [0, 1]
  2. Surface noise:        sigma_surface in [0, 0.5]
  3. Device variation:     sigma_device in [0, 0.1]
  4. Temperature drift:    sigma_temp in [0, 2.0] K
  5. Ensemble size:        n_centers in {5, 10, 20, 50}
  6. Device gain:          sigma_gain in [0, 0.15]
"""

import numpy as np
from dataclasses import dataclass, field


TEMP_HEALTHY = 310.0  # K (37 C)
TEMP_TUMOR = 311.5    # K (38.5 C)
FREQ_SENSITIVITY = -74e3  # Hz/K (dD/dT)

N_FEATURES_PER_CENTER = 5
FEATURE_NAMES = ["T2", "linewidth", "contrast", "frequency", "T1"]

BASELINE_DECOHERENCE = {
    "alpha_T2": 0.0,
    "sigma_surface": 0.0,
    "sigma_device": 0.0,
    "sigma_temp": 0.0,
    "n_centers": 20,
    "sigma_gain": 0.0,
}


@dataclass(frozen=True)
class DecoherenceParams:
    alpha_T2: float = 0.0
    sigma_surface: float = 0.0
    sigma_device: float = 0.0
    sigma_temp: float = 0.0
    n_centers: int = 20
    sigma_gain: float = 0.0

    def to_dict(self):
        return {
            "alpha_T2": self.alpha_T2,
            "sigma_surface": self.sigma_surface,
            "sigma_device": self.sigma_device,
            "sigma_temp": self.sigma_temp,
            "n_centers": self.n_centers,
            "sigma_gain": self.sigma_gain,
        }


DECOHERENCE_AXES = {
    "alpha_T2": np.linspace(0.0, 1.0, 10),
    "sigma_surface": np.linspace(0.0, 0.5, 10),
    "sigma_device": np.linspace(0.0, 0.1, 10),
    "sigma_temp": np.linspace(0.0, 2.0, 10),
    "n_centers": np.array([5, 8, 10, 15, 20, 25, 30, 35, 40, 50]),
    "sigma_gain": np.linspace(0.0, 0.15, 10),
}


def generate_nv_sample(
    temp: float,
    params: DecoherenceParams,
    device_offset: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one sample's NV-diamond feature vector.

    Args:
        temp: True tissue temperature (K).
        params: Decoherence parameters for this condition.
        device_offset: Per-device calibration offset (K), drawn once per device.
        rng: NumPy random generator for reproducibility.

    Returns:
        Feature vector of shape (n_centers * 5,).
    """
    actual_temp = temp + rng.normal(0, params.sigma_temp) + device_offset
    features = np.empty(params.n_centers * N_FEATURES_PER_CENTER)

    for i in range(params.n_centers):
        T2_clean = rng.lognormal(mean=np.log(5e-6), sigma=0.3)
        T2 = T2_clean * np.exp(-3.0 * params.alpha_T2)

        linewidth_base = 1.0 / (np.pi * max(T2, 1e-9))
        freq = FREQ_SENSITIVITY * actual_temp + rng.normal(0, linewidth_base * 0.3)
        linewidth = linewidth_base + rng.normal(0, params.sigma_surface * 1e5)
        contrast = 0.3 * (1.0 - 0.5 * params.alpha_T2) + rng.normal(0, 0.02)
        T1 = rng.lognormal(mean=np.log(1e-3), sigma=0.2)

        surface_noise = rng.normal(0, params.sigma_surface)
        offset = i * N_FEATURES_PER_CENTER
        features[offset + 0] = T2 + surface_noise * 1e-7
        features[offset + 1] = linewidth
        features[offset + 2] = contrast
        features[offset + 3] = freq
        features[offset + 4] = T1

    return features


def generate_dataset(
    n_per_class: int,
    params: DecoherenceParams,
    seed: int,
    n_devices: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a full dataset of NV-diamond samples.

    Simulates n_devices sensors observing the same biology. Each device
    gets a fixed calibration offset drawn from N(0, sigma_device).

    Args:
        n_per_class: Samples per class (healthy/tumor) per device.
        params: Decoherence parameters.
        seed: RNG seed for full reproducibility.
        n_devices: Number of independent sensor devices.

    Returns:
        Tuple of (X, y, device_ids) where:
            X: (n_devices * 2 * n_per_class, n_centers * 5) feature matrix
            y: (n_devices * 2 * n_per_class,) labels (0=healthy, 1=tumor)
            device_ids: (n_devices * 2 * n_per_class,) device index per sample
    """
    rng = np.random.default_rng(seed)
    n_total = n_devices * 2 * n_per_class
    d_features = params.n_centers * N_FEATURES_PER_CENTER

    X = np.empty((n_total, d_features))
    y = np.empty(n_total, dtype=int)
    device_ids = np.empty(n_total, dtype=int)

    device_offsets = rng.normal(0, params.sigma_device, size=n_devices)
    device_gains = rng.normal(1.0, params.sigma_gain, size=n_devices)

    idx = 0
    for dev in range(n_devices):
        for label, temp in [(0, TEMP_HEALTHY), (1, TEMP_TUMOR)]:
            for _ in range(n_per_class):
                X[idx] = generate_nv_sample(temp, params, device_offsets[dev], rng)
                if params.sigma_gain > 0:
                    X[idx] *= device_gains[dev]
                y[idx] = label
                device_ids[idx] = dev
                idx += 1

    return X, y, device_ids


def sweep_single_axis(
    axis_name: str,
    n_per_class: int = 200,
    seed: int = 42,
    n_devices: int = 3,
    baseline_overrides: dict | None = None,
) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Sweep one decoherence axis while holding others at baseline.

    This implements the two-axis design principle from Paper 10:
    vary one effect while holding all others constant.

    Args:
        axis_name: Which axis to sweep (key in DECOHERENCE_AXES).
        n_per_class: Samples per class per device.
        seed: Base RNG seed. Each level gets seed + level_index.
        n_devices: Number of sensor devices.
        baseline_overrides: Override baseline values for held-constant axes.

    Returns:
        List of (axis_value, X, y, device_ids) tuples, one per level.
    """
    levels = DECOHERENCE_AXES[axis_name]
    baseline = dict(BASELINE_DECOHERENCE)
    if baseline_overrides:
        baseline.update(baseline_overrides)

    results = []
    for i, level in enumerate(levels):
        sweep_params = dict(baseline)
        sweep_params[axis_name] = int(level) if axis_name == "n_centers" else float(level)
        params = DecoherenceParams(**sweep_params)
        X, y, dev = generate_dataset(n_per_class, params, seed=seed + i, n_devices=n_devices)
        results.append((float(level), X, y, dev))

    return results
