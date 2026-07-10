# Designed Metrics: Buildable Specs (L1, L2, L3, DC-SAM)

Feature dim d = n_centers * 5 (=100 at n_centers=20).
Simulator returns (X, y, device_ids); device_ids logged per sample.
All scores are computed PER dataset (one scalar/vector per noise level).

================================================================
## L1 — Invariance-Audited Composite Metric (FLAGSHIP BASELINE)
================================================================

### Principle
Report a VECTOR, one component per invariance class deliberately broken.
Each component is closed-form and its invariance is provable.

### Components (all operate on feature matrix X, labels y, device_ids d)

1. Location term  T_loc  (breaks translation invariance -> sees offsets)
   - Compute per-device feature means m_dev = mean(X[d==dev], axis=0).
   - T_loc = mean over device pairs of || m_dev_i - m_dev_j ||_2
     (grand-mean-centered so it measures device SPREAD, not global shift).
   - Provable: NOT invariant to additive per-device offset c_dev
     (m_dev -> m_dev + c_dev changes T_loc). This is the term geodesic lacks.

2. Scale term  T_scale  (breaks scale invariance -> sees gain)
   - s_dev = mean over amplitude features of std(X[d==dev]) (amplitude
     features only, per the gain-physics mask: exclude resonance freq idx 3).
   - T_scale = std(s_dev) / mean(s_dev)   (coefficient of variation across devices).
   - Provable: NOT invariant to multiplicative gain g_dev.

3. Subspace-angle term  T_geo  (the existing geodesic, unchanged)
   - Top-k PCA subspace per device -> Grassmannian geodesic between device
     subspaces (reuse existing grassmannian_geodesic code). Translation-
     AND scale-invariant.

### Composite score
L1_vector = [T_loc, T_scale, T_geo]   (report as vector)
Scalar collapse for rho test: L1_scalar = T_loc + T_scale  (the two terms
designed to see device variation). Report T_geo separately to show it
stays flat (the control-within-the-metric).

### Non-redundancy check (defensibility)
Compute pairwise Spearman between T_loc, T_scale, T_geo across all noise
levels. The parent results predict these are near-orthogonal (different
methods respond to disjoint axes). Report the 3x3 correlation matrix.

### Why it can't cheat
Closed-form -> no free parameters to overfit; specificity on n_centers is
an empirical prediction (adding centers should not inflate T_loc/T_scale).

### Effort: ~30 lines, no training, existing data.

================================================================
## L2 — Tunable-Invariance Dial (MECHANISM PROBE)
================================================================

### Principle
One analytic metric with an equivariance-breaking parameter lambda in [0,1].
lambda=0 -> fully translation/scale-invariant (reproduces geodesic).
lambda=1 -> fully sensitive (reproduces L1_scalar).

### Construction (linear interpolation in the pre-projection step)
For each device subspace computation, before extracting the PCA subspace,
apply a partial de-centering:
   X_dev(lambda) = X_dev - (1 - lambda) * m_dev
- lambda=0: subtracts the full device mean -> geodesic sees no offset (blind).
- lambda=1: keeps the device mean -> offset fully visible (sensitive).
Then compute the geodesic-style score on X_dev(lambda).

Optional second knob for scale (same idea on std): divide amplitude
features by (1 - lambda) * s_dev + lambda * 1.

### The demonstration (this IS the mechanism proof)
Sweep lambda in {0.0, 0.1, ..., 1.0}. For each lambda, compute rho vs
sigma_device across the 100 seeds. Plot rho(lambda). Pre-registered
prediction: rho rises monotonically from ~0 at lambda=0 to >0.8 at lambda=1.

### Effort: L1 + a scalar interpolation; ~15 extra lines.

================================================================
## L3 — Adaptive Metric-Selection Procedure (DEPLOYMENT BASELINE)
================================================================

### Principle
Not a new distance — a PROCEDURE. Given a dataset, estimate the dominant
noise geometry, then route to the existing metric whose invariances match.

### Steps
1. Cheap noise-geometry estimator: compute
   - offset_signal = T_loc (from L1)
   - scale_signal  = T_scale (from L1)
   - curvature_signal = variance explained by top-2 PCA (proxy for alpha_T2)
2. argmax over {offset, scale, curvature} -> dominant axis.
3. Route: offset/scale -> route to L1 (since no existing metric works);
   curvature -> route to bracket norm (which handles alpha_T2).

### The pre-registered result (expected NEGATIVE for existing-only routing)
If L3 is restricted to routing among the 30 EXISTING metrics only, it
CANNOT close the sigma_device blind spot (none are device-sensitive).
This is the point: selection among blind methods can't see. L3 only
succeeds when L1/DC are in its candidate pool -> motivates designed metrics.

### Effort: thin wrapper over L1 + existing per-axis rho table.

================================================================
## DC-SAM — Device-Conditional Structured Autoencoder (FLAGSHIP LEARNED)
================================================================

### Architecture
- Encoder f_phi: MLP d->128->z, ReLU.
- Latent split: z_dev (k_d=4, device-nuisance) + z_sig (k_s=8, L1-sparse).
- Device-conditional prior p(z_dev | dev) = N(mu_dev, sigma^2 I), one
  learned mean per device ID.
- Decoder g_theta: MLP z->128->d, ReLU (reconstruction anti-cheat).

### Loss
L = ||g(f(x)) - x||^2  +  beta * KL(q(z_dev|x) || p(z_dev|dev))  +  lambda * ||z_sig||_1
Start: beta=1.0, lambda=1.0, Adam lr 1e-3, 300 epochs.
All three terms necessary (per pi-SAE 2x2 ablation).

### Score
DCSAM = E_dev[|| mu_dev - mean_mu ||_2] / std(z_sig)

### Anti-cheat (ported faithfulness metrics)
- Specificity: |rho| < 0.3 on n_centers.
- Reconstruction MSE < tau on held-out (high MSE + high score = hallucination).
- Diversity ratio rho ~ 1 (perturbed samples preserve within-device variation).

### Data
- Existing (X,y,device_ids); regenerate at n_devices=8 for the rank condition.

### Effort: ~120 lines; transcribe from pi-SAE code, drop classification head.

================================================================
## Validation Hierarchy (`validation_hierarchy.py`)
================================================================

Every metric (all 30 existing + L1/L2/L3/DC-SAM) gets scored on all
4 levels. The hierarchy applies UNIFORMLY — it's one evaluation
dimension, not a new family of methods.

### Level 1: Intervention effectiveness (IIA)

The sensor analogue of interchange intervention:
1. Pick two samples (x_a from device i, x_b from device j) with
   different labels (hard-example selection: only pairs where the
   swap CAN flip the prediction).
2. Swap the device-attributable component: replace x_a's device-mean
   component with x_b's device-mean component.
3. Re-run the LASSO classifier on the intervened sample.
4. IIA = fraction of swaps where the prediction flips as predicted.

Implementation:
```
def interchange_iia(X, y, device_ids, classifier):
    devices = np.unique(device_ids)
    device_means = {d: X[device_ids == d].mean(axis=0) for d in devices}
    grand_mean = X.mean(axis=0)

    hits, total = 0, 0
    for di in devices:
        for dj in devices:
            if di == dj:
                continue
            mask_i = device_ids == di
            mask_j = device_ids == dj
            # Hard-example selection: cross-label pairs only
            for idx_a in np.where(mask_i & (y == 0))[0]:
                for idx_b in np.where(mask_j & (y == 1))[0][:5]:
                    x_intervened = X[idx_a].copy()
                    x_intervened += (device_means[dj] - device_means[di])
                    pred_orig = classifier.predict([X[idx_a]])[0]
                    pred_new = classifier.predict([x_intervened])[0]
                    if pred_orig != y[idx_b]:
                        total += 1
                        if pred_new == y[idx_b]:
                            hits += 1
    return hits / max(total, 1)
```

Threshold: IIA > 0.5 (necessary, not sufficient).

### Level 2: Faithfulness (diversity + reconstruction)

Diversity ratio rho: after swapping device components, measure whether
per-device variance is preserved (not collapsed to a template).
```
def diversity_ratio(X, device_ids, X_intervened, device_ids_new):
    var_orig = np.mean([X[device_ids == d].var() for d in np.unique(device_ids)])
    var_new = np.mean([X_intervened[device_ids_new == d].var()
                       for d in np.unique(device_ids_new)])
    return var_new / max(var_orig, 1e-12)
```
Threshold: rho > 0.8 (if rho ~ 0, the intervention collapsed to
per-class templates — the memorization failure mode).

Reconstruction MSE: for DC-SAM only, check that autoencoder
reconstruction error stays below tau on held-out data. High MSE +
high score = hallucinated device structure.

### Level 3: Distributional quality (KL/JS + logit difference)

After intervention, measure the FULL distributional shift of the
classifier's output, not just the top-1 label.
```
def distributional_shift(classifier, x_orig, x_intervened):
    p_orig = classifier.predict_proba([x_orig])[0]
    p_new = classifier.predict_proba([x_intervened])[0]
    kl = np.sum(p_orig * np.log(p_orig / np.clip(p_new, 1e-12, None)))
    logit_diff = (np.log(p_new[1] / p_new[0]) -
                  np.log(p_orig[1] / p_orig[0]))
    return kl, logit_diff
```
Thresholds: KL < 2.0, normalized logit_diff in [0, 2]. Values
outside these ranges flag distortion even when IIA = 1.0.

### Level 4: Equivariance (PRIMARY for sigma_device and sigma_gain)

The strongest validator. Directly tests the invariance-blindness law:
does a known device-offset group action produce the PREDICTED
transformation in the metric's recovered structure?

For sigma_device (additive offset, group action c_dev):
1. Compute metric's structure S(X) on original data.
2. Apply a known offset c to all samples from one device:
   X_shifted = X.copy(); X_shifted[device_ids == d] += c
3. Compute S(X_shifted).
4. Equivariance test: does S(X_shifted) - S(X) equal the
   predicted transformation T(c)?
   - For L1's T_loc: predicted shift is ||c|| (exact, closed-form).
   - For geodesic: predicted shift is 0 (invariant — blind).
   - For DC-SAM: predicted shift is mu_dev + c in device latent.

For sigma_gain (multiplicative, group action g_dev):
Same structure, but X_shifted[amplitude_mask] *= g.
Predicted: T_scale shifts by CV of g; T_geo stays flat.

Implementation:
```
def equivariance_accuracy(metric_fn, X, y, device_ids, n_offsets=20):
    devices = np.unique(device_ids)
    rng = np.random.default_rng(42)
    correct = 0
    for _ in range(n_offsets):
        dev = rng.choice(devices)
        offset = rng.normal(0, 0.05, size=X.shape[1])
        X_shifted = X.copy()
        X_shifted[device_ids == dev] += offset

        score_orig = metric_fn(X, y, device_ids)
        score_shifted = metric_fn(X_shifted, y, device_ids)
        delta = abs(score_shifted - score_orig)

        # For a device-sensitive metric: delta should be > threshold
        # For a device-blind metric: delta should be ~ 0
        # "Correct" = delta matches the metric's claimed invariance
        correct += 1 if delta > 0.01 else 0  # simplified
    return correct / n_offsets
```

Threshold: equivariance accuracy > 0.8 for designed metrics on
sigma_device. Existing translation-invariant methods should score
< 0.2 (confirming blindness causally, not just correlationally).

### Scope limitation (pre-committed honesty)

Equivariance testing requires a GROUP ACTION to test against.
sigma_device (additive) and sigma_gain (multiplicative) are clean
group actions. alpha_T2 (curvature collapse) and sigma_surface
(non-group noise) have no natural equivariance target. For these
axes, validation falls back to Levels 1-3. This is stated up front,
not discovered post-hoc.

================================================================
## NIH proposal connection (Extension 4 framing)
================================================================

Extensions 1-3 characterize the blind spot. Extension 4 closes it.

For an R21 (exploratory/developmental):
- Aim 1: Demonstrate that designed invariance-breaking closes the device-
  variation blind spot (L1/L2 preliminary data from this paper).
- Aim 2: Validate on real NV-diamond data from collaborator lab
  (requires data sharing agreement — see docs/data_request_emails.md).

For an R01 (if R21 preliminary data is strong):
- Aim 1: DC-SAM on multi-device real sensor data (n_devices=8+).
- Aim 2: Longitudinal monitoring with adaptive protocol switching (L3).
- Aim 3: Prospective validation in tumor xenograft model.

The L1->L2->DC-SAM ladder maps directly to R21 aims: L1 is the "does
it work at all" preliminary result, L2 is the mechanism proof, DC-SAM
is the "if simple doesn't work, learning can" fallback that justifies
the R01 computational aim.
