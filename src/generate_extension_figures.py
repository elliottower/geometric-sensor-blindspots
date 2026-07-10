"""Generate figures for Extensions 1-3 results.

Fig 6: H_gain — scale-invariance blindness (predicted vs surprise methods)
Fig 7: H_dT — LASSO collapse vs bracket robustness across Delta_T
Fig 8: H_int — asymmetric interaction heatmap (rho vs sigma_dev at each alpha_T2)
Fig 9: H_gain — updated heatmap with sigma_gain column

Usage:
    uv run --with scipy --with numpy --with matplotlib \
        python src/generate_extension_figures.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / "results"
RESULTS = Path(__file__).resolve().parent.parent / "results"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BLUE = "#2166AC"
RED = "#B2182B"
ORANGE = "#D6604D"
TEAL = "#4393C3"
GREEN = "#1B7837"
GRAY = "#969696"


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return data["results"], data.get("meta", {})


def compute_rho_per_seed(results, axis_key="axis"):
    """Returns {method: [rho_per_seed]}."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("error") or r["score"] is None:
            continue
        grouped[r["method"]][r["seed"]].append((r["level"], r["score"]))

    rho_by_method = {}
    for method, seed_data in grouped.items():
        rhos = []
        for seed, pairs in seed_data.items():
            pairs.sort()
            levels = [p[0] for p in pairs]
            scores = [p[1] for p in pairs]
            if len(set(scores)) < 2:
                rhos.append(0.0)
            else:
                rho, _ = stats.spearmanr(levels, scores)
                rhos.append(rho)
        rho_by_method[method] = np.array(rhos)
    return rho_by_method


def get_scores_by_level(results, method):
    by_level = defaultdict(list)
    for r in results:
        if r["method"] == method and r.get("error") is None and r["score"] is not None:
            by_level[r["level"]].append(r["score"])
    return dict(sorted(by_level.items()))


# ── Fig 6: H_gain scale-invariance blindness ────────────────────────

def fig6_gain_blindness(gain_results):
    rho_by_method = compute_rho_per_seed(gain_results)

    predicted_blind = [
        ("grassmannian_geodesic", "Geodesic"),
        ("cka", "CKA"),
        ("berry_phase", "Berry phase"),
        ("procrustes", "Procrustes"),
    ]
    surprise_sensitive = [
        ("sheaf_h1", "Sheaf H1"),
        ("vn_entropy_stability", "VN entropy"),
        ("fidelity_proxy", "Fidelity"),
        ("irm_penalty", "IRM penalty"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    # Left panel: predicted blind
    names, medians, cis_lo, cis_hi = [], [], [], []
    for method, label in predicted_blind:
        rhos = rho_by_method.get(method, np.array([0]))
        med = np.median(rhos)
        boot = np.percentile(rhos, [2.5, 97.5])
        names.append(label)
        medians.append(med)
        cis_lo.append(med - boot[0])
        cis_hi.append(boot[1] - med)

    y_pos = range(len(names))
    colors = [GREEN if abs(m) < 0.15 else ORANGE for m in medians]
    ax1.barh(y_pos, medians, xerr=[cis_lo, cis_hi], color=colors,
             edgecolor="white", capsize=3, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.axvline(0.15, color=RED, linestyle="--", alpha=0.7, label="|rho|=0.15 threshold")
    ax1.axvline(-0.15, color=RED, linestyle="--", alpha=0.7)
    ax1.set_xlabel(r"Median Spearman $\rho$")
    ax1.set_title("Predicted scale-blind (H_gain)")
    ax1.set_xlim(-0.5, 0.5)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(True, alpha=0.2, axis="x")

    # Right panel: surprise sensitive
    names2, medians2, cis_lo2, cis_hi2 = [], [], [], []
    for method, label in surprise_sensitive:
        rhos = rho_by_method.get(method, np.array([0]))
        med = np.median(rhos)
        boot = np.percentile(rhos, [2.5, 97.5])
        names2.append(label)
        medians2.append(med)
        cis_lo2.append(med - boot[0])
        cis_hi2.append(boot[1] - med)

    y_pos2 = range(len(names2))
    ax2.barh(y_pos2, medians2, xerr=[cis_lo2, cis_hi2], color=BLUE,
             edgecolor="white", capsize=3, height=0.6)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(names2)
    ax2.set_xlabel(r"Median Spearman $\rho$")
    ax2.set_title("Surprise scale-sensitive")
    ax2.set_xlim(-1, 1)
    ax2.grid(True, alpha=0.2, axis="x")

    fig.suptitle(r"Extension 1: Scale-invariance blindness on $\sigma_\mathrm{gain}$ (100 seeds)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"fig6_gain_blindness.{ext}")
    plt.close(fig)
    print("  Fig 6 saved")


# ── Fig 7: H_dT — LASSO collapse vs bracket robustness ─────────────

def fig7_delta_t_collapse():
    delta_ts = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5]

    lasso_at_max_deco = []
    lasso_at_min_deco = []

    geo_methods_to_show = [
        ("bracket_norm_raw", "Bracket norm"),
        ("persistent_h0", "Persistent H0"),
        ("transport_stable_bracket", "Transport-stable"),
    ]

    # For each geometric method: score at min and max decoherence
    geo_at_min = {m[0]: [] for m in geo_methods_to_show}
    geo_at_max = {m[0]: [] for m in geo_methods_to_show}

    for dt in delta_ts:
        fname = RESULTS / f"sweep_results_dT_{dt}.json"
        if not fname.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        results, _ = load_results(fname)

        lasso_max = [r["score"] for r in results
                     if r["method"] == "lasso_auc" and r.get("score") is not None
                     and r["level"] >= 0.99]
        lasso_min = [r["score"] for r in results
                     if r["method"] == "lasso_auc" and r.get("score") is not None
                     and r["level"] <= 0.01]
        lasso_at_max_deco.append(np.mean(lasso_max))
        lasso_at_min_deco.append(np.mean(lasso_min))

        for key, _ in geo_methods_to_show:
            scores_lo = [r["score"] for r in results
                         if r["method"] == key and r.get("score") is not None
                         and r["level"] <= 0.01]
            scores_hi = [r["score"] for r in results
                         if r["method"] == key and r.get("score") is not None
                         and r["level"] >= 0.99]
            geo_at_min[key].append(np.mean(scores_lo) if scores_lo else 0)
            geo_at_max[key].append(np.mean(scores_hi) if scores_hi else 0)

    # Normalize each geometric method to fraction of its own max across all conditions
    geo_norm_min = {}
    geo_norm_max = {}
    for key, _ in geo_methods_to_show:
        all_vals = geo_at_min[key] + geo_at_max[key]
        scale = max(abs(v) for v in all_vals) if all_vals else 1.0
        scale = max(scale, 1e-12)
        geo_norm_min[key] = [v / scale for v in geo_at_min[key]]
        geo_norm_max[key] = [v / scale for v in geo_at_max[key]]

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: LASSO AUC collapse ──
    ax1.fill_between(delta_ts, 0.95, 1.05, alpha=0.08, color=GREEN, zorder=0)
    ax1.fill_between(delta_ts, 0.45, 0.5, alpha=0.08, color=RED, zorder=0)

    ax1.plot(delta_ts, lasso_at_min_deco, "o-", color=GREEN, linewidth=2.5,
             markersize=7, zorder=3)
    ax1.plot(delta_ts, lasso_at_max_deco, "s-", color=RED, linewidth=2.5,
             markersize=7, zorder=3)

    ax1.axhline(0.5, color=GRAY, linestyle=":", alpha=0.4)

    legend_left = [
        Line2D([0], [0], color=GREEN, marker="o", linewidth=2, markersize=6,
               label="No decoherence"),
        Line2D([0], [0], color=RED, marker="s", linewidth=2, markersize=6,
               label="Max decoherence"),
        Patch(facecolor=GREEN, alpha=0.15, label=r"Near-perfect (AUC $\geq$ 0.95)"),
        Patch(facecolor=RED, alpha=0.15, label=r"At chance (AUC $\leq$ 0.5)"),
    ]
    ax1.legend(handles=legend_left, fontsize=7.5, loc="lower right", framealpha=0.95,
               title="LASSO classifier", title_fontsize=8)

    ax1.set_xlabel(r"Signal strength $\Delta T$ (K)")
    ax1.set_ylabel(r"LASSO AUC   ($\uparrow$ better)")
    ax1.set_title("Baseline (LASSO) collapses at weak signal", fontsize=10, fontweight="bold")
    ax1.grid(True, alpha=0.15)
    ax1.set_ylim(0.45, 1.05)
    ax1.set_xlim(0.15, 1.55)

    # ── Right: geometric methods at no-deco vs max-deco ──
    ax2.fill_between(delta_ts, 0.95, 1.05, alpha=0.08, color=GREEN, zorder=0)
    ax2.fill_between(delta_ts, 0.45, 0.5, alpha=0.08, color=RED, zorder=0)
    ax2.axhline(0.5, color=GRAY, linestyle=":", alpha=0.4)

    GEO_COLORS = ["#2166AC", "#4393C3", "#6BAED6"]
    GEO_MARKERS_MIN = ["o", "^", "v"]
    GEO_MARKERS_MAX = ["o", "^", "v"]

    for i, (key, label) in enumerate(geo_methods_to_show):
        ax2.plot(delta_ts, geo_norm_min[key], marker=GEO_MARKERS_MIN[i], linestyle="-",
                 color=GREEN, linewidth=2, markersize=6, alpha=0.7 + 0.1 * i, zorder=3)
        ax2.plot(delta_ts, geo_norm_max[key], marker=GEO_MARKERS_MAX[i], linestyle="-",
                 color=RED, linewidth=2, markersize=6, alpha=0.7 + 0.1 * i, zorder=3)

    legend_right = [
        Line2D([0], [0], color=GREEN, marker="o", linewidth=2, markersize=6,
               label="No decoherence"),
        Line2D([0], [0], color=RED, marker="o", linewidth=2, markersize=6,
               label="Max decoherence"),
        Line2D([], [], color="none", label=""),
        Line2D([0], [0], color=GRAY, marker="o", linewidth=0, markersize=6,
               label="Bracket norm"),
        Line2D([0], [0], color=GRAY, marker="^", linewidth=0, markersize=6,
               label="Persistent H0"),
        Line2D([0], [0], color=GRAY, marker="v", linewidth=0, markersize=6,
               label="Transport-stable"),
    ]
    ax2.legend(handles=legend_right, fontsize=7.5, loc="lower right", framealpha=0.95,
               title="Geometric methods", title_fontsize=8)

    ax2.set_xlabel(r"Signal strength $\Delta T$ (K)")
    ax2.set_ylabel(r"Normalized score   ($\uparrow$ better)")
    ax2.set_title("Geometric methods: unaffected by decoherence", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.15)
    ax2.set_ylim(0.45, 1.05)
    ax2.set_xlim(0.15, 1.55)

    fig.suptitle(r"Extension 2: Signal strength ($\Delta T$) sweep — 100 seeds per $\Delta T$",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"fig7_delta_t_collapse.{ext}")
    plt.close(fig)
    print("  Fig 7 saved")


# ── Fig 8: H_int asymmetric interaction ─────────────────────────────

def fig8_interaction_heatmap(grid_results):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    methods = [
        ("bracket_norm_raw", "Bracket norm", "o", "#2166AC"),
        ("persistent_h0", "Persistent H0", "^", "#4393C3"),
        ("transport_stable_bracket", "Transport-stable", "v", "#6BAED6"),
        ("lasso_auc", "LASSO", "s", "#B2182B"),
    ]

    # Index results by (method, axis1_val_rounded, axis2_val_rounded, seed)
    by_method = defaultdict(list)
    for r in grid_results:
        if r.get("error") or r["score"] is None:
            continue
        by_method[r["method"]].append(r)

    t2_levels = sorted({round(r["axis1_val"], 2) for r in grid_results})
    dev_levels = sorted({round(r["axis2_val"], 2) for r in grid_results})

    def rho_device_at_each_t2(method_key):
        """ρ(score, σ_device) at each α_T2 level."""
        grouped = defaultdict(lambda: defaultdict(list))
        for r in by_method[method_key]:
            grouped[round(r["axis1_val"], 2)][r["seed"]].append(
                (r["axis2_val"], r["score"]))
        curve = []
        for t2 in t2_levels:
            rhos = []
            for seed, pairs in grouped[t2].items():
                pairs.sort()
                levels = [p[0] for p in pairs]
                scores = [p[1] for p in pairs]
                if len(set(scores)) < 2:
                    rhos.append(0.0)
                else:
                    rho, _ = stats.spearmanr(levels, scores)
                    rhos.append(rho)
            curve.append(abs(np.median(rhos)) if rhos else 0.0)
        return curve

    def rho_t2_at_each_dev(method_key):
        """ρ(score, α_T2) at each σ_device level."""
        grouped = defaultdict(lambda: defaultdict(list))
        for r in by_method[method_key]:
            grouped[round(r["axis2_val"], 2)][r["seed"]].append(
                (r["axis1_val"], r["score"]))
        curve = []
        for dev in dev_levels:
            rhos = []
            for seed, pairs in grouped[dev].items():
                pairs.sort()
                levels = [p[0] for p in pairs]
                scores = [p[1] for p in pairs]
                if len(set(scores)) < 2:
                    rhos.append(0.0)
                else:
                    rho, _ = stats.spearmanr(levels, scores)
                    rhos.append(rho)
            curve.append(abs(np.median(rhos)) if rhos else 0.0)
        return curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: device sensitivity vs T2 level ──
    ax1.fill_between(t2_levels, 0.95, 1.1, alpha=0.08, color=GREEN, zorder=0)
    ax1.fill_between(t2_levels, -0.05, 0.15, alpha=0.06, color=GRAY, zorder=0)

    for key, label, marker, color in methods:
        curve = rho_device_at_each_t2(key)
        ls = "--" if key == "lasso_auc" else "-"
        ax1.plot(t2_levels, curve, marker=marker, linestyle=ls,
                 color=color, linewidth=2, markersize=6, label=label, zorder=3)

    legend_left = [
        Line2D([0], [0], color=c, marker=m, linewidth=2, markersize=6,
               linestyle="--" if k == "lasso_auc" else "-", label=l)
        for k, l, m, c in methods
    ]
    legend_left.append(Patch(facecolor=GREEN, alpha=0.15, label=r"Strong detection ($|\rho| \geq 0.95$)"))
    legend_left.append(Patch(facecolor=GRAY, alpha=0.12, label=r"Blind ($|\rho| < 0.15$)"))
    ax1.legend(handles=legend_left, fontsize=7, loc="center right", framealpha=0.95)

    ax1.set_xlabel(r"T2 decoherence $\alpha_{T2}$")
    ax1.set_ylabel(r"$|\rho|$ vs $\sigma_\mathrm{device}$   (device-detection ability)")
    ax1.set_title("Can methods detect device variation\nas T2 decoherence increases?",
                  fontsize=10, fontweight="bold")
    ax1.grid(True, alpha=0.15)
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_xlim(-0.03, 1.05)

    # ── Right: T2 sensitivity vs device variation level ──
    ax2.fill_between(dev_levels, 0.95, 1.1, alpha=0.08, color=GREEN, zorder=0)
    ax2.fill_between(dev_levels, -0.05, 0.15, alpha=0.06, color=GRAY, zorder=0)

    for key, label, marker, color in methods:
        curve = rho_t2_at_each_dev(key)
        ls = "--" if key == "lasso_auc" else "-"
        ax2.plot(dev_levels, curve, marker=marker, linestyle=ls,
                 color=color, linewidth=2, markersize=6, label=label, zorder=3)

    legend_right = [
        Line2D([0], [0], color=c, marker=m, linewidth=2, markersize=6,
               linestyle="--" if k == "lasso_auc" else "-", label=l)
        for k, l, m, c in methods
    ]
    legend_right.append(Patch(facecolor=GREEN, alpha=0.15, label=r"Strong detection ($|\rho| \geq 0.95$)"))
    legend_right.append(Patch(facecolor=GRAY, alpha=0.12, label=r"Blind ($|\rho| < 0.15$)"))
    ax2.legend(handles=legend_right, fontsize=7, loc="center right", framealpha=0.95)

    ax2.set_xlabel(r"Device variation $\sigma_\mathrm{device}$")
    ax2.set_ylabel(r"$|\rho|$ vs $\alpha_{T2}$   (T2-detection ability)")
    ax2.set_title("Can methods detect T2 decoherence\nas device variation increases?",
                  fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.15)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlim(dev_levels[0] - 0.02, dev_levels[-1] + 0.02)

    fig.suptitle("Extension 3: Asymmetric axis interaction (50 seeds per grid cell)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"fig8_interaction_asymmetric.{ext}")
    plt.close(fig)
    print("  Fig 8 saved")


# ── Fig 9: Updated heatmap with sigma_gain column ──────────────────

def fig9_extended_heatmap(original_results, gain_results):
    rho_orig = {}
    grouped = defaultdict(lambda: defaultdict(list))
    for r in original_results:
        if r.get("error") or r["score"] is None:
            continue
        grouped[(r["method"], r["axis"])][r["seed"]].append((r["level"], r["score"]))
    for (method, axis), seed_data in grouped.items():
        rhos = []
        for seed, pairs in seed_data.items():
            pairs.sort()
            levels = [p[0] for p in pairs]
            scores = [p[1] for p in pairs]
            if len(set(scores)) < 2:
                rhos.append(0.0)
            else:
                rho, _ = stats.spearmanr(levels, scores)
                rhos.append(rho)
        rho_orig[(method, axis)] = np.median(rhos)

    rho_gain = compute_rho_per_seed(gain_results)

    method_order = [
        "bracket_norm_raw", "transport_stable_bracket", "persistent_h0",
        "cka", "procrustes", "fidelity_proxy", "persistent_sheaf_cohomology",
        "vn_entropy_stability", "berry_phase", "qfi_flatness",
        "spectral_gap_stability", "grassmannian_geodesic",
        "grassmannian_holonomy", "ollivier_ricci_curvature",
        "forman_ricci_curvature", "localized_bracket",
        "sheaf_h1", "sheaf_q_per_edge", "cohens_d_mean",
        "irm_penalty", "ttest_max",
    ]
    method_labels = [
        "Bracket (raw)", "Transport-stable", "Persistent H0",
        "CKA", "Procrustes", "Fidelity proxy", "Persistent sheaf",
        "VN entropy", "Berry phase", "QFI flatness",
        "Spectral gap", "Grassmann. geodesic",
        "Grassmann. holonomy", "Ollivier-Ricci",
        "Forman-Ricci", "Localized bracket",
        "Sheaf H1", "Sheaf Q (per-edge)", "Cohen's d",
        "IRM penalty", "t-test (max)",
    ]
    axis_order = ["alpha_T2", "sigma_surface", "sigma_device", "sigma_temp", "n_centers", "sigma_gain"]
    axis_labels = [r"$\alpha_{T2}$", r"$\sigma_\mathrm{surf}$", r"$\sigma_\mathrm{dev}$",
                   r"$\sigma_\mathrm{temp}$", r"$n_\mathrm{cent}$", r"$\sigma_\mathrm{gain}$"]

    mat = np.zeros((len(axis_order), len(method_order)))
    for i, ax_name in enumerate(axis_order):
        for j, m in enumerate(method_order):
            if ax_name == "sigma_gain":
                rhos = rho_gain.get(m, np.array([0]))
                mat[i, j] = np.median(rhos)
            else:
                mat[i, j] = rho_orig.get((m, ax_name), 0.0)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(np.abs(mat), cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            absval = abs(val)
            weight = "bold" if absval > 0.8 else "normal"
            color = "white" if absval > 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=5.5, fontweight=weight, color=color)

    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(axis_labels)))
    ax.set_yticklabels(axis_labels, fontsize=9)

    # Highlight the new row
    ax.axhline(4.5, color="white", linewidth=2)
    ax.set_title("Extended method--axis sensitivity map (6 axes, 100 seeds)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$|\rho|$ (responsiveness)", fontsize=8)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"fig9_extended_heatmap.{ext}")
    plt.close(fig)
    print("  Fig 9 saved")


def main():
    print("Loading results...")

    gain_path = RESULTS / "sweep_results_gain.json"
    grid_path = RESULTS / "sweep_results_grid.json"
    orig_path = RESULTS / "sweep_results_100seed.json"

    gain_results, _ = load_results(gain_path)
    print(f"  H_gain: {len(gain_results)} results")

    grid_results, _ = load_results(grid_path)
    print(f"  H_int: {len(grid_results)} results")

    original_results, _ = load_results(orig_path)
    print(f"  Original: {len(original_results)} results")

    print("Generating extension figures...")
    fig6_gain_blindness(gain_results)
    fig7_delta_t_collapse()
    fig8_interaction_heatmap(grid_results)
    fig9_extended_heatmap(original_results, gain_results)
    print("Done.")


if __name__ == "__main__":
    main()
