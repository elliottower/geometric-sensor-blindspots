"""Generate all 5 paper figures from sweep results.

Usage:
    uv run --with scipy --with numpy --with matplotlib \
        python src/generate_paper_figures.py results/sweep_results_100seed.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / "results"

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


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return data["results"], data["meta"]


def compute_rho_table(results):
    """Compute Spearman rho(score, level) per (method, axis) across seeds."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("error") or r["score"] is None:
            continue
        grouped[(r["method"], r["axis"])][r["seed"]].append((r["level"], r["score"]))

    rho_table = {}
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
        rho_table[(method, axis)] = {
            "median": np.median(rhos),
            "mean": np.mean(rhos),
            "std": np.std(rhos),
            "rhos": rhos,
        }
    return rho_table


def get_scores_by_level(results, method, axis):
    """Get {level: [scores across seeds]} for one method-axis pair."""
    by_level = defaultdict(list)
    for r in results:
        if r["method"] == method and r["axis"] == axis and r.get("error") is None:
            by_level[r["level"]].append(r["score"])
    return dict(sorted(by_level.items()))


# ── Fig 1: Sensitivity heatmap ──────────────────────────────────────

def fig1_heatmap(results, rho_table, n_seeds):
    method_order = [
        "bracket_norm_raw", "transport_stable_bracket", "persistent_h0",
        "cka", "procrustes", "fidelity_proxy", "persistent_sheaf_cohomology",
        "vn_entropy_stability", "berry_phase", "qfi_flatness",
        "spectral_gap_stability", "grassmannian_geodesic",
        "grassmannian_holonomy", "ollivier_ricci_curvature",
        "forman_ricci_curvature", "localized_bracket",
        "sheaf_q_per_edge", "cohens_d_mean",
        "irm_penalty", "ttest_max",
    ]
    method_labels = [
        "Bracket (raw)", "Transport-stable", "Persistent H0",
        "CKA", "Procrustes", "Fidelity proxy", "Persistent sheaf",
        "VN entropy", "Berry phase", "QFI flatness",
        "Spectral gap", "Grassmann. geodesic",
        "Grassmann. holonomy", "Ollivier-Ricci",
        "Forman-Ricci", "Localized bracket",
        r"Sheaf Q (per-edge)", "Cohen's d",
        "IRM penalty", "t-test (max)",
    ]
    axis_order = ["alpha_T2", "sigma_surface", "sigma_device", "sigma_temp", "n_centers"]
    axis_labels = [r"$\alpha_{T2}$", r"$\sigma_\mathrm{surf}$", r"$\sigma_\mathrm{dev}$",
                   r"$\sigma_\mathrm{temp}$", r"$n_\mathrm{cent}$"]

    mat = np.zeros((len(axis_order), len(method_order)))
    for i, ax in enumerate(axis_order):
        for j, m in enumerate(method_order):
            key = (m, ax)
            if key in rho_table:
                mat[i, j] = rho_table[key]["median"]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(np.abs(mat), cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            absval = abs(val)
            weight = "bold" if absval > 0.8 else "normal"
            color = "white" if absval > 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=6.5, fontweight=weight, color=color)

    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(axis_labels)))
    ax.set_yticklabels(axis_labels, fontsize=9)
    ax.set_title(f"Method--axis sensitivity map (median Spearman $\\rho$, {n_seeds} seeds)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$|\rho|$ (responsiveness)", fontsize=8)

    for path in [OUT / "fig1_sensitivity_heatmap.pdf", OUT / "fig1_sensitivity_heatmap.png"]:
        fig.savefig(path)
    plt.close(fig)
    print(f"  Fig 1 saved")


# ── Fig 2: Geodesic vs bracket on sigma_device ──────────────────────

def fig2_geodesic_vs_bracket(results, rho_table):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    for method, ax_plot, title_prefix in [
        ("bracket_norm_raw", ax1, "Bracket norm (raw)"),
        ("grassmannian_geodesic", ax2, "Grassmannian geodesic"),
    ]:
        by_level = get_scores_by_level(results, method, "sigma_device")
        levels = sorted(by_level.keys())
        means = [np.mean(by_level[l]) for l in levels]
        stds = [np.std(by_level[l]) for l in levels]
        rho_val = rho_table[(method, "sigma_device")]["median"]

        ax_plot.errorbar(levels, means, yerr=stds, fmt="o-", capsize=3,
                         markersize=4, linewidth=1.5, color="#2166AC")
        ax_plot.set_xlabel(r"$\sigma_\mathrm{device}$")
        ax_plot.set_ylabel("Score")
        ax_plot.set_title(f"{title_prefix}\n($\\rho = {rho_val:+.2f}$)")
        ax_plot.grid(True, alpha=0.3)

    fig.suptitle(r"H3a refutation: geodesic vs bracket on $\sigma_\mathrm{device}$",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    for path in [OUT / "fig2_geodesic_vs_bracket.pdf", OUT / "fig2_geodesic_vs_bracket.png"]:
        fig.savefig(path)
    plt.close(fig)
    print(f"  Fig 2 saved")


# ── Fig 3: Curvature phase transition on alpha_T2 ───────────────────

def fig3_curvature_phase(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    for method, ax_plot, label, color in [
        ("ollivier_ricci_curvature", ax1, "Ollivier-Ricci", "#B2182B"),
        ("forman_ricci_curvature", ax2, "Forman-Ricci", "#2166AC"),
    ]:
        by_level = get_scores_by_level(results, method, "alpha_T2")
        levels = sorted(by_level.keys())
        means = [np.mean(by_level[l]) for l in levels]
        stds = [np.std(by_level[l]) for l in levels]

        ax_plot.errorbar(levels, means, yerr=stds, fmt="o-", capsize=3,
                         markersize=4, linewidth=1.5, color=color)
        ax_plot.set_xlabel(r"$\alpha_{T2}$")
        ax_plot.set_ylabel("Curvature score")
        ax_plot.set_title(label)
        ax_plot.grid(True, alpha=0.3)
        ax_plot.axvline(x=0.33, color="gray", linestyle="--", alpha=0.5, label="Collapse threshold")
        ax_plot.legend(fontsize=7)

    fig.suptitle(r"Curvature phase transition on $\alpha_{T2}$", fontsize=11, y=1.02)
    fig.tight_layout()
    for path in [OUT / "fig3_curvature_phase_transition.pdf", OUT / "fig3_curvature_phase_transition.png"]:
        fig.savefig(path)
    plt.close(fig)
    print(f"  Fig 3 saved")


# ── Fig 4: Persistent H0 across all axes ────────────────────────────

def fig4_persistent_h0(results, rho_table):
    axes = ["alpha_T2", "sigma_surface", "sigma_device", "sigma_temp", "n_centers"]
    axis_labels = [r"$\alpha_{T2}$", r"$\sigma_\mathrm{surf}$", r"$\sigma_\mathrm{dev}$",
                   r"$\sigma_\mathrm{temp}$", r"$n_\mathrm{cent}$"]
    colors = ["#2166AC", "#4393C3", "#D6604D", "#92C5DE", "#B2182B"]

    fig, axs = plt.subplots(1, 5, figsize=(16, 3))
    for i, (axis, label, color) in enumerate(zip(axes, axis_labels, colors)):
        by_level = get_scores_by_level(results, "persistent_h0", axis)
        levels = sorted(by_level.keys())
        means = [np.mean(by_level[l]) for l in levels]
        stds = [np.std(by_level[l]) for l in levels]
        rho_val = rho_table[("persistent_h0", axis)]["median"]

        axs[i].errorbar(levels, means, yerr=stds, fmt="o-", capsize=2,
                        markersize=3, linewidth=1.2, color=color)
        axs[i].set_xlabel(label)
        if i == 0:
            axs[i].set_ylabel("Total H0 persistence")
        axs[i].set_title(f"$\\rho = {rho_val:+.2f}$", fontsize=9)
        axs[i].grid(True, alpha=0.3)
        axs[i].tick_params(labelsize=6.5)

    fig.suptitle("Persistent H0 across all five decoherence axes", fontsize=11, y=1.02)
    fig.tight_layout()
    for path in [OUT / "fig4_persistent_h0_generalist.pdf", OUT / "fig4_persistent_h0_generalist.png"]:
        fig.savefig(path)
    plt.close(fig)
    print(f"  Fig 4 saved")


# ── Fig 5: Sheaf split on alpha_T2 ──────────────────────────────────

def fig5_sheaf_split(results, rho_table):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    for method, ax_plot, label, color in [
        ("sheaf_q_per_edge", ax1, r"Sheaf H$^1$ (Cochran's Q)", "#D6604D"),
        ("persistent_sheaf_cohomology", ax2, "Persistent sheaf cohomology", "#2166AC"),
    ]:
        by_level = get_scores_by_level(results, method, "alpha_T2")
        levels = sorted(by_level.keys())
        means = [np.mean(by_level[l]) for l in levels]
        stds = [np.std(by_level[l]) for l in levels]
        rho_val = rho_table[(method, "alpha_T2")]["median"]

        ax_plot.errorbar(levels, means, yerr=stds, fmt="o-", capsize=3,
                         markersize=4, linewidth=1.5, color=color)
        ax_plot.set_xlabel(r"$\alpha_{T2}$")
        ax_plot.set_ylabel("Score")
        ax_plot.set_title(f"{label}\n($\\rho = {rho_val:+.2f}$)")
        ax_plot.grid(True, alpha=0.3)

    fig.suptitle(r"Sheaf cohomology split on $\alpha_{T2}$", fontsize=11, y=1.02)
    fig.tight_layout()
    for path in [OUT / "fig5_sheaf_split.pdf", OUT / "fig5_sheaf_split.png"]:
        fig.savefig(path)
    plt.close(fig)
    print(f"  Fig 5 saved")


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/generate_paper_figures.py <sweep_results.json>")
        sys.exit(1)

    results, meta = load_results(sys.argv[1])
    n_seeds = meta["config"]["n_seeds"]
    print(f"Loaded {len(results)} results ({n_seeds} seeds)")

    rho_table = compute_rho_table(results)
    print(f"Computed rho for {len(rho_table)} method-axis pairs")

    print("Generating figures...")
    fig1_heatmap(results, rho_table, n_seeds)
    fig2_geodesic_vs_bracket(results, rho_table)
    fig3_curvature_phase(results)
    fig4_persistent_h0(results, rho_table)
    fig5_sheaf_split(results, rho_table)
    print("Done.")


if __name__ == "__main__":
    main()
