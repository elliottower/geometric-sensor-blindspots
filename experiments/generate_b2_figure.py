"""Generate the central B2 degradation-curve figure from saved results.

Two-panel layout matching the DI paper's fig4_lod_curve style:
  a) Degradation curve: median rho_geo vs sigma (with bootstrap CI),
     holonomy overlay, negative control reference
  b) Per-seed detection power: BH-significant fraction vs sigma

Run with:
    uv run --no-project --with numpy --with scipy --with matplotlib \
        python experiments/generate_b2_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "experiments" / "results"

# ── NeurIPS style (matching DI paper) ─────────────────────────────
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
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Palette ───────────────────────────────────────────────────────
C_GEO = "#2166AC"       # edge geodesic — strong blue
C_HOL = "#B2182B"       # loop holonomy — brick red
C_NEG = "#878787"       # negative control — neutral gray
C_POWER = "#E66101"     # detection power — warm orange
C_CONFIRM = "#27AE60"   # confirmatory marker — green


def main():
    with open(RESULTS / "gating_b2_orthogonal.json") as f:
        data = json.load(f)

    sigmas = [1.0, 2.0, 3.0, 4.5]
    sigma_keys = ["1.0", "2.0", "3.0", "4.5"]

    # Extract primary geodesic
    geo_med = [data["noise_levels"][s]["primary"]["median_rho_geo"] for s in sigma_keys]
    geo_lo = [data["noise_levels"][s]["primary"]["ci_lo_geo"] for s in sigma_keys]
    geo_hi = [data["noise_levels"][s]["primary"]["ci_hi_geo"] for s in sigma_keys]

    # Extract primary holonomy
    hol_med = [data["noise_levels"][s]["primary"]["median_rho_hol"] for s in sigma_keys]
    hol_lo = [data["noise_levels"][s]["primary"]["ci_lo_hol"] for s in sigma_keys]
    hol_hi = [data["noise_levels"][s]["primary"]["ci_hi_hol"] for s in sigma_keys]

    # Extract negative control geodesic
    neg_med = [data["noise_levels"][s]["negative_control"]["median_rho_geo"] for s in sigma_keys]
    neg_lo = [data["noise_levels"][s]["negative_control"]["ci_lo"] for s in sigma_keys]
    neg_hi = [data["noise_levels"][s]["negative_control"]["ci_hi"] for s in sigma_keys]

    # Extract BH-significant counts
    n_sig = [data["noise_levels"][s]["primary"]["n_sig_bh"] for s in sigma_keys]
    n_total = 50

    # ── Figure ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.6),
                                    gridspec_kw={"wspace": 0.38})

    # ── Panel a: Degradation curve ────────────────────────────────
    # Edge geodesic (primary result)
    geo_err_lo = [m - lo for m, lo in zip(geo_med, geo_lo)]
    geo_err_hi = [hi - m for m, hi in zip(geo_med, geo_hi)]
    ax1.errorbar(sigmas, geo_med, yerr=[geo_err_lo, geo_err_hi],
                 fmt="o-", color=C_GEO, lw=1.3, markersize=4.5,
                 capsize=3, capthick=0.8, elinewidth=0.8,
                 label="Edge geodesic", zorder=5)

    # Loop holonomy (exploratory)
    hol_err_lo = [m - lo for m, lo in zip(hol_med, hol_lo)]
    hol_err_hi = [hi - m for m, hi in zip(hol_med, hol_hi)]
    ax1.errorbar(sigmas, hol_med, yerr=[hol_err_lo, hol_err_hi],
                 fmt="s--", color=C_HOL, lw=1.0, markersize=3.5,
                 capsize=2.5, capthick=0.7, elinewidth=0.7,
                 label="Loop holonomy", zorder=4, alpha=0.85)

    # Negative control
    neg_err_lo = [m - lo for m, lo in zip(neg_med, neg_lo)]
    neg_err_hi = [hi - m for m, hi in zip(neg_med, neg_hi)]
    ax1.errorbar(sigmas, neg_med, yerr=[neg_err_lo, neg_err_hi],
                 fmt="D-", color=C_NEG, lw=0.9, markersize=3,
                 capsize=2, capthick=0.6, elinewidth=0.6,
                 label="Neg. control", zorder=3, alpha=0.7)

    # Zero reference
    ax1.axhline(0, color="0.7", ls="--", lw=0.5, zorder=1)

    # Confirmatory anchor marker
    ax1.axvspan(0.85, 1.15, color=C_CONFIRM, alpha=0.08, zorder=0)

    # Annotate BH counts above geodesic points
    for i, (s, m, n) in enumerate(zip(sigmas, geo_med, n_sig)):
        offset = geo_err_hi[i] + 0.025
        ax1.text(s, m + offset, f"{n}/50",
                 ha="center", va="bottom", fontsize=6.5,
                 color=C_GEO, fontweight="bold")

    ax1.set_xlabel(r"Observation noise $\sigma$")
    ax1.set_ylabel(r"Median $\rho$(geodesic, $\varepsilon_\mathrm{pert}$)")
    ax1.set_xlim(0.5, 5.0)
    ax1.set_ylim(-0.2, 0.7)
    ax1.set_xticks(sigmas)
    ax1.legend(fontsize=6.5, frameon=False, loc="upper right")

    # ── Panel b: Per-seed detection power ─────────────────────────
    power = [n / n_total for n in n_sig]
    ax2.plot(sigmas, power, "o-", color=C_POWER, lw=1.3, markersize=4.5,
             zorder=5, label="Per-seed\nBH-significant")

    # Fill area
    ax2.fill_between(sigmas, 0, power, color=C_POWER, alpha=0.12, zorder=2)

    # Annotate fractions
    for s, p, n in zip(sigmas, power, n_sig):
        offset = 0.04 if p > 0.05 else 0.06
        ha = "left" if s == 2.0 else "center"
        x_nudge = 0.12 if s == 2.0 else 0
        ax2.text(s + x_nudge, p + offset, f"{n}/50",
                 ha=ha, va="bottom", fontsize=6.5,
                 color=C_POWER, fontweight="bold")


    ax2.set_xlabel(r"Observation noise $\sigma$")
    ax2.set_ylabel("Detection rate (BH-corrected)")
    ax2.set_xlim(0.5, 5.0)
    ax2.set_ylim(-0.02, 0.85)
    ax2.set_xticks(sigmas)


    # ── Panel labels ──────────────────────────────────────────────
    ax1.text(-0.15, 1.15, "a", transform=ax1.transAxes, fontsize=11,
             fontweight="bold", va="top")
    ax2.text(-0.15, 1.15, "b", transform=ax2.transAxes, fontsize=11,
             fontweight="bold", va="top")

    fig.savefig(OUT / "b2_degradation_curve.pdf")
    fig.savefig(OUT / "b2_degradation_curve.png")
    plt.close(fig)
    print(f"Saved: {OUT / 'b2_degradation_curve.png'}")
    print(f"Saved: {OUT / 'b2_degradation_curve.pdf'}")


if __name__ == "__main__":
    main()
