#!/usr/bin/env python3
"""Phase-folded ZTF & ATLAS lightcurves — one panel per survey."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

# ── Config ──────────────────────────────────────────────────
PERIOD = 0.257480
NBINS  = 150
FONT   = Path("../../assets/fonts/Inter-VariableFont_opsz,wght.ttf")

C_TEXT     = "#2C241A"
C_TEXT_SEC = "#5C4F3D"
C_BORDER   = "#5C4F3D"
C_ACCENT   = "#B85B3A"

SURVEYS = {
    "ztf": {
        "file": Path("ztf_lc.txt"),
        "out":  Path("ztf_phased.svg"),
        "filters": {
            "zg": {"color": "#558B44", "label": "ZTF g"},
            "zr": {"color": "#C43E2A", "label": "ZTF r"},
            "zi": {"color": "#9A6B92", "label": "ZTF i"},
        },
    },
    "atlas": {
        "file": Path("atlas_lc.txt"),
        "out":  Path("atlas_phased.svg"),
        "filters": {
            "c": {"color": "#4A78A8", "label": "ATLAS cyan"},
            "o": {"color": "#C89B1E", "label": "ATLAS orange"},
        },
    },
}

# ── Font ────────────────────────────────────────────────────
font_manager.fontManager.addfont(str(FONT))
fp_label  = font_manager.FontProperties(fname=str(FONT), size=14, weight="medium")
fp_tick   = font_manager.FontProperties(fname=str(FONT), size=11)
fp_legend = font_manager.FontProperties(fname=str(FONT), size=11)

plt.rcParams.update({
    "text.color":        C_TEXT,
    "axes.labelcolor":   C_TEXT,
    "xtick.color":       C_TEXT_SEC,
    "ytick.color":       C_TEXT_SEC,
    "axes.edgecolor":    C_BORDER,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "svg.fonttype":      "path",
})


def phase_fold(time, flux):
    t0 = time[np.argmin(flux)]
    return ((time - t0) / PERIOD) % 1.0


def bin_phase(phase, flux, flux_err, nbins):
    edges   = np.linspace(0, 1, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx     = np.digitize(phase, edges) - 1

    bf = np.full(nbins, np.nan)
    be = np.full(nbins, np.nan)
    for i in range(nbins):
        m = idx == i
        if m.sum() > 0:
            w = 1.0 / flux_err[m] ** 2
            bf[i] = np.average(flux[m], weights=w)
            be[i] = 1.0 / np.sqrt(w.sum())
    return centers, bf, be


def load_lc(path):
    raw = np.genfromtxt(path, delimiter=",", dtype=None, encoding="utf-8")
    time     = np.array([float(r[0]) for r in raw])
    flux     = np.array([float(r[1]) for r in raw])
    flux_err = np.array([float(r[2]) for r in raw])
    filt     = np.array([str(r[3])   for r in raw])
    return time, flux, flux_err, filt


def make_plot(survey_cfg):
    time, flux, flux_err, filt = load_lc(survey_cfg["file"])
    filters = survey_cfg["filters"]
    present = [f for f in filters if np.any(filt == f)]
    if not present:
        print(f"  No matching filters in {survey_cfg['file']}, skipping.")
        return

    # Use all data for a common t0
    t0 = time[np.argmin(flux)]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Vertical offset between filters so they don't overlap
    offsets = {}
    medians = {}
    for fname in present:
        mask = filt == fname
        medians[fname] = np.nanmedian(flux[mask])
    ref = medians[present[0]]
    for fname in present:
        offsets[fname] = ref - medians[fname]

    for fname in present:
        cfg  = filters[fname]
        mask = filt == fname
        t, f, fe = time[mask], flux[mask] + offsets[fname], flux_err[mask]
        ph = ((t - t0) / PERIOD) % 1.0

        ph2 = np.concatenate([ph, ph + 1.0])
        f2  = np.tile(f, 2)

        bc, bf, be = bin_phase(ph, f, fe, NBINS)
        bc2 = np.concatenate([bc, bc + 1.0])
        bf2 = np.tile(bf, 2) + offsets[fname]
        be2 = np.tile(be, 2)

        ax.scatter(ph2, f2, s=1.0, color=cfg["color"], alpha=0.12,
                   edgecolors="none", rasterized=True, zorder=1)
        ax.errorbar(bc2, bf2, yerr=be2, fmt="o", ms=3.0, lw=0,
                    elinewidth=0.7, color=cfg["color"],
                    markeredgewidth=0, capsize=0, zorder=2,
                    label=cfg["label"])

    leg = ax.legend(prop=fp_legend, frameon=False, loc="upper right",
                    handletextpad=0.4, borderpad=0.3)
    for text in leg.get_texts():
        text.set_color(C_TEXT)

    ax.set_xlabel("Phase", fontproperties=fp_label, labelpad=6, color=C_BORDER)
    ax.set_ylabel("Relative flux", fontproperties=fp_label, labelpad=6, color=C_BORDER)

    ax.set_xlim(-0.02, 2.02)
    ax.tick_params(top=True, right=True)
    ax.grid(False)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fp_tick)
    for spine in ax.spines.values():
        spine.set_color(C_BORDER)

    fig.tight_layout(pad=0.6)
    fig.savefig(survey_cfg["out"], format="svg", transparent=True,
                bbox_inches="tight", pad_inches=0.08, dpi=300)
    print(f"Wrote {survey_cfg['out']}  "
          f"({survey_cfg['out'].stat().st_size / 1024:.0f} kB)")


for name, cfg in SURVEYS.items():
    print(f"Plotting {name} …")
    make_plot(cfg)