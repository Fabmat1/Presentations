#!/usr/bin/env python3
"""Phase-folded TESS lightcurve plot for reveal.js presentation."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

# ── Config ──────────────────────────────────────────────────
PERIOD = 0.257480  # days
NBINS = 150
DATA = Path("tess_lc.txt")
FONT = Path("../../assets/fonts/Inter-VariableFont_opsz,wght.ttf")
OUT = Path("tess_phased.svg")

# Warm Linen palette
C_TEXT     = "#2C241A"
C_TEXT_SEC = "#5C4F3D"
C_BORDER   = "#5C4F3D"
C_ACCENT   = "#B85B3A"

# ── Font ────────────────────────────────────────────────────
font_prop       = font_manager.FontProperties(fname=str(FONT), size=13)
font_prop_label = font_manager.FontProperties(fname=str(FONT), size=14, weight="medium")
font_prop_tick  = font_manager.FontProperties(fname=str(FONT), size=11)

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

# ── Data ────────────────────────────────────────────────────
time, flux, flux_err = np.loadtxt(DATA, delimiter=",", unpack=True)

t0 = time[np.argmin(flux)]
phase = ((time - t0) / PERIOD) % 1.0

# duplicate for two full phases
phase2 = np.concatenate([phase -1, phase])
flux2 = np.tile(flux, 2)
flux_err2 = np.tile(flux_err, 2)

# ── Bin ─────────────────────────────────────────────────────
bin_edges = np.linspace(0, 1, NBINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
indices = np.digitize(phase, bin_edges) - 1

bin_flux = np.full(NBINS, np.nan)
bin_err = np.full(NBINS, np.nan)
for i in range(NBINS):
    mask = indices == i
    if mask.sum() > 0:
        w = 1.0 / flux_err[mask] ** 2
        bin_flux[i] = np.average(flux[mask], weights=w)
        bin_err[i] = 1.0 / np.sqrt(w.sum())

# two-phase binned
bc2 = np.concatenate([bin_centers-1, bin_centers])
bf2 = np.tile(bin_flux, 2)
be2 = np.tile(bin_err, 2)

# ── Plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")

ax.scatter(phase2, flux2, s=1.2, color="#DED7C8", alpha=0.35,
           edgecolors="none", rasterized=True, zorder=1)

ax.errorbar(bc2, bf2, yerr=be2, fmt="o", ms=3.2, lw=0,
            elinewidth=0.8, color=C_ACCENT, markeredgewidth=0,
            capsize=0, zorder=2)

ax.set_xlabel("Phase", fontproperties=font_prop_label, labelpad=6, color=C_BORDER)
ax.set_ylabel("Relative flux", fontproperties=font_prop_label, labelpad=6, color=C_BORDER)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(font_prop_tick)

ax.set_xlim(-1.0, 1.0)
ax.tick_params(top=True, right=True)
ax.grid(False)

for spine in ax.spines.values():
    spine.set_color(C_BORDER)

fig.tight_layout(pad=0.6)
fig.savefig(OUT, format="svg", transparent=True, bbox_inches="tight",
            pad_inches=0.08, dpi=300)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} kB)")