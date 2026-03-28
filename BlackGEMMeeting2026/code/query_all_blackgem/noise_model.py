#!/usr/bin/env python3
"""
BlackGEM photometric noise model — relative RMS vs Gaia G mag.

Physical model (per filter):

    σ_rel(m) = √[ σ_floor² + A·10^{0.4(m−m₀)} + B·10^{0.8(m−m₀)} ]

    • σ_floor  : systematic floor  (scintillation, flat-field, tracking …)
    • A term   : source-photon (Poisson) noise   ∝ 1/√F
    • B term   : sky-background noise             ∝ 1/F
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

# ── configuration ──────────────────────────────────────────────────
MIN_EPOCHS = 5          # minimum detections per source per filter
MAG_BIN    = 0.5        # bin width for the binned-median curve
M_REF      = 15.0       # pivot magnitude for the noise model
SIGMA_CLIP = 4.0        # iterative σ-clip threshold (0 = off)
# ───────────────────────────────────────────────────────────────────


# ━━━ 1. Load & quality-cut ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df = pd.read_csv("output_full.csv")
print(f"Raw rows: {len(df):,}")

mask = (
    (df["FNU_OPT"] > 0)          &
    (df["FLAGS_OPT"] == 0)        &
    (df["FLAGS_MASK"] == 0)       &
    (df["MAGERR_OPT"] < 1.0)     &    # drop very noisy detections
    (df["QC_FLAG"] != "red")           # drop red-flagged images
)
df = df.loc[mask].copy()
print(f"After quality cuts: {len(df):,} rows, "
      f"{df['SOURCE_ID'].nunique():,} unique sources, "
      f"filters: {sorted(df['FILTER'].unique())}")


# ━━━ 2. Compute per-source, per-filter scatter ━━━━━━━━━━━━━━━━━━━
def robust_sigma(x):
    """σ via MAD (median absolute deviation), resilient to outliers."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))

def sigma_clip_mask(x, nsig=SIGMA_CLIP, maxiter=5):
    """Return boolean keep-mask after iterative σ-clipping."""
    keep = np.ones(len(x), dtype=bool)
    for _ in range(maxiter):
        med = np.median(x[keep])
        sig = robust_sigma(x[keep])
        if sig == 0:
            break
        new_keep = np.abs(x - med) < nsig * sig
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return keep

records = []
for (filt, sid), g in df.groupby(["FILTER", "SOURCE_ID"]):
    if len(g) < MIN_EPOCHS:
        continue

    flux = g["FNU_OPT"].values.copy()

    # optional σ-clip
    if SIGMA_CLIP > 0:
        keep = sigma_clip_mask(flux)
        flux = flux[keep]
        if len(flux) < MIN_EPOCHS:
            continue

    med_flux = np.median(flux)
    if med_flux <= 0:
        continue

    records.append({
        "FILTER":       filt,
        "SOURCE_ID":    sid,
        "gmag":         g["PHOT_G_MEAN_MAG"].iloc[0],
        "mean_flux":    med_flux,
        "rms":          np.std(flux, ddof=1) / med_flux,      # classical
        "mad_rms":      robust_sigma(flux)   / med_flux,      # robust
        "n_obs":        len(flux),
        "med_reported": np.median(
            g["FNUERR_OPT"].values / g["FNU_OPT"].values      # pipeline σ_rel
        ),
    })

stats = pd.DataFrame(records)
print(f"\nSources with ≥{MIN_EPOCHS} epochs after σ-clip:")
print(stats.groupby("FILTER")["SOURCE_ID"].count().rename("N_sources"))


# ━━━ 3. Noise model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def noise_model(m, log_floor, log_A, log_B):
    """
    3-component photometric noise model.
    Parameters are log10 values to enforce positivity during fitting.
    """
    dm = m - M_REF
    return np.sqrt(
        10 ** (2 * log_floor)
        + 10 ** log_A * 10 ** (0.4 * dm)
        + 10 ** log_B * 10 ** (0.8 * dm)
    )


# ━━━ 4. Fit & plot per filter ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
filters = sorted(stats["FILTER"].unique())
nf = len(filters)

fig, axes = plt.subplots(1, nf, figsize=(6.5 * nf, 6),
                          squeeze=False, sharey=True)
fit_results = {}

for i, filt in enumerate(filters):
    ax = axes[0, i]
    s  = stats[stats["FILTER"] == filt]
    m  = s["gmag"].values
    σ  = s["mad_rms"].values          # robust scatter

    # ── point cloud ──
    ax.scatter(m, σ, s=2, alpha=0.2, c="steelblue",
               edgecolors="none", rasterized=True, label="per-source (MAD)")

    # ── binned median ──
    bins = np.arange(np.floor(m.min()), np.ceil(m.max()) + MAG_BIN, MAG_BIN)
    med, edges, _  = binned_statistic(m, σ, statistic="median",  bins=bins)
    cnt, _,     _  = binned_statistic(m, σ, statistic="count",   bins=bins)
    p16, _,     _  = binned_statistic(m, σ, statistic=lambda x: np.percentile(x, 16), bins=bins)
    p84, _,     _  = binned_statistic(m, σ, statistic=lambda x: np.percentile(x, 84), bins=bins)
    cen = 0.5 * (edges[:-1] + edges[1:])
    ok  = (~np.isnan(med)) & (cnt >= 5)

    ax.fill_between(cen[ok], p16[ok], p84[ok], color="salmon", alpha=0.25,
                    label="16–84 %ile")
    ax.plot(cen[ok], med[ok], "o-", color="crimson", ms=5, lw=1.5,
            label="binned median", zorder=5)

    # ── fit noise model on binned medians ──
    try:
        popt, pcov = curve_fit(
            noise_model, cen[ok], med[ok],
            p0=[-2.5, -5.0, -8.0],
            bounds=([-6, -10, -14], [0, 0, 0]),
            maxfev=20_000,
        )
        perr = np.sqrt(np.diag(pcov))
        m_grid = np.linspace(bins[0], bins[-1], 500)
        dm     = m_grid - M_REF

        floor = 10 ** popt[0]
        A     = 10 ** popt[1]
        B     = 10 ** popt[2]

        ax.plot(m_grid, noise_model(m_grid, *popt),
                "k-", lw=2.5, label="model (total)", zorder=6)

        # individual components
        ax.axhline(floor, ls=":", color="grey",   lw=1, label=f"floor = {floor:.1e}")
        ax.plot(m_grid, np.sqrt(A * 10**(0.4*dm)),
                ":", color="orange", lw=1, label="photon noise")
        ax.plot(m_grid, np.sqrt(B * 10**(0.8*dm)),
                ":", color="mediumpurple", lw=1, label="sky noise")

        fit_results[filt] = dict(
            log_floor=popt[0], log_A=popt[1], log_B=popt[2],
            floor=floor, A=A, B=B,
            log_floor_err=perr[0], log_A_err=perr[1], log_B_err=perr[2],
        )
        print(f"\n── Filter {filt} ──")
        print(f"  σ_floor = {floor:.2e}   (log = {popt[0]:+.3f} ± {perr[0]:.3f})")
        print(f"  A       = {A:.2e}   (log = {popt[1]:+.3f} ± {perr[1]:.3f})")
        print(f"  B       = {B:.2e}   (log = {popt[2]:+.3f} ± {perr[2]:.3f})")

    except Exception as exc:
        print(f"⚠ Fit failed for filter {filt}: {exc}")

    # ── cosmetics ──
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 2)
    ax.set_xlabel("Gaia $G$ (mag)", fontsize=13)
    if i == 0:
        ax.set_ylabel(r"Relative RMS  $\sigma_F / \langle F \rangle$", fontsize=13)
    ax.set_title(f"filter = {filt}", fontsize=14, weight="bold")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)
    ax.grid(True, which="both", alpha=0.15)

fig.suptitle("BlackGEM photometric noise model", fontsize=15, y=1.02)
plt.tight_layout()
fig.savefig("blackgem_noise_model.png", dpi=200, bbox_inches="tight")
print("\n✓ Saved blackgem_noise_model.png")


# ━━━ 5. Empirical vs pipeline-reported errors ━━━━━━━━━━━━━━━━━━━━
fig2, axes2 = plt.subplots(1, nf, figsize=(5.5 * nf, 5),
                             squeeze=False, sharey=True)
for i, filt in enumerate(filters):
    ax = axes2[0, i]
    s  = stats[stats["FILTER"] == filt]
    ax.scatter(s["med_reported"], s["mad_rms"],
               s=2, alpha=0.2, c="steelblue", edgecolors="none", rasterized=True)
    lims = [3e-4, 3]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="1 : 1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Pipeline median σ_rel", fontsize=12)
    if i == 0:
        ax.set_ylabel("Empirical MAD σ_rel",  fontsize=12)
    ax.set_title(f"filter = {filt}", fontsize=13, weight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.15)

fig2.suptitle("Pipeline errors vs empirical scatter", fontsize=14, y=1.02)
plt.tight_layout()
fig2.savefig("blackgem_empirical_vs_pipeline.png", dpi=200, bbox_inches="tight")
print("✓ Saved blackgem_empirical_vs_pipeline.png")


# ━━━ 6. Save tables ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
stats.to_csv("blackgem_source_scatter.csv", index=False)
pd.DataFrame(fit_results).T.to_csv("blackgem_noise_params.csv",
                                    float_format="%.6e")
print("✓ Saved blackgem_source_scatter.csv, blackgem_noise_params.csv")