#!/usr/bin/env python3
"""
All-sky Mollweide projection: BlackGEM observation-depth heatmap
with hot subdwarf catalogues (Geier+2022) overlay.

Output : blackgem_footprint.png  (transparent, 250 dpi)
Depends: numpy matplotlib astropy astroquery tqdm google-cloud-bigquery

BigQuery cost
─────────────
Scans only RA_CNTR + DEC_CNTR (2 × FLOAT64 = 16 bytes/row).
For ≈500 k images that is ≈8 MB → well under $0.01.
Result is cached locally after the first run.
"""

import os
import sys
import time
import threading
import warnings
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "svg.fonttype":     "none",
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Inter", "Helvetica Neue", "Helvetica",
                         "DejaVu Sans", "Arial"],
    "mathtext.fontset": "dejavusans",
})

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from tqdm import tqdm
import pandas as pd


# ── Palette (--wl-* tokens) ──────────────────────────────────────
BG_PAGE  = "#FAF7F0"
BORDER   = "#DED7C8"
TEXT     = "#2C241A"
TEXT_SEC = "#8A7D6B"
C_BG     = "#2D8A8A"   # BlackGEM  (--wl-c6)
C4       = "#9A6B92"   # known hsd (--wl-c4)
C5       = "#C89B1E"   # cand. hsd (--wl-c5)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".cache_survey")
os.makedirs(CACHE_DIR, exist_ok=True)

# BlackGEM single CCD: 10 560 × 10 560 px · 0.564″/px ≈ 1.654° per side
BLACKGEM_FOV_DEG = 1.654
BQ_TABLE         = "blackgem-full-source-db.blackgem_fullsource_v2.images"


# =================================================================
#  Vectorised Mollweide projection
# =================================================================
def mollweide_xy(lon_rad, lat_rad):
    lon = np.asarray(lon_rad, dtype=np.float64)
    lat = np.asarray(lat_rad, dtype=np.float64)
    _N  = 10_001
    th  = np.linspace(-np.pi / 2, np.pi / 2, _N)
    ph  = np.arcsin(np.clip((2 * th + np.sin(2 * th)) / np.pi, -1, 1))
    theta = np.interp(lat, ph, th)
    x = (2 * np.sqrt(2) / np.pi) * lon * np.cos(theta)
    y = np.sqrt(2) * np.sin(theta)
    return x, y


def mollweide_inv(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inside = (x / (2 * np.sqrt(2)))**2 + (y / np.sqrt(2))**2 <= 1.0
    theta  = np.arcsin(np.clip(y / np.sqrt(2), -1, 1))
    ct     = np.cos(theta)
    ct     = np.where(np.abs(ct) < 1e-10, 1e-10, ct)
    lon    = np.pi * x / (2 * np.sqrt(2) * ct)
    lat    = np.arcsin(np.clip((2 * theta + np.sin(2 * theta)) / np.pi,
                               -1, 1))
    return np.where(inside, lon, np.nan), np.where(inside, lat, np.nan)


def ra_to_moll(ra_deg):
    r = np.asarray(ra_deg, dtype=np.float64)
    return -np.radians(np.where(r > 180, r - 360, r))


# =================================================================
#  Hot-subdwarf catalogues  (Geier+ 2022, J/A+A/662/A40)
# =================================================================
def _get_col(table, *names):
    for n in names:
        if n in table.colnames:
            return np.asarray(table[n], dtype=np.float64)
    raise KeyError(f"None of {names} found in {table.colnames[:20]}")


def load_hotsd():
    cf = os.path.join(CACHE_DIR, "hotsd_candidates.npz")
    kf = os.path.join(CACHE_DIR, "hotsd_known.npz")
    if os.path.exists(cf) and os.path.exists(kf):
        c, k = np.load(cf), np.load(kf)
        return c["ra"], c["dec"], c["sid"], k["ra"], k["dec"], k["sid"]

    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1

    print("  ↓  Downloading J/A+A/662/A40/hotsd …")
    t = Vizier.get_catalogs("J/A+A/662/A40/hotsd")[0]
    ra_c  = _get_col(t, "RA_ICRS", "RAJ2000", "_RAJ2000", "RAdeg")
    dec_c = _get_col(t, "DE_ICRS", "DEJ2000", "_DEJ2000", "DEdeg")
    np.savez_compressed(cf, ra=ra_c, dec=dec_c)
    print(f"     cached {len(ra_c)} candidates")

    print("  ↓  Downloading J/A+A/662/A40/knownhsd …")
    t2 = Vizier.get_catalogs("J/A+A/662/A40/knownhsd")[0]
    ra_k  = _get_col(t2, "RA_ICRS", "RAJ2000", "_RAJ2000", "RAdeg")
    dec_k = _get_col(t2, "DE_ICRS", "DEJ2000", "_DEJ2000", "DEdeg")
    np.savez_compressed(kf, ra=ra_k, dec=dec_k)
    print(f"     cached {len(ra_k)} known")

    # in the Vizier download block, after getting ra_c/dec_c:
    sid_c = np.asarray(t["GaiaEDR3"], dtype=np.int64)
    np.savez_compressed(cf, ra=ra_c, dec=dec_c, sid=sid_c)

    # and for known:
    sid_k = np.asarray(t2["GaiaEDR3"], dtype=np.int64)
    np.savez_compressed(kf, ra=ra_k, dec=dec_k, sid=sid_k)

    return ra_c, dec_c, sid_c, ra_k, dec_k, sid_k


# =================================================================
#  BlackGEM pointings from BigQuery
# =================================================================
def load_blackgem_pointings():
    """Fetch aggregated BlackGEM image centres.

    The query rounds coordinates to 0.01° and groups, so the result
    is compact.  Only RA_CNTR and DEC_CNTR are scanned (16 B/row).
    """
    cache = os.path.join(CACHE_DIR, "blackgem_pointings.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        print(f"  ✓ BlackGEM pointings from cache "
              f"({len(d['ra']):,} unique, {d['n'].sum():,} images)")
        return d["ra"], d["dec"], d["n"]

    try:
        from google.cloud import bigquery
    except ImportError:
        sys.exit(
            "  ✗ google-cloud-bigquery not installed.\n"
            "    pip install google-cloud-bigquery db-dtypes\n"
            "    gcloud auth application-default login")

    client = bigquery.Client()

    query = f"""
    SELECT
        ROUND(RA_CNTR, 2)  AS ra,
        ROUND(DEC_CNTR, 2) AS dec,
        COUNT(*)            AS n
    FROM `{BQ_TABLE}`
    WHERE RA_CNTR  IS NOT NULL
      AND DEC_CNTR IS NOT NULL
      AND RA_CNTR  BETWEEN 0 AND 360
      AND DEC_CNTR BETWEEN -90 AND 90
    GROUP BY 1, 2
    """

    print("  ↓  Querying BlackGEM pointings from BigQuery …")
    job = client.query(query)
    df  = job.to_dataframe()
    ra  = df["ra"].values.astype(np.float64)
    dec = df["dec"].values.astype(np.float64)
    n   = df["n"].values.astype(np.int32)

    np.savez_compressed(cache, ra=ra, dec=dec, n=n)
    billed = job.total_bytes_billed or 0
    print(f"     {len(ra):,} unique pointings, {n.sum():,} images  "
          f"({billed / 1e6:.1f} MB billed)")
    return ra, dec, n


# =================================================================
#  Build observation-count grid
# =================================================================
def build_blackgem_coverage(p_ra, p_dec, p_n, nra=720, ndec=361):
    """Stamp each pointing's FOV onto a regular (RA, Dec) grid."""
    cache = os.path.join(CACHE_DIR, f"blackgem_nobs_{nra}x{ndec}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["ra1d"], d["dec1d"], d["nobs"]

    half  = BLACKGEM_FOV_DEG / 2.0
    ra1d  = np.linspace(0, 360, nra, endpoint=False)
    dec1d = np.linspace(-90, 90, ndec)
    dra   = ra1d[1]  - ra1d[0]
    ddec  = dec1d[1] - dec1d[0]
    nobs  = np.zeros((ndec, nra), dtype=np.float32)

    print(f"  Stamping {len(p_ra):,} BlackGEM FOVs onto "
          f"{nra}×{ndec} grid …")

    for ra, dec, n in tqdm(zip(p_ra, p_dec, p_n),
                           total=len(p_ra), desc="  FOVs"):
        j_lo = max(0,        int(np.floor((dec - half + 90) / ddec)))
        j_hi = min(ndec - 1, int(np.ceil( (dec + half + 90) / ddec)))

        cos_d = np.cos(np.radians(dec))
        if cos_d < 0.01:
            nobs[j_lo:j_hi + 1, :] += n
            continue

        ra_half = half / cos_d
        i_lo = int(np.floor(((ra - ra_half) % 360) / dra)) % nra
        i_hi = int(np.ceil( ((ra + ra_half) % 360) / dra)) % nra

        if i_lo <= i_hi:
            nobs[j_lo:j_hi + 1, i_lo:i_hi + 1] += n
        else:
            nobs[j_lo:j_hi + 1, i_lo:]     += n
            nobs[j_lo:j_hi + 1, :i_hi + 1] += n

    np.savez_compressed(cache, ra1d=ra1d, dec1d=dec1d, nobs=nobs)
    print("  ✓ coverage grid cached")
    return ra1d, dec1d, nobs


# =================================================================
#  Save with progress spinner
# =================================================================
def _save_with_progress(fig, path, **kw):
    exc, done = [None], threading.Event()

    def _w():
        try:
            fig.savefig(path, **kw)
        except Exception as e:
            exc[0] = e
        finally:
            done.set()

    t = threading.Thread(target=_w, daemon=True)
    t.start()
    sp, i, t0 = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", 0, time.time()
    while not done.wait(0.15):
        sys.stdout.write(
            f"\r  {sp[i % len(sp)]} Saving … {time.time() - t0:.1f}s  ")
        sys.stdout.flush()
        i += 1
    sys.stdout.write(
        f"\r  ✓ {path} ({time.time() - t0:.1f}s)          \n")
    sys.stdout.flush()
    if exc[0]:
        raise exc[0]

def plot_obs_histogram(source_ids_c, source_ids_k):
    """Appendix figure: hot subdwarfs vs BlackGEM observation count."""
    csv_path = (
        "/home/fabian/Documents/Doktor/Presentations/"
        "BlackGEMMeeting2026/code/query_all_blackgem/output_full.csv"
    )
    freq = pd.read_csv(csv_path, usecols=["SOURCE_ID"])["SOURCE_ID"].value_counts()

    obs_c = np.array([freq.get(sid, 0) for sid in source_ids_c])
    obs_k = np.array([freq.get(sid, 0) for sid in source_ids_k])

    bins = np.arange(0, 1001, 10)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    h_c, _ = np.histogram(obs_c[obs_c > 0], bins=bins)
    h_k, _ = np.histogram(obs_k[obs_k > 0], bins=bins)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(25/1.5, 12/1.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08})
    fig.patch.set_facecolor(BG_PAGE)
    for ax in (ax_top, ax_bot):
        ax.set_facecolor(BG_PAGE)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.tick_params(colors=TEXT, labelsize=11, length=4)

    ax_top.bar(bin_centres, h_c, width=9, color=C5, alpha=0.82,
               edgecolor="none", zorder=3, label="Hot-sd candidates")
    ax_top.set_ylabel("Candidates", fontsize=13, color=TEXT,
                      fontweight="semibold")

    ax_bot.bar(bin_centres, h_k, width=9, color=C4, alpha=0.82,
               edgecolor="none", zorder=3, label="Known hot subdwarfs")
    ax_bot.set_ylabel("Known hot subdwarfs", fontsize=13, color=TEXT,
                      fontweight="semibold")
    ax_bot.set_xlabel("BlackGEM observation count", fontsize=13,
                      color=TEXT, fontweight="semibold")
    ax_bot.set_xlim(0, 1000)

    pct_levels = [68, 95, 99.5]
    pct_colors = [C_BG, C5, "#C43E2A"]
    pct_styles = ["-", "--", ":"]
    for ax, obs, label in [(ax_top, obs_c, "candidates"),
                           (ax_bot, obs_k, "known")]:
        n_zero  = np.sum(obs == 0)
        n_ge100 = np.sum(obs >= 100)
        med     = np.median(obs[obs > 0]) if np.any(obs > 0) else 0
        pvals   = np.percentile(obs, pct_levels)
        txt = (f"Total: {len(obs):,}    "
               f"No coverage: {n_zero:,} ({n_zero/len(obs)*100:.1f}%)    "
               f"≥100 obs: {n_ge100:,} ({n_ge100/len(obs)*100:.1f}%)    "
               f"Median (covered): {med:.0f}")
        ax.text(0.98, 0.88, txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color=TEXT_SEC)
        ylim = ax.get_ylim()[1]
        for p, v, c, ls in zip(pct_levels, pvals, pct_colors, pct_styles):
            if v > ax.get_xlim()[1]:
                continue
            ax.axvline(v, color=c, alpha=0.7, lw=1.2, ls=ls, zorder=2)
            ax.text(v + 6, ylim * 0.78, f"{p}%\n≤ {v:.0f}",
                    fontsize=8.5, color=c, va="top", fontweight="semibold")

    for ax in (ax_top, ax_bot):
        ax.axvline(100, color=TEXT, alpha=0.25, lw=0.8, ls="--", zorder=2)
        ax.text(110, ax.get_ylim()[1] * 0.90, "100 obs",
                fontsize=8.5, color=TEXT_SEC, va="top")

    _save_with_progress(fig, "blackgem_obs_histogram.svg",
                        format="svg", facecolor=BG_PAGE,
                        bbox_inches="tight")
    plt.close(fig)

# =================================================================
#  MAIN
# =================================================================
def main():
    # ── data ──────────────────────────────────────────────────────
    ra_c, dec_c, sid_c, ra_k, dec_k, sid_k = load_hotsd()
    p_ra, p_dec, p_n         = load_blackgem_pointings()
    ra1d, dec1d, nobs        = build_blackgem_coverage(p_ra, p_dec, p_n)

    # ── coverage fraction (hot subdwarfs inside BlackGEM footprint)
    all_ra  = np.concatenate([ra_c, ra_k])
    all_dec = np.concatenate([dec_c, dec_k])
    ira  = np.clip(np.searchsorted(ra1d, all_ra % 360), 0, len(ra1d) - 1)
    idec = np.clip(np.searchsorted(dec1d, all_dec),      0, len(dec1d) - 1)
    frac = np.mean(nobs[idec, ira] >= 100) * 100
    print(f"  BlackGEM hot-subdwarf coverage: {frac:.1f}%")

    # ── colourmap ─────────────────────────────────────────────────
    r, g, b = to_rgba(C_BG)[:3]
    cmap = LinearSegmentedColormap.from_list("blackgem", [
        (r, g, b, 0.08),
        (r, g, b, 0.22),
        (r, g, b, 0.45),
        (r, g, b, 0.72),
        (r, g, b, 0.92),
    ])

    pos  = nobs[nobs > 0]
    vmax = float(np.percentile(pos, 98)) if len(pos) else 1.0
    vmax = max(vmax, 1.0)

    # ── rasterise via inverse Mollweide ───────────────────────────
    S2 = np.sqrt(2)
    nx_img, ny_img = 1440, 720
    XI, YI = np.meshgrid(np.linspace(-2 * S2, 2 * S2, nx_img),
                         np.linspace(-S2, S2, ny_img))
    lon_i, lat_i = mollweide_inv(XI, YI)
    ra_i  = (-np.degrees(lon_i)) % 360
    dec_i = np.degrees(lat_i)

    ok  = np.isfinite(ra_i)
    img = np.full(XI.shape, np.nan)
    dra  = ra1d[1] - ra1d[0]
    ddec = dec1d[1] - dec1d[0]
    ii = np.round(ra_i[ok]  / dra).astype(int) % len(ra1d)
    jj = np.clip(np.round((dec_i[ok] - dec1d[0]) / ddec).astype(int),
                 0, len(dec1d) - 1)
    img[ok] = nobs[jj, ii]
    img = np.where(img > 0, img, np.nan)

    dec_c = np.clip(dec_c, -89.99, 89.99)
    dec_k = np.clip(dec_k, -89.99, 89.99)

    # ══════════════════════════════════════════════════════════════
    #  FIGURE
    # ══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(7.5, 4.0))
    ax  = fig.add_axes([0.04, 0.12, 0.92, 0.82])
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    pad = 0.06
    ax.set_xlim(-2 * S2 - pad, 2 * S2 + pad)
    ax.set_ylim(-S2 - pad,     S2 + pad)
    ax.axis("off")

    # clip + border ellipses
    clip_e = Ellipse((0, 0), 4 * S2, 2 * S2, transform=ax.transData,
                     fc="none", ec="none")
    ax.add_patch(clip_e)
    ax.add_patch(Ellipse((0, 0), 4 * S2, 2 * S2, transform=ax.transData,
                         fc="none", ec=BORDER, lw=1.0, zorder=8))

    # ── layer 1: BlackGEM heatmap ─────────────────────────────────
    ext = [-2 * S2, 2 * S2, -S2, S2]
    im = ax.imshow(img, extent=ext, origin="lower", cmap=cmap,
                   vmin=0, vmax=vmax, aspect="equal", zorder=2,
                   interpolation="nearest", rasterized=True)
    im.set_clip_path(clip_e)

    # ── grid lines ────────────────────────────────────────────────
    _g = np.linspace(-np.pi, np.pi, 400)
    for d in [-60, -30, 0, 30, 60]:
        gx, gy = mollweide_xy(_g, np.full_like(_g, np.radians(d)))
        ln, = ax.plot(gx, gy, color=TEXT, alpha=0.18, lw=0.35, zorder=4)
        ln.set_clip_path(clip_e)

    _g2 = np.linspace(np.radians(-89.99), np.radians(89.99), 400)
    for h in range(0, 24, 3):
        v = float(ra_to_moll(h * 15.0))
        gx, gy = mollweide_xy(np.full_like(_g2, v), _g2)
        ln, = ax.plot(gx, gy, color=TEXT, alpha=0.18, lw=0.5, zorder=4)
        ln.set_clip_path(clip_e)

    # ── RA / Dec tick labels ──────────────────────────────────────
    def _at(ra_deg, dec_deg):
        x, y = mollweide_xy(ra_to_moll(ra_deg), np.radians(dec_deg))
        return float(x), float(y)

    for h in [0, 3, 6, 9, 15, 18, 21]:
        tx, ty = _at(h * 15, -4)
        ax.text(tx, ty, rf"${h}^{{\rm h}}$", ha="center", va="top",
                fontsize=6.5, color=TEXT_SEC, zorder=10, clip_on=True)
    for d in [-60, -30, 30, 60]:
        tx, ty = _at(359.5, d)
        ax.text(tx, ty, f"{d:+d}°", ha="left", va="center",
                fontsize=5.5, color=TEXT_SEC, zorder=10, clip_on=True)

    # ── layer 2: hot-subdwarf scatter ─────────────────────────────
    cx, cy = mollweide_xy(ra_to_moll(ra_c), np.radians(dec_c))
    ax.plot(cx, cy, "o", ms=0.5, color=C5, alpha=0.45, zorder=5,
            mew=0, ls="none", rasterized=True)

    kx, ky = mollweide_xy(ra_to_moll(ra_k), np.radians(dec_k))
    ax.plot(kx, ky, "o", ms=0.9, color=C4, alpha=0.55, zorder=5.5,
            mew=0, ls="none", rasterized=True)

    # ── legend ────────────────────────────────────────────────────
    handles = [
        Line2D([], [], marker="o", ls="none", ms=5, color=C5,
               alpha=0.7, mew=0, label="Hot-sd candidates (Geier+2022)"),
        Line2D([], [], marker="o", ls="none", ms=6, color=C4,
               alpha=0.8, mew=0, label="Known hot subdwarfs"),
    ]

    # ── annotation ────────────────────────────────────────────────
    stroke_bg = [pe.withStroke(linewidth=2.5, foreground="white")]
    ax.text(0.02, -0.05,
            f"BlackGEM hot-subdwarf coverage: {frac:.1f}%  "
            f"({p_n.sum():,} images · {len(p_ra):,} pointings)",
            transform=ax.transAxes, fontsize=8.5, color=TEXT_SEC,
            va="bottom", ha="left")

    # ── colourbar ─────────────────────────────────────────────────
    cax = fig.add_axes([0.18, 0.04, 0.64, 0.022])
    cb  = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("BlackGEM exposure count",
                 fontsize=7, color=TEXT, labelpad=3)
    cb.ax.tick_params(colors=TEXT, labelsize=6.5, length=2)
    cb.outline.set_edgecolor(BORDER)
    step  = max(1, int(np.ceil(vmax / 8)))
    ticks = np.arange(0, vmax + 1, step)
    cb.set_ticks(ticks)

    # ── save ──────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        _save_with_progress(fig, "blackgem_footprint.png",
                            format="png", transparent=True,
                            bbox_inches="tight", dpi=250)
    plot_obs_histogram(sid_c, sid_k)
    plt.close(fig)


if __name__ == "__main__":
    main()