#!/usr/bin/env python3
"""
All-sky Mollweide projection: TESS sector-depth heatmap, ZTF footprint,
hot subdwarf catalogues (Geier+2022), and per-survey coverage bar chart.

Output: survey_footprints.png (4K, transparent background)

Requirements: numpy matplotlib astropy tesswcs astroquery tqdm
"""

import os
import sys
import time
import threading
import warnings
import numpy as np
from multiprocessing import Pool, cpu_count

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica",
                        "DejaVu Sans", "Arial"],
    "mathtext.fontset": "dejavusans",
})

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Ellipse
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from tqdm import tqdm

# ── Palette ───────────────────────────────────────────────────────
BORDER   = "#DED7C8"
TEXT     = "#2C241A"
TEXT_SEC = "#8A7D6B"
C_ZTF    = "#B85B3A"
C_ATLAS  = "#4A78A8"
C_TESS   = "#558B44"
C4       = "#9A6B92"
C5       = "#C89B1E"

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".cache_survey")
os.makedirs(CACHE_DIR, exist_ok=True)


# =================================================================
#  Vectorized Mollweide projection (replaces mpl's per-path solver)
# =================================================================
def mollweide_xy(lon_rad, lat_rad):
    """(lon, lat) in radians → (x, y) in Mollweide coordinates.
    Uses an analytic lookup table — no Newton-Raphson, no convergence issues."""
    lon = np.asarray(lon_rad, dtype=np.float64)
    lat = np.asarray(lat_rad, dtype=np.float64)

    # Build θ → φ table analytically (invertible, monotonic)
    _N = 10001
    theta_tab = np.linspace(-np.pi / 2, np.pi / 2, _N)
    phi_tab = np.arcsin(np.clip(
        (2 * theta_tab + np.sin(2 * theta_tab)) / np.pi, -1, 1))

    # Invert: given φ (=lat), find θ via interpolation
    theta = np.interp(lat, phi_tab, theta_tab)

    x = (2 * np.sqrt(2) / np.pi) * lon * np.cos(theta)
    y = np.sqrt(2) * np.sin(theta)
    return x, y


def mollweide_inv(x, y):
    """Inverse Mollweide: (x, y) → (lon_rad, lat_rad). NaN outside ellipse."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inside = (x / (2 * np.sqrt(2)))**2 + (y / np.sqrt(2))**2 <= 1.0
    theta = np.arcsin(np.clip(y / np.sqrt(2), -1, 1))
    cos_theta = np.cos(theta)
    cos_theta = np.where(np.abs(cos_theta) < 1e-10, 1e-10, cos_theta)
    lon = np.pi * x / (2 * np.sqrt(2) * cos_theta)
    lat = np.arcsin(np.clip((2 * theta + np.sin(2 * theta)) / np.pi, -1, 1))
    lon = np.where(inside, lon, np.nan)
    lat = np.where(inside, lat, np.nan)
    return lon, lat


def ra_to_moll(ra_deg):
    """RA in degrees → Mollweide longitude in radians (RA increasing left)."""
    r = np.asarray(ra_deg, dtype=np.float64)
    r = np.where(r > 180, r - 360, r)
    return -np.radians(r)


# =================================================================
#  Angular-separation helper (pure numpy, degrees in/out)
# =================================================================
def _angular_sep_deg(ra1, dec1, ra2_scalar, dec2_scalar):
    r1, d1 = np.radians(ra1), np.radians(dec1)
    r2, d2 = np.radians(ra2_scalar), np.radians(dec2_scalar)
    cos_s = (np.sin(d1) * np.sin(d2) +
             np.cos(d1) * np.cos(d2) * np.cos(r1 - r2))
    return np.degrees(np.arccos(np.clip(cos_s, -1, 1)))


# =================================================================
#  Multiprocessing worker for TESS coverage
# =================================================================
_w = {}


def _init_worker(ra_flat, dec_flat, shape):
    _w["ra"]    = ra_flat
    _w["dec"]   = dec_flat
    _w["shape"] = shape


def _process_sector(args):
    sec, bra, bdec, broll = args
    import tesswcs

    ra_f  = _w["ra"]
    dec_f = _w["dec"]
    partial = np.zeros(len(ra_f), dtype=np.float32)

    for cam in range(1, 5):
        for ccd in range(1, 5):
            try:
                wcs = tesswcs.WCS.predict(bra, bdec, broll, cam, ccd)
            except Exception:
                continue
            try:
                cra, cdec = wcs.all_pix2world(1024, 1024, 0)
                cra  = float(np.asarray(cra))
                cdec = float(np.asarray(cdec))

                sep  = _angular_sep_deg(ra_f, dec_f, cra, cdec)
                near = sep < 15.0
                if not np.any(near):
                    continue

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    px, py = wcs.all_world2pix(
                        ra_f[near], dec_f[near], 0)

                nax1 = wcs.pixel_shape[0] if wcs.pixel_shape else 2048
                nax2 = wcs.pixel_shape[1] if wcs.pixel_shape else 2048
                ok = ((px >= -0.5) & (px < nax1 - 0.5) &
                      (py >= -0.5) & (py < nax2 - 0.5) &
                      np.isfinite(px) & np.isfinite(py))
                partial[near] += ok.astype(np.float32)
            except Exception:
                continue

    return partial.reshape(_w["shape"])


# =================================================================
#  Hot subdwarf catalogues (Geier+ 2022, J/A+A/662/A40)
# =================================================================
def _get_col(table, *names):
    for n in names:
        if n in table.colnames:
            return np.asarray(table[n], dtype=np.float64)
    raise KeyError(f"None of {names} found in {table.colnames[:20]}")


def load_hotsd():
    cand_file  = os.path.join(CACHE_DIR, "hotsd_candidates.npz")
    known_file = os.path.join(CACHE_DIR, "hotsd_known.npz")

    if os.path.exists(cand_file) and os.path.exists(known_file):
        c = np.load(cand_file); k = np.load(known_file)
        return c["ra"], c["dec"], k["ra"], k["dec"]

    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1

    print("  ↓  Downloading J/A+A/662/A40/hotsd …")
    cats = Vizier.get_catalogs("J/A+A/662/A40/hotsd")
    t = cats[0]
    ra_c  = _get_col(t, "RA_ICRS", "RAJ2000", "_RAJ2000", "RAdeg")
    dec_c = _get_col(t, "DE_ICRS", "DEJ2000", "_DEJ2000", "DEdeg")
    np.savez_compressed(cand_file, ra=ra_c, dec=dec_c)
    print(f"     cached {len(ra_c)} candidates")

    print("  ↓  Downloading J/A+A/662/A40/knownhsd …")
    cats2 = Vizier.get_catalogs("J/A+A/662/A40/knownhsd")
    t2 = cats2[0]
    ra_k  = _get_col(t2, "RA_ICRS", "RAJ2000", "_RAJ2000", "RAdeg")
    dec_k = _get_col(t2, "DE_ICRS", "DEJ2000", "_DEJ2000", "DEdeg")
    np.savez_compressed(known_file, ra=ra_k, dec=dec_k)
    print(f"     cached {len(ra_k)} known")

    return ra_c, dec_c, ra_k, dec_k


# =================================================================
#  TESS coverage (multiprocessed)
# =================================================================
def build_tess_coverage(nra=720, ndec=361):
    cache = os.path.join(CACHE_DIR, f"tess_nobs_v2_{nra}x{ndec}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["ra1d"], d["dec1d"], d["nobs"]

    import tesswcs

    ra1d  = np.linspace(0, 360, nra, endpoint=False)
    dec1d = np.linspace(-90, 90, ndec)
    RA, DEC = np.meshgrid(ra1d, dec1d)
    nobs = np.zeros(RA.shape, dtype=np.float32)

    ra_flat  = RA.ravel().astype(np.float64)
    dec_flat = DEC.ravel().astype(np.float64)

    ptable = tesswcs.pointings
    seen, rows = set(), []
    for row in ptable:
        sec = int(row["Sector"])
        if sec not in seen:
            seen.add(sec)
            rows.append((sec, float(row["RA"]), float(row["Dec"]),
                         float(row["Roll"])))

    tasks = [(sec, ra, dec, roll) for sec, ra, dec, roll in rows]
    nproc = min(cpu_count(), 16)
    print(f"  Building TESS coverage ({nra}×{ndec}), "
          f"{len(rows)} sectors on {nproc} cores …")

    with Pool(nproc, initializer=_init_worker,
              initargs=(ra_flat, dec_flat, RA.shape)) as pool:
        for partial in tqdm(pool.imap_unordered(_process_sector, tasks),
                            total=len(tasks), desc="  TESS sectors"):
            nobs += partial

    np.savez_compressed(cache, ra1d=ra1d, dec1d=dec1d, nobs=nobs)
    print(f"  ✓ TESS coverage cached → {cache}")
    return ra1d, dec1d, nobs


# =================================================================
#  Save with elapsed-time spinner
# =================================================================
def _save_with_progress(fig, path, **kwargs):
    exc = [None]
    done = threading.Event()

    def _worker():
        try:
            fig.savefig(path, **kwargs)
        except Exception as e:
            exc[0] = e
        finally:
            done.set()

    t = threading.Thread(target=_worker, daemon=True)
    start = time.time()
    t.start()

    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not done.wait(timeout=0.15):
        elapsed = time.time() - start
        sys.stdout.write(
            f"\r  {spinner[i % len(spinner)]} Saving … {elapsed:.1f}s  ")
        sys.stdout.flush()
        i += 1

    elapsed = time.time() - start
    sys.stdout.write(f"\r  ✓ {path} saved ({elapsed:.1f}s)          \n")
    sys.stdout.flush()

    if exc[0] is not None:
        raise exc[0]


# =================================================================
#  MAIN
# =================================================================
def main():
    # ── data ──────────────────────────────────────────────────────
    ra_c, dec_c, ra_k, dec_k = load_hotsd()
    ra1d, dec1d, nobs = build_tess_coverage()

    # ── coverage fractions (hot subdwarfs) ────────────────────────
    all_ra  = np.concatenate([ra_c, ra_k])
    all_dec = np.concatenate([dec_c, dec_k])

    frac_ztf   = np.mean(all_dec > -31) * 100
    frac_atlas = 100.0

    ira  = np.clip(np.searchsorted(ra1d, all_ra % 360), 0, len(ra1d) - 1)
    idec = np.clip(np.searchsorted(dec1d, all_dec),      0, len(dec1d) - 1)
    frac_tess = np.mean(nobs[idec, ira] > 0) * 100

    print(f"  Coverage — ZTF: {frac_ztf:.1f}%  ATLAS: {frac_atlas:.0f}%"
          f"  TESS: {frac_tess:.1f}%")

    # ── TESS colourmap ────────────────────────────────────────────
    r, g, b = to_rgba(C_TESS)[:3]
    cmap_t = LinearSegmentedColormap.from_list("tess", [
        (r, g, b, 0.1),
        (r, g, b, 0.25),
        (r, g, b, 0.50),
        (r, g, b, 0.78),
        (r, g, b, 0.92),
    ])

    vmax = min(float(np.nanmax(nobs)), 40)

    # ── rasterise TESS heatmap via inverse Mollweide ──────────────
    S2 = np.sqrt(2)
    nx_img, ny_img = 1440, 720
    x_img = np.linspace(-2 * S2, 2 * S2, nx_img)
    y_img = np.linspace(-S2, S2, ny_img)
    XI, YI = np.meshgrid(x_img, y_img)
    lon_inv, lat_inv = mollweide_inv(XI, YI)

    ra_inv  = (-np.degrees(lon_inv)) % 360
    dec_inv = np.degrees(lat_inv)

    valid = np.isfinite(ra_inv) & np.isfinite(dec_inv)
    tess_img = np.full(XI.shape, np.nan)
    dra  = ra1d[1] - ra1d[0]
    ddec = dec1d[1] - dec1d[0]
    ira  = np.round(ra_inv[valid] / dra).astype(int) % len(ra1d)
    idec = np.clip(
        np.round((dec_inv[valid] - dec1d[0]) / ddec).astype(int),
        0, len(dec1d) - 1)
    tess_img[valid] = nobs[idec, ira]
    tess_img = np.where(tess_img > 0, tess_img, np.nan)

    dec_c = np.clip(dec_c, -89.99, 89.99)
    dec_k = np.clip(dec_k, -89.99, 89.99)

    # ══════════════════════════════════════════════════════════════
    #  FIGURE  (plain axes — no projection, no per-path solves)
    # ══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16.5, 7.5))
    ax     = fig.add_axes([0.02, 0.06, 0.68, 0.86])
    ax_bar = fig.add_axes([0.72, 0.15, 0.16, 0.68])

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    pad = 0.05
    ax.set_xlim(-2 * np.sqrt(2) - pad, 2 * np.sqrt(2) + pad)
    ax.set_ylim(-np.sqrt(2) - pad, np.sqrt(2) + pad)
    ax.axis("off")

    # invisible clip ellipse + visible boundary
    clip_ell = Ellipse((0, 0), width=4 * np.sqrt(2), height=2 * np.sqrt(2),
                       transform=ax.transData,
                       facecolor="none", edgecolor="none")
    ax.add_patch(clip_ell)

    border_ell = Ellipse((0, 0), width=4 * np.sqrt(2), height=2 * np.sqrt(2),
                         transform=ax.transData,
                         facecolor="none", edgecolor=BORDER, lw=1.4, zorder=8)
    ax.add_patch(border_ell)

    # ── layer 1: ZTF footprint ────────────────────────────────────
    _n = 600
    _lon_b = np.linspace(-np.pi, np.pi, _n)
    xb, yb = mollweide_xy(_lon_b, np.full(_n, np.radians(-31)))

    _dec_r = np.linspace(np.radians(-31), np.radians(89.999), _n // 4)
    xr, yr = mollweide_xy(np.full(len(_dec_r), np.pi), _dec_r)

    xp, yp = mollweide_xy(np.array([0.0]), np.array([np.radians(89.999)]))

    _dec_l = np.linspace(np.radians(89.999), np.radians(-31), _n // 4)
    xl, yl = mollweide_xy(np.full(len(_dec_l), -np.pi), _dec_l)

    poly_x = np.concatenate([xb, xr, xp, xl])
    poly_y = np.concatenate([yb, yr, yp, yl])
    ztf_fill = ax.fill(poly_x, poly_y, color=C_ZTF, alpha=0.1,
                       zorder=1, lw=0)[0]
    ztf_fill.set_clip_path(clip_ell)

    zx, zy = mollweide_xy(_lon_b, np.full(_n, np.radians(-31)))
    ztf_line, = ax.plot(zx, zy, color=C_ZTF, lw=1.5, zorder=6,
                        solid_capstyle="round")
    ztf_line.set_clip_path(clip_ell)

    # ── layer 2: TESS heatmap ─────────────────────────────────────
    extent = [-2 * S2, 2 * S2, -S2, S2]
    im = ax.imshow(tess_img, extent=extent, origin="lower",
                   cmap=cmap_t, vmin=0, vmax=vmax, aspect="equal",
                   zorder=2, interpolation="nearest", rasterized=True)
    im.set_clip_path(clip_ell)

    # ── grid lines (drawn manually) ──────────────────────────────
    _gl = np.linspace(-np.pi, np.pi, 400)
    for dec_deg in [-60, -30, 0, 30, 60]:
        gx, gy = mollweide_xy(_gl, np.full_like(_gl, np.radians(dec_deg)))
        ln, = ax.plot(gx, gy, color=TEXT, alpha=0.2, lw=0.5, zorder=4)
        ln.set_clip_path(clip_ell)

    _gl2 = np.linspace(np.radians(-89.99), np.radians(89.99), 400)
    for h in range(0, 24, 3):
        lon_val = ra_to_moll(h * 15.0)
        gx, gy = mollweide_xy(np.full_like(_gl2, float(lon_val)), _gl2)
        ln, = ax.plot(gx, gy, color=TEXT, alpha=0.2, lw=0.5, zorder=4)
        ln.set_clip_path(clip_ell)

    # ── RA / Dec labels ───────────────────────────────────────────
    def _at(ra_deg, dec_deg):
        x, y = mollweide_xy(ra_to_moll(ra_deg), np.radians(dec_deg))
        return float(x), float(y)

    for h in [0, 3, 6, 9, 15, 18, 21]:
        tx, ty = _at(h * 15.0, -4)
        ax.text(tx, ty, rf"${h}^{{\rm h}}$",
                ha="center", va="top", fontsize=9,
                color=TEXT_SEC, zorder=10, clip_on=True)

    for d in [-60, -30, 30, 60]:
        tx, ty = _at(359.5, d)
        ax.text(tx, ty, f"{d:+d}°",
                ha="left", va="center", fontsize=8,
                color=TEXT_SEC, zorder=10, clip_on=True)

    # ── layer 3: hot subdwarf scatter ─────────────────────────────
    cx, cy = mollweide_xy(ra_to_moll(ra_c), np.radians(dec_c))
    ax.plot(cx, cy, "o", ms=0.8, color=C5, alpha=0.45, zorder=5,
            markeredgewidth=0, linestyle="none", rasterized=True)

    kx, ky = mollweide_xy(ra_to_moll(ra_k), np.radians(dec_k))
    ax.plot(kx, ky, "o", ms=1.4, color=C4, alpha=0.55, zorder=5.5,
            markeredgewidth=0, linestyle="none", rasterized=True)

    # ── direct labels ─────────────────────────────────────────────
    stroke      = [pe.withStroke(linewidth=3.5, foreground="white")]
    stroke_thin = [pe.withStroke(linewidth=2.5, foreground="white")]

    tx, ty = _at(200, -37)
    ax.text(tx, ty, "ZTF  δ > −31°",
            ha="center", va="top", fontsize=10.5, fontweight="bold",
            color=C_ZTF, path_effects=stroke, zorder=11)

    # ── TESS colourbar ────────────────────────────────────────────
    cax = fig.add_axes([0.15, 0.045, 0.42, 0.02])
    cb  = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("TESS observation count (sectors)",
                 fontsize=10, color=TEXT, labelpad=5)
    cb.ax.tick_params(colors=TEXT, labelsize=9, length=3)
    cb.outline.set_edgecolor(BORDER)
    ticks = np.arange(0, vmax + 1, 5)
    if len(ticks) < 3:
        ticks = np.arange(0, vmax + 1, 2)
    cb.set_ticks(ticks)

    # ══════════════════════════════════════════════════════════════
    #  BAR CHART
    # ══════════════════════════════════════════════════════════════
    ax_bar.set_facecolor("none")
    for sp in ax_bar.spines.values():
        sp.set_visible(False)

    surveys = ["ZTF", "ATLAS", "TESS"]
    fracs   = [frac_ztf, frac_atlas, frac_tess]
    colors  = [C_ZTF, C_ATLAS, C_TESS]
    xs      = np.arange(len(surveys))
    bw      = 0.58

    for i, (x, f, c) in enumerate(zip(xs, fracs, colors)):
        bar = FancyBboxPatch(
            (x - bw / 2, 0), bw, f,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=c, alpha=0.82, lw=0, zorder=3, clip_on=False)
        ax_bar.add_patch(bar)
        ax_bar.text(x, f + 2.2, f"{f:.0f}%",
                    ha="center", va="bottom", fontsize=12.5,
                    fontweight="bold", color=c)

    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(surveys, fontsize=11.5, color=TEXT,
                           fontweight="semibold")
    ax_bar.tick_params(axis="x", length=0, pad=6)
    ax_bar.set_yticks([])
    ax_bar.set_ylim(0, 115)
    ax_bar.set_xlim(-0.6, len(surveys) - 0.4)
    ax_bar.set_title("Hot subdwarf\ncoverage", fontsize=12,
                     color=TEXT, fontweight="bold", pad=10,
                     linespacing=1.3)
    ax_bar.axhline(0, color=BORDER, lw=1.2, zorder=2)

    # ── save ──────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        _save_with_progress(fig, "survey_footprints.png",
                            format="png", transparent=True,
                            bbox_inches="tight", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()