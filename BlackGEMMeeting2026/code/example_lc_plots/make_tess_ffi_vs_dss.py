#!/usr/bin/env python3
"""Side-by-side TESS FFI vs DSS2 cutout — pixel scale & crowding."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Tesscut
from astroquery.skyview import SkyView
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, AsinhStretch, PercentileInterval
from reproject import reproject_interp

# ── Config ──────────────────────────────────────────────────
FONT       = Path("../../assets/fonts/Inter-VariableFont_opsz,wght.ttf")
OUT        = Path("tess_vs_dss.svg")
COORD      = SkyCoord("07h37m56.2564261176s", "+31d16m46.736788440s",
                       frame="icrs")
NPIX_TESS  = 11
NPIX_DSS   = 400
TESS_PLATE = 21.0                            # arcsec per TESS pixel
FOV_AMIN   = NPIX_TESS * TESS_PLATE / 60.0

C_DARK     = "#2C241A"
C_TEXT     = "#FAF7F0"
C_TEXT_SEC = "#DED7C8"
C_BORDER   = "#8A7D6B"
C_ACCENT   = "#B85B3A"
C_MARK     = "#5FE0D0"

cmap = LinearSegmentedColormap.from_list("warm_dark", [
    C_DARK, "#3D2E1F", "#7A4A2E", C_ACCENT, "#D4956A", "#F0EBDF", C_TEXT
])
cmap.set_bad(C_DARK)

# ── Font ────────────────────────────────────────────────────
font_manager.fontManager.addfont(str(FONT))
fp_title = font_manager.FontProperties(fname=str(FONT), size=14,
                                       weight="medium")
fp_small = font_manager.FontProperties(fname=str(FONT), size=10)
plt.rcParams["svg.fonttype"] = "path"

import pickle

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── Fetch TESS FFI cutout (cached) ─────────────────────────
tess_cache = CACHE_DIR / "tess_ffi.pkl"
if tess_cache.exists():
    print("Loading TESS FFI from cache …")
    with open(tess_cache, "rb") as f:
        tess_img, tess_wcs, sector = pickle.load(f)
else:
    print("Fetching TESS FFI cutout …")
    cutouts = Tesscut.get_cutouts(coordinates=COORD, size=NPIX_TESS)
    if not cutouts:
        raise RuntimeError("No TESS FFI data for this position.")
    hdu       = cutouts[0]
    flux_cube = hdu[1].data["FLUX"]
    tess_img  = np.nanmedian(flux_cube, axis=0)
    tess_wcs  = WCS(hdu[2].header)
    sector    = hdu[0].header.get("SECTOR", "?")
    with open(tess_cache, "wb") as f:
        pickle.dump((tess_img, tess_wcs, sector), f)
    print(f"  Sector {sector}, {flux_cube.shape[0]} frames — cached.")

# ── Fetch DSS (cached) ─────────────────────────────────────
dss_cache = CACHE_DIR / "dss2_red.pkl"
if dss_cache.exists():
    print("Loading DSS2 Red from cache …")
    with open(dss_cache, "rb") as f:
        dss_img, dss_wcs = pickle.load(f)
else:
    print("Fetching DSS2 Red …")
    dss_list = SkyView.get_images(
        position=COORD, survey="DSS2 Red",
        pixels=NPIX_DSS,
        width=FOV_AMIN * u.arcmin,
        height=FOV_AMIN * u.arcmin,
    )
    dss_img = dss_list[0][0].data.astype(float)
    dss_wcs = WCS(dss_list[0][0].header)
    with open(dss_cache, "wb") as f:
        pickle.dump((dss_img, dss_wcs), f)
    print(f"  {dss_img.shape[0]}×{dss_img.shape[1]} px — cached.")

# ── Reproject TESS → DSS WCS (North-up, same pixel grid) ──
print("Reprojecting TESS → North-up …")
tess_reproj, footprint = reproject_interp(
    (tess_img, tess_wcs), dss_wcs,
    shape_out=dss_img.shape, order=0
)
tess_reproj[footprint < 0.5] = np.nan

# ── Figure ──────────────────────────────────────────────────
fig, (ax_t, ax_d) = plt.subplots(1, 2, figsize=(9.5, 4.2))
fig.patch.set_facecolor(C_DARK)
fig.subplots_adjust(wspace=0.12)

# Left: TESS (reprojected, blocky)
norm_t = ImageNormalize(tess_reproj,
                        interval=PercentileInterval(99.5),
                        stretch=AsinhStretch(a=0.05))
ax_t.imshow(tess_reproj, origin="lower", cmap=cmap, norm=norm_t,
            interpolation="nearest")

# Right: DSS
norm_d = ImageNormalize(dss_img,
                        interval=PercentileInterval(99.5),
                        stretch=AsinhStretch(a=0.05))
ax_d.imshow(dss_img, origin="lower", cmap=cmap, norm=norm_d,
            interpolation="bicubic")

# ── TESS pixel grid on DSS ────────────────────────────────
ny_t, nx_t = tess_img.shape
nsamp = 50
for i in range(nx_t + 1):
    py = np.linspace(-0.5, ny_t - 0.5, nsamp)
    px = np.full(nsamp, i - 0.5)
    sky = tess_wcs.pixel_to_world(px, py)
    xd, yd = dss_wcs.world_to_pixel(sky)
    ax_d.plot(xd, yd, color=C_TEXT_SEC, lw=0.5, alpha=0.4)
for j in range(ny_t + 1):
    px = np.linspace(-0.5, nx_t - 0.5, nsamp)
    py = np.full(nsamp, j - 0.5)
    sky = tess_wcs.pixel_to_world(px, py)
    xd, yd = dss_wcs.world_to_pixel(sky)
    ax_d.plot(xd, yd, color=C_TEXT_SEC, lw=0.5, alpha=0.4)

# Both panels share the DSS pixel grid now
for ax in (ax_t, ax_d):
    ax.set_xlim(-0.5, NPIX_DSS - 0.5)
    ax.set_ylim(-0.5, NPIX_DSS - 0.5)

# ── Target crosshair (same WCS for both panels) ───────────
tx, ty = dss_wcs.world_to_pixel(COORD)
for ax in (ax_t, ax_d):
    ax.plot(tx, ty, "+", color=C_MARK, ms=14, mew=1.8, zorder=5)

# ── Titles ─────────────────────────────────────────────────
ax_t.set_title(f"TESS FFI  ·  Sector {sector}",
               fontproperties=fp_title, color=C_TEXT, pad=10)
ax_d.set_title("DSS2 Red", fontproperties=fp_title, color=C_TEXT, pad=10)

# ── Clean axes ─────────────────────────────────────────────
for ax in (ax_t, ax_d):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(C_DARK)
    for spine in ax.spines.values():
        spine.set_color(C_BORDER)
        spine.set_linewidth(0.6)

# ── Scale bar (DSS panel, bottom-right) ───────────────────
dss_plate = FOV_AMIN * 60.0 / NPIX_DSS      # arcsec per DSS pixel
bar_px = 60.0 / dss_plate                    # 1 arcmin in DSS pixels
ny, nx = dss_img.shape
x0 = nx * 0.95 - bar_px
y0 = ny * 0.06
ax_d.plot([x0, x0 + bar_px], [y0, y0],
          color=C_TEXT_SEC, lw=2, solid_capstyle="butt")
ax_d.text(x0 + bar_px / 2, y0 + ny * 0.04, "1′",
          color=C_TEXT_SEC, ha="center", va="bottom",
          fontproperties=fp_small)
ax_t.plot([x0, x0 + bar_px], [y0, y0],
          color=C_TEXT_SEC, lw=2, solid_capstyle="butt")
ax_t.text(x0 + bar_px / 2, y0 + ny * 0.04, "1′",
          color=C_TEXT_SEC, ha="center", va="bottom",
          fontproperties=fp_small)

# ── Save ────────────────────────────────────────────────────
fig.savefig(OUT, format="svg", bbox_inches="tight",
            pad_inches=0.12, dpi=300, facecolor=C_DARK)
print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} kB)")