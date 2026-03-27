#!/usr/bin/env python3
"""
Animate lightcurve from output.txt and stack with orbit.mp4.
Run from the directory containing both files.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import subprocess, json

from matplotlib import font_manager
font_path = "../../assets/fonts/Inter-VariableFont_opsz,wght.ttf"
font_manager.fontManager.addfont(font_path)
inter = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = inter.get_name()

inter14 = font_manager.FontProperties(fname=font_path, size=20)

# ── palette ───────────────────────────────────────────────
AXIS_CLR = "#5C4F3D"
BG     = "#FAF7F0"
SURFACE = "#F0EBDF"
BORDER = "#DED7C8"
TEXT_S = "#8A7D6B"
C1     = "#B85B3A"

# ── data ──────────────────────────────────────────────────
data = np.loadtxt("output.txt")
phi  = data[:, 0]
flux = data[:, 2]

phi = (phi - phi.min()) / (phi.max() - phi.min())

# two-phase copy: −1 … 1
phi2  = np.concatenate([phi - 1, phi])
flux2 = np.concatenate([flux, flux])
order = np.argsort(phi2)
phi2, flux2 = phi2[order], flux2[order]

# ── probe orbit.mp4 ──────────────────────────────────────
info = json.loads(subprocess.check_output(
    ["ffprobe", "-v", "quiet", "-print_format", "json",
     "-show_streams", "-count_frames", "orbit.mp4"]
))
vs  = next(s for s in info["streams"] if s["codec_type"] == "video")
num, den = map(int, vs["r_frame_rate"].split("/"))
fps = num / den
nf  = int(vs.get("nb_read_frames") or vs.get("nb_frames")
          or round(float(vs.get("duration", 0)) * fps))
print(f"orbit.mp4  {nf} frames  {fps:.1f} fps")

# ── figure ────────────────────────────────────────────────
DPI = 100
fig, ax = plt.subplots(figsize=(1000 / DPI, 550 / DPI), dpi=DPI)
fig.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax.spines[sp].set_color(AXIS_CLR)
    ax.spines[sp].set_linewidth(2.0)
    
ax.tick_params(labelbottom=False, labelleft=False,
               length=6, width=2.0, color=AXIS_CLR, direction="out")
ax.tick_params(labelsize=0, length=0)

ax.set_xlabel("Phase", fontsize=26, color=AXIS_CLR, labelpad=10,
              fontproperties=inter)
ax.set_ylabel("Flux", fontsize=26, color=AXIS_CLR, labelpad=-20,
              fontproperties=inter)

pad = 0.15 * (flux2.max() - flux2.min())
ax.set_xlim(-1, 1)
ax.set_ylim(flux2.min() - pad, flux2.max() + pad)

ax.set_yticks([0.5, 1.0, 1.1])
ax.set_yticklabels(["50%", "100%", "110%"], fontproperties=inter14, color=AXIS_CLR)

ax.tick_params(axis="y", labelleft=True, length=6, width=2.0, color=AXIS_CLR)
ax.tick_params(axis="x", labelbottom=False, length=6, width=2.0, color=AXIS_CLR)

# context phase (dim) and main phase (bright)
ax.plot(phi2, flux2, color=C1, lw=3.5, alpha=0.90, solid_capstyle="round", zorder=2)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.plot(xmax, ymin, ">", color=AXIS_CLR, markersize=10,
        clip_on=False, zorder=10, transform=ax.transData)
ax.plot(xmin, ymax, "^", color=AXIS_CLR, markersize=10,
        clip_on=False, zorder=10, transform=ax.transData)

vline, = ax.plot([0, 0], [ymin, 0], color=TEXT_S, lw=0.7, alpha=0.25, zorder=1)
glow, = ax.plot([], [], "o", color=C1, ms=20, alpha=0.18, zorder=4)
dot,  = ax.plot([], [], "o", color=C1, ms=10, zorder=5,
                mec="white", mew=1.2)

#fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.10)
plt.tight_layout()

# ── animate ───────────────────────────────────────────────
def update(i):
    t = -1 + 2 * i / max(2 * nf - 1, 1)
    f = float(np.interp(t, phi2, flux2))
    
    ymin = ax.get_ylim()[0]
    vline.set_data([t, t], [ymin, f])   # x stays constant, y goes from bottom → f
    glow.set_data([t], [f])
    dot.set_data([t], [f])
    return vline, glow, dot

print("rendering lightcurve …")
anim = FuncAnimation(fig, update, frames=2 * nf, blit=True)
writer = FFMpegWriter(fps=fps, codec="libx264",
                      extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
anim.save("lightcurve.mp4", writer=writer, dpi=DPI,
          savefig_kwargs={"facecolor": SURFACE})
plt.close()
print("✓ lightcurve.mp4")

# ── combine ───────────────────────────────────────────────
print("stacking …")
subprocess.check_call([
    "ffmpeg", "-y",
    "-stream_loop", "1", "-i", "orbit.mp4",
    "-i", "lightcurve.mp4",
    "-filter_complex", "[0:v][1:v]vstack=inputs=2",
    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
    "-shortest",
    "combined.mp4"
])
print("✓ combined.mp4")