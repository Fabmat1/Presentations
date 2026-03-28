#!/usr/bin/env python3
"""
Hot-subdwarf lightcurve coverage in BlackGEM.

Outputs (SVG, transparent):
  blackgem_sdB_coverage_all.svg
  blackgem_sdB_coverage_100plus.svg
  blackgem_sdB_coverage_100plus_proj.svg
"""

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from pathlib import Path

# ─── Font ───────────────────────────────────────────────────────
font_path = "../../assets/fonts/Inter-VariableFont_opsz,wght.ttf"
font_manager.fontManager.addfont(font_path)
inter = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = inter.get_name()
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12})

# ─── Palette ────────────────────────────────────────────────────
C = dict(
    surface="#F0EBDF", border="#DED7C8",
    text="#2C241A", text_sec="#8A7D6B",
    c1="#B85B3A", c2="#4A78A8", c3="#558B44",
    c4="#9A6B92", c5="#C89B1E", c6="#2D8A8A",
    accent="#B85B3A",
)

# ─── Bin schemes ────────────────────────────────────────────────
EDGES_ALL   = [0, 3, 10, 30, 100, np.inf]
LABELS_ALL  = ["1 – 3", "4 – 10", "11 – 30", "31 – 100", "> 100"]
PALETTE_ALL = [C["c1"], C["c2"], C["c3"], C["c4"], C["c5"]]

EDGES_100   = [100, 200, 500, 1000, 2000, np.inf]
LABELS_100  = ["100 – 200", "200 – 500", "500 – 1 k", "1 k – 2 k", "> 2 k"]
PALETTE_100 = [C["c1"], C["c2"], C["c3"], C["c4"], C["c5"]]

# ─── Data ───────────────────────────────────────────────────────
CSV = Path(
    "/home/fabian/Documents/Doktor/Presentations/"
    "BlackGEMMeeting2026/code/query_all_blackgem/output_full.csv"
)
df = pd.read_csv(CSV, parse_dates=["DATE_OBS"])
df["month"] = df["DATE_OBS"].dt.to_period("M")
MONTHS = pd.period_range(df["month"].min(), df["month"].max(), freq="M")


# ════════════════════════════════════════════════════════════════
# Field assignment
# ════════════════════════════════════════════════════════════════

def assign_fields(df_in):
    """
    BlackGEM's OBJECT column is the 5-digit field ID from the
    MLBG_FieldIDs catalogue.
    """
    out = df_in.copy()
    out["FIELD_ID"] = out["OBJECT"].apply(lambda x: f"{int(float(x)):05d}")
    n = out["FIELD_ID"].nunique()
    print(f"Assigned {len(out):,} observations to {n} unique fields")
    return out


# ════════════════════════════════════════════════════════════════
# Strategy inference
# ════════════════════════════════════════════════════════════════

def infer_strategy(df_in):
    """
    Detect the sequential observing campaign from the temporal
    pattern of field visits and estimate its parameters.

    Returns
    -------
    dict
        dwell            – median days actively observing one field
        slot             – median days between starts of consecutive
                           target-field visits (≥ dwell; the excess
                           is time on fields without our stars)
        cadence          – median obs/star/day while field is active
        current          – FIELD_ID being observed now
        remaining_dwell  – days left on current field
        completed        – FIELD_IDs finished in sequential phase
        queue            – FIELD_IDs not yet sequentially visited
        cycle_order      – full ordered list for subsequent cycles
        field_stars      – {FIELD_ID: set(SOURCE_IDs)}
    """
    last_date = df_in["DATE_OBS"].max()

    # ── Per-field activity windows (full dataset) ──────────────
    fs = df_in.groupby("FIELD_ID").agg(
        first=("DATE_OBS", "min"),
        last =("DATE_OBS", "max"),
        n_obs=("DATE_OBS", "size"),
    )

    # ── Detect sequential onset ────────────────────────────────
    # In sequential mode only 1-2 fields are active per month.
    monthly = (df_in
               .groupby(df_in["DATE_OBS"].dt.to_period("M"))
               ["FIELD_ID"].nunique())
    roll = monthly.rolling(3, center=True, min_periods=1).median()
    seq_cand = roll[roll <= 2]
    if not seq_cand.empty:
        seq_start = seq_cand.index[0].to_timestamp()
    else:
        # No clear transition yet – use last 90 days
        seq_start = last_date - pd.Timedelta(days=90)

    # ── Sequential-phase field statistics ──────────────────────
    seq = df_in[df_in["DATE_OBS"] >= seq_start]
    sfs = seq.groupby("FIELD_ID").agg(
        first=("DATE_OBS", "min"),
        last =("DATE_OBS", "max"),
        n_obs=("DATE_OBS", "size"),
    )
    sfs["span"] = (sfs["last"] - sfs["first"]).dt.total_seconds() / 86400
    sfs = sfs.sort_values("first")

    # Current field = most recently observed
    current   = sfs["last"].idxmax()
    completed = [f for f in sfs.index if f != current]

    # ── Dwell time (from completed sequential fields) ──────────
    done = sfs.loc[completed]
    if not done.empty and (done["span"] > 1).any():
        dwell = float(done.loc[done["span"] > 1, "span"].median())
    elif sfs.loc[current, "span"] > 1:
        dwell = float(sfs.loc[current, "span"])
    else:
        dwell = 30.0

    # ── Cadence (obs / star / day while field is active) ───────
    cads = []
    for fid in sfs.index:
        sp = sfs.loc[fid, "span"]
        if sp < 1:
            continue
        per = (seq[seq["FIELD_ID"] == fid]
               .groupby("SOURCE_ID").size() / sp)
        cads.extend(per.tolist())
    cadence = float(np.median(cads)) if cads else 0.5

    # ── Slot: time between consecutive field starts ────────────
    # This naturally includes dead-time on fields that contain
    # none of our targets.
    starts = sfs.sort_values("first")["first"]
    if len(starts) > 1:
        deltas = starts.diff().dt.total_seconds().dropna() / 86400
        slot = float(deltas.median())
    else:
        slot = dwell * 1.5          # assume ~50 % overhead
    slot = max(slot, dwell)         # can't be shorter than dwell

    # ── Remaining dwell on current field ───────────────────────
    elapsed   = (last_date - sfs.loc[current, "first"]).total_seconds() / 86400
    remaining = max(0.0, dwell - elapsed)


    # ── Star ↔ field mapping ───────────────────────────────────
    field_stars = (df_in.groupby("FIELD_ID")["SOURCE_ID"]
                        .apply(set).to_dict())

    # ── Queue: only fields that contain our targets ────────────
    fields_with_targets = {fid for fid, stars in field_stars.items()
                           if len(stars) > 0}
    queue = sorted(
        (fields_with_targets - set(sfs.index)),
        key=lambda f: fs.loc[f, "first"],
    )
    cycle_order = [f for f in (list(sfs.index) + queue)
                   if f in fields_with_targets]

    print(
        f"Strategy  seq. since {seq_start.date()}, "
        f"dwell {dwell:.0f} d, slot {slot:.0f} d, "
        f"cadence {cadence:.2f} obs/star/d\n"
        f"          {len(completed)} completed, "
        f"current {current} ({remaining:.0f} d left), "
        f"{len(queue)} queued, {len(cycle_order)} in cycle"
    )

    return dict(
        dwell=dwell, slot=slot, cadence=cadence,
        current=current, remaining_dwell=remaining,
        completed=completed, queue=queue,
        cycle_order=cycle_order, field_stars=field_stars,
    )


# ════════════════════════════════════════════════════════════════
# Cumulative statistics (observed data – unchanged)
# ════════════════════════════════════════════════════════════════

def cumul(df_in, months, edges, labels, min_pts=0):
    """
    At each month boundary, count per-star observations cumulatively.
    If *min_pts* > 0, only stars whose cumulative count has reached
    that threshold at that month are included.
    """
    rows = []
    for m in months:
        sub = df_in[df_in["month"] <= m]
        if sub.empty:
            rows.append(dict(
                month=m, n=0, **{l: 0 for l in labels},
                med=np.nan, p16=np.nan, p84=np.nan,
                p2=np.nan, p98=np.nan, mx=np.nan))
            continue
        pts = sub.groupby("SOURCE_ID").size()
        if min_pts > 0:
            pts = pts[pts >= min_pts]
        if pts.empty:
            rows.append(dict(
                month=m, n=0, **{l: 0 for l in labels},
                med=np.nan, p16=np.nan, p84=np.nan,
                p2=np.nan, p98=np.nan, mx=np.nan))
            continue
        cats = pd.cut(pts, bins=edges, labels=labels,
                       right=True, include_lowest=True)
        cc = cats.value_counts().reindex(labels, fill_value=0)
        rows.append(dict(
            month=m, n=len(pts), **cc.to_dict(),
            med=pts.median(),
            p16=pts.quantile(0.16), p84=pts.quantile(0.84),
            p2=pts.quantile(0.025), p98=pts.quantile(0.975),
            mx=pts.max()))
    R = pd.DataFrame(rows)
    R["date"] = R["month"].apply(
        lambda p: p.to_timestamp() + pd.Timedelta(days=14))
    return R


# ════════════════════════════════════════════════════════════════
# Projection (field-aware sequential model)
# ════════════════════════════════════════════════════════════════

def project(df_full, last_month, target, edges, labels,
            min_pts=0, strategy=None):
    """
    Project cumulative observation counts forward by modelling the
    sequential field-by-field strategy:

    1.  Finish the current field (remaining dwell).
    2.  Visit each queued field for *dwell* days.
    3.  Cycle through all fields again.

    Between consecutive starts of fields that contain our stars
    *slot* days pass (slot ≥ dwell); the excess is time the
    telescope spends on intermediate fields with no hot subdwarfs.
    """
    if strategy is None:
        strategy = infer_strategy(df_full)
    strat = strategy

    # Base counts (all observations to date)
    star_counts = df_full.groupby("SOURCE_ID").size()

    ref  = last_month.to_timestamp() + pd.Timedelta(days=14)
    end  = (pd.Period(target, freq="M").to_timestamp()
            + pd.Timedelta(days=14))
    max_days = (end - ref).days

    dwell = strat["dwell"]
    gap   = max(0.0, strat["slot"] - dwell)

    # ── Build timeline: (field_id, t_start, t_end) in days ────
    timeline = []
    t = 0.0

    # Current field – remaining dwell
    rd = strat["remaining_dwell"]
    if rd > 0:
        timeline.append((strat["current"], t, t + rd))
    t = rd + gap

    # Queue (first pass) → then full cycles
    for fid in itertools.chain(strat["queue"],
                               itertools.cycle(strat["cycle_order"])):
        if t >= max_days:
            break
        timeline.append((fid, t, t + dwell))
        t += dwell + gap

    # ── Month-by-month projection ─────────────────────────────
    future = pd.period_range(last_month + 1, target, freq="M")
    fstars = strat["field_stars"]
    rows   = []

    for m in future:
        days = (m.to_timestamp()
                + pd.Timedelta(days=14) - ref).total_seconds() / 86400
        proj = star_counts.copy().astype(float)

        for fid, t0, t1 in timeline:
            if t0 >= days:
                break
            active = min(t1, days) - t0
            if active <= 0:
                continue
            stars = fstars.get(fid, set())
            if not stars:
                continue
            mask = proj.index.isin(stars)
            proj.loc[mask] += strat["cadence"] * active

        if min_pts > 0:
            proj = proj[proj >= min_pts]

        if proj.empty:
            rows.append(dict(
                month=m, n=0, **{l: 0 for l in labels},
                med=np.nan, p16=np.nan, p84=np.nan,
                p2=np.nan, p98=np.nan, mx=np.nan))
        else:
            cats = pd.cut(proj, bins=edges, labels=labels,
                           right=True, include_lowest=True)
            cc = cats.value_counts().reindex(labels, fill_value=0)
            rows.append(dict(
                month=m, n=len(proj), **cc.to_dict(),
                med=proj.median(),
                p16=proj.quantile(0.16), p84=proj.quantile(0.84),
                p2=proj.quantile(0.025), p98=proj.quantile(0.975),
                mx=proj.max()))

    P = pd.DataFrame(rows)
    P["date"] = P["month"].apply(
        lambda p: p.to_timestamp() + pd.Timedelta(days=14))
    return P


# ════════════════════════════════════════════════════════════════
# Drawing (unchanged)
# ════════════════════════════════════════════════════════════════

def _style(ax):
    ax.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    ax.tick_params(colors=C["text"])
    ax.grid(axis="y", color=C["border"], lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def draw(R, path, labels, palette, proj=None):
    """Two-panel chart → SVG."""

    active = [l for l in labels
              if R[l].sum() > 0
              or (proj is not None and proj[l].sum() > 0)]
    acols = [palette[labels.index(l)] for l in active]

    all_d = sorted(
        R["date"].tolist()
        + (proj["date"].tolist() if proj is not None else []))
    if len(all_d) > 1:
        span = (all_d[-1] - all_d[0]).days
        bar_w = np.clip(0.8 * span / len(all_d), 12, 28)
    else:
        bar_w = 20

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6.5), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1.3], hspace=0.08))
    fig.patch.set_alpha(0)
    _style(ax1); _style(ax2)

    # ── Top: stacked bars (observed) ───────────────────────────
    bot = np.zeros(len(R))
    for lab, col in zip(active, acols):
        ax1.bar(R["date"], R[lab], bottom=bot, width=bar_w,
                color=col, edgecolor="none", label=lab, zorder=3)
        bot += R[lab].values

    n_now = int(R["n"].iloc[-1])
    if n_now > 0:
        ax1.annotate(
            f"  {n_now:,} stars", xy=(R["date"].iloc[-1], n_now),
            xytext=(6, 2), textcoords="offset points",
            fontsize=10, fontweight="bold", color=C["text"],
            ha="left", va="bottom")

    # ── Top: stacked bars (projected) ──────────────────────────
    if proj is not None and not proj.empty:
        bot_p = np.zeros(len(proj))
        for lab, col in zip(active, acols):
            ax1.bar(proj["date"], proj[lab], bottom=bot_p,
                    width=bar_w, color=col, edgecolor="none",
                    alpha=0.35, zorder=3)
            bot_p += proj[lab].values

        n_end = int(proj["n"].iloc[-1])
        if n_end > 0:
            ax1.annotate(
                f"  {n_end:,} stars",
                xy=(proj["date"].iloc[-1], n_end),
                xytext=(6, 2), textcoords="offset points",
                fontsize=10, fontweight="bold", color=C["text_sec"],
                ha="left", va="bottom")

    ax1.set_ylabel("Cumulative stars observed", color=C["text"])
    ax1.set_ylim(bottom=0)

    leg1 = ax1.legend(
        title="Observations per star", loc="upper left",
        frameon=True, fancybox=False,
        edgecolor=C["border"], facecolor=C["surface"] + "CC",
        title_fontsize=10, fontsize=9)
    leg1.get_title().set_color(C["text"])
    for t in leg1.get_texts():
        t.set_color(C["text"])

    # ── Bottom: depth bands (observed) ─────────────────────────
    v = R.dropna(subset=["med"])
    if not v.empty:
        ax2.fill_between(v["date"], v["p2"], v["p98"],
                         color=C["c2"], alpha=0.12,
                         label="2σ range", zorder=2)
        ax2.fill_between(v["date"], v["p16"], v["p84"],
                         color=C["c2"], alpha=0.28,
                         label="1σ range", zorder=3)
        ax2.plot(v["date"], v["med"], color=C["c2"], lw=2.2,
                 label="Median", zorder=5)
        ax2.plot(v["date"], v["mx"], color=C["accent"],
                 lw=1.3, ls="--", label="Maximum", zorder=4)

    # ── Bottom: depth bands (projected) ────────────────────────
    if proj is not None:
        pv = proj.dropna(subset=["med"])
        if not pv.empty and not v.empty:
            brd = pd.concat([v.iloc[[-1]], pv], ignore_index=True)
            ax2.fill_between(brd["date"], brd["p2"], brd["p98"],
                             color=C["c2"], alpha=0.06, zorder=2)
            ax2.fill_between(brd["date"], brd["p16"], brd["p84"],
                             color=C["c2"], alpha=0.14, zorder=3)
            ax2.plot(brd["date"], brd["med"],
                     color=C["c2"], lw=2.2, alpha=0.5, zorder=5)
            ax2.plot(brd["date"], brd["mx"],
                     color=C["accent"], lw=1.3, ls="--",
                     alpha=0.5, zorder=4)

    ax2.set_yscale("log")
    ax2.set_ylim(bottom=1)
    ax2.set_ylabel("Points per star", color=C["text"])

    # ── Projection boundary ────────────────────────────────────
    if proj is not None and not proj.empty:
        bnd = R["date"].iloc[-1] + pd.Timedelta(days=14)
        for ax in (ax1, ax2):
            ax.axvline(bnd, color=C["text_sec"], ls=":", lw=1, zorder=6)
        ax1.text(bnd + pd.Timedelta(days=10), 0.96, "Projected",
                 transform=ax1.get_xaxis_transform(),
                 color=C["text_sec"], fontsize=9, va="top",
                 fontstyle="italic")

    # ── X-axis formatting ──────────────────────────────────────
    loc = mdates.AutoDateLocator(minticks=5, maxticks=14)
    fmt = mdates.ConciseDateFormatter(loc)
    fmt.formats      = ["%Y", "%b", "%b", "%H:%M", "%H:%M", "%S.%f"]
    fmt.zero_formats = ["",   "%Y", "%b\n%Y", "%b-%d", "%H:%M", "%H:%M"]
    ax2.xaxis.set_major_locator(loc)
    ax2.xaxis.set_major_formatter(fmt)

    leg2 = ax2.legend(
        loc="upper left", ncol=4, frameon=True, fancybox=False,
        edgecolor=C["border"], facecolor=C["surface"] + "CC",
        fontsize=9)
    for t in leg2.get_texts():
        t.set_color(C["text"])

    fig.align_ylabels()
    plt.tight_layout()
    fig.savefig(str(path), transparent=True, bbox_inches="tight")
    print(f"✓  {path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Generate the three figures
# ════════════════════════════════════════════════════════════════
out = CSV.parent

# Assign fields
df = assign_fields(df)

# Infer strategy once (used by the projection)
strat = infer_strategy(df)

# 1 — every star, standard bins
R_all = cumul(df, MONTHS, EDGES_ALL, LABELS_ALL)
draw(R_all, "blackgem_sdB_coverage_all.svg",
     LABELS_ALL, PALETTE_ALL)

# 2 — cumulative ≥ 100 view, fine bins up to > 2 k
R100 = cumul(df, MONTHS, EDGES_100, LABELS_100, min_pts=100)
draw(R100, "blackgem_sdB_coverage_100plus.svg",
     LABELS_100, PALETTE_100)

# 3 — same + projection → end of 2028
P100 = project(df, MONTHS[-1], "2028-12",
               EDGES_100, LABELS_100, min_pts=100, strategy=strat)
draw(R100,  "blackgem_sdB_coverage_100plus_proj.svg",
     LABELS_100, PALETTE_100, proj=P100)


P_all = project(df, MONTHS[-1], "2028-12",
                EDGES_ALL, LABELS_ALL, min_pts=0, strategy=strat)
draw(R_all,"blackgem_sdB_coverage_all_proj.svg",
     LABELS_ALL, PALETTE_ALL, proj=P_all)