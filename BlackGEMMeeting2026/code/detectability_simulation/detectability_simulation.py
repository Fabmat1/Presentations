#!/usr/bin/env python3
"""
HW Vir detection probability simulation using LCURVE.

For each G-mag bin (8–23), generates 10 random eclipsing sdB+dM systems,
runs LCURVE for a physical model lightcurve, then simulates observations
with TESS, ZTF, ATLAS and BlackGEM (4 N-scenarios).

Variability detection: chi-squared p < 5 %
Period recovery:       Lomb-Scargle peak within 1 % of true period (or harmonic)

Outputs one SVG per BlackGEM-N scenario + debug diagnostic plots.
"""

import numpy as np
import json
import subprocess
import tempfile
import os
import sys
from scipy import stats
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from astropy.timeseries import LombScargle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from multiprocessing import Pool, cpu_count
from functools import partial

try:
    from c_functions.gls_wrapper import GLS_AVAILABLE, gls_power
except ImportError:
    GLS_AVAILABLE = False

if not GLS_AVAILABLE:
    from astropy.timeseries import LombScargle

np.random.seed(2024)

# ═════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════
G_GRAV = 6.674e-11
MSUN   = 1.989e30
RSUN   = 6.957e8

# ═════════════════════════════════════════════════════════
# Colour palette
# ═════════════════════════════════════════════════════════
C = dict(
    bg="#F0EBDF", bdr="#DED7C8", txt="#2C241A", sec="#8A7D6B",
    TESS="#558B44", ZTF="#B85B3A", ATLAS="#4A78A8", BG="#2D8A8A",
)

# ═════════════════════════════════════════════════════════
# Noise model  (CCD equation per exposure)
# ═════════════════════════════════════════════════════════
def _F0(lam_nm, dlam_nm):
    """Photon flux from AB=0 source [ph s⁻¹ m⁻²]."""
    return 3.631e-23 * dlam_nm * 1e-9 / (6.626e-34 * lam_nm * 1e-9)

INSTR = {
    "TESS": dict(
        D=0.105, eta=0.50, texp=20,
        lam=786.5, dlam=400,
        sky_ppix_s=75, npix=15, RN=10,
        floor=0.0002, dG=0.5,
    ),
    "ZTF": dict(
        D=1.22, eta=0.40, texp=30,
        lam=475, dlam=130,
        sky_mu=21.5, pscl=1.01,
        npix=28, RN=10,
        floor=0.005, dG=0.0,
    ),
    "ATLAS": dict(
        D=0.50, eta=0.30, texp=30,
        lam=690, dlam=260,
        sky_mu=21.0, pscl=1.86,
        npix=23, RN=5,
        floor=0.005, dG=0.15,
    ),
    "BlackGEM": dict(
        D=0.65, eta=0.35, texp=90,
        lam=475, dlam=130,
        sky_mu=22.0, pscl=0.56,
        npix=20, RN=5,
        floor=0.003, dG=0.0,
    ),
}


def sig_flux(G, key):
    """Fractional-flux 1-σ uncertainty per single exposure at Gaia G."""
    if key == "ZTF":
        y0, A, k, x0 = 1.05, 6.20089, 1.12832, 19.35670
        return (y0 + A * np.exp(k * (G - x0))) / 100.0

    elif key == "ATLAS":
        a, b, c = 2.8119e-08, 0.833650, 1.5512e-03
        return max(a * np.exp(b * G) + c, 0.015)

    elif key == "TESS":
        a, b, c = 2.2115e-07, 0.857555, 7.9433e-03
        return a * np.exp(b * G) + c

    elif key == "BlackGEM":
        return np.sqrt(2.520e-05
                       + 8.614e-05 * 10.0 ** (0.4 * (G - 15.0))
                       + 5.875e-07 * 10.0 ** (0.8 * (G - 15.0)))

    raise ValueError(f"Unknown instrument key: {key}")


# ═════════════════════════════════════════════════════════
# HW Vir system generator
# ═════════════════════════════════════════════════════════
def roche_r(q):
    """Eggleton Roche-lobe radius in units of orbital separation a."""
    q13, q23 = q ** (1 / 3), q ** (2 / 3)
    return 0.49 * q23 / (0.6 * q23 + np.log(1 + q13))


def _gen_hwvir_rng(rng):
    """Same as gen_hwvir but uses an explicit RandomState."""
    for _ in range(300):
        M1 = 0.47
        M2 = np.clip(rng.normal(0.12, 0.04), 0.05, 0.20)
        P_d = rng.uniform(1, 10) / 24.0
        R1 = np.clip(rng.normal(0.20, 0.02), 0.14, 0.28)
        R2 = max(0.06, 0.0324 + 0.9343 * M2 + 0.0374 * M2 ** 2)
        T1 = rng.uniform(25000, 35000)
        T2 = 2400 + 6000 * M2

        a  = ((G_GRAV * (M1 + M2) * MSUN * (P_d * 86400) ** 2)
              / (4 * np.pi ** 2)) ** (1 / 3)
        aR = a / RSUN
        r1, r2, q = R1 / aR, R2 / aR, M2 / M1

        if r2 > 0.90 * roche_r(q):   continue
        if r1 > 0.90 * roche_r(1/q): continue
        if r1 + r2 > 1.0:            continue

        i_crit = np.degrees(np.arccos(min(r1 + r2, 1.0)))
        i_lo   = max(70.0, i_crit)
        if i_lo >= 90.0: continue
        i = rng.uniform(i_lo, 90.0)

        vs = 2 * np.pi * a * np.sin(np.radians(i)) / (P_d * 86400) / 1e3
        return dict(M1=M1, M2=M2, R1=R1, R2=R2, T1=T1, T2=T2,
                    P_d=P_d, i=i, r1=r1, r2=r2, q=q, vs=vs)
    return None


def _make_ground_t_rng(rng, N, baseline):
    """Same as make_ground_t but uses an explicit RandomState."""
    nights = np.floor(rng.uniform(0, baseline, N))
    return np.sort(nights + rng.uniform(0.30, 0.70, N))


def gen_hwvir():
    """Draw a random eclipsing sdB + dM system.  Returns dict or None."""
    for _ in range(300):
        M1 = 0.47
        M2 = np.clip(np.random.normal(0.12, 0.04), 0.05, 0.20)
        P_d = np.random.uniform(1, 10) / 24.0
        R1 = np.clip(np.random.normal(0.20, 0.02), 0.14, 0.28)
        R2 = max(0.06, 0.0324 + 0.9343 * M2 + 0.0374 * M2 ** 2)
        T1 = np.random.uniform(25000, 35000)
        T2 = 2400 + 6000 * M2

        a  = ((G_GRAV * (M1 + M2) * MSUN * (P_d * 86400) ** 2)
              / (4 * np.pi ** 2)) ** (1 / 3)
        aR = a / RSUN
        r1, r2, q = R1 / aR, R2 / aR, M2 / M1

        if r2 > 0.90 * roche_r(q):   continue
        if r1 > 0.90 * roche_r(1/q): continue
        if r1 + r2 > 1.0:            continue

        i_crit = np.degrees(np.arccos(min(r1 + r2, 1.0)))
        i_lo   = max(70.0, i_crit)
        if i_lo >= 90.0: continue
        i = np.random.uniform(i_lo, 90.0)

        vs = 2 * np.pi * a * np.sin(np.radians(i)) / (P_d * 86400) / 1e3
        return dict(M1=M1, M2=M2, R1=R1, R2=R2, T1=T1, T2=T2,
                    P_d=P_d, i=i, r1=r1, r2=r2, q=q, vs=vs)
    return None


def _gen_reflection_rng(rng):
    """Draw a random reflection-effect sdB + dM system (non-eclipsing).
    Periods up to 20 hours, inclinations < 75 degrees."""
    for _ in range(300):
        M1 = 0.47
        M2 = np.clip(rng.normal(0.12, 0.04), 0.05, 0.20)
        P_d = rng.uniform(1, 20) / 24.0
        R1 = np.clip(rng.normal(0.20, 0.02), 0.14, 0.28)
        R2 = max(0.06, 0.0324 + 0.9343 * M2 + 0.0374 * M2 ** 2)
        T1 = rng.uniform(25000, 35000)
        T2 = 2400 + 6000 * M2

        a  = ((G_GRAV * (M1 + M2) * MSUN * (P_d * 86400) ** 2)
              / (4 * np.pi ** 2)) ** (1 / 3)
        aR = a / RSUN
        r1, r2, q = R1 / aR, R2 / aR, M2 / M1

        if r2 > 0.90 * roche_r(q):   continue
        if r1 > 0.90 * roche_r(1/q): continue
        if r1 + r2 > 1.0:            continue

        i = rng.uniform(20.0, 75.0)

        vs = 2 * np.pi * a * np.sin(np.radians(i)) / (P_d * 86400) / 1e3
        return dict(M1=M1, M2=M2, R1=R1, R2=R2, T1=T1, T2=T2,
                    P_d=P_d, i=i, r1=r1, r2=r2, q=q, vs=vs)
    return None


def _wd_radius(M_wd):
    """White dwarf radius [R_sun] from Nauenberg (1972) mass-radius relation."""
    M_Ch = 1.44
    return 0.0126 * ((M_wd / M_Ch) ** (-2.0 / 3.0)
                      - (M_wd / M_Ch) ** (2.0 / 3.0)) ** 0.5


def _gen_ellipsoidal_rng(rng):
    """Draw a random ellipsoidal sdB + WD system.
    M2 = 0.4–0.8 Msun WD, periods up to 0.25 days."""
    for _ in range(300):
        M1 = 0.47
        M2 = rng.uniform(0.4, 0.8)
        P_d = rng.uniform(0.04, 0.25)
        R1 = np.clip(rng.normal(0.20, 0.02), 0.14, 0.28)
        R2 = _wd_radius(M2)
        T1 = rng.uniform(25000, 35000)
        T2 = rng.uniform(10000, 25000)

        a  = ((G_GRAV * (M1 + M2) * MSUN * (P_d * 86400) ** 2)
              / (4 * np.pi ** 2)) ** (1 / 3)
        aR = a / RSUN
        r1, r2, q = R1 / aR, R2 / aR, M2 / M1

        if r2 > 0.90 * roche_r(q):   continue
        if r1 > 0.90 * roche_r(1/q): continue
        if r1 + r2 > 1.0:            continue

        i = rng.uniform(20.0, 80.0)

        vs = 2 * np.pi * a * np.sin(np.radians(i)) / (P_d * 86400) / 1e3
        return dict(M1=M1, M2=M2, R1=R1, R2=R2, T1=T1, T2=T2,
                    P_d=P_d, i=i, r1=r1, r2=r2, q=q, vs=vs)
    return None

# ═════════════════════════════════════════════════════════
# System-type registry
# ═════════════════════════════════════════════════════════
SYS_GENERATORS = {
    "hwvir":       _gen_hwvir_rng,
    "reflection":  _gen_reflection_rng,
    "ellipsoidal": _gen_ellipsoidal_rng,
}

SYS_LIMITS = {
    "hwvir":       dict(min_var=0.30, max_var=np.inf, max_regen=20),
    "reflection":  dict(min_var=0.05, max_var=0.30,   max_regen=50),
    "ellipsoidal": dict(min_var=0.005, max_var=0.02,  max_regen=50),
}

SYS_MIRROR = {
    "hwvir":       "0",
    "reflection":  "1",
    "ellipsoidal": "0",
}

# ═════════════════════════════════════════════════════════
# LCURVE JSON builder + runner
# ═════════════════════════════════════════════════════════
def _v(val):
    return f"{val}  0.1  0.001  0  1"


def make_cfg(s, wl, out):
    dummy = os.path.abspath(os.path.join(os.path.dirname(out), "dummy_data.txt"))
    x = np.linspace(0, 1, 200)
    with open(dummy, "w") as f:
        for xi in x:
            f.write(f"{xi:.8f} 1 1 1 1 1\n")
    mp = {
        "q":              f"{s['q']}  {2*s['q']:.6f}  {s['q']*0.01:.6f}  0 1",
        "iangle":         f"{s['i']:.4f}  1  0.1  0 1",
        "r1":             f"{s['r1']:.6f}  {s['r1']*0.3:.6f}  {s['r1']*0.01:.6f}  0 1",
        "r2":             f"{s['r2']:.6f}  {s['r2']*0.3:.6f}  {s['r2']*0.01:.6f}  0 1",
        "velocity_scale": f"{s['vs']:.4f}  {s['vs']*0.5:.4f}  {s['vs']*0.02:.4f}  0 1",
        "t1":             f"{s['T1']:.1f}  {s['T1']*0.5:.1f}  {s['T1']*0.005:.1f}  0 1",
        "t2":             f"{s['T2']:.1f}  {s['T2']*0.5:.1f}  {s['T2']*0.02:.1f}  0 1",
        "ldc1_1": _v(0.35), "ldc2_1": _v(0.65),
        "ldc1_2": _v(0.20), "ldc2_2": _v(-0.10),
        "ldc1_3": _v(-0.10),"ldc2_3": _v(0.05),
        "ldc1_4": _v(0.03), "ldc2_4": _v(-0.02),
        "beam_factor1": _v(4.0), "beam_factor2": _v(1.0),
        "t0":  "0.0  0.1  1e-5  0 1",
        "period": "1  0.001  1e-8  0 1",
        "pdot": "0 0.01 1e-5 0 1", "deltat": "0 0.001 0.0001 0 1",
        "gravity_dark1": _v(0.25), "gravity_dark2": _v(0.08),
        "absorb": _v(1.0),
        "cphi3": "0 0.05 0.01 0 1", "cphi4": "0 0.05 0.01 0 1",
        "spin1": _v(1), "spin2": _v(1),
        "slope": "0 0.01 1e-5 0 1", "quad": "0 0.01 1e-5 0 1",
        "cube":  "0 0.01 1e-5 0 1", "third": "0 0.01 1e-5 0 1",
        "rdisc1": "0 0.01 0.001 0 1", "rdisc2": "0 0.01 0.02 0 1",
        "height_disc": "0 0.01 1e-5 0 1", "beta_disc": "0 0.01 1e-5 0 1",
        "temp_disc": "0 50 40 0 1", "texp_disc": "0 0.2 0.001 0 1",
        "lin_limb_disc": "0 0.02 0.0001 0 1",
        "quad_limb_disc": "0 0.02 0.0001 0 1",
        "radius_spot": "0 0.01 0.01 0 1", "length_spot": "0 0.01 0.005 0 1",
        "height_spot": "0 0.01 1e-5 0 1",  "expon_spot": "0 0.2 0.1 0 1",
        "epow_spot": "0 0.01 0.01 0 1",    "angle_spot": "0 5 2 0 1",
        "yaw_spot": "0 5 2 0 1",           "temp_spot": "0 500 200 0 1",
        "tilt_spot": "0 5 2 0 1",          "cfrac_spot": "0 0.05 0.008 0 1",
        "stsp11_long": "0 0 0 0 0", "stsp11_lat": "0 0 0 0 0",
        "stsp11_fwhm": "0 0 0 0 0", "stsp11_tcen": "0 0 0 0 0",
        "stsp21_long": "0 0 0 0 0", "stsp21_lat": "0 0 0 0 0",
        "stsp21_fwhm": "0 0 0 0 0", "stsp21_tcen": "0 0 0 0 0",
        "delta_phase": "1e-7",
        "nlat1f": "40", "nlat2f": "80",
        "nlat1c": "20", "nlat2c": "40",
        "npole": "1", "nlatfill": "2", "nlngfill": "2",
        "lfudge": "0", "llo": "90", "lhi": "-90",
        "phase1": "0.1", "phase2": "0.4",
        "roche1": "1", "roche2": "1",
        "eclipse1": "1", "eclipse2": "1",
        "glens1": "0", "use_radii": "1",
        "gdark_bolom1": "1", "gdark_bolom2": "1",
        "mucrit1": "0", "mucrit2": "0",
        "limb1": "Claret", "limb2": "Claret",
        "mirror": "0",
        "add_disc": "0", "nrad": "40", "opaque": "0",
        "add_spot": "0", "nspot": "0",
        "iscale": "0",
        "wavelength": str(wl),
        "tperiod":    str(s["P_d"]),
    }
    return {
        "time1": 0, "time2": 1, "ntime": 1500,
        "expose": 0, "ndivide": 1, "noise": 0,
        "seed": int(np.random.randint(1, 99999)),
        "nfile": 1,
        "data_file_path": dummy,
        "output_file_path": out,
        "plot_device": "null",
        "residual_offset": 0.0, "autoscale": True,
        "sstar1": 1, "sstar2": 1, "sdisc": 0, "sspot": 0, "ssfac": 1,
        "star1_type": "sd", "star2_type": "ms",
        "use_priors": False, "true_period": s["P_d"],
        "model_parameters": mp,
    }


def run_lcurve(cfg, d):
    """Execute lcurve_re, return (phase, flux) arrays or (None, None)."""
    p = os.path.abspath(os.path.join(d, "config.json"))
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    try:
        r = subprocess.run(
            ["lcurve_re", p],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:
        print(f"    ⚠ LCURVE exec error: {exc}")
        return None, None
    if r.returncode != 0:
        print(f"    ⚠ LCURVE rc={r.returncode}: {r.stderr[:200]}")
        return None, None
    op = os.path.abspath(cfg["output_file_path"])
    if not os.path.exists(op):
        return None, None
    try:
        data = np.loadtxt(op)
        return data[:, 0], data[:, 2]
    except Exception:
        return None, None


# ═════════════════════════════════════════════════════════
# Timestamp generators
# ═════════════════════════════════════════════════════════
_dt = 20.0 / 86400.0
TESS_T = np.concatenate([np.arange(0, 27, _dt),
                          np.arange(28, 55, _dt)])


def make_ground_t(N, baseline):
    nights = np.floor(np.random.uniform(0, baseline, N))
    return np.sort(nights + np.random.uniform(0.30, 0.70, N))


BL_MAP = {"ZTF": 1826, "ATLAS": 2190}
BG_BL  = {10: 365, 35: 365, 500: 180, 1000: 30}


# ═════════════════════════════════════════════════════════
# Detection tests
# ═════════════════════════════════════════════════════════
def var_detected(fl, err, thresh=0.001):
    n = len(fl)
    if n < 3:
        return False
    w = 1.0 / err**2
    wmean = np.sum(fl * w) / np.sum(w)
    chi2  = np.sum(((fl - wmean) / err) ** 2)
    return stats.chi2.sf(chi2, n - 1) < thresh


def per_found(t, fl, err, Ptrue, tol=0.01):
    n = len(t)
    if n < 6:
        return False
    T = np.ptp(t)
    if T < 0.05:
        return False
    min_p = max(np.min(np.diff(np.sort(t))) * 2, 0.015)
    max_p = min(T / 2, 3.0)
    min_p = min(min_p, Ptrue * 0.4)
    max_p = max(max_p, Ptrue * 2.5)
    if max_p <= min_p:
        return False
    f0 = 1.0 / max_p
    df = 1.0 / (20.0 * T)
    Nf = min(int(np.ceil((1.0 / min_p - f0) / df)) + 1, 2_000_000)
    if Nf < 10:
        return False
    if GLS_AVAILABLE:
        power = gls_power(
            t, fl, err, f0=f0, df=df, Nf=Nf,
            normalization=0, fit_mean=True,
            center_data=True, nterms=1,
        )
        freqs = f0 + df * np.arange(Nf)
    else:
        freqs = f0 + df * np.arange(Nf)
        ls = LombScargle(t, fl, err)
        power = ls.power(freqs, method="fast")
    bp = 1.0 / freqs[np.argmax(power)]
    for h in (0.5, 1.0, 2.0):
        if abs(bp - Ptrue * h) / (Ptrue * h) < tol:
            return True
    return False


# ═════════════════════════════════════════════════════════
# Simulation parameters
# ═════════════════════════════════════════════════════════
GBINS  = np.arange(8, 23)
GC     = GBINS + 0.5
NT     = 25
BGN    = [10, 35, 500, 1000]
FIXED  = {"TESS": len(TESS_T), "ZTF": 1300, "ATLAS": 3600}
WL     = 475.0


# ═════════════════════════════════════════════════════════
# Worker function  (★ returns model LC + noise levels for debug)
# ═════════════════════════════════════════════════════════
def _process_one_trial(args):
    """Worker for one (bin, trial).  Returns result dict with debug data.
    Accepts 4-tuple (ib, g0, it, seed) for backward compat (defaults to hwvir)
    or 5-tuple (ib, g0, it, seed, sys_type)."""
    if len(args) == 5:
        ib, g0, it, seed, sys_type = args
    else:
        ib, g0, it, seed = args
        sys_type = "hwvir"

    rng = np.random.RandomState(seed)
    G = rng.uniform(g0, g0 + 1)

    gen_func  = SYS_GENERATORS[sys_type]
    limits    = SYS_LIMITS[sys_type]
    min_var   = limits["min_var"]
    max_var   = limits["max_var"]
    max_regen = limits["max_regen"]
    mirror_val = SYS_MIRROR[sys_type]

    sp = None
    ph_m = fl_m = None
    for _attempt in range(max_regen):
        sp = gen_func(rng)
        if sp is None:
            continue

        td = os.path.abspath(os.path.join(".", ".tmp",
                             f"{sys_type}_bin{ib}_trial{it}"))
        os.makedirs(td, exist_ok=True)
        out = os.path.join(td, "output.txt")
        dummy = os.path.join(td, "dummy_data.txt")
        with open(dummy, "w") as f:
            f.write("0.0 1.0 0.001\n0.5 1.0 0.001\n")

        cfg = make_cfg(sp, WL, out)
        cfg["data_file_path"] = dummy
        cfg["seed"] = int(rng.randint(1, 99999))
        cfg["model_parameters"]["mirror"] = mirror_val

        ph_m, fl_m = run_lcurve(cfg, td)
        if ph_m is None:
            continue

        fl_m /= np.median(fl_m)
        ptp = np.ptp(fl_m)
        if ptp >= min_var and ptp < max_var:
            break
        else:
            ph_m = fl_m = None

    if ph_m is None or sp is None:
        return None

    ph_e = np.concatenate([ph_m - 1, ph_m, ph_m + 1])
    fl_e = np.concatenate([fl_m, fl_m, fl_m])
    model = interp1d(ph_e, fl_e, kind="linear",
                     bounds_error=False, fill_value=1.0)

    P  = sp["P_d"]
    ed = np.ptp(fl_m)

    result = {"ib": ib, "it": it, "G": G, "P": P, "ed": ed,
              "system": sp, "ph_model": ph_m, "fl_model": fl_m,
              "sys_type": sys_type}

    def _test(t, inst_key):
        phases = (t % P) / P
        ft = model(phases)
        s  = sig_flux(G, inst_key)
        fo = ft + rng.normal(0, s, len(t))
        fe = np.full(len(t), s)
        return var_detected(fo, fe), per_found(t, fo, fe, P), s

    d, p, s = _test(TESS_T, "TESS")
    result.update(det_TESS=d, per_TESS=p, sig_TESS=s)

    t_ztf = _make_ground_t_rng(rng, FIXED["ZTF"], BL_MAP["ZTF"])
    d, p, s = _test(t_ztf, "ZTF")
    result.update(det_ZTF=d, per_ZTF=p, sig_ZTF=s)

    t_atl = _make_ground_t_rng(rng, FIXED["ATLAS"], BL_MAP["ATLAS"])
    d, p, s = _test(t_atl, "ATLAS")
    result.update(det_ATLAS=d, per_ATLAS=p, sig_ATLAS=s)

    for bn in BGN:
        t_bg = _make_ground_t_rng(rng, bn, BG_BL[bn])
        d, p, s = _test(t_bg, "BlackGEM")
        result[f"det_BG{bn}"] = d
        result[f"per_BG{bn}"] = p
        result[f"sig_BG{bn}"] = s

    return result

# ═════════════════════════════════════════════════════════
# ★  DEBUG PLOTTING FUNCTIONS  (all new)
# ═════════════════════════════════════════════════════════
def _style_ax(ax):
    """Common debug-axis cosmetics."""
    ax.set_facecolor(C["bg"])
    ax.tick_params(colors=C["txt"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(C["bdr"]); spine.set_linewidth(1.0)


def plot_debug_noise_model():
    """σ_flux vs Gaia G for every survey, with typical eclipse depths."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(C["bg"]); _style_ax(ax)

    Garr = np.linspace(7.5, 23.5, 400)
    for key, col in [("TESS", C["TESS"]), ("ZTF", C["ZTF"]),
                     ("ATLAS", C["ATLAS"]), ("BlackGEM", C["BG"])]:
        sigma = np.array([sig_flux(g, key) for g in Garr])
        ax.semilogy(Garr, sigma, color=col, lw=2.5, label=key)

    for depth, lab in [(0.30, "30 % eclipse"), (0.05, "5 %"),
                       (0.01, "1 %")]:
        ax.axhline(depth, color=C["sec"], lw=0.7, ls=":", alpha=0.5)
        ax.text(23.6, depth, lab, fontsize=7, color=C["sec"], va="center")

    ax.set_xlabel("Gaia G  [mag]", fontsize=13, color=C["txt"])
    ax.set_ylabel("σ_flux  (fractional, single exposure)",
                  fontsize=13, color=C["txt"])
    ax.set_title("Per-exposure noise model — typical HW Vir eclipse depths",
                 fontsize=13, color=C["txt"], pad=10)
    ax.legend(fontsize=11, frameon=True, facecolor=C["bg"],
              edgecolor=C["bdr"])
    ax.set_xlim(7.5, 24); ax.set_ylim(5e-5, 5)
    fig.tight_layout()
    fig.savefig("debug_noise_model.png", dpi=150,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print("✓  saved debug_noise_model.png")


def plot_debug_lightcurves(valid_results, sys_label="HW Vir",
                           file_prefix="debug_lightcurves"):
    """
    Grid of phase-folded lightcurves.
    Columns = representative G magnitudes.
    Rows    = Model, TESS, ZTF, ATLAS, BlackGEM(N=35).
    """
    target_ibs = [0, 2, 5, 8, 12]
    examples = {}
    for r in valid_results:
        ib = r["ib"]
        if ib in target_ibs and ib not in examples:
            examples[ib] = r
    cols = sorted(examples.keys())
    ncols = len(cols)
    if ncols == 0:
        print(f"⚠  no examples for {file_prefix}"); return

    surv_info = [("TESS",     "TESS",     C["TESS"]),
                 ("ZTF",      "ZTF",      C["ZTF"]),
                 ("ATLAS",    "ATLAS",    C["ATLAS"]),
                 ("BlackGEM\n(N = 35)", "BlackGEM", C["BG"])]
    nrows = 1 + len(surv_info)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.3 * ncols, 2.3 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor(C["bg"])

    for j, ib in enumerate(cols):
        r = examples[ib]
        ph_m = np.asarray(r["ph_model"])
        fl_m = np.asarray(r["fl_model"])
        G, P, syspar = r["G"], r["P"], r["system"]

        ph_ext = np.concatenate([ph_m - 1, ph_m, ph_m + 1])
        fl_ext = np.concatenate([fl_m, fl_m, fl_m])
        mfn = interp1d(ph_ext, fl_ext, kind="linear",
                       bounds_error=False, fill_value=1.0)

        ax0 = axes[0, j]; ax0.set_facecolor(C["bg"])
        ax0.plot(ph_m, fl_m, color=C["txt"], lw=0.9)
        ax0.set_title(
            f"G = {G:.1f}   P = {P*24:.2f} h   Δ = {r['ed']:.4f}\n"
            f"i = {syspar['i']:.1f}°   T₁ = {syspar['T1']:.0f} K   "
            f"R₁/a = {syspar['r1']:.3f}   R₂/a = {syspar['r2']:.4f}",
            fontsize=6.5, color=C["txt"], pad=4)
        if j == 0:
            ax0.set_ylabel("Model", fontsize=9, fontweight="bold",
                           color=C["txt"])
        ax0.set_xlim(0, 1)
        ax0.tick_params(labelsize=5, colors=C["txt"])
        for spine in ax0.spines.values():
            spine.set_color(C["bdr"]); spine.set_linewidth(0.7)

        rng_dbg = np.random.RandomState(77777 + ib)
        for i, (lab, inst_key, col) in enumerate(surv_info):
            ax = axes[1 + i, j]; ax.set_facecolor(C["bg"])

            if inst_key == "TESS":
                t = TESS_T
            elif inst_key == "ZTF":
                t = _make_ground_t_rng(rng_dbg, FIXED["ZTF"],
                                       BL_MAP["ZTF"])
            elif inst_key == "ATLAS":
                t = _make_ground_t_rng(rng_dbg, FIXED["ATLAS"],
                                       BL_MAP["ATLAS"])
            else:
                t = _make_ground_t_rng(rng_dbg, 35, BG_BL[35])

            phases = (t % P) / P
            sig = sig_flux(G, inst_key)
            fo = mfn(phases) + rng_dbg.normal(0, sig, len(t))
            Npts = len(t)

            if Npts > 500:
                nb = min(200, Npts // 5)
                be = np.linspace(0, 1, nb + 1)
                bc = 0.5 * (be[:-1] + be[1:])
                dig = np.clip(np.digitize(phases, be) - 1, 0, nb - 1)
                bf = np.array([np.median(fo[dig == k])
                               if np.any(dig == k) else np.nan
                               for k in range(nb)])
                ok = ~np.isnan(bf)
                ax.plot(bc[ok], bf[ok], '.', ms=1.5, color=col,
                        alpha=0.85)
            else:
                ax.scatter(phases, fo, s=8, color=col, alpha=0.55,
                           edgecolors="none")

            xf = np.linspace(0, 1, 500)
            ax.plot(xf, mfn(xf), color="crimson", lw=0.5, alpha=0.5)

            snr_var = r["ed"] / sig if sig > 0 else 0
            ax.text(0.97, 0.06,
                    f"N = {Npts:,}   σ = {sig:.4f}\n"
                    f"SNR$_{{var}}$ = {snr_var:.1f}",
                    transform=ax.transAxes, fontsize=5, color=C["sec"],
                    ha="right", va="bottom", family="monospace")

            if j == 0:
                ax.set_ylabel(lab, fontsize=8, fontweight="bold",
                              color=C["txt"])
            if 1 + i == nrows - 1:
                ax.set_xlabel("Phase", fontsize=7, color=C["txt"])
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=5, colors=C["txt"])
            for spine in ax.spines.values():
                spine.set_color(C["bdr"]); spine.set_linewidth(0.7)

    fig.suptitle(f"Debug — simulated {sys_label} phase-folded lightcurves",
                 fontsize=12, color=C["txt"], y=1.005)
    fig.tight_layout()
    fname = f"{file_prefix}.png"
    fig.savefig(fname, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"✓  saved {fname}")

def plot_debug_raw_detections(nvalid, det_f, per_f, det_b, per_b,
                              sys_label="HW Vir",
                              file_prefix="debug_raw_detections"):
    """
    Raw (unsmoothed) fractions with 95 % Wilson CIs.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.patch.set_facecolor(C["bg"])

    def _wilson(k, n):
        if n == 0:
            return 0.0, 0.0, 0.0
        p = k / n; z = 1.96
        denom = 1 + z ** 2 / n
        ctr = (p + z ** 2 / (2 * n)) / denom
        hw = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
        return p, max(0, ctr - hw), min(1, ctr + hw)

    palette = [
        ("TESS",  det_f["TESS"],  per_f["TESS"],  C["TESS"],  -0.18),
        ("ZTF",   det_f["ZTF"],   per_f["ZTF"],   C["ZTF"],   -0.06),
        ("ATLAS", det_f["ATLAS"], per_f["ATLAS"], C["ATLAS"],  0.06),
        ("BG-35", det_b[35],      per_b[35],      C["BG"],     0.18),
    ]

    for ax, which in [(ax1, "det"), (ax2, "per")]:
        _style_ax(ax)
        for nm, dv, pv, col, dx in palette:
            raw = dv if which == "det" else pv
            x = GC + dx
            elo, ehi = [], []
            for ib in range(len(GBINS)):
                n = int(nvalid[ib])
                k = int(round(raw[ib] * n)) if n > 0 else 0
                _, lo, hi = _wilson(k, n)
                elo.append(raw[ib] - lo)
                ehi.append(hi - raw[ib])
            ax.errorbar(x, raw, yerr=[elo, ehi],
                        fmt="o-", ms=4, lw=1.3, color=col, label=nm,
                        capsize=2.5, capthick=0.8)
        ax.set_ylim(-0.05, 1.12)
        ax.legend(fontsize=9, frameon=True, facecolor=C["bg"],
                  edgecolor=C["bdr"], loc="upper right")
        lab = "Variability detection" if which == "det" else "Period recovery"
        ax.set_ylabel(lab, fontsize=12, color=C["txt"])

    ax2.set_xlabel("Gaia G  [mag]", fontsize=13, color=C["txt"])

    ax3 = ax2.twinx()
    ax3.bar(GC, nvalid, width=0.7, alpha=0.12, color=C["sec"])
    ax3.set_ylabel("N valid trials", fontsize=9, color=C["sec"])
    ax3.set_ylim(0, NT * 2.5)
    ax3.tick_params(colors=C["sec"], labelsize=8)

    fig.suptitle(
        f"Debug — {sys_label} raw (unsmoothed) detection fractions  ·  95 % Wilson CI",
        fontsize=13, color=C["txt"])
    fig.tight_layout()
    fname = f"{file_prefix}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"✓  saved {fname}")
def plot_debug_eclipse_params(valid_results, sys_label="HW Vir",
                              file_prefix="debug_eclipse_params"):
    """Distributions of variation amplitude, period, inclination."""
    eds  = [r["ed"]             for r in valid_results]
    Ps   = [r["P"] * 24        for r in valid_results]
    incs = [r["system"]["i"]   for r in valid_results]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.patch.set_facecolor(C["bg"])

    for ax, data, xl, tl in [
        (a1, eds,  "Peak-to-peak variation",  "Variation amplitudes"),
        (a2, Ps,   "Period  [h]",             "Orbital periods"),
        (a3, incs, "Inclination  [°]",        "Inclinations"),
    ]:
        _style_ax(ax)
        ax.hist(data, bins=20, color=C["BG"], edgecolor=C["bdr"],
                alpha=0.75)
        ax.set_xlabel(xl, fontsize=11, color=C["txt"])
        ax.set_ylabel("Count", fontsize=11, color=C["txt"])
        ax.set_title(tl, fontsize=12, color=C["txt"])

    fig.suptitle(f"{sys_label} — system parameter distributions",
                 fontsize=13, color=C["txt"])
    fig.tight_layout()
    fname = f"{file_prefix}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"✓  saved {fname}")

def _accumulate_results(results):
    """Accumulate trial results into detection/period fraction arrays."""
    det_f = {s: np.zeros(len(GBINS)) for s in FIXED}
    per_f = {s: np.zeros(len(GBINS)) for s in FIXED}
    det_b = {n: np.zeros(len(GBINS)) for n in BGN}
    per_b = {n: np.zeros(len(GBINS)) for n in BGN}
    nvalid = np.zeros(len(GBINS))

    for r in results:
        if r is None:
            continue
        ib = r["ib"]
        nvalid[ib] += 1
        for s in FIXED:
            det_f[s][ib] += r[f"det_{s}"]
            per_f[s][ib] += r[f"per_{s}"]
        for bn in BGN:
            det_b[bn][ib] += r[f"det_BG{bn}"]
            per_b[bn][ib] += r[f"per_BG{bn}"]

    mask = nvalid > 0
    for s in FIXED:
        det_f[s][mask] /= nvalid[mask]
        per_f[s][mask] /= nvalid[mask]
    for bn in BGN:
        det_b[bn][mask] /= nvalid[mask]
        per_b[bn][mask] /= nvalid[mask]

    return nvalid, det_f, per_f, det_b, per_b


def _make_publication_plots(nvalid, det_f, per_f, det_b, per_b,
                            sys_label, file_prefix):
    """Generate one SVG per BlackGEM-N scenario for a given system type."""

    def smooth(y, sigma=0.75):
        return np.clip(gaussian_filter1d(y.astype(float), sigma), 0, 1)

    for bn in BGN:
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("#00000000")

        curves = [
            ("TESS",     det_f["TESS"],  per_f["TESS"],  C["TESS"],
             f"TESS  (N = {FIXED['TESS']:,})"),
            ("ZTF",      det_f["ZTF"],   per_f["ZTF"],   C["ZTF"],
             f"ZTF  (N = {FIXED['ZTF']:,})"),
            ("ATLAS",    det_f["ATLAS"], per_f["ATLAS"], C["ATLAS"],
             f"ATLAS  (N = {FIXED['ATLAS']:,})"),
            ("BlackGEM", det_b[bn],      per_b[bn],      C["BG"],
             f"BlackGEM  (N = {bn:,})"),
        ]

        for _, d, p, col, _ in curves:
            ax.plot(GC, smooth(d), color=col, lw=2.5, ls="-",  zorder=3)
            ax.plot(GC, smooth(p), color=col, lw=2.5, ls="--", zorder=3)

        handles = [mlines.Line2D([], [], color=col, lw=2.5, label=lab)
                   for _, _, _, col, lab in curves]
        handles += [
            mlines.Line2D([], [], color=C["sec"], lw=2, ls="-",
                          label="Variability detection"),
            mlines.Line2D([], [], color=C["sec"], lw=2, ls="--",
                          label="Period recovery"),
        ]
        leg = ax.legend(
            handles=handles,
            loc="lower left",
            fontsize=10, frameon=True, fancybox=False,
            edgecolor=C["bdr"], framealpha=0.95,
            facecolor=C["bg"],
            borderpad=0.7, labelspacing=0.40,
        )
        for t in leg.get_texts():
            t.set_color(C["txt"])

        ax.set_xlim(8, 23)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xlabel("Gaia G  [mag]", fontsize=14, color=C["txt"])
        ax.set_ylabel("Detection / recovery fraction",
                       fontsize=14, color=C["txt"])
        ax.set_title(f"{sys_label} systems — LCURVE simulation",
                     fontsize=15, color=C["txt"], pad=14)
        ax.tick_params(colors=C["txt"], labelsize=12,
                       direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_color(C["bdr"]); spine.set_linewidth(1.2)

        fig.tight_layout()
        fn = f"{file_prefix}_bgN{bn:04d}.svg"
        fig.savefig(fn, format="svg", transparent=True,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"✓  saved {fn}")

# ═════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":

    SYS_CONFIGS = [
        {
            "type":         "hwvir",
            "label":        "HW Vir",
            "file_prefix":  "hwvir_lcurve",
            "debug_prefix": "debug_hwvir",
        },
        {
            "type":         "reflection",
            "label":        "Reflection effect",
            "file_prefix":  "reflection_lcurve",
            "debug_prefix": "debug_reflection",
        },
        {
            "type":         "ellipsoidal",
            "label":        "Ellipsoidal",
            "file_prefix":  "ellipsoidal_lcurve",
            "debug_prefix": "debug_ellipsoidal",
        },
    ]

    print("=" * 56)
    print("  Detection probability simulation  (LCURVE-based)")
    print(f"  System types: {[sc['label'] for sc in SYS_CONFIGS]}")
    print(f"  Workers: {cpu_count()}   GLS: {GLS_AVAILABLE}")
    print("=" * 56)

    from tqdm import tqdm
    nworkers = min(cpu_count(), 12)

    all_accumulated = {}

    for sc in SYS_CONFIGS:
        sys_type     = sc["type"]
        sys_label    = sc["label"]
        file_prefix  = sc["file_prefix"]
        debug_prefix = sc["debug_prefix"]

        print(f"\n{'─' * 56}")
        print(f"  Simulating: {sys_label}")
        print(f"{'─' * 56}")

        master_rng = np.random.RandomState(2024 + hash(sys_type) % 10000)
        tasks = []
        for ib, g0 in enumerate(GBINS):
            for it in range(NT):
                tasks.append((ib, g0, it,
                              master_rng.randint(0, 2 ** 31),
                              sys_type))

        results = []
        with Pool(nworkers) as pool:
            for r in tqdm(pool.imap_unordered(_process_one_trial, tasks),
                          total=len(tasks),
                          desc=f"{sys_label} sims",
                          ncols=80, unit="trial"):
                results.append(r)

        nvalid, det_f, per_f, det_b, per_b = _accumulate_results(results)
        valid_results = [r for r in results if r is not None]

        all_accumulated[sys_type] = dict(
            nvalid=nvalid, det_f=det_f, per_f=per_f,
            det_b=det_b, per_b=per_b, valid_results=valid_results,
        )

        print(f"\n  {sys_label} valid trials per bin: {nvalid}")

        if sys_type == "hwvir":
            print(f"\n── Per-trial diagnostic ({sys_label}, G = 8–10 bins) ──")
            for r in sorted(
                [r for r in valid_results if r["ib"] <= 1],
                key=lambda x: (x["ib"], x["it"]),
            ):
                print(
                    f"  ib={r['ib']} it={r['it']:2d}  G={r['G']:.2f}  "
                    f"P={r['P']*24:.2f}h  Δ={r['ed']:.3f}  "
                    f"TESS(det={int(r['det_TESS'])},per={int(r['per_TESS'])}) "
                    f"ZTF(det={int(r['det_ZTF'])},per={int(r['per_ZTF'])}) "
                    f"σ_T={r['sig_TESS']:.5f}  σ_Z={r['sig_ZTF']:.5f}"
                )

    # ═══════════════════════════════════════════════════════
    # Debug plots
    # ═══════════════════════════════════════════════════════
    print(f"\n── Generating debug plots ──")
    plot_debug_noise_model()

    for sc in SYS_CONFIGS:
        sys_type     = sc["type"]
        sys_label    = sc["label"]
        debug_prefix = sc["debug_prefix"]
        acc = all_accumulated[sys_type]

        if acc["valid_results"]:
            plot_debug_lightcurves(
                acc["valid_results"],
                sys_label=sys_label,
                file_prefix=f"{debug_prefix}_lightcurves",
            )
            plot_debug_eclipse_params(
                acc["valid_results"],
                sys_label=sys_label,
                file_prefix=f"{debug_prefix}_params",
            )
        plot_debug_raw_detections(
            acc["nvalid"], acc["det_f"], acc["per_f"],
            acc["det_b"], acc["per_b"],
            sys_label=sys_label,
            file_prefix=f"{debug_prefix}_raw_detections",
        )

    # ═══════════════════════════════════════════════════════
    # Publication plots
    # ═══════════════════════════════════════════════════════
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica",
                            "Arial", "DejaVu Sans"],
        "font.size": 12,
    })

    for sc in SYS_CONFIGS:
        sys_type    = sc["type"]
        sys_label   = sc["label"]
        file_prefix = sc["file_prefix"]
        acc = all_accumulated[sys_type]

        _make_publication_plots(
            acc["nvalid"], acc["det_f"], acc["per_f"],
            acc["det_b"], acc["per_b"],
            sys_label=sys_label,
            file_prefix=file_prefix,
        )

    # ═══════════════════════════════════════════════════════
    # Save all fractions
    # ═══════════════════════════════════════════════════════
    save_dict = {"GC": GC}
    for sc in SYS_CONFIGS:
        sys_type = sc["type"]
        acc = all_accumulated[sys_type]
        pfx = sys_type
        save_dict[f"{pfx}_nvalid"] = acc["nvalid"]
        for s in FIXED:
            save_dict[f"{pfx}_det_{s}"]  = acc["det_f"][s]
            save_dict[f"{pfx}_per_{s}"]  = acc["per_f"][s]
        for n in BGN:
            save_dict[f"{pfx}_det_BG{n}"] = acc["det_b"][n]
            save_dict[f"{pfx}_per_BG{n}"] = acc["per_b"][n]

    np.savez("detection_fractions.npz", **save_dict)
    print("\n✓  saved detection_fractions.npz")
    print("\nDone.")