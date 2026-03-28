import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.lines as mlines

# ═══════════════════════════════════════════════════════════════
# Colour palette
# ═══════════════════════════════════════════════════════════════
PAL = dict(
    ZTF='#B85B3A', ATLAS='#4A78A8', TESS='#558B44', BG='#2D8A8A',
    txt='#2C241A', sec='#8A7D6B', bdr='#DED7C8', srf='#F0EBDF',
)

# ═══════════════════════════════════════════════════════════════
# Photon‐flux zero point  (AB = 0, top‐hat bandpass)
# F0 = f_ν(0) · Δλ / (λ · h)   [ph s⁻¹ m⁻²]
# ═══════════════════════════════════════════════════════════════
_h, _c, _AB0 = 6.626e-34, 3.0e8, 3.631e-23          # SI

def F0(lam_nm, dlam_nm):
    return _AB0 * dlam_nm / (lam_nm * _h)             # ph s⁻¹ m⁻²

# ═══════════════════════════════════════════════════════════════
# Per‐exposure noise models   σ(G)  [mag]
# ═══════════════════════════════════════════════════════════════

def _sig_ground(G, D, eta, texp, pscl, npix,
                mu_sky, RN, floor, lam, dlam, dG):
    m   = np.asarray(G, float) + dG
    f0  = F0(lam, dlam)
    A   = np.pi * (D / 2.0)**2
    Ns  = f0 * A * eta * 10.0**(-0.4 * m) * texp
    Nk  = f0 * A * eta * 10.0**(-0.4 * mu_sky) * pscl**2 * texp  # per pix
    var = Ns + npix * (Nk + RN**2)
    snr = Ns / np.sqrt(np.clip(var, 1.0, None))
    sig = 1.0857 / np.clip(snr, 1e-10, None)
    return np.hypot(sig, floor)

def sig_tess(G):
    T    = np.asarray(G, float) + 0.5                 # hot‐sdB colour term
    C    = 1.5e4 * 10.0**(-0.4 * (T - 10.0))         # e⁻ s⁻¹
    Ns   = C * 20.0                                    # 20‐s cadence
    npix = 15
    Nsky = npix * 75.0 * 20.0                          # zodiacal
    Nrn  = npix * 100.0                                # (10 e⁻)²
    var  = Ns + Nsky + Nrn
    snr  = Ns / np.sqrt(np.clip(var, 1.0, None))
    sig  = 1.0857 / np.clip(snr, 1e-10, None)
    return np.hypot(sig, 2e-4)                         # 0.2 mmag floor

sig_ztf = lambda G: _sig_ground(G, 1.22, .40, 30,
                                 1.01, 25, 21.5, 10, .004,
                                 475, 130, 0.0)
sig_atl = lambda G: _sig_ground(G, 0.50, .30, 30,
                                 1.86, 23, 21.0,  5, .005,
                                 690, 260, 0.2)
sig_bg  = lambda G: _sig_ground(G, 0.65, .35, 90,
                                 0.56, 20, 22.0,  5, .003,
                                 475, 130, 0.0)

# ═══════════════════════════════════════════════════════════════
# Catalogue completeness (sigmoid around single‐exposure limit)
# ═══════════════════════════════════════════════════════════════

def completeness(G, Glim, w):
    if Glim is None:
        return np.ones_like(np.asarray(G, float))
    return norm.cdf((Glim - np.asarray(G, float)) / w)

# ═══════════════════════════════════════════════════════════════
# HW Vir representative system
# ═══════════════════════════════════════════════════════════════
DELTA = 0.30      # primary eclipse depth  [mag]
FDUTY = 0.10      # eclipse duty cycle

# ═══════════════════════════════════════════════════════════════
# Detection probabilities  (BLS framework)
#   • variability detection  – threshold  5
#   • period recovery        – threshold  7.5  + phase‐coverage penalty
# ═══════════════════════════════════════════════════════════════

def P_detect(sig, Neff):
    nin = np.clip(Neff * FDUTY, 0, None)
    snr = DELTA * np.sqrt(nin) / np.clip(sig, 1e-10, None)
    return norm.cdf(snr - 5.0)

def P_period(sig, Neff):
    nin = np.clip(Neff * FDUTY, 0, None)
    snr = DELTA * np.sqrt(nin) / np.clip(sig, 1e-10, None)
    cov = 1.0 - np.exp(-nin / 5.0)
    return norm.cdf(snr - 7.5) * cov

# ═══════════════════════════════════════════════════════════════
# Survey table   (name, σ‐func, N_epochs, G_lim, comp‐width)
# ═══════════════════════════════════════════════════════════════
SURVEYS = [
    ('TESS',     sig_tess, 233_280, 18.0, 1.00),
    ('ZTF',      sig_ztf,    1_300, 20.8, 0.35),
    ('ATLAS',    sig_atl,    3_600, 19.3, 0.35),
    ('BlackGEM', sig_bg,      None, 22.0, 0.35),
]
CKEY = {'TESS': 'TESS', 'ZTF': 'ZTF', 'ATLAS': 'ATLAS', 'BlackGEM': 'BG'}

# ═══════════════════════════════════════════════════════════════
# Magnitude grid & BlackGEM scenarios
# ═══════════════════════════════════════════════════════════════
G     = np.linspace(8.0, 22.0, 2000)
BG_Ns = [10, 35, 500, 1000]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial',
                        'DejaVu Sans'],
    'font.size': 12,
})

# ── helper: find G where curve crosses 0.5 ───────────────────
def cross50(Garr, Parr):
    idx = np.where(Parr < 0.5)[0]
    return Garr[idx[0]] if len(idx) else np.nan

# ═══════════════════════════════════════════════════════════════
# Generate one plot per BlackGEM N
# ═══════════════════════════════════════════════════════════════

for bgN in BG_Ns:

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('#00000000')

    # ── curves ────────────────────────────────────────────────
    for name, sfn, N0, Glim, w in SURVEYS:
        N   = bgN if name == 'BlackGEM' else N0
        sig = sfn(G)
        Ne  = N * completeness(G, Glim, w)

        pd = P_detect(sig, Ne)
        pp = P_period(sig, Ne)

        col = PAL[CKEY[name]]
        ax.plot(G, pd, color=col, lw=2.5, zorder=3)
        ax.plot(G, pp, color=col, lw=2.5, ls='--', zorder=3)

        # optional: print 50 % crossings
        g50d = cross50(G, pd)
        g50p = cross50(G, pp)
        tag  = f'{name:8s}  N={N:>7,}'
        print(f'  {tag}   detect 50 %: G={g50d:5.1f}   '
              f'period 50 %: G={g50p:5.1f}')

    # ── 50 % reference line ──────────────────────────────────
    ax.axhline(0.5, color=PAL['sec'], lw=0.7, ls=':', zorder=1)

    # ── legend ───────────────────────────────────────────────
    hdl = []
    for name, _, N0, _, _ in SURVEYS:
        N = bgN if name == 'BlackGEM' else N0
        hdl.append(mlines.Line2D(
            [], [], color=PAL[CKEY[name]], lw=2.5,
            label=f'{name}  (N = {N:,})'))
    hdl.append(mlines.Line2D([], [], color=PAL['sec'], lw=2,
               ls='-',  label='Variability detection'))
    hdl.append(mlines.Line2D([], [], color=PAL['sec'], lw=2,
               ls='--', label='Period recovery'))

    leg = ax.legend(
        handles=hdl, loc='upper right', fontsize=10,
        frameon=True, fancybox=False,
        edgecolor=PAL['bdr'], framealpha=0.92,
        borderpad=0.7, labelspacing=0.40)
    for t in leg.get_texts():
        t.set_color(PAL['txt'])

    # ── axes & labels ────────────────────────────────────────
    ax.set_xlim(8, 22)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel('Gaia G  [mag]', fontsize=14, color=PAL['txt'])
    ax.set_ylabel('Detection / recovery probability',
                  fontsize=14, color=PAL['txt'])
    ax.set_title('HW Vir systems — analytical detection model',
                 fontsize=15, color=PAL['txt'], pad=14)
    ax.tick_params(colors=PAL['txt'], labelsize=12,
                   direction='in', top=True, right=True)
    for sp in ax.spines.values():
        sp.set_color(PAL['bdr'])
        sp.set_linewidth(1.2)
    ax.grid(True, ls=':', lw=0.7, color=PAL['bdr'], alpha=0.55)

    # ── parameter annotation ─────────────────────────────────
    ax.text(
        0.98, 0.03,
        f'δ = {DELTA} mag  ·  duty cycle = {FDUTY:.0%}  ·  P ≈ 0.1 d\n'
        f'BlackGEM g-band 90 s  |  N = {bgN}',
        transform=ax.transAxes, fontsize=9.5, color=PAL['sec'],
        va='bottom', ha='right',
        bbox=dict(fc='white', ec='none', alpha=0.55, pad=3))

    fig.tight_layout()
    fname = f'hwvir_bgN{bgN:04d}.png'
    fig.savefig(fname, dpi=220, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {fname}\n')