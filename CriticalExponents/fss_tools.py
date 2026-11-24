# helper functions
# used by fss_run.py  and fss_analyse.py 


import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyerrors as pe


# paths 

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
BIN_DIR = BASE_DIR / "bin"
LOG_DIR = BASE_DIR / "logs"
TEMPS_DIR = BASE_DIR / "temps"
SRC = BASE_DIR / "wolff.cpp"


# constants
TC = 2 / np.log(1 + np.sqrt(2))#
TC_5DP = round(TC, 5)# 2.26919
T_DECIMALS = 5 # wolff.cpp currently rounds temperatures for its seeds to 5dp


# temperature grid
def make_temps(L, Tc=TC, n_coarse=20, n_fine=21, width=4.0, extra_hi=2.0,
               lo=1.8, hi=2.8):
    coarse = np.linspace(lo, hi, n_coarse)
    step = 2 * width / (n_fine - 1)                         # in units of 1/L
    n_hi = int(round(extra_hi / step))
    fine = Tc + (-width + step * np.arange(n_fine + n_hi)) / L   # includes Tc exactly
    keep = np.min(np.abs(coarse[:, None] - fine[None, :]), axis=1) > step / (2 * L)
    temps = np.concatenate([coarse[keep], fine])
    return np.unique(np.round(temps, T_DECIMALS))


# folder layout
#   data/L<L>/nsamples<n>/T<T 5dp>/{run000.dat, ..., burnin_run000.dat, meta.txt}

def t_name(T):
    return f"T{T:.{T_DECIMALS}f}"


def t_dir(L, T, nsamples, data_dir=DATA_DIR):
    return Path(data_dir) / f"L{L}" / f"nsamples{nsamples}" / t_name(T)


def is_done(L, T, nsamples, data_dir=DATA_DIR):
    # wolff.cpp writes meta.txt last, only exists for finished temps
    return (t_dir(L, T, nsamples, data_dir) / "meta.txt").exists()


def binary(L):
    return BIN_DIR / f"wolff_L{L}"


# loading
# now temperatures come from the folders rather than linspace, so any grid works

def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            parts = line.split(None, 1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1].strip()
    return meta


def read_trace(fname):
    # reads our run003.dat
    d = pd.read_csv(fname, sep=r"\s+", comment="#", header=None,
                    engine="c", dtype=np.float64).to_numpy()
    if d.ndim != 2 or d.shape[1] != 4:
        raise ValueError(f"{fname} must have 4 columns (sample M E blockClusterSum)")
    return d[:, 1].copy(), d[:, 2].copy(), d[:, 3].copy()


def available_Ls(nsamples, data_dir=DATA_DIR):
    # golbs lattice sizes with finished temps
    Ls = []
    for p in Path(data_dir).glob(f"L*/nsamples{nsamples}"):
        if any(p.glob("T*/meta.txt")):
            Ls.append(int(p.parent.name[1:]))
    return sorted(Ls)


def done_temps(L, nsamples, data_dir=DATA_DIR):
    # since only meta.tct exist for finished runs
    base = Path(data_dir) / f"L{L}" / f"nsamples{nsamples}"
    return sorted(float(p.parent.name[1:]) for p in base.glob("T*/meta.txt"))


def load_L(L, nsamples, data_dir=DATA_DIR):
    # All finished temperatures for a L
    # indexed [T][run].
    temps = done_temps(L, nsamples, data_dir)
    M, E, S, metas = [], [], [], []
    for T in temps:
        tdir = t_dir(L, T, nsamples, data_dir)
        meta = read_meta(tdir / "meta.txt")
        n_runs = int(meta["nRuns"])
        ms, es, ss = [], [], []
        for run in range(n_runs):
            m, e, s = read_trace(tdir / f"run{run:03d}.dat")
            ms.append(m); es.append(e); ss.append(s)
        M.append(ms); E.append(es); S.append(ss); metas.append(meta)
    return dict(L=L, N=L * L, temps=np.array(temps), M=M, E=E, S=S, meta=metas)


def eq_start(S_run, N, n_eq_sw):
    #Index of the first row after reaching equilibrium
    return int(np.searchsorted(np.cumsum(S_run) / N, n_eq_sw, side="right"))


# fits (from my Wolff_Algorithm/WolffCriticalV2.py)

def fit_Tc(invL, Tstar):
    # T*(L) = Tc + a/L
    slope, intercept = np.polyfit(invL, Tstar, 1)
    return slope, intercept


# weighted power-law fit y = A L^p; returns exponent, its error, ln A
def powerlaw_fit(L, y, err, Lmin=0):
    L, y, err = map(np.asarray, (L, y, err))
    m = (L >= Lmin) & np.isfinite(y) & np.isfinite(err) & (err > 0)
    p, cov = np.polyfit(np.log(L[m]), np.log(y[m]), 1,
                        w=y[m] / err[m], cov="unscaled")
    return p[0], np.sqrt(cov[0, 0]), p[1]


def chi2_red(y, yfit, err, npar):
    m = np.isfinite(y) & np.isfinite(err)
    return np.sum(((y[m] - yfit[m]) / err[m]) ** 2) / (m.sum() - npar)


# pyerrors fits on points with known errors

# points at different (L, T) come from independent runs, so each becomes its
# own Gaussian Obs (pe.cov_Obs); the name must be unique to that point
# (e.g. "U_L32_T2.26919"), reusing a name for the same point is fine

def point_obs(value, err, name):
    return pe.cov_Obs(float(value), float(err) ** 2, name)


def poly_fit(x, y, err, names, deg):
    """Least-squares polynomial c0 + c1 x + ... through Obs points.
    returns (list of coefficient Obs, chi2/dof)."""
    yo = [point_obs(v, e, n) for v, e, n in zip(y, err, names)]
    func = lambda a, x: sum(a[k] * x ** k for k in range(deg + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pe.fits.least_squares(np.asarray(x, float), yo, func, silent=True)
    return list(res.fit_parameters), res.chisquare_by_dof


def ev(o):
    """value and error of a derived Obs."""
    o.gamma_method()
    return o.value, o.dvalue


# peak finder (from WolffCriticalV2.peak), now with pyerrors errors:
# search for the maximum only within k/L of Tc (the fine patch, which runs to
# +6/L on the high side), fit a parabola
# through 2w+1 points around it, vertex and height inherit the point errors.
# falls back to the grid maximum (T error nan) if the parabola is not a maximum
# or its vertex lies outside the fitted points

def peak(temps, y, err, L, name, Tc=TC, w=2, k=6.0):
    temps, y, err = map(np.asarray, (temps, y, err))
    near = np.where(np.abs(temps - Tc) <= k / L)[0]
    i = near[np.argmax(y[near])]
    lo, hi = max(i - w, near[0]), min(i + w + 1, near[-1] + 1)
    fallback = dict(T=temps[i], dT=np.nan, y=y[i], dy=err[i], ok=False)
    if hi - lo < 3:
        return fallback
    x = temps[lo:hi] - temps[i]                      # centred for a stable fit
    names = [f"{name}_L{L}_{t_name(T)}" for T in temps[lo:hi]]
    (c0, c1, c2), chi2dof = poly_fit(x, y[lo:hi], err[lo:hi], names, 2)
    if c2.value >= 0:
        return fallback
    xp = -c1 / (2 * c2)
    if not (x[0] <= xp.value <= x[-1]):
        return fallback
    Tp, dTp = ev(xp)
    ymax, dymax = ev(c0 - c1 ** 2 / (4 * c2))
    return dict(T=temps[i] + Tp, dT=dTp, y=ymax, dy=dymax, ok=True, chi2dof=chi2dof)


def binder_crossing(r1, r2, Tc=TC, width=1.0):
    """Crossing of U(T) for two lattice sizes.
    r1, r2: per-L results with temps, U, U_err. Each U(T) is fitted with a
    parabola in x = T - Tc over |x| <= width/L, the crossing nearest Tc is
    solved from the difference (stable quadratic formula).
    returns T, dT, U*, dU*."""
    coefs = []
    for r in (r1, r2):
        t = np.asarray(r["temps"])
        sel = np.abs(t - Tc) <= width / r["L"] + 1e-9
        names = [f"U_L{r['L']}_{t_name(T)}" for T in t[sel]]
        c, _ = poly_fit(t[sel] - Tc, r["U"][sel], r["U_err"][sel], names, 2)
        coefs.append(c)
    (a0, a1, a2), (b0, b1, b2) = coefs
    A, B, C = a2 - b2, a1 - b1, a0 - b0
    D = B * B - 4 * A * C
    if D.value < 0:
        return dict(T=np.nan, dT=np.nan, U=np.nan, dU=np.nan)
    sgn = 1.0 if B.value >= 0 else -1.0
    x = -2 * C / (B + sgn * D ** 0.5)                # root closest to 0
    Ustar = a0 + a1 * x + a2 * x * x
    xv, dx = ev(x)
    Uv, dU = ev(Ustar)
    return dict(T=Tc + xv, dT=dx, U=Uv, dU=dU)
