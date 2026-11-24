# Finite-size scaling analysis of the Wolff data (2D Ising)

# for every (L, T) ran we use pyerrors replicas 

# primitives  |m|, m^2, m^4, e, e^2, m^2 e, m^4 e      (m, e per spin)

# derived
"""
chi    = N (<m^2> - <|m|>^2) / T
cv     = N (<e^2> - <e>^2) / T^2
 U      = 1 - <m^4> / (3 <m^2>^2)
dU/db  = (1 - U) N (<m^4 e>/<m^4> - 2 <m^2 e>/<m^2> + <e>)
"""

"""
per L:  values at Tc, and chi / cv / dU/db peaks (parabola, pyerrors errors)
across L:  Binder crossings, exponents from power-law fits, Cv vs ln L, collapse
"""

#examples
"""
    python fss_analyse.py                    # every L with data, nsamples 10000
    python fss_analyse.py --ls 16 24 32      # pilot only
    python fss_analyse.py --lmin 24          # drop the smallest L from the headline fits
    python fss_analyse.py --no-cache --show
"""

import argparse
import pickle
import warnings

import matplotlib
import numpy as np
import pyerrors as pe

from fss_tools import (BASE_DIR, TC, TC_5DP, available_Ls, binder_crossing,
                       chi2_red, done_temps, eq_start, fit_Tc, load_L, peak,
                       powerlaw_fit)


# settings
N_SAMPLES = 10_000
S_WINDOW = 2.0 # pyerrors windowing parameter 
N_EQ_SW = 100  # extra burnin sweeps to discard (already have 200)
CACHE_VERSION = "v1" # bump to invalidate every cache file
FIG_DIR = BASE_DIR / "figs"
CACHE_DIR = BASE_DIR / "cache"

# exact 2D Ising values
EXACT = dict(Tc=TC, nu=1.0, gamma_nu=1.75, beta_nu=0.125, inv_nu=1.0, Ustar=0.61069)

OBS = ["m", "chi", "e", "cv", "U", "dUdb"]


# per-temperature analysis (pyerrors)
def analyse_T(m_runs, e_runs, T, N, ens):
    """m_runs, e_runs: per-run arrays of m and e per spin. returns value/err dict."""
    names = [f"{ens}|r{r}" for r in range(len(m_runs))]
    am = [np.abs(x) for x in m_runs]
    m2r = [x ** 2 for x in m_runs]
    m4r = [x ** 4 for x in m_runs]

    m1 = pe.Obs(am, names)
    m2 = pe.Obs(m2r, names)
    m4 = pe.Obs(m4r, names)
    e1 = pe.Obs([x for x in e_runs], names)
    e2 = pe.Obs([x ** 2 for x in e_runs], names)
    m2e = pe.Obs([a * b for a, b in zip(m2r, e_runs)], names)
    m4e = pe.Obs([a * b for a, b in zip(m4r, e_runs)], names)

    chi = N * (m2 - m1 * m1) / T
    cv = N * (e2 - e1 * e1) / T ** 2
    U = 1 - m4 / (3 * m2 * m2)
    dUdb = (1 - U) * N * (m4e / m4 - 2 * m2e / m2 + e1)

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # constant series at low T can warn
        for key, o in zip(OBS, (m1, chi, e1, cv, U, dUdb)):
            o.gamma_method(S=S_WINDOW)
            out[key], out[key + "_err"] = o.value, o.dvalue
    # integrated autocorrelation time of |m|, in samples (rows)
    out["tau_m"] = m1.e_tauint[ens]
    return out


def analyse_L(L, nsamples, use_cache=True):
    #Per-T results for one L, cached until the set of finished temperatures changes
    temps_now = done_temps(L, nsamples)
    cache = CACHE_DIR / f"{CACHE_VERSION}_L{L}_ns{nsamples}_neq{N_EQ_SW}_S{S_WINDOW}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            res = pickle.load(f)
        if np.array_equal(res["temps"], temps_now):
            return res

    print(f"  analysing L={L} ({len(temps_now)} temperatures)...", flush=True)
    d = load_L(L, nsamples)
    N = d["N"]
    rows = []
    for i, T in enumerate(d["temps"]):
        m_runs, e_runs = [], []
        for r in range(len(d["M"][i])):
            s0 = eq_start(d["S"][i][r], N, N_EQ_SW)
            m_runs.append(d["M"][i][r][s0:] / N)
            e_runs.append(d["E"][i][r][s0:] / N)
        rows.append(analyse_T(m_runs, e_runs, T, N, f"L{L}_T{T:.5f}"))

    res = dict(L=L, N=N, temps=d["temps"], n_runs=len(d["M"][0]),
               kT=np.array([int(m["kT"].split()[0]) for m in d["meta"]]))
    for key in rows[0]:
        res[key] = np.array([r[key] for r in rows])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(res, f)
    return res


# finite-size quantities per L
def at_Tc(r, key):
    i = np.where(np.isclose(r["temps"], TC_5DP))[0]
    if len(i) == 0:
        return np.nan, np.nan
    return r[key][i[0]], r[key + "_err"][i[0]]


def fss_table(results):
    Ls = np.array(sorted(results))
    t = {k: [] for k in ["chi_Tc", "m_Tc", "dUdb_Tc", "U_Tc", "tau_Tc"]}
    for name in ("chi", "cv", "dUdb"):
        for k in ("Tpk", "Tpk_err", "max", "max_err", "ok"):
            t[f"{name}_{k}"] = []
    for L in Ls:
        r = results[L]
        for key in ("chi", "m", "dUdb", "U"):
            v, e = at_Tc(r, key)
            t[f"{key}_Tc"].append((v, e))
        i = np.where(np.isclose(r["temps"], TC_5DP))[0]
        t["tau_Tc"].append(r["tau_m"][i[0]] if len(i) else np.nan)
        for name in ("chi", "cv", "dUdb"):
            p = peak(r["temps"], r[name], r[name + "_err"], L, name)
            t[f"{name}_Tpk"].append(p["T"]); t[f"{name}_Tpk_err"].append(p["dT"])
            t[f"{name}_max"].append(p["y"]); t[f"{name}_max_err"].append(p["dy"])
            t[f"{name}_ok"].append(p["ok"])
    out = {"L": Ls}
    for k, v in t.items():
        if k.endswith("_Tc") and k != "tau_Tc":
            arr = np.array(v)
            out[k], out[k + "_err"] = arr[:, 0], arr[:, 1]
        else:
            out[k] = np.array(v)
    return out


def crossings(results):
    Ls = sorted(results)
    rows = []
    for L1, L2 in zip(Ls[:-1], Ls[1:]):
        c = binder_crossing(results[L1], results[L2])
        rows.append(dict(L1=L1, L2=L2, **c))
    return rows


def wmean(v, e):
    v, e = np.asarray(v), np.asarray(e)
    m = np.isfinite(v) & np.isfinite(e) & (e > 0)
    w = 1 / e[m] ** 2
    return np.sum(w * v[m]) / w.sum(), 1 / np.sqrt(w.sum())


# fits
def fits(ft, lmin):
    L = ft["L"]
    f = {}
    # gamma/nu, beta/nu, 1/nu
    f["gamma_nu_max"] = powerlaw_fit(L, ft["chi_max"], ft["chi_max_err"], lmin)
    f["gamma_nu_Tc"] = powerlaw_fit(L, ft["chi_Tc"], ft["chi_Tc_err"], lmin)
    p, dp, lnA = powerlaw_fit(L, ft["m_Tc"], ft["m_Tc_err"], lmin)
    f["beta_nu_Tc"] = (-p, dp, lnA)
    f["inv_nu_max"] = powerlaw_fit(L, ft["dUdb_max"], ft["dUdb_max_err"], lmin)
    f["inv_nu_Tc"] = powerlaw_fit(L, ft["dUdb_Tc"], ft["dUdb_Tc_err"], lmin)

    # Cv max: a + b ln L (alpha = 0) against a power law
    sel = L >= lmin
    lnL, cv, dcv = np.log(L[sel]), ft["cv_max"][sel], ft["cv_max_err"][sel]
    (b, a), cov = np.polyfit(lnL, cv, 1, w=1 / dcv, cov="unscaled")
    f["cv_log"] = (a, b, np.sqrt(cov[1, 1]), np.sqrt(cov[0, 0]),
                   chi2_red(cv, a + b * lnL, dcv, 2))
    ap, dap, lnAp = powerlaw_fit(L, ft["cv_max"], ft["cv_max_err"], lmin)
    f["cv_pow"] = (ap, dap, lnAp, chi2_red(cv, np.exp(lnAp) * L[sel] ** ap, dcv, 2))

    # pseudocritical temperatures T*(L) = Tc + a/L (assumes nu = 1, the 2D value)
    for name in ("chi", "cv"):
        s = sel & np.isfinite(ft[f"{name}_Tpk"])
        f[f"Tstar_{name}"] = fit_Tc(1 / L[s], ft[f"{name}_Tpk"][s])
    return f


def lmin_scan(ft, Lmins):
    rows = []
    for lmin in Lmins:
        if np.sum(ft["L"] >= lmin) < 3:
            break
        f = fits(ft, lmin)
        rows.append((lmin, f))
    return rows


# plots

def colours(Ls):
    cmap = matplotlib.colormaps["viridis"]
    return {L: cmap(i / max(1, len(Ls) - 1)) for i, L in enumerate(Ls)}


def plot_observables(results, plt, tag):
    Ls = sorted(results)
    col = colours(Ls)
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    panels = [("m", r"$\langle|m|\rangle$"), ("chi", r"$\chi$"), ("U", r"$U$"),
              ("e", r"$\langle e\rangle$"), ("cv", r"$C_V$"), ("dUdb", r"$dU/d\beta$")]
    for a, (key, lab) in zip(ax.flat, panels):
        for L in Ls:
            r = results[L]
            a.errorbar(r["temps"], r[key], yerr=r[key + "_err"], fmt="o-", ms=3,
                       lw=0.8, capsize=1.5, color=col[L], label=f"L={L}")
        a.axvline(TC, color="k", ls=":", lw=1)
        a.set_xlabel("T"); a.set_ylabel(lab); a.grid(alpha=0.25)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Wolff observables (dotted: exact $T_c$)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fss_observables_{tag}.pdf")
    return fig


def plot_binder(results, cross, plt, tag):
    Ls = sorted(results)
    col = colours(Ls)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for L in Ls:
        r = results[L]
        s = np.abs(r["temps"] - TC) <= 0.15
        ax[0].errorbar(r["temps"][s], r["U"][s], yerr=r["U_err"][s], fmt="o-", ms=3,
                       lw=0.8, capsize=1.5, color=col[L], label=f"L={L}")
    ax[0].axvline(TC, color="k", ls=":", lw=1, label=r"exact $T_c$")
    ax[0].axhline(EXACT["Ustar"], color="gray", ls="--", lw=1, label=r"$U^*=0.6107$")
    ax[0].set_xlabel("T"); ax[0].set_ylabel("U"); ax[0].legend(fontsize=8)
    ax[0].set_title("Binder cumulant near $T_c$"); ax[0].grid(alpha=0.25)

    x = [1 / np.sqrt(c["L1"] * c["L2"]) for c in cross]
    ax[1].errorbar(x, [c["T"] for c in cross], yerr=[c["dT"] for c in cross],
                   fmt="o", capsize=3, label="crossing T")
    ax[1].axhline(TC, color="k", ls=":", label=r"exact $T_c$")
    for c, xi in zip(cross, x):
        ax[1].annotate(f"{c['L1']}/{c['L2']}", (xi, c["T"]), fontsize=7,
                       textcoords="offset points", xytext=(4, 4))
    ax[1].set_xlabel(r"$1/\sqrt{L_1 L_2}$"); ax[1].set_ylabel(r"$T_\times$")
    ax[1].set_title("Crossings of consecutive sizes"); ax[1].grid(alpha=0.25)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fss_binder_{tag}.pdf")
    return fig


def plot_exponents(results, ft, f, lmin, plt, tag):
    L = ft["L"]
    Lf = np.linspace(L.min(), L.max(), 200)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))

    # T*(L)
    a = ax[0, 0]
    x = np.linspace(0, 1.05 / L.min(), 100)
    for name, lab, mk in (("chi", r"$\chi$ peak", "o"), ("cv", r"$C_V$ peak", "s")):
        a.errorbar(1 / L, ft[f"{name}_Tpk"], yerr=ft[f"{name}_Tpk_err"], fmt=mk,
                   capsize=2, label=lab)
        slope, Tinf = f[f"Tstar_{name}"]
        a.plot(x, Tinf + slope * x, "--", lw=1, label=rf"{lab}: $T_\infty={Tinf:.4f}$")
    a.axhline(TC, color="k", ls=":", label=rf"exact $T_c={TC:.5f}$")
    a.set_xlabel("1/L"); a.set_ylabel(r"$T^*(L)$"); a.legend(fontsize=7)
    a.set_title("Pseudocritical temperature shift"); a.grid(alpha=0.25)

    # chi
    a = ax[0, 1]
    for key, lab, mk, fk in (("chi_max", r"$\chi_{\max}$", "o", "gamma_nu_max"),
                             ("chi_Tc", r"$\chi(T_c)$", "s", "gamma_nu_Tc")):
        p, dp, lnA = f[fk]
        a.errorbar(L, ft[key], yerr=ft[key + "_err"], fmt=mk, capsize=2,
                   label=rf"{lab}: $\gamma/\nu={p:.3f}\pm{dp:.3f}$")
        a.plot(Lf, np.exp(lnA) * Lf ** p, "--", lw=1)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("L"); a.set_ylabel(r"$\chi$")
    a.set_title(r"Susceptibility (exact $\gamma/\nu = 1.75$)"); a.legend(fontsize=8)
    a.grid(alpha=0.25, which="both")

    # |m|(Tc)
    a = ax[0, 2]
    p, dp, lnA = f["beta_nu_Tc"]
    a.errorbar(L, ft["m_Tc"], yerr=ft["m_Tc_err"], fmt="o", capsize=2,
               label=rf"$\beta/\nu={p:.4f}\pm{dp:.4f}$")
    a.plot(Lf, np.exp(lnA) * Lf ** (-p), "--", lw=1)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("L")
    a.set_ylabel(r"$\langle|m|\rangle(T_c)$")
    a.set_title(r"Magnetisation (exact $\beta/\nu = 0.125$)"); a.legend(fontsize=8)
    a.grid(alpha=0.25, which="both")

    # dU/dbeta
    a = ax[1, 0]
    for key, lab, mk, fk in (("dUdb_max", r"max $dU/d\beta$", "o", "inv_nu_max"),
                             ("dUdb_Tc", r"$dU/d\beta(T_c)$", "s", "inv_nu_Tc")):
        p, dp, lnA = f[fk]
        a.errorbar(L, ft[key], yerr=ft[key + "_err"], fmt=mk, capsize=2,
                   label=rf"{lab}: $1/\nu={p:.3f}\pm{dp:.3f}$")
        a.plot(Lf, np.exp(lnA) * Lf ** p, "--", lw=1)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("L"); a.set_ylabel(r"$dU/d\beta$")
    a.set_title(r"Binder slope (exact $1/\nu = 1$)"); a.legend(fontsize=8)
    a.grid(alpha=0.25, which="both")

    # Cv max vs ln L
    a = ax[1, 1]
    ca, cb, dca, dcb, c2log = f["cv_log"]
    ap, dap, lnAp, c2pow = f["cv_pow"]
    a.errorbar(np.log(L), ft["cv_max"], yerr=ft["cv_max_err"], fmt="o", capsize=2,
               label="simulation")
    a.plot(np.log(Lf), ca + cb * np.log(Lf), "--", lw=1,
           label=rf"$a+b\ln L$: $b={cb:.3f}\pm{dcb:.3f}$, $\chi^2_r={c2log:.2f}$")
    a.plot(np.log(Lf), np.exp(lnAp) * Lf ** ap, ":", lw=1,
           label=rf"$AL^{{\alpha/\nu}}$: ${ap:.3f}$, $\chi^2_r={c2pow:.2f}$")
    a.set_xlabel("ln L"); a.set_ylabel(r"$C_{V,\max}$")
    a.set_title(r"Specific heat (exact: $\alpha=0$, log divergence)"); a.legend(fontsize=8)
    a.grid(alpha=0.25)

    # collapse with exact exponents
    a = ax[1, 2]
    col = colours(list(L))
    for Li in L:
        r = results[Li]
        a.plot((r["temps"] - TC) * Li ** EXACT["inv_nu"],
               r["chi"] * Li ** (-EXACT["gamma_nu"]), "o", ms=3, color=col[Li],
               label=f"L={Li}")
    a.set_xlim(-6, 6)
    a.set_xlabel(r"$(T-T_c)\,L^{1/\nu}$"); a.set_ylabel(r"$\chi\,L^{-\gamma/\nu}$")
    a.set_title("Data collapse (exact exponents)"); a.legend(fontsize=7)
    a.grid(alpha=0.25)

    fig.suptitle(f"2D Ising finite-size scaling, Wolff  (fits use L >= {lmin})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fss_exponents_{tag}.pdf")
    return fig


# print results
def report(ft, f, cross, scan, lmin):
    out = []
    w = out.append
    w("2D Ising finite-size scaling (Wolff)")
    w(f"L = {[int(x) for x in ft['L']]},  headline fits use L >= {lmin}\n")

    w(f"{'L':>5} {'chi(Tc)':>16} {'|m|(Tc)':>18} {'U(Tc)':>16} {'tau_|m|(Tc)':>12}")
    for i, L in enumerate(ft["L"]):
        w(f"{L:>5} {ft['chi_Tc'][i]:>9.3f}±{ft['chi_Tc_err'][i]:<6.3f}"
          f" {ft['m_Tc'][i]:>10.5f}±{ft['m_Tc_err'][i]:<7.5f}"
          f" {ft['U_Tc'][i]:>9.5f}±{ft['U_Tc_err'][i]:<6.5f} {ft['tau_Tc'][i]:>12.2f}")
    w("  (tau in recorded samples)")

    w(f"\n{'L':>5} {'T*_chi':>18} {'chi_max':>16} {'T*_cv':>18} {'cv_max':>14}")
    for i, L in enumerate(ft["L"]):
        flag = "" if ft["chi_ok"][i] and ft["cv_ok"][i] and ft["dUdb_ok"][i] else "  (grid fallback)"
        w(f"{L:>5} {ft['chi_Tpk'][i]:>10.5f}±{ft['chi_Tpk_err'][i]:<7.5f}"
          f" {ft['chi_max'][i]:>9.3f}±{ft['chi_max_err'][i]:<6.3f}"
          f" {ft['cv_Tpk'][i]:>10.5f}±{ft['cv_Tpk_err'][i]:<7.5f}"
          f" {ft['cv_max'][i]:>7.4f}±{ft['cv_max_err'][i]:<6.4f}{flag}")

    w("\nBinder crossings (consecutive sizes)")
    for c in cross:
        w(f"  L={c['L1']:>3}/{c['L2']:<3}  T = {c['T']:.5f} ± {c['dT']:.5f}"
          f"   U* = {c['U']:.4f} ± {c['dU']:.4f}")
    good = [c for c in cross if c["L1"] >= lmin]
    if good:
        Tm, dTm = wmean([c["T"] for c in good], [c["dT"] for c in good])
        Um, dUm = wmean([c["U"] for c in good], [c["dU"] for c in good])
        w(f"  weighted mean (L1 >= {lmin}): Tc = {Tm:.5f} ± {dTm:.5f},"
          f"  U* = {Um:.4f} ± {dUm:.4f}")

    w("\nresults                     measured              exact")
    w(f"  Tc   chi peaks -> 1/L=0    {f['Tstar_chi'][1]:.5f}               {TC:.5f}")
    w(f"  Tc   cv peaks  -> 1/L=0    {f['Tstar_cv'][1]:.5f}               {TC:.5f}")
    for k, lab, ex in (("gamma_nu_max", "gamma/nu  (chi_max)", 1.75),
                       ("gamma_nu_Tc", "gamma/nu  (chi(Tc))", 1.75),
                       ("beta_nu_Tc", "beta/nu   (|m|(Tc))", 0.125),
                       ("inv_nu_max", "1/nu      (max dU/db)", 1.0),
                       ("inv_nu_Tc", "1/nu      (dU/db(Tc))", 1.0)):
        p, dp, _ = f[k]
        w(f"  {lab:<24}  {p:.4f} ± {dp:.4f}       {ex}")
    ca, cb, dca, dcb, c2log = f["cv_log"]
    ap, dap, _, c2pow = f["cv_pow"]
    b_exact = 8 / np.pi * (np.log(1 + np.sqrt(2)) / 2) ** 2     # Ferdinand-Fisher
    w(f"  Cv_max = a + b ln L       b = {cb:.4f} ± {dcb:.4f}  chi2_r = {c2log:.2f}"
      f"   (exact b = (8/pi) Kc^2 = {b_exact:.4f})")
    w(f"  Cv_max = A L^p            p = {ap:.4f} ± {dap:.4f}  chi2_r = {c2pow:.2f}")
    g, dg, _ = f["gamma_nu_Tc"]
    b, db, _ = f["beta_nu_Tc"]
    w(f"  hyperscaling 2 beta/nu + gamma/nu = {2 * b + g:.4f} ± {np.hypot(2 * db, dg):.4f}"
      f"   (d = 2)")

    w("\nL_min scan (stability of the fits)")
    w(f"  {'Lmin':>5} {'g/nu max':>16} {'g/nu Tc':>16} {'b/nu Tc':>16} {'1/nu max':>16} {'1/nu Tc':>16}")
    for lm, fs in scan:
        cells = [f"{fs[k][0]:.4f}±{fs[k][1]:.4f}" for k in
                 ("gamma_nu_max", "gamma_nu_Tc", "beta_nu_Tc", "inv_nu_max", "inv_nu_Tc")]
        w(f"  {lm:>5} " + " ".join(f"{c:>16}" for c in cells))
    return "\n".join(out)



# main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsamples", type=int, default=N_SAMPLES)
    ap.add_argument("--ls", type=int, nargs="+", help="lattice sizes (default: all with data)")
    ap.add_argument("--lmin", type=int, default=0, help="smallest L in the headline fits")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--show", action="store_true", help="open the figures")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ls = args.ls or available_Ls(args.nsamples)
    
    if len(Ls) < 3:
        raise SystemExit(f"need at least 3 lattice sizes with data, found {Ls}")
    print(f"L = {Ls}, nsamples = {args.nsamples}")

    results = {L: analyse_L(L, args.nsamples, not args.no_cache) for L in Ls}
    ft = fss_table(results)
    lmin = args.lmin or min(Ls)
    f = fits(ft, lmin)
    cross = crossings(results)
    scan = lmin_scan(ft, sorted(Ls))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"L{min(Ls)}-{max(Ls)}_ns{args.nsamples}"
    plot_observables(results, plt, tag)
    plot_binder(results, cross, plt, tag)
    plot_exponents(results, ft, f, lmin, plt, tag)

    text = report(ft, f, cross, scan, lmin)
    print("\n" + text)
    
    (FIG_DIR / f"fss_results_{tag}.txt").write_text(text + "\n")
    print(f"\nfigures and results saved in {FIG_DIR}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
