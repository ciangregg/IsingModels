## this was based on my plotting my wolff.cpp before I got the burn in and kt set up

from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyerrors as pe
from scipy.special import ellipk

# --------------------------------------------------------------------------
# 1. settings
# --------------------------------------------------------------------------
L = 100
N = L * L
nsweeps = 10000                      # sweep-equivalents per run (used in folder name)

DATA_ROOT = Path("efficient/data")   
OUT_DIR = Path("efficient/figs/cmp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

Tc = 2 / np.log(1 + np.sqrt(2))      # about 2.269
S_WINDOW = 2.0                       # pyerrors windowing parameter

RUN_EQ_PLOT = True                   # set False once n_eq values are chosen
USE_CACHE = True                    # pickle the analysis so plot tweaks are quick

# n_eq is in SWEEP-EQUIVALENTS (1 sweep = N spin updates).
# dx = x-offset of the markers, as a fraction of the T grid spacing
ALGOS = {
    "met": dict(folder="met_results", subdir="nt10", n_eq=1000, label="Metropolis",
                color="tab:blue", marker="o", dx=0, fixed_size=True),
    "wolff": dict(folder="wolff_results", subdir="nt10", n_eq=1000, label="Wolff",
                  color="tab:red", marker="s", dx=+0, fixed_size=False),
}


# --------------------------------------------------------------------------
# 2. loader
# --------------------------------------------------------------------------
def read_meta(base):
    meta = {}
    with open(base / "metadata.txt") as f:
        for line in f:
            parts = line.split(None, 1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1].strip()
    return meta


def find_base(cfg):
    """<root>/<folder>/L<L>/nt<...>/nsweeps<nsweeps>. cfg['subdir'] = None -> auto-detect nt*."""
    parent = DATA_ROOT / cfg["folder"] / f"L{L}"
    if cfg["subdir"] is not None:
        return parent / cfg["subdir"] / f"nsweeps{nsweeps}"
    matches = sorted(parent.glob(f"nt*/nsweeps{nsweeps}"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one nt*/nsweeps{nsweeps} under {parent}, "
                         f"found {[str(m) for m in matches]}; set 'subdir' in ALGOS")
    return matches[0]


def load(algo):
    """Return dict with temps and nested lists M[i][run], E[i][run], S[i][run]
    (1D arrays, lengths may differ between runs and temperatures).

    For Metropolis, S (cluster size) is filled with N: one sweep touches N spins.
    For Wolff, S is column 3 of the file (column 4 is the step count nsweeps_eq).
    """
    cfg = ALGOS[algo]
    base = find_base(cfg)
    meta = read_meta(base)
    nT = int(meta["nT"])
    n_runs = int(meta["nRuns"])
    Tmin, Tmax = float(meta["Tmin"]), float(meta["Tmax"])
    temps = np.linspace(Tmin, Tmax, nT)

    M = [[None] * n_runs for _ in range(nT)]
    E = [[None] * n_runs for _ in range(nT)]
    S = [[None] * n_runs for _ in range(nT)]

    for i, T in enumerate(temps):
        for run in range(n_runs):
            fname = base / f"T{T:.5f}" / f"run{run:03d}.dat"
            d = pd.read_csv(fname, sep=r"\s+", header=None, engine="c",
                            dtype=np.float64).to_numpy()
            M[i][run] = d[:, 1].copy()
            E[i][run] = d[:, 2].copy()
            if cfg["fixed_size"]:
                S[i][run] = np.full(d.shape[0], float(N))
            else:
                # columns: step, M, E, cluster size, nsweeps_eq (= number of rows)
                if d.shape[1] < 5:
                    raise ValueError(f"{fname} has {d.shape[1]} columns, expected 5 "
                                     f"(step M E size nsweeps_eq)")
                if not np.all(d[:, 4] == d.shape[0]):
                    raise ValueError(f"{fname}: column 4 ({d[0, 4]:.0f}) != number of rows "
                                     f"({d.shape[0]}); truncated file?")
                S[i][run] = d[:, 3].copy()
                if S[i][run].min() < 1 or S[i][run].max() > N:
                    raise ValueError(f"{fname}: column 3 is not a cluster size "
                                     f"(min {S[i][run].min()}, max {S[i][run].max()})")

    return dict(algo=algo, temps=temps, nT=nT, n_runs=n_runs,
                Tmin=Tmin, Tmax=Tmax, M=M, E=E, S=S)


def check_grids(data):
    """Fail loudly if the two algorithms were not run on the same grid."""
    a, b = list(data.values())
    for key in ("nT", "n_runs"):
        if a[key] != b[key]:
            raise ValueError(f"{key} differs between algorithms: {a[key]} vs {b[key]}")
    if not np.allclose(a["temps"], b["temps"]):
        raise ValueError("temperature grids differ between algorithms")


def sweep_clock(S_run):
    """Cumulative sweep-equivalents after each row of a run."""
    return np.cumsum(S_run) / N


def eq_start(S_run, n_eq_sw):
    """Index of the first row kept after discarding n_eq_sw sweep-equivalents."""
    return int(np.searchsorted(sweep_clock(S_run), n_eq_sw, side="right"))


# --------------------------------------------------------------------------
# 3. equilibration plot
# --------------------------------------------------------------------------
def plot_equilibration(data, T_targets=(1.8, 2.1, Tc, 2.8)):
    """x axis is sweep-equivalents so the two algorithms are directly comparable."""
    temps = next(iter(data.values()))["temps"]
    idxs = [int(np.argmin(np.abs(temps - T))) for T in T_targets]

    fig, ax = plt.subplots(2, len(idxs), figsize=(4 * len(idxs), 7), squeeze=False)
    for j, idx in enumerate(idxs):
        for algo, d in data.items():
            cfg = ALGOS[algo]
            for run in range(d["n_runs"]):
                x = sweep_clock(d["S"][idx][run])
                lab = cfg["label"] if run == 0 else None
                ax[0, j].plot(x, np.abs(d["M"][idx][run]) / N,
                              color=cfg["color"], lw=0.6, alpha=0.4, label=lab)
                ax[1, j].plot(x, d["E"][idx][run] / N,
                              color=cfg["color"], lw=0.6, alpha=0.4)
            for r in (0, 1):
                ax[r, j].axvline(cfg["n_eq"], color=cfg["color"], ls="--", lw=1)
        ax[0, j].set_title(f"T = {temps[idx]:.3f}")
        ax[0, j].set_ylabel("|m|")
        ax[1, j].set_ylabel("e")
        ax[1, j].set_xlabel("sweep-equivalents")
        for r in (0, 1):
            ax[r, j].set_xscale("log")
    ax[0, 0].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cmp_equilibration_L{L}_nsweeps{nsweeps}.pdf")
    plt.close()


# --------------------------------------------------------------------------
# 4. analysis (pyerrors)
# --------------------------------------------------------------------------
def analyse(d, cfg):
    """All errors from pyerrors.

    Build primitive Obs (m, m^2, e, e^2) with one replica per run (replicas may
    have different lengths), then form
      m   = <|m|>
      chi = N (<m^2> - <|m|>^2) / T
      e   = <e>
      cv  = N (<e^2> - <e>^2) / T^2
    by arithmetic, so correlations between the primitives are tracked.
    tau is in steps (rows); tau_*_sw converts to sweep-equivalents via the mean
    fraction of the lattice flipped per step.
    """
    temps, n_runs, n_eq_sw = d["temps"], d["n_runs"], cfg["n_eq"]
    nT = d["nT"]
    keys = ["m", "chi", "e", "cv", "tau_m", "tau_e", "win_m", "win_e", "frac",
            "n_samples", "sweeps_used"]
    res = {k: np.zeros(nT) for k in keys}
    res.update({k + "_err": np.zeros(nT) for k in ["m", "chi", "e", "cv", "tau_m", "tau_e"]})

    for i, T in enumerate(temps):
        ens = f"T{T:.5f}"
        names = [f"{ens}|r{r}" for r in range(n_runs)]

        m_run, e_run, s_run = [], [], []
        for r in range(n_runs):
            s0 = eq_start(d["S"][i][r], n_eq_sw)
            if len(d["M"][i][r]) - s0 < 10:
                raise ValueError(f"{d['algo']} T={T:.4f} run {r}: only "
                                 f"{len(d['M'][i][r]) - s0} rows left after n_eq={n_eq_sw} sweeps")
            m_run.append(np.abs(d["M"][i][r][s0:]) / N)
            e_run.append(d["E"][i][r][s0:] / N)
            s_run.append(d["S"][i][r][s0:])

        m1 = pe.Obs([x for x in m_run], names)
        m2 = pe.Obs([x ** 2 for x in m_run], names)
        e1 = pe.Obs([x for x in e_run], names)
        e2 = pe.Obs([x ** 2 for x in e_run], names)

        chi = N * (m2 - m1 * m1) / T
        cv = N * (e2 - e1 * e1) / T ** 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # constant series at low T can warn
            for o in (m1, e1, chi, cv):
                o.gamma_method(S=S_WINDOW)

        res["m"][i], res["m_err"][i] = m1.value, m1.dvalue
        res["e"][i], res["e_err"][i] = e1.value, e1.dvalue
        res["chi"][i], res["chi_err"][i] = chi.value, chi.dvalue
        res["cv"][i], res["cv_err"][i] = cv.value, cv.dvalue

        # replicas "T...|r<k>" are grouped by pyerrors into one ensemble "T..."
        res["tau_m"][i], res["tau_m_err"][i] = m1.e_tauint[ens], m1.e_dtauint[ens]
        res["tau_e"][i], res["tau_e_err"][i] = e1.e_tauint[ens], e1.e_dtauint[ens]
        res["win_m"][i], res["win_e"][i] = m1.e_windowsize[ens], e1.e_windowsize[ens]

        s_all = np.concatenate(s_run)
        res["frac"][i] = s_all.mean() / N                     # lattice fraction per step
        res["n_samples"][i] = np.mean([len(x) for x in m_run])  # rows per run
        res["sweeps_used"][i] = np.mean([x.sum() / N for x in s_run])

    for k in ("m", "e"):
        res[f"tau_{k}_sw"] = res[f"tau_{k}"] * res["frac"]
        res[f"tau_{k}_sw_err"] = res[f"tau_{k}_err"] * res["frac"]

    res["temps"] = temps
    res["n_runs"] = n_runs
    return res


def get_results(algo, d):
    cfg = ALGOS[algo]
    cache = OUT_DIR / f"cache_{algo}_L{L}_ns{nsweeps}_neq{cfg['n_eq']}.pkl"
    if USE_CACHE and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    res = analyse(d, cfg)
    with open(cache, "wb") as f:
        pickle.dump(res, f)
    return res


# --------------------------------------------------------------------------
# 5. exact references (infinite lattice, J = k_B = 1)
# --------------------------------------------------------------------------
def exact_m(T):
    T = np.asarray(T, dtype=float)
    out = np.full_like(T, np.nan)
    below = T < Tc
    out[below] = (1 - np.sinh(2 / T[below]) ** -4) ** (1 / 8)
    return out


def exact_e(T):
    T = np.asarray(T, dtype=float)
    b = 1 / T
    k = 2 * np.sinh(2 * b) / np.cosh(2 * b) ** 2
    kp = 2 * np.tanh(2 * b) ** 2 - 1
    with np.errstate(all="ignore"):
        e = -(1 / np.tanh(2 * b)) * (1 + (2 / np.pi) * kp * ellipk(k ** 2))
    e[~np.isfinite(e)] = np.nan
    return e


# --------------------------------------------------------------------------
# 6. plots
# --------------------------------------------------------------------------
# (row, col, key, title); same layout as V2 so figures pair up
PANELS = [(0, 0, "e", "Energy per spin"),
          (0, 1, "m", "Magnetisation per spin $|m|$"),
          (1, 0, "cv", r"$c_v$"),
          (1, 1, "chi", r"Susceptibility $\chi$")]


def tag(results):
    n_runs = next(iter(results.values()))["n_runs"]
    return f"L{L}_nsweeps{nsweeps}_nruns{n_runs}"


def plot_observables(results):
    temps = next(iter(results.values()))["temps"]
    dT = temps[1] - temps[0]
    Td = np.linspace(temps[0], temps[-1], 600)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for r, c, key, title in PANELS:
        a = ax[r, c]
        for algo, res in results.items():
            cfg = ALGOS[algo]
            a.errorbar(temps + cfg["dx"] * dT, res[key], yerr=res[key + "_err"],
                       fmt=cfg["marker"], ms=4, color=cfg["color"], capsize=2,
                       label=cfg["label"])
        if key == "e":
            a.plot(Td, exact_e(Td), "k-", lw=0.8, label="exact (L = inf)")
        if key == "m":
            a.plot(Td, exact_m(Td), "k-", lw=0.8, label="exact (L = inf)")
        a.axvline(Tc, color="green", ls=":", label=fr"$T_c={Tc:.2f}$")
        a.set_title(title)
        a.set_xlabel("T")
        a.set_ylabel(title)
        a.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cmp_observables_{tag(results)}.pdf")
    plt.close()


def plot_pulls(results):
    temps = results["met"]["temps"]
    w, m = results["wolff"], results["met"]

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for r, c, key, title in PANELS:
        a = ax[r, c]
        sigma = np.sqrt(w[key + "_err"] ** 2 + m[key + "_err"] ** 2)
        with np.errstate(all="ignore"):
            pull = (w[key] - m[key]) / sigma
        a.plot(temps, pull, "ko", ms=4)
        a.axhline(0, color="k", lw=0.8)
        for y in (1, -1):
            a.axhline(y, color="grey", ls="--", lw=0.8)
        for y in (2, -2):
            a.axhline(y, color="grey", ls=":", lw=0.8)
        a.axvline(Tc, color="green", ls=":")
        frac_in = np.nanmean(np.abs(pull) < 1)
        a.set_title(f"{title}  ({100 * frac_in:.0f}% within 1$\\sigma$)")
        a.set_xlabel("T")
        a.set_ylabel(r"(Wolff $-$ Metropolis) / $\sigma$")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cmp_pulls_{tag(results)}.pdf")
    plt.close()


def plot_tau(results):
    temps = next(iter(results.values()))["temps"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, k, name in [(ax[0], "m", r"$|m|$"), (ax[1], "e", r"$e$")]:
        for algo, res in results.items():
            cfg = ALGOS[algo]
            a.errorbar(temps, res[f"tau_{k}_sw"], yerr=res[f"tau_{k}_sw_err"],
                       fmt=cfg["marker"], ms=4, color=cfg["color"], capsize=2,
                       label=f"{cfg['label']} (sweep-equivalents)")
            if not ALGOS[algo]["fixed_size"]:
                a.plot(temps, res[f"tau_{k}"], "--", color=cfg["color"], lw=1,
                       label=f"{cfg['label']} (raw cluster steps)")
        a.axvline(Tc, color="green", ls=":", label=fr"$T_c={Tc:.2f}$")
        a.set_yscale("log")
        a.set_xlabel("T")
        a.set_ylabel(rf"$\tau_{{int}}$ ({name})")
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cmp_tau_{tag(results)}.pdf")
    plt.close()


# --------------------------------------------------------------------------
# 7. main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    data = {algo: load(algo) for algo in ALGOS}
    check_grids(data)

    if RUN_EQ_PLOT:
        plot_equilibration(data)
        print("wrote equilibration plot; check n_eq (sweep-equivalents) in ALGOS, "
              "then set RUN_EQ_PLOT = False")

    results = {algo: get_results(algo, data[algo]) for algo in ALGOS}

    for algo, res in results.items():
        j = np.argmin(np.abs(res["temps"] - Tc))
        print(f"{algo}: near Tc  frac/step = {res['frac'][j]:.3f}, "
              f"rows/run = {res['n_samples'][j]:.0f}, "
              f"sweeps used/run = {res['sweeps_used'][j]:.0f}")

    plot_observables(results)
    plot_pulls(results)
    plot_tau(results)
    print("done")