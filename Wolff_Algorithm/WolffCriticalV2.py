import math
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# folders relative to this file, not the terminal's working directory
BASE_DIR = Path(__file__).resolve().parent
OBS_DIR = BASE_DIR / "plots" / "observables" / "wolff"
CRIT_DIR = BASE_DIR / "plots" / "critical"
CACHE_DIR = BASE_DIR / "cache"
for _d in (OBS_DIR, CRIT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# numba and numpy's RNG seed
@njit
def seed_numba(s):
    np.random.seed(s)

np.random.seed(13)
seed_numba(13)


## ising
@njit
def energy(s, J, h):
    # 2D Ising energy with periodic boundary conditions.
    E = 0.0
    N = s.shape[0]
    for i in range(N):
        for j in range(N):
            E -= J * s[i, j] * (
                s[(i + 1) % N, j] +
                s[i, (j + 1) % N]
            )
            E -= h * s[i, j]
    return E


## auto correlation and tau

def autocorr(x):
    # Normalised autocorrelation phi(t) via FFT
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)                 # zero-pad to avoid wraparound
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0]

def tau_int(x, c=6.0):
    # Integrated autocorrelation time with Sokal automatic windowing
    # found=False means the window condition was never met
    phi = autocorr(x)
    taus = np.cumsum(phi) - 0.5
    W = np.arange(len(taus))
    ok = W >= c * taus
    found = bool(ok.any())
    m = np.argmax(ok) if found else len(taus) - 1
    return taus[m], found


## wolff algorithm and simulation

@njit
def wolff_step(s, L, J, T, E, M, mark, stack):
    p_add = 1.0 - math.exp(-2.0 * J / T)

    # (step 1) random seed spin
    x0 = np.random.randint(L)
    y0 = np.random.randint(L)
    spin = s[x0, y0]
    stack[0] = x0 * L + y0
    mark[x0, y0] = True
    n = 1
    head = 0

    # (step 2) build the cluster (spins not flipped yet, `mark` defines membership)
    while head < n:
        site = stack[head]
        head += 1
        x = site // L
        y = site % L
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = (x + di) % L
            ny = (y + dj) % L
            if s[nx, ny] == spin and not mark[nx, ny]:
                if np.random.rand() < p_add:
                    mark[nx, ny] = True
                    stack[n] = nx * L + ny
                    n += 1

    # energy change from boundary sites only
    dE = 0.0
    for i in range(n):
        x = stack[i] // L
        y = stack[i] % L
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = (x + di) % L
            ny = (y + dj) % L
            if not mark[nx, ny]:
                if s[nx, ny] == spin:
                    dE += 2.0 * J
                else:
                    dE -= 2.0 * J

    # flip cluster and clear marks (O(n))
    for i in range(n):
        x = stack[i] // L
        y = stack[i] % L
        s[x, y] = -spin
        mark[x, y] = False

    # flipping n spins of value `spin` changes M by -2*spin*n
    return E + dE, M - 2 * spin * n, n


#  whole block of steps inside numba, no Python loop or spins.sum()
@njit
def run_block(s, L, J, T, E, M, mark, stack, nsteps):
    mags = np.empty(nsteps, np.int64)
    ens = np.empty(nsteps, np.float64)
    ns = np.empty(nsteps, np.int64)
    for i in range(nsteps):
        E, M, n = wolff_step(s, L, J, T, E, M, mark, stack)
        mags[i] = M
        ens[i] = E
        ns[i] = n
    return E, M, mags, ens, ns


def adaptive_wolff_run(T, L, J=1.0,
                       block_size=2000, k=1000, discard_mult=10,
                       max_steps=200_000, short_warmup=200,
                       Tc=None, verbose=False):
    # Run Wolff at T, extending the run until the post-discard series is
    # at least k*tau_int long, or max_steps is hit.  tau_sweeps = tau*<n>/N
    h = 0.0
    Nspins = L * L

    if Tc is None:
        Tc = 2 * abs(J) / np.log(1 + np.sqrt(2))

    if T < Tc:
        spins = np.ones((L, L), dtype=np.int64)          # cold start
    else:
        spins = np.random.choice(np.array([-1, 1]), size=(L, L)).astype(np.int64)

    E_current = energy(spins, J, h)
    M_current = int(spins.sum())                          #  computed once

    mark = np.zeros((L, L), dtype=np.bool_)
    stack = np.empty(Nspins, dtype=np.int64)

    # short fixed warmup, not recorded
    E_current, M_current, _, _, _ = run_block(
        spins, L, J, T, E_current, M_current, mark, stack, short_warmup)

    mags = np.empty(0, np.int64)
    energies = np.empty(0, np.float64)
    ns = np.empty(0, np.int64)
    total_steps = short_warmup
    current_block = int(block_size)
    ok = False

    while True:
        E_current, M_current, mb, eb, nb = run_block(
            spins, L, J, T, E_current, M_current, mark, stack, current_block)
        mags = np.concatenate([mags, mb])
        energies = np.concatenate([energies, eb])
        ns = np.concatenate([ns, nb])
        total_steps += current_block

        tau_hat, found = tau_int(np.abs(mags))
        discard_n = int(min(discard_mult * tau_hat, len(mags) // 2))
        usable_len = len(mags) - discard_n

        if verbose:
            print(f" L={L}  T={T:.3f} steps={total_steps} tau~{tau_hat:.1f} "
                  f"found={found} usable_len={usable_len} need={k * tau_hat:.0f}")

        # success is checked first, so a run that just made it is not discarded
        if found and usable_len >= k * tau_hat:
            ok = True
            break
        if total_steps >= max_steps:
            ok = False
            break

        current_block *= 2

    mags_arr = mags[discard_n:]
    energies_arr = energies[discard_n:] / Nspins
    ns_arr = ns[discard_n:]

    tau_final, found_final = tau_int(np.abs(mags_arr))
    n_eff = len(mags_arr) / (2 * tau_final) if tau_final > 0 else np.nan
    nbar = ns_arr.mean()
    tau_sweeps = tau_final * nbar / Nspins

    mf = mags_arr.astype(float)
    m = np.mean(np.abs(mf)) / Nspins
    chi = (np.mean(mf**2) - np.mean(np.abs(mf))**2) / (T * Nspins)
    E = np.mean(energies_arr)
    cv = np.var(energies_arr) * Nspins / (T ** 2)

    return dict(T=T,
                tau=tau_final, tau_found=found_final, n_eff=n_eff,
                tau_sweeps=tau_sweeps, nbar=nbar,
                m=m, chi=chi, E=E, cv=cv,
                total_steps=total_steps, discard_n=discard_n,
                ok=ok and found_final)


# bigger defaults; fixed first block so we rarely need doubling
def tau_at_Tc(L, n_runs=20, k=20000, max_steps=5_000_000, block_size=50_000):
    Tc = 2 / np.log(1 + np.sqrt(2))
    vals = []
    for _ in range(n_runs):
        r = adaptive_wolff_run(Tc, L, 1.0, k=k, Tc=Tc,
                               max_steps=max_steps, block_size=block_size)
        if r["ok"]:
            vals.append(r["tau_sweeps"])
    vals = np.array(vals)
    if len(vals) < 2:
        print(f" L={L}: only {len(vals)} converged runs, raise max_steps")
        return np.nan, np.nan, len(vals)
    err = vals.std(ddof=1) / np.sqrt(len(vals))
    print(f" L={L}: tau_sweeps = {vals.mean():.3f} ± {err:.3f} "
          f"({len(vals)}/{n_runs} converged)")
    return vals.mean(), err, len(vals)


def make_temps(L, Tc, n_coarse=20, n_fine=21, width=4.0, lo=1.8, hi=2.8):
    coarse = np.linspace(lo, hi, n_coarse)
    fine = Tc + np.linspace(-width, width, n_fine) / L    # includes Tc exactly
    return np.unique(np.round(np.concatenate([coarse, fine]), 6))


def wolff_sim(L=10, temps=None, J=1.0, h=0.0, n_runs=5, k=1000,
              max_steps=500_000, short_warmup=200):
    Tc = 2 * abs(J) / np.log(1 + np.sqrt(2))

    if temps is None:
        temps = make_temps(L, Tc)
    temps = np.asarray(temps)

    E_mean, E_std = [], []
    cv_mean, cv_std = [], []
    m_mean, m_std = [], []
    chi_mean, chi_std = [], []
    tau_mean, tau_std = [], []
    tausw_mean, tausw_std = [], []
    ok_frac = []

    for T in temps:
        runs = [adaptive_wolff_run(T, L, J, k=k, Tc=Tc, max_steps=max_steps,
                                   short_warmup=short_warmup)
                for _ in range(n_runs)]

        for lst_m, lst_s, key in [(E_mean, E_std, "E"), (cv_mean, cv_std, "cv"),
                                  (m_mean, m_std, "m"), (chi_mean, chi_std, "chi"),
                                  (tau_mean, tau_std, "tau"),
                                  (tausw_mean, tausw_std, "tau_sweeps")]:
            vals = np.array([r[key] for r in runs])
            lst_m.append(vals.mean())
            # standard ERROR of the mean (names kept as *_std so the rest
            # of the code is unchanged, but these now hold the SE)
            lst_s.append(vals.std(ddof=1) / np.sqrt(n_runs))

        ok_frac.append(np.mean([r["ok"] for r in runs]))
        print(f" L={L}, T={T:.4f} done, ran {n_runs} runs, ok={ok_frac[-1]:.0%}")

    ok_frac = np.array(ok_frac)

    # ---- observables ----
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    panels = [(ax[0, 0], m_mean, m_std, 'red', "Magnetism"),
              (ax[1, 0], cv_mean, cv_std, 'blue', r"$c_v$ vs T"),
              (ax[0, 1], chi_mean, chi_std, 'purple', "Susceptibility"),
              (ax[1, 1], E_mean, E_std, 'orange', "Energy per spin")]
    for a, y, err, col, name in panels:
        a.errorbar(temps, y, yerr=err, fmt='o', color=col, capsize=3)
        a.axvline(Tc, color='green', label=fr"$T_c={Tc:.2f}$")
        a.set_title(name); a.set_xlabel("T"); a.set_ylabel(name)
        a.legend()
    plt.tight_layout()
    plt.savefig(OBS_DIR / f"wolff_observables_L{L}_k{k}_nruns{n_runs}.pdf")   
    plt.close()                                                               

    # ---- tau: steps and sweep-equivalents ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].errorbar(temps, tau_mean, yerr=tau_std, fmt='o', color='black', capsize=3)
    ax[0].set_ylabel(r"$\tau_{int}$ (Wolff steps)")
    ax[1].errorbar(temps, tausw_mean, yerr=tausw_std, fmt='o', color='gray', capsize=3)
    ax[1].set_ylabel(r"$\tau_{int}$ (sweep-equivalents)")
    for a in ax:
        a.axvline(Tc, color='green', label=fr"$T_c={Tc:.2f}$")
        a.set_xlabel("T"); a.legend()
    plt.tight_layout()
    plt.savefig(OBS_DIR / f"wolff_tau_L{L}_k{k}_nruns{n_runs}.pdf")           
    plt.close()                                                               

    return dict(temps=temps, E_mean=E_mean, E_std=E_std,
                cv_mean=cv_mean, cv_std=cv_std,
                m_mean=m_mean, m_std=m_std,
                chi_mean=chi_mean, chi_std=chi_std,
                tau_mean=tau_mean, tau_std=tau_std,
                tausw_mean=tausw_mean, tausw_std=tausw_std,
                ok_frac=ok_frac)


## plot fitting

### only works for susceptibilty not cv, L^7/4 vs L^0 
#  search for the maximum only within k/L of Tc (the fine patch), so a noisy
# coarse-grid point far from Tc can no longer win the argmax
def peak(temps, y, Tc, L, w=2, k=4.0):
    temps, y = np.asarray(temps), np.asarray(y)
    near = np.where(np.abs(temps - Tc) <= k / L)[0]
    i = near[np.argmax(y[near])]
    lo, hi = max(i - w, near[0]), min(i + w + 1, near[-1] + 1)
    a, b, c = np.polyfit(temps[lo:hi], y[lo:hi], 2)
    if a >= 0:                                    # not a maximum: use grid max
        return temps[i], y[i]
    Tp = -b / (2 * a)
    if not (temps[lo] <= Tp <= temps[hi - 1]):    # vertex outside fitted window
        return temps[i], y[i]
    return Tp, np.polyval([a, b, c], Tp)

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

# load from disk if present, otherwise build and save.
# Delete the .pkl (or change the settings in its filename) when settings change.
def cached(path, build):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    out = build()
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


#  everything that runs simulations lives under the main guard
if __name__ == "__main__":

    Tc = 2 / np.log(1 + np.sqrt(2))

    # Ls for finite-size scaling: 40 roughly even sizes from 10 to 80
    Ls = np.linspace(50, 100, 7, dtype=int).tolist()

    N_RUNS, K = 10, 1000                  # temperature sweeps (chi, Cv peaks)
    TAU_RUNS, TAU_K = 20, 20000           # dedicated tau(Tc) runs

    #  one cache file per L, settings in the name
    res = {}
    for L in Ls:
        res[L] = cached(CACHE_DIR / f"sweep_L{L}_n{N_RUNS}_k{K}.pkl",
                        lambda L=L: wolff_sim(L=L, n_runs=N_RUNS, k=K))

    tauTc = {}
    for L in Ls:
        tauTc[L] = cached(CACHE_DIR / f"tauTc_L{L}_n{TAU_RUNS}_k{TAU_K}.pkl",
                          lambda L=L: tau_at_Tc(L, n_runs=TAU_RUNS, k=TAU_K))

    # finite-size quantities
    Tchi, Tcv = [], []
    chimax, chimax_err, cvmax = [], [], []

    for L in Ls:
        r = res[L]
        t = np.asarray(r["temps"])

        Tp, yp = peak(t, r["chi_mean"], Tc, L)         
        Tchi.append(Tp)
        chimax.append(yp)
        # error on chi_max: SE at the grid point nearest the peak
        chimax_err.append(r["chi_std"][int(np.argmin(np.abs(t - Tp)))])

        Tp, yp = peak(t, r["cv_mean"], Tc, L)           
        Tcv.append(Tp)
        cvmax.append(yp)

    tau_c = np.array([tauTc[L][0] for L in Ls])
    tau_err = np.array([tauTc[L][1] for L in Ls])

    Tchi = np.asarray(Tchi)
    Tcv = np.asarray(Tcv)
    chimax = np.asarray(chimax)
    chimax_err = np.asarray(chimax_err)
    cvmax = np.asarray(cvmax)
    Ls_arr = np.asarray(Ls)
    invL = 1 / Ls_arr

    # fits
    a_chi, Tc_chi = fit_Tc(invL, Tchi)
    a_cv, Tc_cv = fit_Tc(invL, Tcv)

    n_large = 5
    large = np.argsort(Ls)[-n_large:]
    a_chi_large, Tc_chi_large = fit_Tc(invL[large], Tchi[large])
    a_cv_large, Tc_cv_large = fit_Tc(invL[large], Tcv[large])

    # weighted exponent fits, with a scan over the smallest L included
    print("\ngamma/nu scan (expected 1.75):")
    for Lmin in (10, 20, 30, 40):
        g_, dg_, _ = powerlaw_fit(Ls_arr, chimax, chimax_err, Lmin)
        print(f"    Lmin={Lmin:3d}: {g_:.3f} ± {dg_:.3f}")

    print("\nz scan (reference ~0.25):")
    for Lmin in (10, 20, 30, 40):
        z_, dz_, _ = powerlaw_fit(Ls_arr, tau_c, tau_err, Lmin)
        print(f"    Lmin={Lmin:3d}: {z_:.3f} ± {dz_:.3f}")

    # headline fits (all L); change Lmin here once you have read the scans
    gamma_over_nu, dg, log_A = powerlaw_fit(Ls_arr, chimax, chimax_err, 0)
    A_chi = np.exp(log_A)
    z, dz, log_tau0 = powerlaw_fit(Ls_arr, tau_c, tau_err, 0)
    tau0 = np.exp(log_tau0)

    # power law vs a + b ln L for tau
    good = np.isfinite(tau_c) & np.isfinite(tau_err)
    b_log, a_log = np.polyfit(np.log(Ls_arr[good]), tau_c[good], 1,
                              w=1 / tau_err[good])
    chi2_pow = chi2_red(tau_c, tau0 * Ls_arr**z, tau_err, 2)
    chi2_log = chi2_red(tau_c, a_log + b_log * np.log(Ls_arr), tau_err, 2)

    # Plots
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    xfit = np.linspace(0, invL.max() * 1.05, 300)

    ax[0].scatter(invL, Tchi, s=55, label=r"$\chi$ peak")
    ax[0].scatter(invL, Tcv, s=55, label=r"$C_V$ peak")
    ax[0].plot(xfit, Tc_chi + a_chi * xfit, "--", label=rf"$\chi$: $T_\infty={Tc_chi:.4f}$")
    ax[0].plot(xfit, Tc_cv + a_cv * xfit, "--", label=rf"$C_V$: $T_\infty={Tc_cv:.4f}$")
    ax[0].axhline(Tc, linestyle=":", linewidth=2, label=rf"exact $T_c={Tc:.6f}$")
    ax[0].set_xlabel(r"$1/L$")
    ax[0].set_ylabel(r"$T^*(L)$")
    ax[0].set_title(r"Finite-size shift of pseudocritical temperature")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)

    Lfit = np.linspace(min(Ls), max(Ls), 300)

    #  susceptibility: errorbar, uncertainty in legend, reference normalised
    # over all points (not pinned to the noisy first point)
    ax[1].errorbar(Ls, chimax, yerr=chimax_err, fmt="o", ms=5, capsize=2,
                   label="simulation")
    ax[1].plot(Lfit, A_chi * Lfit**gamma_over_nu, "--",
               label=rf"fit: $\gamma/\nu={gamma_over_nu:.3f}\pm{dg:.3f}$")
    A_theory = np.exp(np.mean(np.log(chimax) - 1.75 * np.log(Ls_arr)))
    ax[1].plot(Lfit, A_theory * Lfit**1.75, ":", label=r"2D Ising: $\gamma/\nu=1.75$")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$L$")
    ax[1].set_ylabel(r"$\chi_{\max}$")
    ax[1].set_title(r"Susceptibility finite-size scaling")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25, which="both")

    #  dynamic scaling
    ax[2].errorbar(Ls, tau_c, yerr=tau_err, fmt="o", ms=5, capsize=2,
                   label="simulation")
    ax[2].plot(Lfit, tau0 * Lfit**z, "--", label=rf"fit: $z={z:.3f}\pm{dz:.3f}$")
    ax[2].plot(Lfit, a_log + b_log * np.log(Lfit), "-.", label=r"$a+b\ln L$")
    A_ref = np.exp(np.nanmean(np.log(tau_c) - 0.25 * np.log(Ls_arr)))
    ax[2].plot(Lfit, A_ref * Lfit**0.25, ":", label=r"reference: $z\approx0.25$")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel(r"$L$")
    ax[2].set_ylabel(r"$\tau_{\rm sweeps}(T_c)$")
    ax[2].set_title(r"Dynamic finite-size scaling")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.25, which="both")

    plt.tight_layout()

    # short filename instead of listing every L
    tag = f"L{Ls[0]}-{Ls[-1]}x{len(Ls)}"
    plt.savefig(CRIT_DIR / f"wolff_critical_z{z:.4f}_gamma_over_nu{gamma_over_nu:.4f}_{tag}.pdf")
    plt.show()

    # Results
    print("\n2D Ising Finite Size Scaling Results")

    print(f"\nExact Tc:")
    print(f"    Tc = {Tc:.6f}")

    print("\nPseudocritical temperature extrapolation:")
    print(f"    chi peak, all L:       Tc = {Tc_chi:.5f}")
    print(f"    chi peak, largest L:   Tc = {Tc_chi_large:.5f}")
    print(f"\n    Cv peak, all L:        Tc = {Tc_cv:.5f}")
    print(f"    Cv peak, largest L:    Tc = {Tc_cv_large:.5f}")

    print("\nSusceptibility scaling:")
    print(f"    gamma/nu = {gamma_over_nu:.4f} ± {dg:.4f}")
    print(f"    expected = 1.7500")

    print("\nDynamic scaling:")
    print(f"    z = {z:.4f} ± {dz:.4f}")
    print(f"    reference Wolff value ~ 0.25")
    print(f"    reduced chi2: power law {chi2_pow:.2f}, a+b ln L {chi2_log:.2f}")