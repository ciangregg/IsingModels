import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# numba and numpy's RNG seed
@njit
def seed_numba(s):
    np.random.seed(s)

np.random.seed(13)
seed_numba(13)

#Lattice size for the runs
#Ls for finite-size scaling.
#Ls = [10, 14, 18, 24, 32, 40, 48, 56, 64, 70, 76, 80, 84, 90]
#Ls = [10, 14]
#Ls = list(range(10, 101, 2))   # [10, 12, 14, ..., 100]

Ls = np.linspace(10, 80, 40, dtype=int).tolist()   # 40 roughly even sizes from 10 to 100



## ising
@njit
def energy(s, J, h):
    #2D Ising energy with periodic boundary conditions.
    
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


## auto corralation and Tau

def autocorr(x):
    #Normalised autocorrelation phi(t) (or chi(t)) via FFT
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)                 # zero-pad to avoid wraparound
    acf = np.fft.irfft(f * np.conj(f))[:n]
    #acf /= np.arange(n, 0, -1)                  # divide by number of pairs (n - t)
    return acf / acf[0]

def tau_int(x, c=6.0):
    #Integrated autocorrelation time with Sokal automatic windowing
    #found=False means the window condition was never met
    phi = autocorr(x)
    taus = np.cumsum(phi) - 0.5
    W = np.arange(len(taus))
    ok = W >= c * taus
    found = bool(ok.any())
    m = np.argmax(ok) if found else len(taus) - 1
    return taus[m], found


def tau_at_Tc(L, n_runs=10, k=5000, max_steps=2_000_000):
    Tc = 2 / np.log(1 + np.sqrt(2))
    vals = []
    for _ in range(n_runs):
        r = adaptive_wolff_run(Tc, L, 1.0, k=k, Tc=Tc, max_steps=max_steps)
        if r["ok"]:
            vals.append(r["tau_sweeps"])
    vals = np.array(vals)
    if len(vals) < 2:
        print(f" L={L}: only {len(vals)} converged runs, raise max_steps")
        return np.nan, np.nan, len(vals)
    print(f" L={L}: tau_sweeps = {vals.mean():.3f} ± {vals.std(ddof=1)/np.sqrt(len(vals)):.3f} "
          f"({len(vals)}/{n_runs} converged)")
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)), len(vals)

## wolff alogrithm and simulation

@njit
def wolff_step(s, L, J, T, E, mark, stack):
    # wolff prob
    p_add = 1.0 - math.exp(-2.0 * J / T)

    # (step 1) random seed spin the first of the cluster
    x0 = np.random.randint(L)
    y0 = np.random.randint(L)
    spin = s[x0, y0]
    stack[0] = x0 * L + y0
    mark[x0, y0] = True
    n = 1
    head = 0
    
    # (step 2) build the cluster
    # grow (spins are NOT flipped yet, so `mark` alone defines membership)
    while head < n:
        site = stack[head]
        head += 1
        
        #get our coords
        # x= site // L
        # y= site % L
        # ie for (3,7), x=37//7 = 3 and y=37%10=7

        x = site // L
        y = site % L
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = (x + di) % L
            ny = (y + dj) % L
            
            # if our spin matches the first and if its not stamped
            if s[nx, ny] == spin and not mark[nx, ny]:
                if np.random.rand() < p_add:
                    mark[nx, ny] = True # add to the cluser
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
                    dE += 2.0 * J      # Spins dont agree after spin 
                else:
                    dE -= 2.0 * J      # Spins do agree after spin

    # flip cluster and clear marks (O(n))
    for i in range(n):
        x = stack[i] // L
        y = stack[i] % L
        s[x, y] = -spin
        mark[x, y] = False

    return E + dE, n


def adaptive_wolff_run(T, L, J=1,
                       block_size=2000, k=1000, discard_mult=10,
                       max_steps=200_000, short_warmup=200,
                       Tc=None, verbose=False):
    # Run Wolff at temperature T, adaptively extending the run until the (post-discard) series is long enough relative to tau_int, or until max_steps is hit.
    # tau_sweeps = tau * <n> / N
    h=0
    Nspins = L * L

    if Tc is None:
        Tc = 2 * abs(J) / np.log(1 + np.sqrt(2))

    # cold start
    if T < Tc:
        spins = np.ones((L, L), dtype=np.int64)
    else:
        spins = np.random.choice(np.array([-1, 1]), size=(L, L)).astype(np.int64)

    E_current = energy(spins, J=J, h=h)

    mark = np.zeros((L, L), dtype=np.bool_)
    stack = np.empty(Nspins, dtype=np.int64)

    # short fixed warmup, not recorded
    for _ in range(short_warmup):
        E_current, n = wolff_step(spins, L, J, T, E_current, mark, stack)

    mags = []
    energies = []
    ns = []
    total_steps = short_warmup
    current_block = int(block_size)

    tau_hat = np.inf # start with with tau=infinity
    found = False
    ok = False

    while True:
        for _ in range(current_block):
            E_current, n = wolff_step(spins, L, J, T, E_current, mark, stack)
            mags.append(spins.sum())
            energies.append(E_current)
            ns.append(n)
            
        total_steps += current_block

        series = np.abs(np.array(mags))
        tau_hat, found = tau_int(series)

        discard_n = int(min(discard_mult * tau_hat, len(series) // 2))
        usable_len = len(series) - discard_n

        if verbose:
            print(f" L={L}  T={T:.3f} stepsprint={total_steps} tau~{tau_hat:.1f} "
                  f"found={found} usable_len={usable_len} need={k * tau_hat:.0f}")

        if total_steps >= max_steps:
            ok = False
            break

        if found and usable_len >= k * tau_hat:
            ok = True
            break

        current_block *= 2

    # slice to usable portion
    mags_arr = np.array(mags[discard_n:])
    energies_arr = np.array(energies[discard_n:]) / Nspins
    ns_arr = np.array(ns[discard_n:])

    # recompute tau on the cleaned series
    tau_final, found_final = tau_int(np.abs(mags_arr))
    n_eff = len(mags_arr) / (2 * tau_final) if tau_final > 0 else np.nan
    nbar = ns_arr.mean()
    tau_sweeps = tau_final * nbar / Nspins

    # observables
    m = np.mean(np.abs(mags_arr)) / Nspins
    chi = (np.mean(mags_arr**2) - np.mean(np.abs(mags_arr))**2) / (T * Nspins)
    E = np.mean(energies_arr)
    cv = np.var(energies_arr) * Nspins / (T ** 2)

    return dict(T=T, mags=mags_arr, energies=energies_arr,
                tau=tau_final, tau_found=found_final, n_eff=n_eff,
                tau_sweeps=tau_sweeps, nbar=nbar,
                m=m, chi=chi, E=E, cv=cv,
                total_steps=total_steps, discard_n=discard_n,
                ok=ok and found_final)


def wolff_sim(L=10, temps=None, J=1.0, h=0.0, n_runs=5, k=1000, max_steps=500_000, short_warmup=200):
    
    Tc = 2 * abs(J) / np.log(1 + np.sqrt(2))

    if temps is None:
        temps = make_temps(L, Tc)
    temps = np.asarray(temps)

    print("Critical temp")
    print(Tc)

    E_mean, E_std = [], []
    cv_mean, cv_std = [], []
    m_mean, m_std = [], []
    chi_mean, chi_std = [], []
    tau_mean, tau_std = [], []
    tausw_mean, tausw_std = [], []
    ok_frac = []

    for T in temps:
        runs = [adaptive_wolff_run(T, L, J, k=k, Tc=Tc, max_steps=500_000, short_warmup=200) for _ in range(n_runs)]

        for lst_m, lst_s, key in [(E_mean, E_std, "E"), (cv_mean, cv_std, "cv"),
                                  (m_mean, m_std, "m"), (chi_mean, chi_std, "chi"),
                                  (tau_mean, tau_std, "tau"),
                                  (tausw_mean, tausw_std, "tau_sweeps")]:
            vals = np.array([r[key] for r in runs])
            lst_m.append(vals.mean())
            lst_s.append(vals.std())

        ok_frac.append(np.mean([r["ok"] for r in runs]))
        print(f" L={L}, T={T:.3f} done, ran {n_runs} runs, ok={ok_frac[-1]:.0%}")

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
    #plots/observables/wolff
    plt.savefig(f"Wolff_Algorithm/plots/observables/wolff/wolff_observables_L{L}_k{k}_nruns{n_runs}.pdf")
    plt.show()

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
    #Wolff_Algorithm/plots/observables/wolff
    plt.savefig(f"Wolff_Algorithm/plots/observables/wolff/wolff_tau_L{L}_k{k}_nruns{n_runs}.pdf")
    plt.show()

    return dict(temps=temps, E_mean=E_mean, E_std=E_std,
                cv_mean=cv_mean, cv_std=cv_std,
                m_mean=m_mean, m_std=m_std,
                chi_mean=chi_mean, chi_std=chi_std,
                tau_mean=tau_mean, tau_std=tau_std,
                tausw_mean=tausw_mean, tausw_std=tausw_std,
                ok_frac=ok_frac)

def make_temps(L, Tc, n_coarse=20, n_fine=21, width=4.0, lo=1.8, hi=2.8):
    coarse = np.linspace(lo, hi, n_coarse)
    fine   = Tc + np.linspace(-width, width, n_fine) / L    # includes Tc exactly
    return np.unique(np.round(np.concatenate([coarse, fine]), 6))


## plot fitting
def peak(temps, y, w=2):
    temps, y = np.asarray(temps), np.asarray(y)
    i = int(np.argmax(y))
    lo, hi = max(i - w, 0), min(i + w + 1, len(y))
    a, b, c = np.polyfit(temps[lo:hi], y[lo:hi], 2)
    if a >= 0:                          # not a maximum: fall back to the grid max
        return temps[i], y[i]
    Tp = -b / (2 * a)
    if not (temps[lo] <= Tp <= temps[hi - 1]):   # vertex outside the fitted window
        return temps[i], y[i]
    return Tp, np.polyval([a, b, c], Tp)

def fit_Tc(invL, Tstar):
    # Linear finite-size scaling fit
    # T*(L) = Tc + a/L
    
    slope, intercept = np.polyfit(invL, Tstar, 1)
    return slope, intercept


### Running our simulation

Tc = 2 / np.log(1 + np.sqrt(2))



res = { L: wolff_sim(L=L,n_runs=10,k=1000) for L in Ls }


# finite-size quantities

Tchi, Tcv = [] , []
chimax, cvmax = [], []
tau_c = []

for L in Ls:
    
    r = res[L]
    t = np.asarray(r["temps"])

    # Susceptibility peak
    Tp, yp = peak(t, r["chi_mean"])
    Tchi.append(Tp)
    chimax.append(yp)

    # Specific heat peak
    Tp, yp = peak(t, r["cv_mean"])
    Tcv.append(Tp)
    cvmax.append(yp)
    '''
    # Interpolate autocorrelation time to the exact Tc
    tau = np.interp(Tc,t, r["tausw_mean"])
    tau_c.append(tau)
    '''
tauTc = {L: tau_at_Tc(L) for L in Ls}
tau_c   = np.array([tauTc[L][0] for L in Ls])    # means
tau_err = np.array([tauTc[L][1] for L in Ls])    # standard errors

# Convert to arrays
Tchi = np.asarray(Tchi)
Tcv = np.asarray(Tcv)
chimax = np.asarray(chimax)
cvmax = np.asarray(cvmax)
tau_c = np.asarray(tau_c)
invL = 1 / np.asarray(Ls)


# fits
a_chi, Tc_chi = fit_Tc(invL, Tchi)
a_cv, Tc_cv = fit_Tc(invL, Tcv)


# Also fit only the largest 5 lattice sizes.
# This is useful for checking whether the extrapolation is changing significantly when small systems are removed.

n_large = 5
large = np.argsort(Ls)[-n_large:]
a_chi_large, Tc_chi_large = fit_Tc(invL[large], Tchi[large])
a_cv_large, Tc_cv_large = fit_Tc(invL[large], Tcv[large])

# chi exponet
# chi_max ~ L^(gamma/nu)
gamma_over_nu, log_A = np.polyfit( np.log(Ls), np.log(chimax),1 )
A_chi = np.exp(log_A)

# dynamics exponent
# tau ~ L^z
z, log_tau0 = np.polyfit(np.log(Ls),np.log(tau_c),1 )
tau0 = np.exp(log_tau0)

# Plots
fig, ax = plt.subplots(1, 3, figsize=(17, 5))

# Pseudocritical temperature
xfit = np.linspace(0,invL.max() * 1.05, 300)

# Data
ax[0].scatter(invL,Tchi,s=55,label=r"$\chi$ peak")
ax[0].scatter(invL,Tcv,s=55,label=r"$C_V$ peak")

# Full-range fits
ax[0].plot(xfit,Tc_chi + a_chi * xfit,"--", label=rf"$\chi$: $T_\infty={Tc_chi:.4f}$")
ax[0].plot(xfit, Tc_cv + a_cv * xfit, "--", label=rf"$C_V$: $T_\infty={Tc_cv:.4f}$")

# Exact critical temperature
ax[0].axhline(Tc,linestyle=":",linewidth=2, label=rf"exact $T_c={Tc:.6f}$")
ax[0].set_xlabel(r"$1/L$")
ax[0].set_ylabel(r"$T^*(L)$")
ax[0].set_title(r"Finite-size shift of pseudocritical temperature")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.25)


# Susceptibility scaling

Lfit = np.linspace(min(Ls), max(Ls),300)


ax[1].loglog(Ls,chimax,"o",markersize=7, label="simulation")
ax[1].loglog( Lfit, A_chi * Lfit**gamma_over_nu, "--", label=rf"fit: $\gamma/\nu={gamma_over_nu:.3f}$")

# Reference theoretical exponent, normalised to first point
A_theory = chimax[0] / Ls[0]**1.75

ax[1].loglog(Lfit, A_theory * Lfit**1.75, ":",label=r"2D Ising: $\gamma/\nu=1.75$")
ax[1].set_xlabel(r"$L$")
ax[1].set_ylabel(r"$\chi_{\max}$")
ax[1].set_title(r"Susceptibility finite-size scaling")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.25, which="both")


# Dynamic scaling

ax[2].loglog(Ls, tau_c, "o", markersize=7,label="simulation")
ax[2].loglog( Lfit, tau0 * Lfit**z, "--", label=rf"fit: $z={z:.3f}$")
# Reference Wolff value
tau_ref = tau_c[0] / Ls[0]**0.25
ax[2].loglog(Lfit, tau_ref * Lfit**0.25, ":", label=r"reference: $z\approx0.25$")
ax[2].set_xlabel(r"$L$")
ax[2].set_ylabel(r"$\tau_{\rm sweeps}(T_c)$")
ax[2].set_title(r"Dynamic finite-size scaling")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.25, which="both")

plt.tight_layout()

name = "_".join(map(str, Ls))   # Ls = [1, 2, 3] becomes "1_2_3"
#Wolff_Algorithm/plots/
plt.savefig(f"Wolff_Algorithm/plots/critical/wolff_critical_z{z:.4f}_gamma_over_nu{gamma_over_nu:.4f}_Ls{name}.pdf")
plt.show()

# Results

print("2D Ising Finite Size Scaling Results")

print(f"\nExact Tc:")
print(f"    Tc = {Tc:.6f}")

print("\nPseudocritical temperature extrapolation:")
print(f"    chi peak, all L:       Tc = {Tc_chi:.5f}")
print(f"    chi peak, largest L:   Tc = {Tc_chi_large:.5f}")

print(f"\n    Cv peak, all L:        Tc = {Tc_cv:.5f}")
print(f"    Cv peak, largest L:    Tc = {Tc_cv_large:.5f}")

print("\nSusceptibility scaling:")
print(f"    gamma/nu = {gamma_over_nu:.4f}")
print(f"    expected = 1.7500")

print("\nDynamic scaling:")
print(f"    z = {z:.4f}")
print(f"    reference Wolff value ~ 0.25")

