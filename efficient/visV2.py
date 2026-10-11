from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyerrors as pe


#  settings (must match the C++ run) 
L = 100
nsweeps = 10000
n_eq = nsweeps // 20          # discard first 10% as equilibration
N = L * L

BASE = Path(f"efficient/data/met_results/L{L}/nsweeps{nsweeps}")
OBS_DIR = Path("efficient/figs")
OBS_DIR.mkdir(parents=True, exist_ok=True)

# grid from metadata.txt ----
meta = {}
with open(BASE / "metadata.txt") as f:
    for line in f:
        parts = line.split(None, 1)
        if len(parts) == 2:
            meta[parts[0]] = parts[1].strip()

nT = int(meta["nT"])
n_runs = int(meta["nRuns"])
temps = np.linspace(float(meta["Tmin"]), float(meta["Tmax"]), nT)
Tc = 2 / np.log(1 + np.sqrt(2))   # about 2.269


# ---- observables for one run ----
def observables(M, E, T):
    M, E = M[n_eq:], E[n_eq:]
    m = np.mean(np.abs(M)) / N
    chi = (np.mean(M**2) - np.mean(np.abs(M))**2) / (T * N)
    cv = np.var(E) / (N * T**2)
    e = np.mean(E) / N
    return m, chi, cv, e

# auto correlation fns from cluster.ipynb
def autocorr(x):
    #Normalised autocorrelation phi(t) (or chi(t)) via FFT
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)                 # zero-pad to avoid wraparound
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= np.arange(n, 0, -1)                  # divide by number of pairs (n - t)
    return acf / acf[0]

def tau_int(x, c=6.0):
    phi = autocorr(x)
    taus = np.cumsum(phi) - 0.5
    W = np.arange(len(taus))
    ok = W >= c * taus
    found = bool(ok.any())                      # local variable, shadows the global array
    m = np.argmax(ok) if found else len(taus) - 1
    return taus[m], found

tau = np.zeros((nT, n_runs, 2))               
found = np.ones((nT, n_runs, 2), dtype=bool)

# results[temperature, run, observable]
results = np.zeros((nT, n_runs, 4))
for i, T in enumerate(temps):
    for run in range(n_runs):
        fname = BASE / f"T{T:.5f}" / f"run{run:03d}.dat"
        M, E = np.loadtxt(fname, usecols=(1, 2), unpack=True)
        results[i, run] = observables(M, E, T)
        for k, x in enumerate((np.abs(M[n_eq:]), E[n_eq:])):
            tau[i, run, k], found[i, run, k] = tau_int(x)
    print(f"T={T:.3f}  m={results[i, :, 0].mean():.4f}  e={results[i, :, 3].mean():.4f}")

# mean over runs, standard error of the mean
mean = results.mean(axis=1)
err = results.std(axis=1, ddof=1) / np.sqrt(n_runs)
m_mean, chi_mean, cv_mean, E_mean = mean.T
m_std, chi_std, cv_std, E_std = err.T

# ---- observables ----
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
panels = [(ax[0, 1], m_mean, m_std, 'red', "Magnetism per spin"),
          (ax[1, 0], cv_mean, cv_std, 'blue', r"$c_v$"),
          (ax[1, 1], chi_mean, chi_std, 'purple', r"Susceptibility $\chi$"),
          (ax[0, 0], E_mean, E_std, 'orange', "Energy per spin")]
for a, y, e, col, name in panels:
    a.errorbar(temps, y, yerr=e, fmt='o', color=col, capsize=3)
    a.axvline(Tc, color='green', label=fr"$T_c={Tc:.2f}$")
    a.set_title(name); a.set_xlabel("T"); a.set_ylabel(name)
    a.legend()
plt.tight_layout()
plt.savefig(OBS_DIR / f"met_observables_L{L}_nsweeps{nsweeps}_nruns{n_runs}.pdf")
plt.close()

tau_mean = tau.mean(axis=1)
tau_std = tau.std(axis=1, ddof=1) / np.sqrt(n_runs)
bad = ~found.all(axis=1)                       # any run failed the window test

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, k, col, name in [(ax[0], 0, 'black', r"$|M|$"), (ax[1], 1, 'gray', r"$E$")]:
    a.errorbar(temps, tau_mean[:, k], yerr=tau_std[:, k], fmt='o', color=col, capsize=3)
    a.plot(temps[bad[:, k]], tau_mean[bad[:, k], k], 'rx', ms=10, label="window not met")
    a.axvline(Tc, color='green', label=fr"$T_c={Tc:.2f}$")
    a.set_xlabel("T"); a.set_ylabel(rf"$\tau_{{int}}$ ({name}, sweeps)")
    a.legend()
plt.tight_layout()
plt.savefig(OBS_DIR / f"met_tau_L{L}_nsweeps{nsweeps}_nruns{n_runs}.pdf")
plt.close()
