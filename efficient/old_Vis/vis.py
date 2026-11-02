import os
import numpy as np
import matplotlib.pyplot as plt

L = 101
N = L * L
nsweeps = 10000
n_eq = nsweeps // 10          # discard first 10% as equilibration

nT, Tmin, Tmax = 50, 1.8, 2.8
temps = np.linspace(Tmin, Tmax, nT)   # same as Tmin + (Tmax-Tmin)*t/(nT-1)

e_mean, cv, m_mean, chi = [], [], [], []

for T in temps:
    fname = f"efficient/data/met_results_T{T:.4f}L{L}nsweeps{nsweeps}.dat"
    step, Mags, Energies = np.loadtxt(fname, unpack=True)

    M_eq = Mags[n_eq:]
    E_eq = Energies[n_eq:]

    e_mean.append(np.mean(E_eq) / N)
    cv.append(np.var(E_eq) / (N * T**2))
    m_mean.append(np.mean(np.abs(M_eq)) / N)
    chi.append((np.mean(M_eq**2) - np.mean(np.abs(M_eq))**2) / (T * N))

    print(f"T={T:.3f}  e={e_mean[-1]:.4f}  m={m_mean[-1]:.4f}")
    

Tc = 2 / np.log(1 + np.sqrt(2))   # about 2.269

fig, ax = plt.subplots(2, 2, figsize=(10, 8))
data = [(ax[0, 0], m_mean, r"$\langle |m| \rangle$", "red"),
        (ax[0, 1], chi,    r"$\chi$",                "purple"),
        (ax[1, 0], cv,     r"$c_v$",                 "blue"),
        (ax[1, 1], e_mean, r"$e$",                   "orange")]

for a, y, label, col in data:
    a.plot(temps, y, "o-", color=col)
    a.axvline(Tc, color="green", ls="--", label=fr"$T_c={Tc:.3f}$")
    a.set_xlabel("T")
    a.set_ylabel(label)
    a.legend()
    a.grid()

plt.tight_layout()
os.makedirs("efficient/figs", exist_ok=True)
plt.savefig(f"efficient/figs/observables_L{L}.pdf")
plt.show()