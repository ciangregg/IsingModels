import os
import numpy as np
import matplotlib.pyplot as plt

L=100
N=L*L

T=2.0

step, Mags, Energies = np.loadtxt(f"efficient/data/met_results_T{T}L{L}nsweeps100000.dat", unpack=True)

mags= Mags/N
energies=Energies/N

E = np.mean(energies)
cv = np.var(energies) * N / (T ** 2)
m = np.mean(np.abs(Mags)) / N
chi = np.var(np.abs(Mags)) / (T * N)

print(f"e = {E:.4f}, cv = {cv:.4f}, m = {m:.4f}, chi = {chi:.4f}")

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax[0].plot(step, energies, color="orange")
ax[0].set_ylabel("e (energy per spin)")
ax[0].grid()

ax[1].plot(step, mags, color="red")
ax[1].set_ylabel("m (magnetisation per spin)")
ax[1].set_xlabel("sweep")
ax[1].grid()

ax[0].set_title(f"Metropolis, L={L}, T={T}")
plt.tight_layout()

os.makedirs("efficient/figs/e-m", exist_ok=True)
plt.savefig(f"efficient/figs/e-m/e-m_T{T}L{L}.pdf")
plt.show()


