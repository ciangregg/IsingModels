import numpy as np
import matplotlib.pyplot as plt
from visV4 import load, sweep_clock, ALGOS, N #  (same folder as visV4.py)

T_target = 1.8        
data = {a: load(a) for a in ALGOS}
idx = int(np.argmin(np.abs(data["met"]["temps"] - T_target)))

edges = np.logspace(np.log10(200), np.log10(1e4), 25)   # 200 = n_eq
centers = np.sqrt(edges[:-1] * edges[1:])

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for algo, d in data.items():
    col, lab = ALGOS[algo]["color"], ALGOS[algo]["label"]
    xs = [sweep_clock(d["S"][idx][r]) for r in range(d["n_runs"])]
    ys = [np.abs(d["M"][idx][r]) / N for r in range(d["n_runs"])]

    for x, y in zip(xs, ys):
        ax[0].plot(x, y, color=col, lw=0.5, alpha=0.25)   
        ax[1].plot(x, y, color=col, lw=0.5, alpha=0.25)   
    # pooled stats per log bin
    std, rng, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = np.concatenate([y[(x >= lo) & (x < hi)] for x, y in zip(xs, ys)])
        std.append(v.std()); rng.append(v.max() - v.min()); cnt.append(len(v))
    ax[2].plot(centers, std, "o-", color=col, label=lab)

ax[0].set_xscale("log"); ax[0].set_title("raw, log m (current)")
ax[0].set_xlim(1, 1e4)
ax[1].set_title("raw, linear m")
ax[2].set_xscale("log"); ax[2].set_title("std of samples per log bin")
ax[2].set_ylim(bottom=0); ax[2].legend()
for a in ax: a.set_xlabel("sweep-equivalents")
plt.tight_layout(); plt.savefig("efficient/figs/check_display.pdf", dpi=150)
print("samples per bin:", cnt)