import numpy as np
from visV4 import load, eq_start, ALGOS, Tc, N

def window_stats(d, idx, w=500, after_neq=True, per_run=False):
    algo = d["algo"]
    T = d["temps"][idx]
    print(f"\n=== {algo}  T={T:.4f} ===")
    means, stds = [], []
    for r in range(d["n_runs"]):
        s0 = eq_start(d["S"][idx][r], ALGOS[algo]["n_eq"]) if after_neq else 0
        m = np.abs(d["M"][idx][r][s0:]) / N
        k = len(m) // w
        blocks = m[:k * w].reshape(k, w)
        mu, sd = blocks.mean(1), blocks.std(1)
        means.append(mu[:18]); stds.append(sd[:18])   # keep common window count
        if per_run:
            print(r, mu.round(4), sd.round(4))
    means, stds = np.array(means), np.array(stds)
    # across-run average per window: flat = stationary
    print("mean of |m| per window (avg over runs):", means.mean(0).round(4))
    print("std  of |m| per window (avg over runs):", stds.mean(0).round(4))
    print(f"first-third vs last-third std: {stds[:, :6].mean():.4f} vs {stds[:, -6:].mean():.4f}")
        

if __name__ == "__main__":
    data = {algo: load(algo) for algo in ALGOS}
    temps = data["met"]["temps"]
    for T_target in (1.8, 2.1, Tc, 2.8):
        idx = int(np.argmin(np.abs(temps - T_target)))
        for algo in ("met", "wolff"):
            window_stats(data[algo], idx, w=500)