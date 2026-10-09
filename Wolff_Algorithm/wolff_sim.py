"""
wolff_sim.py  --  Wolff cluster simulation of the 2D Ising model.

Runs simulations and saves the RAW time series (|M| source, energy, cluster
size) to .npz files, one file per (mode, L, T, run).  Analysis lives elsewhere
and imports `load_run`, `autocorr`, `tau_int` from here.

Two independent policies are chosen per run:

    length  : Fixed(n_steps)  |  Adaptive(k, block_size, max_steps, c)
    burnin  : FixedBurnin(n_discard)  |  AdaptiveBurnin(mult)

The burn-in policy only decides the stored `discard_n`; the full series
(everything after the short warmup) is always saved, so analysis can override it.

Example:
    python wolff_sim.py --mode fixed --Ls 16 32 64 --runs 10 --steps 100000
    python wolff_sim.py --mode adaptive --Ls 50 100 --runs 10 --k 1000
    python wolff_sim.py --mode adaptive --temps tc --runs 20 --k 20000 \
                        --block 50000 --max-steps 5000000 --Ls 50 100
"""
import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numba import njit

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

J_DEFAULT = 1.0
TC = 2.0 / math.log(1 + math.sqrt(2))      # exact Tc for J = 1


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Fixed:
    n_steps: int = 100_000


@dataclass(frozen=True)
class Adaptive:
    k: float = 1000.0            # want usable_len >= k * tau_int
    block_size: int = 2000       # first block; doubles each round
    max_steps: int = 200_000
    c: float = 6.0               # Sokal window constant


@dataclass(frozen=True)
class FixedBurnin:
    n_discard: int = 1000


@dataclass(frozen=True)
class AdaptiveBurnin:
    mult: float = 10.0           # discard mult * tau_int steps


def _discard(burnin, tau_hat, n):
    d = burnin.mult * tau_hat if isinstance(burnin, AdaptiveBurnin) else burnin.n_discard
    return int(min(d, n // 2))


# --------------------------------------------------------------------------
# numba kernels
# --------------------------------------------------------------------------
@njit(cache=True)
def seed_numba(s):
    np.random.seed(s)


@njit(cache=True)
def energy(s, J, h):
    # 2D Ising energy with periodic boundary conditions
    E = 0.0
    N = s.shape[0]
    for i in range(N):
        for j in range(N):
            E -= J * s[i, j] * (s[(i + 1) % N, j] + s[i, (j + 1) % N])
            E -= h * s[i, j]
    return E


@njit(cache=True)
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


@njit(cache=True)
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


# --------------------------------------------------------------------------
# autocorrelation (used by the adaptive policies, and importable by analysis)
# --------------------------------------------------------------------------
def autocorr(x):
    # Normalised autocorrelation phi(t) via FFT
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)                 # zero-pad to avoid wraparound
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0]


def tau_int(x, c=6.0):
    # Integrated autocorrelation time with Sokal automatic windowing.
    # found=False means the window condition was never met.
    phi = autocorr(x)
    taus = np.cumsum(phi) - 0.5
    W = np.arange(len(taus))
    ok = W >= c * taus
    found = bool(ok.any())
    m = np.argmax(ok) if found else len(taus) - 1
    return taus[m], found


# --------------------------------------------------------------------------
# one simulation run
# --------------------------------------------------------------------------
def make_seed(base_seed, L, T, run):
    # deterministic, independent seed per (L, T, run); fits numba's uint32 seed
    ss = np.random.SeedSequence([int(base_seed), int(L), int(round(T * 1e6)), int(run)])
    return int(ss.generate_state(1)[0])


def run_series(T, L, seed, length=None, burnin=None, J=J_DEFAULT,
               short_warmup=200, verbose=False):
    """Run Wolff at (T, L).  Returns a dict of raw arrays + metadata.

    Saved series starts AFTER the short warmup and is never trimmed;
    `discard_n` is what the burn-in policy would discard.
    `energies` are TOTAL energies (divide by L*L for per-spin).
    """
    length = length or Fixed()
    burnin = burnin or AdaptiveBurnin()
    h = 0.0
    Nspins = L * L

    seed_numba(seed)
    rng = np.random.default_rng(seed)

    # cold start below Tc, hot start above
    start = "cold" if T < TC else "hot"
    if start == "cold":
        spins = np.ones((L, L), dtype=np.int64)
    else:
        spins = (2 * rng.integers(0, 2, size=(L, L)) - 1).astype(np.int64)

    E = energy(spins, J, h)
    M = int(spins.sum())
    mark = np.zeros((L, L), dtype=np.bool_)
    stack = np.empty(Nspins, dtype=np.int64)

    # short fixed warmup, not recorded
    E, M, _, _, _ = run_block(spins, L, J, T, E, M, mark, stack, short_warmup)
    total_steps = short_warmup

    if isinstance(length, Fixed):
        E, M, mags, ens, ns = run_block(spins, L, J, T, E, M, mark, stack,
                                        int(length.n_steps))
        total_steps += int(length.n_steps)
        tau_hat, found = tau_int(np.abs(mags))
        discard_n = _discard(burnin, tau_hat, len(mags))
        ok = found                       # window condition met at least once
        c_used = 6.0

    elif isinstance(length, Adaptive):
        c_used = length.c
        mags = np.empty(0, np.int64)
        ens = np.empty(0, np.float64)
        ns = np.empty(0, np.int64)
        cur = int(length.block_size)
        while True:
            E, M, mb, eb, nb = run_block(spins, L, J, T, E, M, mark, stack, cur)
            mags = np.concatenate([mags, mb])
            ens = np.concatenate([ens, eb])
            ns = np.concatenate([ns, nb])
            total_steps += cur

            tau_hat, found = tau_int(np.abs(mags), length.c)
            discard_n = _discard(burnin, tau_hat, len(mags))
            usable = len(mags) - discard_n

            if verbose:
                print(f" L={L} T={T:.3f} steps={total_steps} tau~{tau_hat:.1f} "
                      f"found={found} usable={usable} need={length.k * tau_hat:.0f}")

            # success checked first, so a run that just made it is not discarded
            if found and usable >= length.k * tau_hat:
                ok = True
                break
            if total_steps >= length.max_steps:
                ok = False
                break
            cur *= 2
    else:
        raise TypeError(f"unknown length policy: {length!r}")

    # tau on the post-discard series (what analysis will typically use)
    tau_final, found_final = tau_int(np.abs(mags[discard_n:]), c_used)
    usable_len = len(mags) - discard_n

    return dict(
        mags=mags, energies=ens, ns=ns,
        L=L, T=T, J=J, seed=seed, start=start, short_warmup=short_warmup,
        mode=type(length).__name__.lower(),
        length_policy=json.dumps(asdict(length)),
        burnin_policy=json.dumps({"type": type(burnin).__name__, **asdict(burnin)}),
        discard_n=discard_n, tau_hat=float(tau_hat), found=bool(found),
        tau_final=float(tau_final), found_final=bool(found_final),
        usable_len=usable_len,
        n_over_tau=float(usable_len / tau_final) if tau_final > 0 else float("nan"),
        total_steps=total_steps, ok=bool(ok and found_final),
    )


# --------------------------------------------------------------------------
# saving / loading
# --------------------------------------------------------------------------
def run_path(mode, L, T, run, data_dir=DATA_DIR):
    return Path(data_dir) / mode / f"L{L}" / f"T{T:.6f}_run{run:03d}.npz"


def save_run(path, out):
    # write to a temp file then rename, so a crash never leaves a half file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def load_run(path):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    for k in list(d):
        if d[k].ndim == 0:
            d[k] = d[k].item()
    return d


# --------------------------------------------------------------------------
# sweep driver
# --------------------------------------------------------------------------
def make_temps(L, Tc=TC, n_coarse=20, n_fine=21, width=4.0, lo=1.8, hi=2.8):
    coarse = np.linspace(lo, hi, n_coarse)
    fine = Tc + np.linspace(-width, width, n_fine) / L    # includes Tc exactly
    return np.unique(np.round(np.concatenate([coarse, fine]), 6))


def _task(args):
    path, T, L, run, length, burnin, J, short_warmup, base_seed = args
    if Path(path).exists():
        return path, "skip", None
    seed = make_seed(base_seed, L, T, run)
    out = run_series(T, L, seed, length, burnin, J, short_warmup)
    save_run(path, out)
    return path, "done", out["ok"]


def run_sweep(Ls, temps_for_L, n_runs, length, burnin, J=J_DEFAULT,
              short_warmup=200, base_seed=13, workers=None, data_dir=DATA_DIR):
    mode = type(length).__name__.lower()
    tasks = []
    for L in Ls:
        for T in temps_for_L(L):
            for r in range(n_runs):
                tasks.append((run_path(mode, L, T, r, data_dir), float(T), int(L), r,
                              length, burnin, J, short_warmup, base_seed))
    todo = [t for t in tasks if not Path(t[0]).exists()]
    print(f"{len(tasks)} runs total, {len(tasks) - len(todo)} already on disk, "
          f"{len(todo)} to do")
    if not todo:
        return

    # compile once in the parent so workers load the cache instead of racing
    run_series(2.0, 4, 0, Fixed(10), FixedBurnin(0), J, 10)

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    n_done = 0
    n_bad = 0

    def report(res):
        nonlocal n_done, n_bad
        n_done += 1
        if res[1] == "done" and not res[2]:
            n_bad += 1
        if n_done % 10 == 0 or n_done == len(todo):
            print(f" {n_done}/{len(todo)} done  (not ok: {n_bad})", flush=True)

    if workers == 1:
        for t in todo:
            report(_task(t))
    else:
        with Pool(workers) as pool:
            for res in pool.imap_unordered(_task, todo, chunksize=1):
                report(res)


def _parse_temps(spec):
    if spec[0] == "sweep":
        return lambda L: make_temps(L)
    if spec[0] == "tc":
        return lambda L: [round(TC, 6)]
    vals = [round(float(x), 6) for x in spec]
    return lambda L: vals


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["fixed", "adaptive"], default="fixed")
    p.add_argument("--Ls", type=int, nargs="+", required=True)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--temps", nargs="+", default=["sweep"],
                   help="'sweep', 'tc', or explicit temperatures")
    # length
    p.add_argument("--steps", type=int, default=100_000, help="fixed: steps per run")
    p.add_argument("--k", type=float, default=1000, help="adaptive: need N >= k*tau")
    p.add_argument("--block", type=int, default=2000, help="adaptive: first block")
    p.add_argument("--max-steps", type=int, default=200_000, help="adaptive cap")
    p.add_argument("--c", type=float, default=6.0, help="adaptive: window constant")
    # burn-in
    p.add_argument("--burnin", choices=["fixed", "adaptive"], default="adaptive")
    p.add_argument("--burnin-n", type=int, default=1000)
    p.add_argument("--burnin-mult", type=float, default=10.0)
    # misc
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=13, help="base seed")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    a = p.parse_args()

    length = (Fixed(a.steps) if a.mode == "fixed"
              else Adaptive(a.k, a.block, a.max_steps, a.c))
    burnin = (FixedBurnin(a.burnin_n) if a.burnin == "fixed"
              else AdaptiveBurnin(a.burnin_mult))

    run_sweep(a.Ls, _parse_temps(a.temps), a.runs, length, burnin,
              short_warmup=a.warmup, base_seed=a.seed, workers=a.workers,
              data_dir=a.data_dir)


if __name__ == "__main__":
    main()