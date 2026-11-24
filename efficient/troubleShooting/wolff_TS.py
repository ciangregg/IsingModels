from pathlib import Path
import math
import time

import numpy as np


# Settings, matched to wolff.cpp

L = 100
N = L * L
J = 1

TEMPS = [1.800, 2.106, 2.269, 2.800]
NSWEEPS = 10_000
NRUNS = 1

BURN_IN_SAMPLES = 200
PILOT_EQ_SWEEPS = 300
PILOT_MEAS_SWEEPS = 100

BASE_SEED = 13
DATA_DIR = Path("efficient/troubleShooting/data/python")

TC = 2.0 * abs(J) / math.log(1.0 + math.sqrt(2.0))



# Initialisation and soem observables

def initialise_spins(T, rng):
    """Cold start below Tc; random hot start at/above Tc."""
    if T < TC:
        return np.ones(N, dtype=np.int8)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=N)


def magnetisation(spins):
    return int(np.sum(spins))


def energy(spins):
    """
    Total energy, using the same flattened-neighbour convention
    as the supplied C++ implementation.
    """
    idx = np.arange(N)
    right = (idx + 1) % N
    up = (idx + L) % N
    return int(-J * np.sum(spins * (spins[right] + spins[up])))


def nn_sum(spins, i):
    """Four neighbours, with the same indexing as wolff.cpp."""
    right = (i + 1) % N
    left = (i - 1) % N
    up = (i + L) % N
    down = (i - L) % N
    return int(spins[right] + spins[left] + spins[up] + spins[down])


# Wolff cluster update

def wolff_step(spins, rng, p_add, cur_m, cur_e, stack):
    """
    Grow and flip one Wolff cluster.

    M and E are updated incrementally, following wolff.cpp.
    Returns updated M, updated E and cluster size.
    """
    seed = int(rng.integers(N))
    old_spin = int(spins[seed])
    new_spin = -old_spin

    cluster_size = 1
    stack.clear()
    stack.append(seed)

    # Flip the seed immediately so it cannot be added twice.
    cur_e += 2 * J * old_spin * nn_sum(spins, seed)
    spins[seed] = new_spin

    while stack:
        current = stack.pop()

        # the same flattened indexing as the C++ code.
        neighbours = (
            (current + 1) % N,
            (current - 1) % N,
            (current + L) % N,
            (current - L) % N,
        )

        for nn in neighbours:
            if spins[nn] == old_spin and rng.random() < p_add:
                stack.append(nn)
                cluster_size += 1
                cur_e += 2 * J * old_spin * nn_sum(spins, nn)
                spins[nn] = new_spin

    cur_m += 2 * new_spin * cluster_size
    return cur_m, cur_e, cluster_size


# Random streams and pilot calibration

def make_rng(stage, t_index, run_index):
    """
    Independent reproducible streams, with separate pilot/production
    stages. NumPy's generator is not bit-for-bit identical to mt19937
    plus C++ seed_seq, but the stream structure is analogous.
    """
    seed = np.random.SeedSequence(
        [BASE_SEED, L, stage, t_index, run_index]
    )
    return np.random.default_rng(seed)


def calibrate(T, t_index, p_add):
    """Pilot equilibration and mean cluster-size calibration."""
    rng = make_rng(0, t_index, 0)
    spins = initialise_spins(T, rng)
    cur_m = magnetisation(spins)
    cur_e = energy(spins)
    stack = []

    # Pilot equilibration, budgeted by number of flipped spins.
    flips = 0
    eq_budget = PILOT_EQ_SWEEPS * N
    while flips < eq_budget:
        cur_m, cur_e, size = wolff_step(
            spins, rng, p_add, cur_m, cur_e, stack
        )
        flips += size

    # Measure mean cluster size, again using a spin-flip budget.
    flips = 0
    steps = 0
    meas_budget = PILOT_MEAS_SWEEPS * N
    while flips < meas_budget:
        cur_m, cur_e, size = wolff_step(
            spins, rng, p_add, cur_m, cur_e, stack
        )
        flips += size
        steps += 1

    nbar = flips / steps
    kT = max(1, int(round(N / nbar)))
    return nbar, kT


# 5. Output helpers

def write_metadata(base_dir, nbar_kt):
    base_dir.mkdir(parents=True, exist_ok=True)
    with (base_dir / "metadata.txt").open("w") as f:
        f.write(f"L {L}\n")
        f.write(f"nT {len(TEMPS)}\n")
        f.write(f"Tmin {TEMPS[0]}\n")
        f.write(f"Tmax {TEMPS[-1]}\n")
        f.write("temps " + " ".join(f"{T:g}" for T in TEMPS) + "\n")
        f.write(f"nRuns {NRUNS}\n")
        f.write(f"nsweeps {NSWEEPS}  (samples per run, ~N flipped spins each)\n")
        f.write(f"burnInSamples {BURN_IN_SAMPLES}\n")
        f.write(f"pilotEqSweeps {PILOT_EQ_SWEEPS}\n")
        f.write(f"pilotMeasSweeps {PILOT_MEAS_SWEEPS}\n")
        f.write(f"baseSeed {BASE_SEED}\n")
        f.write("engine numpy.default_rng (PCG64)\n")
        f.write("seeding SeedSequence{baseSeed, L, stage, t_index, run_index}\n")
        f.write("data columns: sample M E blockClusterSum (M, E are totals)\n")
        f.write("# per-temperature calibration: T kT nbar\n")
        for T, kT, nbar in nbar_kt:
            f.write(f"{T:.5f} {kT} {nbar:.10g}\n")


def write_trace(path, kind, T, run, t_index, kT, nbar, rows):
    with path.open("w") as f:
        f.write(f"# {kind}\n")
        f.write(f"# L {L}\n")
        f.write(f"# N {N}\n")
        f.write(f"# T {T:.5f}\n")
        f.write(f"# run {run}\n")
        f.write(f"# seed_seq {{{BASE_SEED}, {L}, 1, {t_index}, {run}}}\n")
        f.write(f"# start {'cold' if T < TC else 'hot'}\n")
        f.write(f"# kT {kT}   (wolff steps per sample)\n")
        f.write(f"# nbar {nbar:.10g}\n")
        f.write(f"# burnInSamples {BURN_IN_SAMPLES}\n")
        f.write(f"# nrows {len(rows)}\n")
        f.write("# columns: sample M E blockClusterSum\n")
        for sample, m, e, block_sum in rows:
            f.write(f"{sample} {m} {e} {block_sum}\n")


# 6. Main simulation

def main():
    base_dir = (
        DATA_DIR / f"L{L}" / f"nsweeps{NSWEEPS}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    calibrations = []
    for t_index, T in enumerate(TEMPS):
        p_add = 1.0 - math.exp(-(2.0 * J) / T)
        nbar, kT = calibrate(T, t_index, p_add)
        calibrations.append((T, kT, nbar))
        print(
            f"T={T:.3f}  nbar={nbar:.3f}  kT={kT}  "
            f"samples/run={NSWEEPS}"
        )

    write_metadata(base_dir, calibrations)

    for t_index, (T, kT, nbar) in enumerate(calibrations):
        p_add = 1.0 - math.exp(-(2.0 * J) / T)
        t_dir = base_dir / f"T{T:.5f}"
        t_dir.mkdir(parents=True, exist_ok=True)

        for run in range(NRUNS):
            start_time = time.time()
            rng = make_rng(1, t_index, run)
            spins = initialise_spins(T, rng)
            cur_m = magnetisation(spins)
            cur_e = energy(spins)
            stack = []

            # Burn-in; save the trace only for run zero.
            burn_rows = []
            for b in range(BURN_IN_SAMPLES):
                block_sum = 0
                for _ in range(kT):
                    cur_m, cur_e, size = wolff_step(
                        spins, rng, p_add, cur_m, cur_e, stack
                    )
                    block_sum += size
                if run == 0:
                    burn_rows.append((b, cur_m, cur_e, block_sum))

            if run == 0:
                write_trace(
                    t_dir / "burnin_run000.dat",
                    "burn-in trace (not used for statistics)",
                    T, run, t_index, kT, nbar, burn_rows
                )

            # Production: one recorded row per kT cluster updates.
            rows = []
            print_every = max(1, NSWEEPS // 100)

            for sample in range(NSWEEPS):
                block_sum = 0
                for _ in range(kT):
                    cur_m, cur_e, size = wolff_step(
                        spins, rng, p_add, cur_m, cur_e, stack
                    )
                    block_sum += size

                rows.append((sample, cur_m, cur_e, block_sum))

                if (sample + 1) % print_every == 0 or sample + 1 == NSWEEPS:
                    pct = 100.0 * (sample + 1) / NSWEEPS
                    print(
                        f"\r  run {run + 1}/{NRUNS}  {pct:5.1f}%",
                        end="", flush=True
                    )

            print(f"  ({time.time() - start_time:.1f} s)")

            # Check incremental bookkeeping against full recalculation.
            if cur_e != energy(spins) or cur_m != magnetisation(spins):
                raise RuntimeError(
                    f"Bookkeeping mismatch at T={T} run {run}"
                )

            write_trace(
                t_dir / f"run{run:03d}.dat",
                "production samples",
                T, run, t_index, kT, nbar, rows
            )

        print()

    print(f"Python Wolff data written to: {base_dir}")


if __name__ == "__main__":
    main()
