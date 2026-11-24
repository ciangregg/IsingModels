"""Run the Wolff FSS simulations: compile one binary per L, then run every
(L, T) as its own job in parallel, skipping temperatures that are already done.

examples
    python fss_run.py --dry-run                 # show the plan, run nothing
    python fss_run.py --ls 16 24 32             # pilot
    python fss_run.py                           # full sweep with the settings below
    python fss_run.py --nsamples 20000 --workers 3

Seeds depend only on (L, T, run), so splitting the work into one job per
(L, T) gives exactly the same data as running a whole L in one go.
Ctrl+C stops cleanly: finished temperatures are kept, rerun to carry on.
"""

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fss_tools import (BIN_DIR, DATA_DIR, LOG_DIR, SRC, TEMPS_DIR, binary,
                       is_done, make_temps, t_name)


# settings (we can override all in the terminal)
#LS = [16, 24, 32, 48, 64, 96, 128,256]
LS = [16, 24, 32, 48, 64, 96, 128, 160]

N_RUNS = 10
N_SAMPLES = 10_000  # samples per run, N flipped spins each
WORKERS = max(1, (os.cpu_count() or 2) - 1)

CXX = "g++"
CXXFLAGS = ["-std=c++17", "-O3", "-march=native", "-Wall", "-Wextra"]
LDLIBS = ["-lgsl", "-lgslcblas", "-lm"]

# extra flags without editing this file, e.g. FSS_CXXFLAGS="-I/opt/gsl/include"
CXXFLAGS += shlex.split(os.environ.get("FSS_CXXFLAGS", ""))
if "FSS_LDLIBS" in os.environ:
    LDLIBS = shlex.split(os.environ["FSS_LDLIBS"])


# compile
def compile_L(L, force=False):
    #Build bin/wolff_L<L>, only if missing or older than wolff.cpp
    exe = binary(L)
    if not force and exe.exists() and exe.stat().st_mtime >= SRC.stat().st_mtime:
        return
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [CXX, *CXXFLAGS, f"-DL={L}", str(SRC), "-o", str(exe), *LDLIBS]
    print(f"compiling L={L}: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"compile failed for L={L}:\n{res.stderr}")


# jobs
running = set() # live processes, for Ctrl+C
running_lock = threading.Lock()


def cost(L, nruns, nsamples):
    #Relative cost of one (L, T) job: spin flips ~ N * samples * runs
    return L * L * nsamples * nruns


def run_job(L, T, nruns, nsamples):
    #Run one (L, T) and return (L, T, returncode, seconds, log path)
    tag = f"L{L}_{t_name(T)}"
    temps_file = TEMPS_DIR / f"{tag}.txt"
    log_file = LOG_DIR / f"{tag}.log"
    temps_file.write_text(f"{T:.5f}\n")

    cmd = [str(binary(L)), str(temps_file), str(nruns), str(nsamples), str(DATA_DIR)]
    t0 = time.time()
    with open(log_file, "w") as log:
        log.write("# " + " ".join(cmd) + "\n")
        log.flush()
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        with running_lock:
            running.add(p)
        rc = p.wait()
        with running_lock:
            running.discard(p)

    # wolff.cpp writes meta.txt last so a zero exit without it is still a failure
    if rc == 0 and not is_done(L, T, nsamples):
        rc = -999
    return L, T, rc, time.time() - t0, log_file


def fmt_time(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


# main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ls", type=int, nargs="+", default=LS, help="lattice sizes")
    ap.add_argument("--nruns", type=int, default=N_RUNS)
    ap.add_argument("--nsamples", type=int, default=N_SAMPLES)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--dry-run", action="store_true", help="show the plan, run nothing")
    ap.add_argument("--force-compile", action="store_true", help="rebuild all binaries")
    args = ap.parse_args()

    Ls = sorted(set(args.ls), reverse=True)        # largest first

    # plan
    jobs, total_cost = [], 0
    print(f"nRuns={args.nruns}  nsamples={args.nsamples}  workers={args.workers}")
    print(f"{'L':>5} {'temps':>6} {'done':>5} {'todo':>5} {'share of cost':>14}")
    plan = []
    for L in Ls:
        temps = make_temps(L)
        todo = [T for T in temps if not is_done(L, T, args.nsamples)]
        c = cost(L, args.nruns, args.nsamples) * len(todo)
        plan.append((L, len(temps), len(temps) - len(todo), len(todo), c))
        total_cost += c
        jobs += [(L, T) for T in todo]
    for L, n, done, todo, c in plan:
        share = c / total_cost if total_cost else 0
        print(f"{L:>5} {n:>6} {done:>5} {todo:>5} {share:>13.1%}")
    print(f"{len(jobs)} jobs to run")

    if args.dry_run or not jobs:
        return

    for d in (DATA_DIR, LOG_DIR, TEMPS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for L in sorted({L for L, _ in jobs}):
        compile_L(L, force=args.force_compile)

    # run
    t_start = time.time()
    done_cost, job_secs, failures = 0, 0.0, []
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = [pool.submit(run_job, L, T, args.nruns, args.nsamples) for L, T in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            L, T, rc, secs, log = fut.result()
            done_cost += cost(L, args.nruns, args.nsamples)
            job_secs += secs
            elapsed = time.time() - t_start
            # time left: remaining work at the measured seconds-per-flip of
            # finished jobs, shared over the workers (fewer if few jobs remain)
            # (elapsed/done would overestimate early on, while many jobs are
            # part-way through but none have finished)
            n_left = len(jobs) - i
            eta = (total_cost - done_cost) * (job_secs / done_cost) / max(1, min(args.workers, n_left))
            status = "ok" if rc == 0 else f"FAILED (exit {rc}), see {log}"
            if rc != 0:
                failures.append((L, T, log))
            print(f"[{i}/{len(jobs)}] L={L:<4} T={T:.5f}  {fmt_time(secs)}  {status}"
                  f"   elapsed {fmt_time(elapsed)}, ~{fmt_time(eta)} left", flush=True)
    except KeyboardInterrupt:
        print("\nstopping: cancelling queued jobs and killing running ones...")
        pool.shutdown(wait=False, cancel_futures=True)
        with running_lock:
            for p in running:
                p.terminate()
        print("finished temperatures are kept; rerun the same command to carry on")
        sys.exit(1)
    pool.shutdown()

    print(f"\nall jobs finished in {fmt_time(time.time() - t_start)}")
    if failures:
        print(f"{len(failures)} failed:")
        for L, T, log in failures:
            print(f"  L={L} T={T:.5f}  log: {log}")
        sys.exit(1)


if __name__ == "__main__":
    main()