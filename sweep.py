"""Run every line of a task file (train.py arguments) with N processes sharing one GPU.

A run whose done.json exists is skipped by train.py, so a resubmitted sweep resumes.
Each run's stdout/stderr goes to <logdir>/<line number>.log.
usage: python sweep.py tasks/run01.txt --parallel 8 --out results --logdir logs/run01
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("tasks")
ap.add_argument("--parallel", type=int, default=8)
ap.add_argument("--out", default="results")
ap.add_argument("--logdir", default="logs")
a = ap.parse_args()
os.makedirs(a.logdir, exist_ok=True)
lines = [ln.strip() for ln in open(a.tasks) if ln.strip() and not ln.startswith("#")]
here = os.path.dirname(os.path.abspath(__file__))
queue = list(enumerate(lines, 1))
running, failed = {}, 0
t0 = time.time()
print(f"{len(lines)} runs, {a.parallel} at a time", flush=True)
while queue or running:
    while queue and len(running) < a.parallel:
        i, ln = queue.pop(0)
        log = open(os.path.join(a.logdir, f"{i:03d}.log"), "w")
        cmd = [sys.executable, os.path.join(here, "train.py"), *shlex.split(ln), "--out", a.out]
        running[i] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=here), log)
    time.sleep(5)
    for i in list(running):
        p, log = running[i]
        if p.poll() is not None:
            log.close()
            failed += p.returncode != 0
            print(f"[{time.time() - t0:7.0f}s] task {i} exit {p.returncode}", flush=True)
            del running[i]
print(f"SWEEP DONE failed={failed} seconds={time.time() - t0:.0f}", flush=True)
sys.exit(1 if failed else 0)
