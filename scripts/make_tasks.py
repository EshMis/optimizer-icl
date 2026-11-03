"""Write a task file: the cross product of optimizers x pool sizes x seeds.

usage: python scripts/make_tasks.py tasks/run02_main.txt --opt adamw:1e-3:1e-3 muon:0.02:1e-3 \
         --M 0 16 64 256 1024 4096 16384 --seeds 0 1 2 3 4 5 --steps 30000 --comment "Run 2 ..."
Each --opt entry is name:lr:aux_lr. pool_seed = seed, so seed s pairs the two optimizers
on the same pool, init and batches.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--opt", nargs="+", required=True)
ap.add_argument("--M", nargs="+", type=int, required=True)
ap.add_argument("--seeds", nargs="+", type=int, required=True)
ap.add_argument("--steps", type=int, required=True)
ap.add_argument("--extra", default="", help="appended to every line, e.g. '--wd 0.1 --tag wd'")
ap.add_argument("--comment", default="")
a = ap.parse_args()
lines = []
for s in a.seeds:                      # seed-major so a partial sweep still has complete pairs
    for M in a.M:
        for spec in a.opt:
            o, lr, aux = spec.split(":")
            lines.append(f"--opt {o} --lr {lr} --aux_lr {aux} --M {M} --seed {s} --pool_seed {s} "
                         f"--steps {a.steps} {a.extra}".strip())
with open(a.out, "w") as f:
    if a.comment:
        f.write(f"# {a.comment}\n")
    f.write("\n".join(lines) + "\n")
print(len(lines), "runs ->", a.out)
