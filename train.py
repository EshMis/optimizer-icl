"""Train one transformer on in-context linear regression with one optimizer.

Writes <out>/<run_id>/: config.json, baselines.json, metrics.jsonl (one row per eval,
flushed as it goes), ckpt/step_*.pt at log-spaced steps, and done.json LAST.
Seeds: `pool_seed` fixes the task pool and eval prompts, `seed` fixes init and the training
data stream. Runs that differ only in optimizer see identical pools, inits and batches.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from icl.data import make_pool, sample
from icl.metrics import EvalSet, evaluate
from icl.model import ICLTransformer
from icl.optim import Routed


def run_id(c):
    m = "inf" if c["M"] == 0 else str(c["M"])
    tag = f"{c['opt']}_lr{c['lr']:g}_aux{c['aux_lr']:g}_wd{c['wd']:g}_M{m}_s{c['seed']}_p{c['pool_seed']}"
    tag += f"_L{c['n_layer']}D{c['n_embd']}H{c['n_head']}_d{c['d']}K{c['K']}_T{c['steps']}_B{c['batch']}_{c['sched']}"
    if c.get("route", "all") != "all":
        tag += f"_route{c['route']}"
    if c["opt"] == "signum" and c.get("fro_match"):
        tag += "_fm"
    elif c.get("ns_steps", 5) != 5 or c.get("fro_match"):
        tag += f"_ns{c['ns_steps']}" + ("fm" if c.get("fro_match") else "")
    return tag + (f"_{c['tag']}" if c["tag"] else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", default="adamw", choices=["adamw", "muon", "signum", "sgdm", "adamw_fn"])
    ap.add_argument("--lr", type=float, default=1e-3, help="LR for block weight matrices")
    ap.add_argument("--aux_lr", type=float, default=1e-3, help="AdamW LR for all other parameters")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--M", type=int, default=0, help="task pool size; 0 = fresh task every prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool_seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--noise_var", type=float, default=0.25)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--n_head", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--sched", default="const", choices=["const", "cosine"])
    ap.add_argument("--n_eval_log", type=int, default=60, help="log-spaced evals (plus every 1%%)")
    ap.add_argument("--n_ckpt", type=int, default=10)
    ap.add_argument("--route", default="all", choices=["all", "attn", "mlp"], help="which block matrices get --opt; the rest get AdamW at --aux_lr")
    ap.add_argument("--ns_steps", type=int, default=5, help="Muon Newton-Schulz iterations (0 = normalised momentum)")
    ap.add_argument("--fro_match", action="store_true", help="Muon: rescale every update to full Muon's Frobenius norm")
    ap.add_argument("--eval_pct", type=float, default=1.0, help="evaluate every this many percent of training (plus log-spaced)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c = vars(ap.parse_args())
    rid = run_id(c)
    od = os.path.join(c["out"], rid)
    if os.path.exists(os.path.join(od, "done.json")):
        print("already done", rid)
        return
    os.makedirs(os.path.join(od, "ckpt"), exist_ok=True)
    dev = torch.device(c["device"])
    torch.backends.cuda.matmul.allow_tf32 = False   # keep fp32 so distances to ridge are not precision noise
    torch.manual_seed(c["seed"])

    model = ICLTransformer(c["d"], c["K"], c["n_layer"], c["n_embd"], c["n_head"]).to(dev)
    c["n_params"] = sum(p.numel() for p in model.parameters())
    W0 = {n: p.detach().clone() for n, p in model.hidden_matrices().items()}
    opt = Routed(model, c["opt"], c["lr"], c["aux_lr"], c["wd"], route=c["route"], ns_steps=c["ns_steps"], fro_match=c["fro_match"])
    hid = model.hidden_matrices()
    prev = {n: p.detach().clone() for n, p in hid.items()}
    is_attn = torch.tensor([n.split(".")[2] in ("q", "k", "v", "o") for n in hid], device=dev)
    path_acc = torch.zeros(2, device=dev)   # cumulative Frobenius length of the weight path: [attn, mlp]
    pool = make_pool(c["d"], c["M"], c["pool_seed"])
    pool_dev = pool.to(dev) if pool is not None else None
    ev = EvalSet(c, pool, dev)
    json.dump(c, open(os.path.join(od, "config.json"), "w"), indent=1)
    json.dump(ev.baselines(), open(os.path.join(od, "baselines.json"), "w"))

    T = c["steps"]
    evals = set(np.unique(np.geomspace(10, T, c["n_eval_log"]).astype(int)).tolist())
    evals |= set(range(0, T + 1, max(1, int(T * c["eval_pct"] / 100))))
    ckpts = set(np.unique(np.geomspace(100, T, c["n_ckpt"]).astype(int)).tolist()) | {T}
    gen = torch.Generator(device=dev).manual_seed(1_000_003 * c["seed"] + 7)
    mf = open(os.path.join(od, "metrics.jsonl"), "w")
    ema, t0, status = None, time.time(), "ok"

    for step in range(T + 1):
        if step in evals:
            row = dict(step=step, train_loss=ema, elapsed=time.time() - t0, path_attn=path_acc[0].item(), path_mlp=path_acc[1].item(),
                       **evaluate(model, ev, W0))
            mf.write(json.dumps(row) + "\n")
            mf.flush()
        if step in ckpts:
            torch.save(model.state_dict(), os.path.join(od, "ckpt", f"step_{step}.pt"))
        if step == T:
            break
        if step < c["warmup"]:
            mult = (step + 1) / c["warmup"]
        elif c["sched"] == "cosine":
            mult = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - c["warmup"]) / (T - c["warmup"])))
        else:
            mult = 1.0
        opt.set_lr_mult(mult)
        x, y, _ = sample(c["batch"], c["K"], c["d"], c["noise_var"], pool_dev, gen, dev)
        loss = ((model(x, y) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            d = torch.stack([(p - prev[n]).norm() for n, p in hid.items()])
            path_acc[0] += d[is_attn].sum()
            path_acc[1] += d[~is_attn].sum()
            for n, p in hid.items():
                prev[n].copy_(p)
        lv = loss.item() if step % 50 == 0 else None
        if lv is not None:
            if not math.isfinite(lv):
                status = f"diverged at step {step}"
                break
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
        if step % 5000 == 0:
            print(f"{rid} step {step} loss {ema:.4f} {step / max(1e-9, time.time() - t0):.0f} it/s", flush=True)
    mf.close()
    json.dump(dict(status=status, seconds=time.time() - t0, steps=step), open(os.path.join(od, "done.json"), "w"))
    print("DONE", rid, status, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
