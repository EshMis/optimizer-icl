"""Train one transformer on in-context substitution ciphers with one optimizer.

Same layout as train.py: <out>/<run_id>/ with config.json, baselines.json, metrics.jsonl and
done.json. Metrics per evaluation, on prompts from the pool ("id") and fresh ciphers ("ood"):
  ce_{set}, acc_{set}            cross-entropy and accuracy over all positions
  ce_{set}_seen / _unseen        split by whether p_k already appeared in the prompt
  acc_{set}_seen / _unseen
  alpha_{set}                    least-squares weight of the model's probabilities between the
                                 in-context decoder (0) and the pool memoriser (1)
Baselines hold the same quantities for the two reference predictors.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from icl.cipher import icl_preds, make_pool, pool_preds, sample
from icl.cipher_model import CipherTransformer
from icl.optim import Routed


def run_id(c):
    m = "inf" if c["M"] == 0 else str(c["M"])
    tag = f"cipher_{c['opt']}_lr{c['lr']:g}_aux{c['aux_lr']:g}_M{m}_s{c['seed']}_p{c['pool_seed']}"
    tag += f"_L{c['n_layer']}D{c['n_embd']}H{c['n_head']}_V{c['V']}K{c['K']}_T{c['steps']}_B{c['batch']}"
    if c["route"] != "all":
        tag += f"_route{c['route']}"
    return tag + (f"_{c['tag']}" if c["tag"] else "")


def split_stats(logp, target, seen, prefix):
    """CE and accuracy over all / seen / unseen positions."""
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    acc = (logp.argmax(-1) == target).float()
    out = {f"ce_{prefix}": nll.mean().item(), f"acc_{prefix}": acc.mean().item()}
    for name, m in (("seen", seen), ("unseen", ~seen)):
        if m.any():
            out[f"ce_{prefix}_{name}"] = nll[m].mean().item()
            out[f"acc_{prefix}_{name}"] = acc[m].mean().item()
    return out


class EvalSet:
    def __init__(self, c, pool, device, n=1024, seed=10_000):
        g = torch.Generator(device=device).manual_seed(seed + c["pool_seed"])
        V, K = c["V"], c["K"]
        pool_dev = pool.to(device) if pool is not None else None
        self.sets = {}
        for name, p in (("id", pool_dev), ("ood", None)):
            plain, code, perm = sample(n, K, V, p, g, device)
            oh = F.one_hot(plain, V).float()
            seen = ((torch.cumsum(oh, 1) - oh).gather(-1, plain.unsqueeze(-1)).squeeze(-1) > 0)
            q_icl = icl_preds(plain, code, V)
            q_mem = pool_preds(plain, code, pool_dev) if pool_dev is not None else q_icl
            self.sets[name] = dict(plain=plain, code=code, seen=seen, icl=q_icl, mem=q_mem)

    def baselines(self):
        out = {}
        for name, s in self.sets.items():
            for ref in ("icl", "mem"):
                out.update(split_stats(torch.log(s[ref].clamp_min(1e-9)), s["code"], s["seen"], f"{ref}_{name}"))
        return out


@torch.no_grad()
def evaluate(model, ev):
    model.eval()
    out = {}
    for name, s in ev.sets.items():
        logits = model(s["plain"], s["code"])
        logp = F.log_softmax(logits, -1)
        out.update(split_stats(logp, s["code"], s["seen"], name))
        p = logp.exp()
        d = s["mem"] - s["icl"]
        den = (d ** 2).sum().item()
        out[f"alpha_{name}"] = (((p - s["icl"]) * d).sum().item() / den) if den > 1e-9 else float("nan")
        out[f"dist_icl_{name}"] = ((p - s["icl"]) ** 2).sum(-1).mean().item()
        out[f"dist_mem_{name}"] = ((p - s["mem"]) ** 2).sum(-1).mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", default="adamw", choices=["adamw", "muon", "signum", "sgdm", "adamw_fn"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux_lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--M", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool_seed", type=int, default=0)
    ap.add_argument("--V", type=int, default=16, help="alphabet size")
    ap.add_argument("--K", type=int, default=24, help="pairs per prompt")
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--n_head", type=int, default=2)
    ap.add_argument("--steps", type=int, default=15_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--route", default="all", choices=["all", "attn", "mlp"])
    ap.add_argument("--eval_pct", type=float, default=2.0)
    ap.add_argument("--n_eval_log", type=int, default=60)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c = vars(ap.parse_args())
    rid = run_id(c)
    od = os.path.join(c["out"], rid)
    if os.path.exists(os.path.join(od, "done.json")):
        print("already done", rid)
        return
    os.makedirs(od, exist_ok=True)
    dev = torch.device(c["device"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(c["seed"])

    model = CipherTransformer(c["V"], c["K"], c["n_layer"], c["n_embd"], c["n_head"]).to(dev)
    c["n_params"] = sum(p.numel() for p in model.parameters())
    opt = Routed(model, c["opt"], c["lr"], c["aux_lr"], c["wd"], route=c["route"])
    pool = make_pool(c["V"], c["M"], c["pool_seed"])
    pool_dev = pool.to(dev) if pool is not None else None
    ev = EvalSet(c, pool, dev)
    json.dump(c, open(os.path.join(od, "config.json"), "w"), indent=1)
    json.dump(ev.baselines(), open(os.path.join(od, "baselines.json"), "w"))

    T = c["steps"]
    evals = set(np.unique(np.geomspace(10, T, c["n_eval_log"]).astype(int)).tolist())
    evals |= set(range(0, T + 1, max(1, int(T * c["eval_pct"] / 100))))
    gen = torch.Generator(device=dev).manual_seed(1_000_003 * c["seed"] + 7)
    mf = open(os.path.join(od, "metrics.jsonl"), "w")
    ema, t0, status = None, time.time(), "ok"
    for step in range(T + 1):
        if step in evals:
            mf.write(json.dumps(dict(step=step, train_loss=ema, elapsed=time.time() - t0, **evaluate(model, ev))) + "\n")
            mf.flush()
        if step == T:
            break
        opt.set_lr_mult((step + 1) / c["warmup"] if step < c["warmup"] else 1.0)
        plain, code, _ = sample(c["batch"], c["K"], c["V"], pool_dev, gen, dev)
        logits = model(plain, code)
        loss = F.cross_entropy(logits.reshape(-1, c["V"]), code.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            lv = loss.item()
            if not math.isfinite(lv):
                status = f"diverged at step {step}"
                break
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
        if step % 5000 == 0:
            print(f"{rid} step {step} loss {ema:.4f}", flush=True)
    mf.close()
    json.dump(dict(status=status, seconds=time.time() - t0, steps=step), open(os.path.join(od, "done.json"), "w"))
    print("DONE", rid, status, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
