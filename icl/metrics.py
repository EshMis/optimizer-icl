"""Measurements taken at every evaluation step.

Fixed evaluation prompts are drawn once per run: `id` prompts use tasks from the training
pool, `ood` prompts use fresh tasks. The two reference predictors are computed on the same
prompts once, so every checkpoint is compared against identical targets.

alpha: least-squares weight in model ~ alpha * dMMSE + (1 - alpha) * ridge. alpha = 1 is
pure memorisation of the pool, alpha = 0 is in-context regression.
"""
import math

import torch

from .data import dmmse_preds, ridge_preds, sample


class EvalSet:
    def __init__(self, cfg, pool, device, n=2048, seed=10_000):
        g = torch.Generator(device=device).manual_seed(seed + cfg["pool_seed"])
        K, d, nv = cfg["K"], cfg["d"], cfg["noise_var"]
        self.sets = {}
        pool_dev = pool.to(device) if pool is not None else None
        for name, p in (("id", pool_dev), ("ood", None)):
            x, y, w = sample(n, K, d, nv, p, g, device)
            r = ridge_preds(x, y, nv)
            dm = dmmse_preds(x, y, pool_dev, nv) if pool_dev is not None else r
            self.sets[name] = dict(x=x, y=y, w=w, ridge=r, dmmse=dm)

    def baselines(self):
        out = {}
        for name, s in self.sets.items():
            for ref in ("ridge", "dmmse"):
                out[f"{ref}_{name}_mse_pos"] = ((s[ref] - s["y"]) ** 2).mean(0).tolist()
        return out


def _spec(W):
    s = torch.linalg.svdvals(W.float())
    p = s / s.sum()
    erank = math.exp(-(p * torch.log(p + 1e-12)).sum().item())
    srank = (s.pow(2).sum() / s[0].pow(2)).item()
    return srank, erank, s[0].item()


@torch.no_grad()
def evaluate(model, ev, W0, n_attn=256):
    model.eval()
    out = {}
    for name, s in ev.sets.items():
        f = model(s["x"], s["y"])
        mse_pos = ((f - s["y"]) ** 2).mean(0)
        out[f"mse_{name}_pos"] = mse_pos.tolist()
        out[f"mse_{name}_last"] = mse_pos[-1].item()
        out[f"mse_{name}_mean"] = mse_pos.mean().item()
        sl = slice(1, None)  # position 0 has no context: every predictor answers ~0
        r, dm = s["ridge"][:, sl], s["dmmse"][:, sl]
        fm = f[:, sl]
        out[f"dist_ridge_{name}"] = ((fm - r) ** 2).mean().item()
        out[f"dist_dmmse_{name}"] = ((fm - dm) ** 2).mean().item()
        den = ((dm - r) ** 2).sum().item()
        out[f"alpha_{name}"] = (((fm - r) * (dm - r)).sum().item() / den) if den > 1e-9 else float("nan")
        # alpha per context length (how memorisation vs regression mix changes with k)
        num_k = ((f - s["ridge"]) * (s["dmmse"] - s["ridge"])).sum(0)
        den_k = ((s["dmmse"] - s["ridge"]) ** 2).sum(0).clamp_min(1e-9)
        out[f"alpha_{name}_pos"] = (num_k / den_k).tolist()

    # attention: entropy and mass on y-tokens, read at the x-query tokens that make predictions
    s = ev.sets["ood"]
    att = []
    model(s["x"][:n_attn], s["y"][:n_attn], attn_out=att)
    T = att[0].shape[-1]
    q_idx = torch.arange(2, T, 2, device=att[0].device)              # x-tokens with context
    is_y = (torch.arange(T, device=att[0].device) % 2 == 1).float()
    ent, ymass = [], []
    for p in att:                                                     # (B,H,T,T)
        pq = p[:, :, q_idx]
        ent.append((-(pq * torch.log(pq + 1e-12)).sum(-1)).mean().item())
        ymass.append((pq * is_y).sum(-1).mean().item())
    out["attn_entropy"] = ent
    out["attn_ymass"] = ymass

    # spectra of block weight matrices and of their change since initialisation
    spec = {}
    for n, W in model.hidden_matrices().items():
        a = _spec(W)
        b = _spec(W - W0[n])
        spec[n.replace("blocks.", "").replace(".weight", "")] = [round(v, 5) for v in (*a, *b)]
    out["spec"] = spec  # name: [srank, erank, s1, dsrank, derank, ds1]
    model.train()
    return out
