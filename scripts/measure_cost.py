"""Measure what each optimizer costs per training step, on the training model and data.

For every width and optimizer setting, on one GPU, one setting at a time:
  - FLOPs, counted by torch.utils.flop_counter.FlopCounterMode, for the model (forward +
    backward) and for each optimizer step. The counter sees matrix products only; the
    element-wise work of AdamW, Signum and SGD is not counted, so their optimizer FLOPs read
    as zero and their cost shows up in the timings instead.
  - Time, by CUDA events: the model's forward + backward, the step of the optimizer under
    study (block matrices), and the AdamW step for everything else (the same in every setting).
    Median over timed steps after warm-up.
  - Memory: bytes of optimizer state held for the block matrices.

usage: python scripts/measure_cost.py reports/figs/cost [--steps 200] [--widths 64 128 256 512]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icl.data import sample  # noqa: E402
from icl.model import ICLTransformer  # noqa: E402
from icl.optim import Routed  # noqa: E402

SETTINGS = [  # (key, label, opt, lr, ns_steps, fro_match)
    ("sgdm", "SGD with momentum", "sgdm", 0.1, 5, False),
    ("adamw", "AdamW", "adamw", 3e-4, 5, False),
    ("adamw_fn", "Adam direction, Muon step size", "adamw_fn", 0.005, 5, False),
    ("signum", "Signum", "signum", 3e-4, 5, False),
    ("muon_ns0fm", "Muon, 0 Newton-Schulz steps", "muon", 0.005, 0, True),
    ("muon_ns2fm", "Muon, 2 Newton-Schulz steps", "muon", 0.005, 2, True),
    ("muon", "Muon (5 Newton-Schulz steps)", "muon", 0.005, 5, False),
]


def state_bytes(opt):
    n = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                n += v.numel() * v.element_size()
    return n


def measure(width, key, opt_name, lr, ns, fm, steps, dev):
    torch.manual_seed(0)
    model = ICLTransformer(8, 16, 8, width, 2).to(dev)
    opt = Routed(model, opt_name, lr, 3e-4, ns_steps=ns, fro_match=fm)
    gen = torch.Generator(device=dev).manual_seed(1)

    def batch():
        x, y, _ = sample(256, 16, 8, 0.25, None, gen, dev)
        return x, y

    def fwd_bwd(x, y):
        loss = ((model(x, y) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()

    # FLOPs (after one step so optimizer state exists)
    x, y = batch(); fwd_bwd(x, y); opt.step()
    x, y = batch()
    with FlopCounterMode(display=False) as fc:
        fwd_bwd(x, y)
    f_model = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        opt.hidden.step()
    f_hidden = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        opt.aux.step()
    f_aux = fc.get_total_flops()

    # time
    ev = lambda: torch.cuda.Event(enable_timing=True)
    t_model, t_hidden, t_aux = [], [], []
    for i in range(steps + 30):
        x, y = batch()
        e0, e1, e2, e3 = ev(), ev(), ev(), ev()
        e0.record(); fwd_bwd(x, y); e1.record(); opt.hidden.step(); e2.record(); opt.aux.step(); e3.record()
        torch.cuda.synchronize()
        if i >= 30:
            t_model.append(e0.elapsed_time(e1)); t_hidden.append(e1.elapsed_time(e2)); t_aux.append(e2.elapsed_time(e3))
    n_block = sum(p.numel() for p in model.hidden_matrices().values())
    return dict(width=width, setting=key, n_params=sum(p.numel() for p in model.parameters()), n_block=n_block,
                flops_model=f_model, flops_opt=f_hidden, flops_aux=f_aux,
                ms_model=float(np.median(t_model)), ms_opt=float(np.median(t_hidden)), ms_aux=float(np.median(t_aux)),
                state_bytes=state_bytes(opt.hidden))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figs")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512])
    a = ap.parse_args()
    os.makedirs(a.figs, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False      # as in training
    res = []
    for W in a.widths:
        for key, _, o, lr, ns, fm in SETTINGS:
            r = measure(W, key, o, lr, ns, fm, a.steps, dev)
            res.append(r)
            print(json.dumps(r), flush=True)
    json.dump(dict(device=torch.cuda.get_device_name(), torch=torch.__version__, steps=a.steps, results=res),
              open(os.path.join(a.figs, "cost.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
