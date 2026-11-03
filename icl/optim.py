"""Optimizer routing: block weight matrices get the optimizer under study, everything else
(read-in, read-out, positional embedding, LayerNorms, biases) always gets AdamW.

So "adamw" vs "muon" differ only in how the attention and MLP matrices are updated.
Muon follows Keller Jordan's reference (github.com/KellerJordan/Muon): Nesterov momentum,
5 quintic Newton-Schulz steps in bf16, update scaled by sqrt(max(1, rows/cols)).
Same-shaped matrices are orthogonalised in one batched call.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda else G.float()   # bf16 matmul is slow on CPU
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if tall:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0,
                 orthogonalize=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay, orthogonalize=orthogonalize))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            by_shape = {}
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.lerp_(p.grad, 1 - g["momentum"])
                u = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf.clone()
                by_shape.setdefault(tuple(p.shape), []).append((p, u))
            for shape, items in by_shape.items():
                U = torch.stack([u for _, u in items])
                if g["orthogonalize"]:
                    U = zeropower_via_newtonschulz5(U, g["ns_steps"]) * max(1, shape[0] / shape[1]) ** 0.5
                else:  # "signum" ablation: same momentum, elementwise sign instead of orthogonalisation
                    U = U.sign()
                for (p, _), du in zip(items, U):
                    if g["weight_decay"]:
                        p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.add_(du.to(p.dtype), alpha=-g["lr"])


class Routed:
    """Two optimizers behind one step()/zero_grad(), with a shared LR multiplier for schedules."""

    def __init__(self, model, opt, lr, aux_lr, weight_decay=0.0, betas=(0.9, 0.999)):
        hidden = model.hidden_matrices()
        hid_ids = {id(p) for p in hidden.values()}
        rest = [p for p in model.parameters() if id(p) not in hid_ids]
        hp = list(hidden.values())
        if opt == "adamw":
            self.hidden = torch.optim.AdamW(hp, lr=lr, betas=betas, weight_decay=weight_decay)
        elif opt == "muon":
            self.hidden = Muon(hp, lr=lr, weight_decay=weight_decay)
        elif opt == "signum":
            self.hidden = Muon(hp, lr=lr, weight_decay=weight_decay, orthogonalize=False)
        elif opt == "sgdm":
            self.hidden = torch.optim.SGD(hp, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        else:
            raise ValueError(opt)
        self.aux = torch.optim.AdamW(rest, lr=aux_lr, betas=betas, weight_decay=0.0)
        self.opts = [self.hidden, self.aux]
        for o in self.opts:
            for g in o.param_groups:
                g["base_lr"] = g["lr"]

    def set_lr_mult(self, m):
        for o in self.opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * m

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self):
        for o in self.opts:
            o.zero_grad(set_to_none=True)
