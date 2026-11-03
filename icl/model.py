"""GPT-2-style transformer over interleaved (x, y) tokens, as in Garg et al. 2022.

Tokens are x_1, y_1, x_2, y_2, ...; y_i is embedded as the d-vector (y_i, 0, ..., 0). A
linear read-in maps tokens to the residual stream, a linear read-out maps it to a scalar,
and the prediction for y_k is read at the x_k token. Q, K, V and O are separate matrices
so each is a 2-D parameter Muon can orthogonalise on its own.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H = H
        self.ln1 = nn.LayerNorm(D)
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        self.ln2 = nn.LayerNorm(D)
        self.fc = nn.Linear(D, 4 * D)
        self.proj = nn.Linear(4 * D, D)

    def forward(self, h, attn_out=None):
        B, T, D = h.shape
        a = self.ln1(h)
        q, k, v = (m(a).view(B, T, self.H, D // self.H).transpose(1, 2) for m in (self.q, self.k, self.v))
        if attn_out is None:
            z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.H)
            mask = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
            p = s.masked_fill(~mask, float("-inf")).softmax(-1)
            attn_out.append(p.detach())
            z = p @ v
        h = h + self.o(z.transpose(1, 2).reshape(B, T, D))
        return h + self.proj(F.gelu(self.fc(self.ln2(h))))


class ICLTransformer(nn.Module):
    def __init__(self, d, K, n_layer=8, n_embd=128, n_head=2):
        super().__init__()
        self.d = d
        self.read_in = nn.Linear(d, n_embd)
        self.pos = nn.Parameter(torch.zeros(2 * K, n_embd))
        self.blocks = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.read_out = nn.Linear(n_embd, 1)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * n_layer) if name.endswith(("o.weight", "proj.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    @staticmethod
    def interleave(x, y):
        B, K, d = x.shape
        ytok = torch.zeros_like(x)
        ytok[..., 0] = y
        return torch.stack([x, ytok], dim=2).reshape(B, 2 * K, d)

    def forward(self, x, y, attn_out=None, return_hidden=False):
        h = self.read_in(self.interleave(x, y)) + self.pos
        for blk in self.blocks:
            h = blk(h, attn_out)
        h = self.ln_f(h)
        pred = self.read_out(h[:, ::2]).squeeze(-1)    # prediction for y_k at token x_k
        return (pred, h[:, ::2]) if return_hidden else pred

    def hidden_matrices(self):
        """The 2-D weights inside transformer blocks: what Muon is routed to."""
        return {n: p for n, p in self.named_parameters() if p.dim() == 2 and n.startswith("blocks.")}
