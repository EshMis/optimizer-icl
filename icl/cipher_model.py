"""The same GPT-2-style transformer as icl/model.py, with token embeddings in and logits out,
for the substitution-cipher task. Plain letters use ids 0..V-1, coded letters V..2V-1.
The prediction for c_k is read at the p_k token.
"""
import math

import torch
import torch.nn as nn

from .model import Block


class CipherTransformer(nn.Module):
    def __init__(self, V, K, n_layer=8, n_embd=128, n_head=2):
        super().__init__()
        self.V = V
        self.tok = nn.Embedding(2 * V, n_embd)
        self.pos = nn.Parameter(torch.zeros(2 * K, n_embd))
        self.blocks = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * n_layer) if name.endswith(("o.weight", "proj.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, plain, code, attn_out=None):
        B, K = plain.shape
        ids = torch.stack([plain, code + self.V], dim=2).reshape(B, 2 * K)
        h = self.tok(ids) + self.pos
        for blk in self.blocks:
            h = blk(h, attn_out)
        return self.head(self.ln_f(h[:, ::2]))        # (B,K,V) logits at plain tokens

    def hidden_matrices(self):
        return {n: p for n, p in self.named_parameters() if p.dim() == 2 and n.startswith("blocks.")}
