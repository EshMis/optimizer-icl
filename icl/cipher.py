"""In-context substitution ciphers: the discrete (letters) analogue of in-context regression.

A task is a hidden permutation (cipher) of an alphabet of V letters. A prompt is K plaintext
letters drawn uniformly with replacement and their coded letters. The model reads
p_1, c_1, p_2, c_2, ... and predicts each c_k from the pairs before it.

A run fixes a pool of M ciphers drawn once from `pool_seed` (M = 0: a fresh cipher per prompt).

Two Bayes-optimal reference predictors, as ridge and dMMSE are for regression:
- icl_preds: uniform prior over ALL permutations (the in-context decoder). If p_k appeared
  earlier, its code is known. Otherwise the code is uniform over the coded letters not yet
  used. Nothing can beat this on a new cipher.
- pool_preds: uniform prior over the POOL (the memoriser). Average the predictions of every
  pool cipher consistent with the pairs seen so far.
"""
import torch


def make_pool(V, M, pool_seed):
    if M == 0:
        return None
    g = torch.Generator().manual_seed(pool_seed)
    return torch.stack([torch.randperm(V, generator=g) for _ in range(M)])


def sample(B, K, V, pool, gen, device):
    """Return plain (B,K) long, code (B,K) long, perm (B,V) long."""
    if pool is None:
        perm = torch.argsort(torch.rand(B, V, generator=gen, device=device), dim=1)
    else:
        perm = pool[torch.randint(pool.shape[0], (B,), generator=gen, device=device)]
    plain = torch.randint(V, (B, K), generator=gen, device=device)
    code = torch.gather(perm, 1, plain)
    return plain, code, perm


def icl_preds(plain, code, V):
    """(B,K,V) probabilities from the uniform-over-all-permutations posterior."""
    B, K = plain.shape
    dev = plain.device
    seen_p = torch.zeros(B, K, V, device=dev)         # plain letters seen before position k
    seen_c = torch.zeros(B, K, V, device=dev)         # coded letters seen before position k
    known = torch.full((B, K), -1, dtype=torch.long, device=dev)   # code of p_k if seen before
    oh_p = torch.nn.functional.one_hot(plain, V).float()
    oh_c = torch.nn.functional.one_hot(code, V).float()
    cum_p = torch.cumsum(oh_p, 1) - oh_p              # exclusive
    cum_c = torch.cumsum(oh_c, 1) - oh_c
    seen_p = (cum_p > 0).float()
    seen_c = (cum_c > 0).float()
    # map plain -> code from earlier pairs: table (B,K,V) built cumulatively
    table = torch.zeros(B, K, V, V, device=dev)
    pair = oh_p.unsqueeze(-1) * oh_c.unsqueeze(-2)    # (B,K,V,V)
    table = torch.cumsum(pair, 1) - pair
    row = torch.gather(table, 2, plain.view(B, K, 1, 1).expand(B, K, 1, V)).squeeze(2)  # (B,K,V)
    is_known = row.sum(-1) > 0
    unused = 1.0 - seen_c
    uni = unused / unused.sum(-1, keepdim=True).clamp_min(1.0)
    return torch.where(is_known.unsqueeze(-1), (row > 0).float(), uni)


def pool_preds(plain, code, pool, chunk=64):
    """(B,K,V) probabilities from the uniform-over-pool posterior (pure memorisation)."""
    B, K = plain.shape
    M, V = pool.shape
    out = torch.empty(B, K, V, device=plain.device)
    P = pool.to(plain.device)
    for s in range(0, B, chunk):
        pl, cd = plain[s:s + chunk], code[s:s + chunk]
        b = pl.shape[0]
        # pred[b,k,m] = P[m, plain[b,k]]
        pred = P.T[pl]                                 # (b,K,M)
        match = (pred == cd.unsqueeze(-1)).float()     # consistent with pair k
        cons = torch.cumprod(match, dim=1)
        cons = torch.cat([torch.ones_like(cons[:, :1]), cons[:, :-1]], dim=1)   # pairs before k
        oh = torch.nn.functional.one_hot(pred, V).float()                        # (b,K,M,V)
        num = (cons.unsqueeze(-1) * oh).sum(2)
        out[s:s + b] = num / cons.sum(-1, keepdim=True).clamp_min(1e-9)
    return out
