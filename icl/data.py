"""In-context linear regression data (Raventos et al. 2023 setup).

A run fixes a task pool of M weight vectors w_m ~ N(0, I_d), drawn once from `pool_seed`.
M = 0 means an unlimited pool: every prompt gets a fresh w. A prompt is K pairs
(x_i, y_i) with x_i ~ N(0, I_d) and y_i = w.x_i + eps_i, eps_i ~ N(0, noise_var).
"""
import torch


def make_pool(d, M, pool_seed):
    if M == 0:
        return None
    g = torch.Generator().manual_seed(pool_seed)
    return torch.randn(M, d, generator=g)


def sample(B, K, d, noise_var, pool, gen, device):
    """Return x (B,K,d), y (B,K), w (B,d). `pool` lives on `device` or is None."""
    x = torch.randn(B, K, d, generator=gen, device=device)
    if pool is None:
        w = torch.randn(B, d, generator=gen, device=device)
    else:
        idx = torch.randint(pool.shape[0], (B,), generator=gen, device=device)
        w = pool[idx]
    y = torch.einsum("bkd,bd->bk", x, w) + noise_var ** 0.5 * torch.randn(B, K, generator=gen, device=device)
    return x, y, w


def ridge_preds(x, y, noise_var):
    """Bayes-optimal prediction for w ~ N(0, I): posterior-mean ridge with lambda = noise_var.

    pred[:, k] uses pairs 0..k-1 and x_k; pred[:, 0] = 0 (prior mean)."""
    B, K, d = x.shape
    out = torch.zeros(B, K, device=x.device, dtype=torch.float64)
    xd, yd = x.double(), y.double()
    eye = torch.eye(d, device=x.device, dtype=torch.float64)
    for k in range(1, K):
        X, Y = xd[:, :k], yd[:, :k]
        A = X.transpose(1, 2) @ X + noise_var * eye
        w = torch.linalg.solve(A, (X.transpose(1, 2) @ Y.unsqueeze(-1))).squeeze(-1)
        out[:, k] = (xd[:, k] * w).sum(-1)
    return out.float()


def dmmse_preds(x, y, pool, noise_var, chunk_elems=2 ** 25):
    """Bayes-optimal prediction when w is uniform over the finite pool (pure memorisation)."""
    B, K, d = x.shape
    M = pool.shape[0]
    out = torch.empty(B, K, device=x.device)
    bc = max(1, chunk_elems // (K * M))
    P = pool.to(x.device)
    for s in range(0, B, bc):
        xs, ys = x[s:s + bc], y[s:s + bc]
        f = xs @ P.T                                    # (b,K,M) every pool member's prediction
        r = (ys.unsqueeze(-1) - f) ** 2
        # log-likelihood of pairs strictly before position k
        ll = -torch.cumsum(r, dim=1) / (2 * noise_var)
        ll = torch.cat([torch.zeros_like(ll[:, :1]), ll[:, :-1]], dim=1)
        post = torch.softmax(ll, dim=-1)
        out[s:s + bc] = (post * f).sum(-1)
    return out
