import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnLayer(nn.Module):
    """Transformer encoder layer that exposes per-head attention weights."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        h = self.n1(x)
        a, w = self.attn(h, h, h, need_weights=return_attn,
                         average_attn_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return (x, w) if return_attn else (x, None)


class ConnectomeTransformer(nn.Module):
    """
    Graph transformer for structural connectomes.

    use_identity=False -> Variant A: connection profile only
    use_identity=True  -> Variant B: + learned node identity embedding
    """

    def __init__(self, n_nodes, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, dropout=0.1, use_identity=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.use_identity = use_identity

        self.proj = nn.Linear(n_nodes, d_model)
        if use_identity:
            self.identity = nn.Embedding(n_nodes, d_model)
            nn.init.normal_(self.identity.weight, std=0.02)

        self.layers = nn.ModuleList([
            AttnLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.edge_head = nn.Sequential(
            nn.Linear(2 * d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, 1))

    def encode(self, A, return_attn=False):
        """A: (B, N, N) masked adjacency. Returns node embeddings."""
        h = self.proj(A)
        if self.use_identity:
            ids = torch.arange(self.n_nodes, device=A.device)
            h = h + self.identity(ids).unsqueeze(0)

        attns = []
        for layer in self.layers:
            h, w = layer(h, return_attn=return_attn)
            if return_attn:
                attns.append(w)          # (B, heads, N, N)
        return self.norm(h), attns

    def predict_edges(self, h):
        """Symmetric edge prediction from node embeddings."""
        B, N, D = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        e1 = self.edge_head(torch.cat([hi, hj], -1)).squeeze(-1)
        e2 = self.edge_head(torch.cat([hj, hi], -1)).squeeze(-1)
        return (e1 + e2) / 2

    def forward(self, A_masked, return_attn=False):
        h, attns = self.encode(A_masked, return_attn)
        return self.predict_edges(h), attns


def mask_edges(A, ratio=0.15, generator=None):
    """
    Zero a random subset of upper-triangular entries (mirrored).
    Returns masked input and a boolean mask of held-out positions.
    """
    B, N, _ = A.shape
    dev = A.device
    iu = torch.triu_indices(N, N, offset=1, device=dev)
    n_pairs = iu.shape[1]
    n_mask = int(ratio * n_pairs)

    m = torch.zeros(B, N, N, dtype=torch.bool, device=dev)
    for b in range(B):
        sel = torch.randperm(n_pairs, device=dev,
                             generator=generator)[:n_mask]
        r, c = iu[0, sel], iu[1, sel]
        m[b, r, c] = True
        m[b, c, r] = True

    return A.masked_fill(m, 0.0), m


def masked_loss(pred, target, mask):
    """MSE over held-out entries only."""
    return F.mse_loss(pred[mask], target[mask])