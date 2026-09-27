"""Optional PyTorch float64 lanes: variogram pair accumulation and batched kriging solves on a CUDA device.

PyTorch is imported lazily, so installing GeoCond does not install a GPU runtime (``pip install geocond[cuda]`` adds
it). Both lanes implement exactly the definitions of the NumPy reference and are held to it by tests:

- **Variogram pairs.** The same unordered pairs, the same membership rules (half-open lag bins, inclusive angle and
  bandwidth tolerances to a relative 1e-12, coincident pairs excluded), tiled over rows, with counts, separations and
  squared differences accumulated per bin in float64. Counts are identical; sums may differ from the reference in the
  last bits because the GPU accumulates atomically in no fixed order (agreement within 1e-12 relative).
- **Kriging systems.** Neighbourhoods are selected on the CPU by the same ``Neighborhood``; the systems of targets with
  the same number of neighbours are assembled and solved as one batch: covariances evaluated on the device, a batched
  Cholesky factor, the ordinary constraints through the Schur complement, and the same failure rules (condition limit,
  materially negative variance). Only point supports run on the device; other supports are refused rather than
  silently routed to another method.
"""

import numpy as np
from numpy.typing import NDArray

from .covariance import CovarianceModel
from .validation import ValidationError


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only without the optional dependency
        raise ValidationError("the torch backend needs PyTorch: pip install geocond[cuda]") from error
    return torch


def device_of(device: str | None):
    torch = _torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def correlation_torch(torch, family: str, r):
    if family == "exponential":
        return torch.exp(-3.0 * r)
    if family == "spherical":
        return torch.where(r < 1.0, 1.0 - 1.5 * r + 0.5 * r**3, torch.zeros_like(r))
    return torch.exp(-3.0 * r * r)


def covariance_torch(torch, model: CovarianceModel, h, a: int = 0, b: int = 0, dev=None):
    """C_ab at separations ``h`` (..., 3), a float64 tensor; the nugget only at exactly zero separation."""
    out = torch.where((h == 0).all(dim=-1), torch.tensor(float(model.nugget[a, b]), dtype=torch.float64, device=h.device),
                      torch.zeros(h.shape[:-1], dtype=torch.float64, device=h.device))
    for c in model.components:
        if c.sill[a, b] != 0.0:
            rot = torch.as_tensor(np.array(c.rotation), dtype=torch.float64, device=h.device)
            inv = torch.as_tensor(1.0 / c.ranges, dtype=torch.float64, device=h.device)
            r = torch.linalg.vector_norm((h @ rot) * inv, dim=-1)
            out = out + float(c.sill[a, b]) * correlation_torch(torch, c.family, r)
    return out


def variogram_pairs_torch(coordinates, first, second, edges, direction, angle_tolerance, bandwidth, device,
                          retain_pairs, rows_per_tile=2048):
    """Counts, separation sums, product sums and (optionally) the pairs of every bin, over all unordered pairs."""
    torch = _torch()
    dev = device_of(device)
    x = torch.as_tensor(np.array(coordinates), dtype=torch.float64, device=dev)
    za = torch.as_tensor(np.array(first), dtype=torch.float64, device=dev)
    zb = torch.as_tensor(np.array(second), dtype=torch.float64, device=dev)
    e = torch.as_tensor(np.array(edges), dtype=torch.float64, device=dev)
    nb = len(edges) - 1
    n = len(coordinates)
    counts = torch.zeros(nb, dtype=torch.float64, device=dev)
    seps = torch.zeros(nb, dtype=torch.float64, device=dev)
    prods = torch.zeros(nb, dtype=torch.float64, device=dev)
    roots = torch.zeros(nb, dtype=torch.float64, device=dev)
    d = None if direction is None else torch.as_tensor(direction, dtype=torch.float64, device=dev)
    cos_t = None if angle_tolerance is None else float(np.cos(np.radians(angle_tolerance)))
    kept = []
    coincident = 0
    cols = torch.arange(n, device=dev)
    for start in range(0, n, rows_per_tile):
        rows = torch.arange(start, min(n, start + rows_per_tile), device=dev)
        upper = cols[None, :] > rows[:, None]
        h = x[None, :, :] - x[rows][:, None, :]
        dist = torch.linalg.vector_norm(h, dim=-1)
        zero = upper & (dist == 0.0)
        coincident += int(zero.sum().item())
        keep = upper & (dist != 0.0)
        if d is not None:
            along = torch.abs(h @ d)
            if cos_t is not None:
                keep &= along >= dist * cos_t - 1e-12 * dist
            if bandwidth is not None:
                across = torch.sqrt(torch.clamp(dist**2 - along**2, min=0.0))
                keep &= across <= bandwidth + 1e-12 * torch.clamp(dist, min=1.0)
        bins = torch.searchsorted(e, dist.contiguous(), right=True) - 1
        keep &= (bins >= 0) & (bins < nb)
        i_idx, j_idx = torch.nonzero(keep, as_tuple=True)
        b = bins[i_idx, j_idx]
        gi = rows[i_idx]
        counts += torch.bincount(b, minlength=nb).to(torch.float64)
        seps += torch.bincount(b, weights=dist[i_idx, j_idx], minlength=nb)
        prods += torch.bincount(b, weights=(za[gi] - za[j_idx]) * (zb[gi] - zb[j_idx]), minlength=nb)
        roots += torch.bincount(b, weights=torch.sqrt(torch.abs(za[gi] - za[j_idx])), minlength=nb)
        if retain_pairs:
            kept.append((gi.cpu().numpy(), j_idx.cpu().numpy(), b.cpu().numpy()))
    pairs = None
    if retain_pairs:
        pairs = tuple(np.concatenate([k[m] for k in kept]).astype(np.int64) if kept else np.zeros(0, np.int64)
                      for m in range(3))
    return (counts.cpu().numpy().astype(np.int64), seps.cpu().numpy(), prods.cpu().numpy(), roots.cpu().numpy(),
            pairs, coincident)


def solve_batch_torch(torch, dev, model, method, X, V, Z, E, T, tv, means, max_condition, negative_tol):
    """Point-support systems of one neighbourhood size, batched: X (B, n, 3) observation coordinates, V (B, n)
    variables, Z (B, n) values, E (B, n) measurement-error variances, T (B, 3) targets. Returns means (B, k), error covariances (B, k, k), weights (B, n, k),
    constraint residuals (B,) and a status code per system (0 ok, 1 ill conditioned, 2 negative variance)."""
    B, n, _ = X.shape
    k = len(tv)
    x = torch.as_tensor(X, dtype=torch.float64, device=dev)
    v = torch.as_tensor(V, dtype=torch.int64, device=dev)
    z = torch.as_tensor(Z, dtype=torch.float64, device=dev)
    t = torch.as_tensor(T, dtype=torch.float64, device=dev)
    err_var = torch.as_tensor(E, dtype=torch.float64, device=dev)
    p = model.n_variables
    h = x[:, None, :, :] - x[:, :, None, :]
    C = torch.zeros((B, n, n), dtype=torch.float64, device=dev)
    for a in range(p):
        for b in range(p):
            mask = (v[:, :, None] == a) & (v[:, None, :] == b)
            if mask.any():
                C = torch.where(mask, covariance_torch(torch, model, h, a, b), C)
    C = C + torch.diag_embed(err_var)
    ht = t[:, None, :] - x
    c = torch.zeros((B, n, k), dtype=torch.float64, device=dev)
    for col, target in enumerate(tv):
        for a in range(p):
            mask = v == a
            if mask.any():
                c[:, :, col] = torch.where(mask, covariance_torch(torch, model, ht, a, target), c[:, :, col])
    zero3 = torch.zeros((1, 3), dtype=torch.float64, device=dev)
    C00 = torch.tensor([[float(covariance_torch(torch, model, zero3, a, b)[0]) for b in tv] for a in tv],
                       dtype=torch.float64, device=dev)
    eig = torch.linalg.eigvalsh(C)
    status = torch.zeros(B, dtype=torch.int64, device=dev)
    bad = (eig[:, 0] <= 0) | (eig[:, -1] > max_condition * eig[:, 0])
    status[bad] = 1
    Cs = torch.where(bad[:, None, None], torch.eye(n, dtype=torch.float64, device=dev).expand(B, n, n), C)
    L = torch.linalg.cholesky(Cs)
    Ci_c = torch.cholesky_solve(c, L)
    if method == "simple":
        W = Ci_c
        m = torch.as_tensor(means, dtype=torch.float64, device=dev)
        pred = m[list(tv)][None, :] + torch.einsum("bnk,bn->bk", W, z - m[v])
        resid = torch.zeros(B, dtype=torch.float64, device=dev)
        lin = torch.linalg.matrix_norm(C @ W - c) / (torch.linalg.matrix_norm(C) * torch.linalg.matrix_norm(W) + torch.linalg.matrix_norm(c))
    else:
        present = list(range(p))
        F = torch.stack([(v == a).to(torch.float64) for a in present], dim=2)  # (B, n, p); absent columns are zero
        has = F.sum(dim=1) > 0  # (B, p)
        F = torch.where(has[:, None, :], F, torch.zeros_like(F))
        f0 = torch.tensor([[1.0 if a == target else 0.0 for target in tv] for a in present], dtype=torch.float64,
                          device=dev).expand(B, p, k)
        Ci_F = torch.cholesky_solve(F, L)
        S = F.transpose(1, 2) @ Ci_F
        eye = torch.eye(p, dtype=torch.float64, device=dev).expand(B, p, p)
        S = torch.where(has[:, :, None] & has[:, None, :], S, eye * (~has[:, :, None]).to(torch.float64))
        rhs = F.transpose(1, 2) @ Ci_c - torch.where(has[:, :, None], f0, torch.zeros_like(f0))
        lam = torch.linalg.solve(S, rhs)
        W = Ci_c - Ci_F @ lam
        pred = torch.einsum("bnk,bn->bk", W, z)
        top = C @ W + F @ lam - c
        lin = torch.sqrt(torch.linalg.matrix_norm(top) ** 2 + torch.linalg.matrix_norm(F.transpose(1, 2) @ W - torch.where(has[:, :, None], f0, torch.zeros_like(f0))) ** 2)
        lin = lin / (torch.linalg.matrix_norm(C) * torch.linalg.matrix_norm(W) + torch.linalg.matrix_norm(c))
        resid = torch.amax(torch.abs(F.transpose(1, 2) @ W - torch.where(has[:, :, None], f0, torch.zeros_like(f0))),
                           dim=(1, 2))
    err = C00[None] - W.transpose(1, 2) @ c - c.transpose(1, 2) @ W + W.transpose(1, 2) @ C @ W
    var = torch.diagonal(err, dim1=1, dim2=2)
    sill = float(np.max(np.diag(model.total_sill)))
    negative = (var < -negative_tol * torch.clamp(torch.diagonal(C00)[None, :], min=sill)).any(dim=1)
    status[(status == 0) & negative] = 2
    return (pred.cpu().numpy(), err.cpu().numpy(), W.cpu().numpy(), resid.cpu().numpy(), lin.cpu().numpy(),
            status.cpu().numpy(), eig[:, 0].cpu().numpy(), eig[:, -1].cpu().numpy())


def check_point_supports(supports) -> NDArray[np.float64]:
    if not all(s.kind == "point" for s in supports):
        raise ValidationError("the torch backend runs point supports only; use backend='numpy' for integrated supports")
    return np.array([s.center for s in supports])
