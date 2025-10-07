import torch

def hyper_eff_adjacency(I, w_e, D_e):
    """
    W_eff = I diag(w_e) D_e^{-1} I^T
    I: (|V|, |E|) in {0,1}
    w_e: (|E|,)
    D_e: (|E|, |E|) diagonal with r(e)
    """
    # Safe inverse of diagonal D_e
    if D_e.ndim == 2:
        d = torch.diagonal(D_e)
    else:
        d = D_e
    De_inv = torch.diag(1.0 / (d + 1e-9))
    return I @ (torch.diag(w_e) @ De_inv) @ I.T

def normalized_hyper_L(I, w_e, r_e, device):
    """
    Normalized hypergraph Laplacian:
      L = I - D_v^{-1/2} W_eff D_v^{-1/2}
    with D_v = diag(I w_e).
    """
    I = I.to(device)
    w_e = w_e.to(device)
    r_e = r_e.to(device)

    W_eff = hyper_eff_adjacency(I, w_e, torch.diag(r_e)).to(device)

    # Node degrees from hyperedges
    d_v = (I @ w_e)  # (|V|,)

    # D_v^{-1/2} with safe handling for zeros
    invsqrt = torch.zeros_like(d_v)
    mask = d_v > 0
    invsqrt[mask] = torch.rsqrt(d_v[mask] + 1e-9)
    Dv_inv_half = torch.diag(invsqrt)

    Iden = torch.eye(I.shape[0], device=device)
    L = Iden - Dv_inv_half @ W_eff @ Dv_inv_half

    # Numerical hygiene: symmetrize and clamp diagonal nonnegative
    L = 0.5 * (L + L.T)
    L.diagonal().clamp_min_(0.0)
    return L

def multi_L(blocks, coeffs):
    # blocks: list of L_* ; coeffs: same length, nonneg
    L = torch.zeros_like(blocks[0])
    for Li, ci in zip(blocks, coeffs):
        L = L + ci * Li
    return L

def top_eigs(L, k=None):
    # Dense eigen-decomp (ascending). For large |V|, switch to Lanczos.
    evals, evecs = torch.linalg.eigh(L)
    return evals, evecs
