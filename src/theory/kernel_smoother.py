import torch

@torch.no_grad()
def gaussian_kernel(X: torch.Tensor, Y: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    """
    Isotropic Gaussian (RBF) kernel: K_ij = exp(-||x_i - y_j||^2 / (2 h^2))
    X: (N, D), Y: (M, D) on same device/dtype.
    """
    X = torch.nn.functional.normalize(X, dim=-1) if X.ndim == 2 else X
    Y = torch.nn.functional.normalize(Y, dim=-1) if Y.ndim == 2 else Y
    # squared euclidean via (x - y)^2 = ||x||^2 + ||y||^2 - 2 x·y
    x2 = (X * X).sum(dim=-1, keepdim=True)           # (N,1)
    y2 = (Y * Y).sum(dim=-1, keepdim=True).T         # (1,M)
    dist2 = (x2 + y2 - 2.0 * (X @ Y.T)).clamp_min(0)
    K = torch.exp(-dist2 / (2.0 * (h ** 2)))
    return K

@torch.no_grad()
def row_stochastic(K: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Make a kernel row-stochastic: each row sums to 1.
    """
    denom = K.sum(dim=1, keepdim=True).clamp_min(eps)
    return K / denom
