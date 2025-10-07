import torch

def diffusion_kernel(L, tau):
    # K_Tt = exp(-tau * L)
    return torch.linalg.matrix_exp(-tau * L)

def apply_semantic_diffusion(c, L, tau):
    # <c, exp(-2 τ L) c>
    K = torch.linalg.matrix_exp(-2.0 * tau * L)
    return (c.unsqueeze(0) @ K @ c.unsqueeze(-1)).squeeze()
