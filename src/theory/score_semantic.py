import torch

def d_sem_pointwise(th_KK, th_full):
    # Eq.(KL1) positive-part log-diff
    return torch.clamp((th_KK+1e-12).log() - (th_full+1e-12).log(), min=0.0)

