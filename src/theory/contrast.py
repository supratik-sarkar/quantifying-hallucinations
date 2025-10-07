import torch

def contrast_vec(vx_idx, vk_idx, deg):
    # degree-matched, null-mode-projected contrast (simplified)
    # c = e_{vx} - e_{vk} ; normalize by sqrt(deg)
    c = torch.zeros_like(deg)
    c[vx_idx]=1.0; c[vk_idx]-=1.0
    # degree weighting
    d = torch.clamp(deg, min=1e-9)
    c = c / torch.sqrt(d)
    # projection to 1^⊥ (remove null mode)
    c = c - c.mean()
    return c
