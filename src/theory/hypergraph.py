import torch

def pairwise_sem_diffs(dvals):
    # For a hyperedge with nodes indices idx, return sum_{a,b} |d[a]-d[b]|
    # dvals: (r,)  (Δ_{ε,h}(x_a|p_a))
    r = dvals.shape[0]
    diffs = dvals.unsqueeze(0).repeat(r,1) - dvals.unsqueeze(1).repeat(1,r)
    return diffs.abs().sum()

def w_Tt_for_hyperedge(dvals, Tvals, eta):
    num = pairwise_sem_diffs(dvals)
    den = Tvals.sum() + 1e-9
    return torch.exp(-eta * num / den).clamp(0,1)
