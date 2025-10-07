import torch

def selector_K_topk(emb, K_idx, x_idx):
    """
    Π_𝕂(x): identity on K; otherwise map to nearest in K by cosine sim.
    emb: (N,D), K_idx: list/1D tensor of indices∈K, x_idx: index of x
    """
    if x_idx in set(K_idx): return x_idx
    x = emb[x_idx:x_idx+1]
    K = emb[K_idx]
    sim = (x @ K.T) / (x.norm(dim=-1, keepdim=True)*K.norm(dim=-1, keepdim=True)+1e-9)
    j = sim.argmax(dim=-1).item()
    return int(K_idx[j])
