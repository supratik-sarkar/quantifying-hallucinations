import torch, torch.nn.functional as F

def entropy_baseline(logits):
    p = F.softmax(logits, -1)
    return -(p * (p+1e-9).log()).sum(-1)

def logprob_gap_baseline(logits):
    top2, _ = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1)
    if top2.shape[-1]<2: return torch.zeros(logits.shape[0], device=logits.device)
    return top2[...,0]-top2[...,1]
