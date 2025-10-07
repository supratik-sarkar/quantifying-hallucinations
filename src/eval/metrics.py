import torch

def accuracy_from_probs(probs, y_true_idx):
    # probs: (B,C); y_true_idx: (B,)
    preds = probs.argmax(dim=-1)
    return (preds == y_true_idx).float().mean().item()

def fpr_at_tpr(scores_pos, scores_neg, tpr=0.95):
    # simplistic ROC slice; scores higher=more positive
    import numpy as np
    sp = np.array(scores_pos); sn = np.array(scores_neg)
    ths = np.linspace(min(sp.min(), sn.min()), max(sp.max(), sn.max()), 200)
    best_fpr=1.0
    for th in ths:
        tp = (sp>=th).mean()
        if tp>=tpr:
            fp = (sn>=th).mean()
            best_fpr=min(best_fpr, fp)
    return best_fpr
