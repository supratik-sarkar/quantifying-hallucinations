import torch, torch.nn.functional as F

class SurrogateBoltzmann:
    """
    Unifies f_p over a finite candidate set C via energies:
       f_p(c|x) ∝ exp(-E(c;x,p)/T)
    We expose: logits over candidates, entropy, and top-k.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    @torch.no_grad()
    def probs_from_energies(self, E):  # E: (B, C)
        logits = -E / max(self.temperature, 1e-6)
        return F.softmax(logits, dim=-1), logits

    @torch.no_grad()
    def entropy(self, probs):
        eps = 1e-8
        return -(probs * (probs+eps).log()).sum(-1)

