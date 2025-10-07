import torch
from src.theory.energy import energy_gap_spectral

def test_energy_bounds():
    L=torch.eye(8)*0.1
    evals, evecs = torch.linalg.eigh(L)
    c=torch.randn(8); m,M=0.5,2.0
    Elo,Ehi = energy_gap_spectral(c, evals, evecs, (m,M), tau=1.0)
    assert Elo<=Ehi and Elo>=0
