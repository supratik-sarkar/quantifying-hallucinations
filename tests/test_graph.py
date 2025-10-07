import torch
from src.theory.laplacian import normalized_hyper_L

def test_hyper_lap():
    V,E=32,8
    I=torch.zeros(V,E);
    for e in range(E): I[torch.randperm(V)[:4],e]=1.0
    w=torch.rand(E); r=torch.full((E,),4.0)
    L=normalized_hyper_L(I,w,r,"cpu")
    evals,_=torch.linalg.eigh(L)
    assert (evals>=-1e-6).all()
