import torch
from src.theory.kernel_smoother import gaussian_kernel, T_h
from src.theory.score_semantic import d_sem_pointwise

def test_score_semantic():
    a=torch.randn(8,16); K=gaussian_kernel(a,a,1.0)
    q=torch.rand(8); Th=T_h(q,K)
    d=d_sem_pointwise(Th, Th+0.1)
    assert (d>=0).all()
