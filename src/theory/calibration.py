import torch

def good_turing_missing_mass(freq1_count, N):
    # simple GT: prob mass of unseen ≈ n1 / N
    if N<=0: return 0.0
    return float(freq1_count) / float(N)

def kv_schedule_upper_tau(m, c_norm2, theta_KV, lam_max):
    # τ ≤ (1/(2 λ_max)) log( m * ||c||^2 / θ_KV )  (from Eq.(KV_embed))
    num = (m * c_norm2) / max(theta_KV, 1e-12)
    if num <= 1.0: return 0.0
    return float(0.5/lam_max * torch.log(torch.tensor(num)).item())
