import torch

def smooth_density_mixture(fp_vals, rho_vals, eps):
    # tilde f_{p,ε} = (1-ε) f_p + ε ρ  ; assume fp, rho over finite C normalized
    return (1-eps)*fp_vals + eps*rho_vals

