import torch

def energy_gap_spectral(c, evals, evecs, coeff_bounds, tau):
    # Implements Eq.(energy_diff_eigexp) with ζ_i(t,τ) = w_i e^{-2τλ_i}, w_i∈[m,M]
    # Returns lower and upper CF-bound energies.
    uiTc = (evecs.T @ c)  # mode projections
    uiTc2 = uiTc**2
    lamb = evals
    m, M = coeff_bounds
    expfac = torch.exp(-2.0 * tau * lamb)
    E_lo = (m * expfac * uiTc2)[1:].sum()  # skip i=0 null
    E_hi = (M * expfac * uiTc2)[1:].sum()
    return E_lo, E_hi
