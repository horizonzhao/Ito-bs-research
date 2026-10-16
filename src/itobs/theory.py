"""Theoretical predictions that the simulations are tested against."""
import numpy as np
from . import bs


def predicted_hedge_std(S0, K, T, r, sigma, n_rebal, drift=None, n_u=200, n_z=4001):
    """Predicted std of discounted P&L of a delta-hedged short call in a TRUE Black-Scholes world.

    Each rebalancing step leaves the error  e_k = -1/2 Gamma_k S_k^2 sigma^2 dt (Z_k^2 - 1)  (Ito's second-order term).
    The e_k are martingale differences with Var(Z^2 - 1) = 2, so
        Var = sum_k e^{-2 r t_k} E[(1/2 Gamma S^2 sigma^2 dt)^2] * 2
            ~ (sigma^4 dt / 2) * integral_0^T e^{-2rt} E[Gamma_t^2 S_t^4] dt.
    The inner expectation integrates over the lognormal law of S_t (drift defaults to r); the time integral uses
    tau = T u^2 to remove the 1/sqrt(tau) endpoint singularity.
    """
    drift = r if drift is None else drift
    dt = T / n_rebal
    u, w = np.polynomial.legendre.leggauss(n_u)
    u = 0.5 * (u + 1); w = 0.5 * w                       # nodes on (0, 1)
    z = np.linspace(-9, 9, n_z); pdf = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi); dz = z[1] - z[0]
    total = 0.0
    for ui, wi in zip(u, w):
        tau = T * ui**2; t = T - tau
        S_t = S0 * np.exp((drift - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)
        g = bs.gamma(S_t, K, tau, r, sigma)
        inner = np.sum(g**2 * S_t**4 * pdf) * dz
        total += wi * np.exp(-2 * r * t) * inner * 2 * T * ui   # d tau = 2 T u du
    var = 0.5 * sigma**4 * dt * total
    return float(np.sqrt(var))
