"""Heston (1993) stochastic volatility: semi-analytic call price and Monte Carlo simulation.

dS = mu S dt + sqrt(v) S dW1
dv = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho
Risk-neutral pricing uses mu = r.
"""
import numpy as np
from scipy.integrate import quad


def _char_fn(u, T, v0, kappa, theta, xi, rho):
    """E[exp(i u X)], X = ln(S_T / F). 'Little Heston trap' form (numerically stable)."""
    b = kappa - 1j * rho * xi * u
    d = np.sqrt(b**2 + xi**2 * (1j * u + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    D = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    C = kappa * theta / xi**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    return np.exp(C + D * v0)


def call_price(S0, K, T, r, v0, kappa, theta, xi, rho):
    """Lewis (2001) single-integral representation."""
    k = np.log(S0 / K) + r * T
    f = lambda u: (np.exp(1j * u * k) * _char_fn(u - 0.5j, T, v0, kappa, theta, xi, rho)).real / (u**2 + 0.25)
    integral, _ = quad(f, 0, np.inf, limit=200)
    return S0 - np.sqrt(S0 * K) * np.exp(-r * T / 2) / np.pi * integral


def simulate(S0, T, mu, v0, kappa, theta, xi, rho, n_steps, n_paths, rng):
    """Full-truncation Euler. Returns the full S path array (n_paths, n_steps+1)."""
    dt = T / n_steps
    S = np.empty((n_paths, n_steps + 1)); S[:, 0] = S0
    lnS = np.full(n_paths, np.log(S0)); v = np.full(n_paths, v0)
    for k in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        lnS += (mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
        S[:, k + 1] = np.exp(lnS)
    return S
