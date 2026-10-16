"""Part 2: martingale experiments."""
import numpy as np


def brownian_paths(T, n_steps, n_paths, rng):
    dt = T / n_steps
    W = np.cumsum(np.sqrt(dt) * rng.standard_normal((n_paths, n_steps)), axis=1)
    return np.linspace(0, T, n_steps + 1), np.hstack([np.zeros((n_paths, 1)), W])


def discounted_price_mean(S0, drift, sigma, r, t_grid, n, rng):
    """E[e^{-rt} S_t] for GBM with given drift, on a time grid (exact sampling)."""
    Z = rng.standard_normal(n)
    out = []
    for t in t_grid:
        S_t = S0 * np.exp((drift - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * Z)
        out.append(np.exp(-r * t) * S_t.mean())
    return np.array(out)
