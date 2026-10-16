"""Part 1: Ito's lemma experiments."""
import numpy as np


def gbm_terminal(S0, mu, sigma, T, n, rng):
    """Exact GBM terminal values, obtained by applying Ito's lemma to ln S."""
    W_T = np.sqrt(T) * rng.standard_normal(n)
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)


def euler_gbm(S0, mu, sigma, T, n_steps, n_paths, rng):
    """Euler-Maruyama discretisation of dS = mu S dt + sigma S dW (no Ito lemma used)."""
    dt = T / n_steps
    S = np.full(n_paths, S0)
    for _ in range(n_steps):
        S = S + mu * S * dt + sigma * S * np.sqrt(dt) * rng.standard_normal(n_paths)
    return S


def wdw_integrals(T, n_steps, n_paths, rng):
    """Left-endpoint (Ito), right-endpoint, and closed-form values of int_0^T W dW."""
    dt = T / n_steps
    dW = np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    W = np.cumsum(dW, axis=1)
    W_left = np.hstack([np.zeros((n_paths, 1)), W[:, :-1]])
    left = np.sum(W_left * dW, axis=1)
    right = np.sum(W * dW, axis=1)
    closed = 0.5 * W[:, -1] ** 2 - 0.5 * T
    return left, right, closed
