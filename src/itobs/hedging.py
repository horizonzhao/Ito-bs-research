"""Part 4: discrete delta hedging of a short call."""
import numpy as np
from . import bs


def hedge_pnl(S0, K, T, r, sigma, true_mu, n_rebal, n_paths, rng):
    """Sell one call at the BS price, delta-hedge n_rebal times, return discounted terminal P&L."""
    price0 = bs.call_price(S0, K, T, r, sigma)
    dt = T / n_rebal
    Z = rng.standard_normal((n_paths, n_rebal))
    S = np.full(n_paths, S0)
    delta = bs.delta(S, K, T, r, sigma)
    cash = price0 - delta * S
    for k in range(1, n_rebal + 1):
        cash *= np.exp(r * dt)
        S = S * np.exp((true_mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, k - 1])
        if k < n_rebal:
            new = bs.delta(S, K, T - k * dt, r, sigma)
            cash -= (new - delta) * S
            delta = new
    final = cash + delta * S - np.maximum(S - K, 0)
    return final * np.exp(-r * T)


def hedge_pnl_with_proxy(S0, K, T, r, sigma, true_mu, n_rebal, n_paths, rng):
    """Like hedge_pnl, but also returns the 'gamma-term' proxy
        proxy = - sum_k e^{-r t_{k+1}} * 1/2 Gamma_k S_k^2 sigma^2 dt (Z_k^2 - 1)
    which is the second-order (Ito) term the delta hedge leaves behind each step."""
    price0 = bs.call_price(S0, K, T, r, sigma)
    dt = T / n_rebal
    Z = rng.standard_normal((n_paths, n_rebal))
    S = np.full(n_paths, S0)
    delta = bs.delta(S, K, T, r, sigma)
    cash = price0 - delta * S
    proxy = np.zeros(n_paths)
    for k in range(1, n_rebal + 1):
        g = bs.gamma(S, K, T - (k - 1) * dt, r, sigma)
        proxy -= np.exp(-r * k * dt) * 0.5 * g * S**2 * sigma**2 * dt * (Z[:, k - 1] ** 2 - 1)
        cash *= np.exp(r * dt)
        S = S * np.exp((true_mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, k - 1])
        if k < n_rebal:
            new = bs.delta(S, K, T - k * dt, r, sigma)
            cash -= (new - delta) * S
            delta = new
    pnl = np.exp(-r * T) * (cash + delta * S - np.maximum(S - K, 0))
    return pnl, proxy


def hedge_on_paths(S_paths, K, T, r, sigma_h, premium):
    """Delta-hedge (BS delta at vol sigma_h) a short call on GIVEN paths S_paths (n_paths, n_steps+1).
    The call is sold for `premium`. Returns discounted P&L."""
    n_rebal = S_paths.shape[1] - 1
    dt = T / n_rebal
    delta = bs.delta(S_paths[:, 0], K, T, r, sigma_h)
    cash = premium - delta * S_paths[:, 0]
    for k in range(1, n_rebal + 1):
        cash *= np.exp(r * dt)
        if k < n_rebal:
            new = bs.delta(S_paths[:, k], K, T - k * dt, r, sigma_h)
            cash -= (new - delta) * S_paths[:, k]
            delta = new
    return np.exp(-r * T) * (cash + delta * S_paths[:, -1] - np.maximum(S_paths[:, -1] - K, 0))
