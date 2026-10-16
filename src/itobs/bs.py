"""Black-Scholes closed form and Greeks (European call, no dividends)."""
import numpy as np
from scipy.stats import norm


def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def delta(S, K, T, r, sigma):
    return norm.cdf(_d1d2(S, K, T, r, sigma)[0])


def gamma(S, K, T, r, sigma):
    d1, _ = _d1d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(S, K, T, r, sigma):
    """dV/dt (calendar time), so it is negative for a long call."""
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
