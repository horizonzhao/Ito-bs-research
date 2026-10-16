"""Implied volatility: invert Black-Scholes, build a smile, load real quotes from CSV."""
import numpy as np
from scipy.optimize import brentq
from . import bs


def implied_vol(price, S, K, T, r, lo=1e-4, hi=3.0):
    """Invert the BS call price. Returns nan if the price violates no-arbitrage bounds."""
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if not (intrinsic + 1e-10 < price < S):
        return np.nan
    return brentq(lambda s: bs.call_price(S, K, T, r, s) - price, lo, hi, xtol=1e-10)


def put_to_call(put_price, S, K, T, r):
    """Put-call parity: C = P + S - K e^{-rT}. Lets you use OTM puts for the low strikes."""
    return put_price + S - K * np.exp(-r * T)


def smile_from_quotes(df, S, r):
    """df columns: strike, maturity (years), price, type ('C' or 'P'). Returns df with 'iv'."""
    out = df.copy()
    calls = [p if t == "C" else put_to_call(p, S, k, T, r)
             for p, t, k, T in zip(df["price"], df["type"], df["strike"], df["maturity"])]
    out["iv"] = [implied_vol(c, S, k, T, r) for c, k, T in zip(calls, df["strike"], df["maturity"])]
    return out
