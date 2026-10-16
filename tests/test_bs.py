import numpy as np
import pytest
from itobs import bs

P = dict(K=100.0, T=1.0, r=0.03, sigma=0.25)


def test_put_call_parity():
    S = 90.0
    call = bs.call_price(S, **P)
    put = call - S + P["K"] * np.exp(-P["r"] * P["T"])      # parity defines the put; check it is a valid put
    d1 = (np.log(S / P["K"]) + (P["r"] + .5 * P["sigma"]**2) * P["T"]) / (P["sigma"] * np.sqrt(P["T"]))
    from scipy.stats import norm
    put_direct = P["K"] * np.exp(-P["r"] * P["T"]) * norm.cdf(-(d1 - P["sigma"])) - S * norm.cdf(-d1)
    assert put == pytest.approx(put_direct, abs=1e-10)


def test_no_arbitrage_bounds():
    for S in (50, 100, 200):
        c = bs.call_price(S, **P)
        assert max(S - P["K"] * np.exp(-P["r"] * P["T"]), 0) <= c <= S


def test_delta_gamma_match_finite_differences():
    S, h = 105.0, 1e-3
    fd_delta = (bs.call_price(S + h, **P) - bs.call_price(S - h, **P)) / (2 * h)
    fd_gamma = (bs.call_price(S + h, **P) - 2 * bs.call_price(S, **P) + bs.call_price(S - h, **P)) / h**2
    assert bs.delta(S, **P) == pytest.approx(fd_delta, abs=1e-6)
    assert bs.gamma(S, **P) == pytest.approx(fd_gamma, abs=1e-5)


def test_theta_matches_finite_difference_in_calendar_time():
    S, h = 100.0, 1e-5
    # V_t = -dV/dT
    fd = -(bs.call_price(S, P["K"], P["T"] + h, P["r"], P["sigma"]) - bs.call_price(S, P["K"], P["T"] - h, P["r"], P["sigma"])) / (2 * h)
    assert bs.theta(S, **P) == pytest.approx(fd, abs=1e-5)


def test_closed_form_satisfies_black_scholes_pde():
    S = np.linspace(60, 150, 50)
    res = bs.theta(S, **P) + .5 * P["sigma"]**2 * S**2 * bs.gamma(S, **P) + P["r"] * S * bs.delta(S, **P) - P["r"] * bs.call_price(S, **P)
    assert np.abs(res).max() < 1e-10
