import numpy as np
import pytest
from itobs import bs, heston, implied_vol as iv

H = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)


def test_heston_reduces_to_bs_when_vol_of_vol_vanishes():
    p = heston.call_price(100, 100, 1.0, 0.03, 0.04, 2.0, 0.04, 1e-3, 0.0)
    assert p == pytest.approx(bs.call_price(100, 100, 1.0, 0.03, 0.20), abs=1e-3)


def test_heston_semi_analytic_matches_monte_carlo():
    rng = np.random.default_rng(20)
    pay = np.concatenate([np.exp(-0.03) * np.maximum(heston.simulate(100, 1.0, 0.03, **H, n_steps=200, n_paths=50_000, rng=rng)[:, -1] - 100, 0)
                          for _ in range(3)])
    exact = heston.call_price(100, 100, 1.0, 0.03, **H)
    assert abs(pay.mean() - exact) / exact < 0.015      # sampling error + small Euler bias


def test_heston_price_respects_no_arbitrage_and_monotone_in_strike():
    prices = [heston.call_price(100, k, 0.5, 0.03, **H) for k in (80, 90, 100, 110, 120)]
    assert all(np.diff(prices) < 0)
    assert all(p >= max(100 - k * np.exp(-0.015), 0) for p, k in zip(prices, (80, 90, 100, 110, 120)))


@pytest.mark.parametrize("sigma", [0.1, 0.25, 0.6])
@pytest.mark.parametrize("K", [80, 100, 130])
def test_implied_vol_roundtrip(sigma, K):
    price = bs.call_price(100, K, 0.7, 0.03, sigma)
    assert iv.implied_vol(price, 100, K, 0.7, 0.03) == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_arbitrage_violating_price():
    assert np.isnan(iv.implied_vol(0.0, 100, 100, 1.0, 0.03))
    assert np.isnan(iv.implied_vol(150.0, 100, 100, 1.0, 0.03))


def test_negative_rho_gives_downward_sloping_smile():
    ivs = [iv.implied_vol(heston.call_price(100, k, 1.0, 0.03, **H), 100, k, 1.0, 0.03) for k in (80, 100, 120)]
    assert ivs[0] > ivs[1] > ivs[2]
