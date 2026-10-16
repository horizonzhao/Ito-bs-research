import numpy as np
import pytest
from itobs import bs, hedging, heston, theory

A = (100.0, 100.0, 1.0, 0.03, 0.20)


def test_hedge_error_decays_like_inverse_sqrt_N():
    rng = np.random.default_rng(30)
    Ns = [16, 64, 256]
    sd = [hedging.hedge_pnl(*A, 0.03, N, 8000, rng).std() for N in Ns]
    slope = np.polyfit(np.log(Ns), np.log(sd), 1)[0]
    assert -0.58 < slope < -0.42


def test_hedge_error_matches_theoretical_prediction():
    rng = np.random.default_rng(31)
    for N in (64, 256):
        p = hedging.hedge_pnl(*A, 0.03, N, 15_000, rng)
        assert p.std() == pytest.approx(theory.predicted_hedge_std(*A, N), rel=0.06)


def test_gamma_term_explains_pnl():
    rng = np.random.default_rng(32)
    p, proxy = hedging.hedge_pnl_with_proxy(*A, 0.03, 128, 8000, rng)
    assert np.corrcoef(p, proxy)[0, 1] > 0.97


def test_mean_pnl_is_zero_and_independent_of_true_mu():
    rng = np.random.default_rng(33)
    for mu in (-0.2, 0.03, 0.4):
        p = hedging.hedge_pnl(*A, mu, 252, 12_000, rng)
        assert abs(p.mean()) < 0.06                       # small relative to std ~0.4; O(dt) effect allowed


def test_misspecified_hedge_mean_equals_premium_minus_fair_value():
    rng = np.random.default_rng(34)
    H = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    paths = heston.simulate(100, 1.0, 0.03, **H, n_steps=100, n_paths=12_000, rng=rng)
    sh = 0.25
    premium = bs.call_price(100, 100, 1.0, 0.03, sh)
    p = hedging.hedge_on_paths(paths, 100, 1.0, 0.03, sh, premium)
    predicted = premium - heston.call_price(100, 100, 1.0, 0.03, **H)
    assert abs(p.mean() - predicted) < 4 * p.std() / np.sqrt(len(p))


def test_misspecification_error_does_not_vanish_with_N():
    rng = np.random.default_rng(35)
    H = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    prem = bs.call_price(100, 100, 1.0, 0.03, 0.2)
    out = {}
    for N in (32, 256):
        paths = heston.simulate(100, 1.0, 0.03, **H, n_steps=N, n_paths=6000, rng=rng)
        out[N] = hedging.hedge_on_paths(paths, 100, 1.0, 0.03, 0.2, prem).std()
    bs_256 = hedging.hedge_pnl(*A, 0.03, 256, 6000, rng).std()
    assert out[256] > 0.7 * out[32]          # barely improves (BS world would improve by sqrt(8) ~ 2.8x)
    assert out[256] > 3 * bs_256
