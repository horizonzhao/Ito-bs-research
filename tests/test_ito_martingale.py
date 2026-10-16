import numpy as np
import pytest
from itobs import ito, martingale


def test_ito_log_drift_has_minus_half_sigma_squared():
    rng = np.random.default_rng(1)
    mu, sigma, T = 0.12, 0.25, 1.0
    S = ito.gbm_terminal(100, mu, sigma, T, 400_000, rng)
    m = np.log(S / 100).mean()
    se = np.log(S / 100).std() / np.sqrt(len(S))
    assert abs(m - (mu - .5 * sigma**2) * T) < 4 * se
    assert abs(m - mu * T) > 10 * se                 # and the naive mu*T is clearly rejected


def test_mean_of_S_grows_at_mu_not_mu_minus_half_sigma_sq():
    rng = np.random.default_rng(2)
    S = ito.gbm_terminal(100, 0.12, 0.25, 1.0, 1_000_000, rng)
    assert S.mean() == pytest.approx(100 * np.exp(0.12), rel=3e-3)


def test_euler_scheme_agrees_with_ito_solution():
    rng = np.random.default_rng(3)
    S = ito.euler_gbm(100, 0.12, 0.25, 1.0, 250, 40_000, rng)
    target = (0.12 - .5 * 0.25**2)
    assert np.log(S / 100).mean() == pytest.approx(target, abs=0.01)


def test_left_endpoint_integral_has_mean_zero_right_endpoint_adds_T():
    rng = np.random.default_rng(4)
    left, right, closed = ito.wdw_integrals(1.0, 400, 30_000, rng)
    se = left.std() / np.sqrt(len(left))
    assert abs(left.mean()) < 4 * se
    assert abs(right.mean() - 1.0) < 4 * right.std() / np.sqrt(len(right))
    assert np.sqrt(np.mean((left - closed) ** 2)) < 0.1      # Ito formula W^2/2 - t/2


def test_W2_minus_t_is_martingale_but_W2_is_not():
    rng = np.random.default_rng(5)
    t, W = martingale.brownian_paths(1.0, 100, 40_000, rng)
    M = W**2 - t
    assert np.abs(M.mean(0)).max() < 4 * M.std(0).max() / np.sqrt(40_000)
    assert (W**2).mean(0)[-1] == pytest.approx(1.0, abs=0.03)        # E[W_1^2] = 1 != 0


def test_discounted_price_is_martingale_only_under_Q():
    rng = np.random.default_rng(6)
    tg = np.array([0.5, 1.0])
    q = martingale.discounted_price_mean(100, 0.03, 0.25, 0.03, tg, 600_000, rng)
    p = martingale.discounted_price_mean(100, 0.12, 0.25, 0.03, tg, 600_000, rng)
    assert np.allclose(q, 100, rtol=3e-3)
    assert p[-1] > 106
