import numpy as np
import pytest
from itobs import bs, pricing

A = (100.0, 100.0, 1.0, 0.03, 0.25)


def test_monte_carlo_under_Q_within_confidence_interval():
    rng = np.random.default_rng(10)
    price, ci, _ = pricing.mc_price(*A, 400_000, rng)
    assert abs(price - bs.call_price(*A)) < 1.5 * ci          # 95% CI -> allow slack for a single seed


def test_monte_carlo_under_real_world_measure_is_wrong():
    rng = np.random.default_rng(11)
    wrong, ci, _ = pricing.mc_price(*A, 400_000, rng, drift=0.12)
    assert wrong - bs.call_price(*A) > 10 * ci


def test_pde_matches_closed_form():
    assert pricing.pde_price(*A) == pytest.approx(bs.call_price(*A), abs=5e-3)


def test_binomial_converges_at_order_one():
    exact = bs.call_price(*A)
    e1 = abs(pricing.binomial_price(*A, 100) - exact)
    e2 = abs(pricing.binomial_price(*A, 400) - exact)
    assert e2 < e1 / 2.5                                       # ~4x smaller for 4x steps (order 1), with slack
    assert e2 < 0.01
