"""Part 3: pricing a European call several ways."""
import numpy as np
from scipy.linalg import solve_banded


def mc_price(S0, K, T, r, sigma, n, rng, drift=None):
    """Monte Carlo price. drift=None -> risk-neutral (r). Passing drift=mu is the WRONG measure."""
    d = r if drift is None else drift
    Z = rng.standard_normal(n)
    S_T = S0 * np.exp((d - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    pay = np.exp(-r * T) * np.maximum(S_T - K, 0)
    return pay.mean(), 1.96 * pay.std() / np.sqrt(n), pay


def pde_price(S0, K, T, r, sigma, M=400, Nt=400, Smax_mult=4):
    """Crank-Nicolson for V_t + 0.5 s^2 S^2 V_SS + r S V_S - r V = 0 (time-to-maturity form)."""
    Smax = Smax_mult * K
    S = np.linspace(0, Smax, M + 1)
    dt = T / Nt
    V = np.maximum(S - K, 0)
    i = np.arange(1, M)
    a = 0.25 * dt * (sigma**2 * i**2 - r * i)
    b = -0.5 * dt * (sigma**2 * i**2 + r)
    c = 0.25 * dt * (sigma**2 * i**2 + r * i)
    ab = np.zeros((3, M - 1))
    ab[0, 1:] = -c[:-1]
    ab[1, :] = 1 - b
    ab[2, :-1] = -a[1:]
    for step in range(Nt):
        tau = (step + 1) * dt
        rhs = a * V[:-2] + (1 + b) * V[1:-1] + c * V[2:]
        V_up = Smax - K * np.exp(-r * tau)
        rhs[-1] += c[-1] * V_up
        V = np.concatenate([[0.0], solve_banded((1, 1), ab, rhs), [V_up]])
    return np.interp(S0, S, V)


def binomial_price(S0, K, T, r, sigma, n):
    """Cox-Ross-Rubinstein tree. q is exactly the 'risk-neutral probability'."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    S_T = S0 * u ** np.arange(n, -1, -1) * d ** np.arange(0, n + 1)
    V = np.maximum(S_T - K, 0)
    for _ in range(n):
        V = np.exp(-r * dt) * (q * V[:-1] + (1 - q) * V[1:])
    return V[0]
