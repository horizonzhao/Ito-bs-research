"""Section B: numerical foundations (Ito, martingales, pricing, Greeks).

Run before run_research.py when reproducing the full story; outputs fig1-5 and results.json.
"""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from itobs import bs, ito, martingale, pricing, hedging

FIG = pathlib.Path(__file__).parent / "figures"; FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
S0, K, T, r, sigma, mu = 100.0, 100.0, 1.0, 0.03, 0.25, 0.12
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25, "font.size": 10})
R = {"params": dict(S0=S0, K=K, T=T, r=r, sigma=sigma, mu=mu)}

# ---------------- Part 1: Ito's lemma ----------------
S_T = ito.gbm_terminal(S0, mu, sigma, T, 500_000, rng)
lnret = np.log(S_T / S0)
S_euler = ito.euler_gbm(S0, mu, sigma, T, 500, 50_000, rng)
left, right, closed = ito.wdw_integrals(T, 500, 50_000, rng)
R["ito"] = dict(
    mean_log_return_sim=lnret.mean(), mean_log_return_ito=(mu - .5 * sigma**2) * T, naive_mu_T=mu * T,
    mean_S_T=S_T.mean(), theory_mean_S_T=S0 * np.exp(mu * T), median_S_T=float(np.median(S_T)),
    euler_mean_log=float(np.log(S_euler / S0).mean()),
    wdw_left_mean=left.mean(), wdw_right_mean=right.mean(), wdw_left_vs_closed_rmse=float(np.sqrt(np.mean((left - closed) ** 2))))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].hist(lnret, bins=150, density=True, alpha=.55, label="simulated")
xs = np.linspace(-1, 1.2, 500)
ax[0].plot(xs, norm.pdf(xs, (mu - .5 * sigma**2) * T, sigma * np.sqrt(T)), "r", lw=2, label=r"Normal($\mu-\sigma^2/2,\ \sigma^2T$)")
ax[0].axvline(mu * T, color="k", ls="--", label=r"naive mean $\mu T$")
ax[0].axvline(lnret.mean(), color="r", ls=":", label="simulated mean")
ax[0].set(title=r"(a) $\ln(S_T/S_0)$: the $-\sigma^2/2$ shift", xlabel="log return"); ax[0].legend(fontsize=8)
ax[1].hist(S_T, bins=150, density=True, alpha=.55, color="tab:green")
ax[1].axvline(S_T.mean(), color="k", ls="--", label=f"mean = {S_T.mean():.1f}")
ax[1].axvline(np.median(S_T), color="r", ls="--", label=f"median = {np.median(S_T):.1f}")
ax[1].set(title="(b) Lognormal $S_T$: mean > median (volatility drag)", xlabel="$S_T$"); ax[1].legend(fontsize=8)
ax[2].hist(left, bins=120, alpha=.55, label=f"left endpoint (Ito), mean={left.mean():+.2f}")
ax[2].hist(right, bins=120, alpha=.55, label=f"right endpoint (cheating), mean={right.mean():+.2f}")
ax[2].set(title=r"(c) $\int_0^T W\,dW$: peeking adds $+T$", xlabel="value"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "fig1_ito_lemma.png"); plt.close()

# ---------------- Part 2: martingales ----------------
t, W = martingale.brownian_paths(T, 500, 50_000, rng)
EW2 = (W**2).mean(0); EM = (W**2 - t).mean(0)
tg = np.linspace(0.01, T, 40)
dP = martingale.discounted_price_mean(S0, mu, sigma, r, tg, 1_000_000, rng)
dQ = martingale.discounted_price_mean(S0, r, sigma, r, tg, 1_000_000, rng)
R["martingale"] = dict(
    E_W2_at_T=float(EW2[-1]), E_W2_minus_t_at_T=float(EM[-1]), max_abs_E_W2_minus_t=float(np.abs(EM).max()),
    disc_price_P_at_T=float(dP[-1]), disc_price_Q_at_T=float(dQ[-1]))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(t, EW2, label=r"$E[W_t^2]$  (not a martingale)")
ax[0].plot(t, t, "k--", lw=1, label="$t$")
ax[0].plot(t, EM, label=r"$E[W_t^2 - t]$  (martingale)")
ax[0].set(title=r"(a) Subtracting $t$ removes the trend", xlabel="t"); ax[0].legend()
ax[1].plot(tg, dP, "o-", ms=3, label=rf"under $P$ (drift $\mu$={mu}): has trend")
ax[1].plot(tg, dQ, "s-", ms=3, label=rf"under $Q$ (drift $r$={r}): flat = martingale")
ax[1].axhline(S0, color="k", ls="--", lw=1)
ax[1].set(title=r"(b) $E[e^{-rt}S_t]$: martingale only under $Q$", xlabel="t"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "fig2_martingale.png"); plt.close()

# ---------------- Part 3: pricing ----------------
exact = float(bs.call_price(S0, K, T, r, sigma))
mc, mc_ci, pay = pricing.mc_price(S0, K, T, r, sigma, 2_000_000, rng)
mc_wrong, _, _ = pricing.mc_price(S0, K, T, r, sigma, 2_000_000, rng, drift=mu)
pde = float(pricing.pde_price(S0, K, T, r, sigma))
tree = float(pricing.binomial_price(S0, K, T, r, sigma, 1000))
R["pricing"] = dict(closed_form=exact, monte_carlo_Q=mc, monte_carlo_ci95=mc_ci, pde_crank_nicolson=pde,
                    binomial_1000=tree, wrong_monte_carlo_under_P=mc_wrong)

ns = np.unique(np.logspace(2, np.log10(len(pay)), 30).astype(int))
err = [abs(pay[:m].mean() - exact) for m in ns]
Ns = [10, 20, 50, 100, 200, 500, 1000]
terr = [abs(pricing.binomial_price(S0, K, T, r, sigma, n) - exact) for n in Ns]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
names = ["Closed form", "MC under Q", "PDE (CN)", "Binomial", r"MC under $P$ (wrong)"]
vals = [exact, mc, pde, tree, mc_wrong]
bars = ax[0].bar(names, vals, color=["k", "tab:blue", "tab:green", "tab:orange", "tab:red"])
for b, v in zip(bars, vals): ax[0].text(b.get_x() + b.get_width() / 2, v + .15, f"{v:.3f}", ha="center", fontsize=8)
ax[0].set(title="(a) Same price, four ways; the wrong measure fails", ylim=(0, max(vals) * 1.15)); ax[0].tick_params(axis="x", labelsize=7, rotation=15)
ax[1].loglog(ns, err, "o-", ms=3, label="MC |error|")
ax[1].loglog(ns, 25 / np.sqrt(ns), "k--", label=r"$\propto 1/\sqrt{n}$")
ax[1].set(title="(b) Monte Carlo convergence", xlabel="paths"); ax[1].legend()
ax[2].loglog(Ns, terr, "o-", label="binomial |error|")
ax[2].loglog(Ns, 3 / np.array(Ns), "k--", label=r"$\propto 1/N$")
ax[2].set(title="(c) Binomial tree convergence", xlabel="steps N"); ax[2].legend()
plt.tight_layout(); plt.savefig(FIG / "fig3_pricing.png"); plt.close()

# ---------------- Part 4: hedging ----------------
freqs = [4, 12, 52, 252, 1000]
pnl = {f: hedging.hedge_pnl(S0, K, T, r, sigma, mu, f, 20_000, rng) for f in freqs}
mus = [-0.2, 0.0, 0.12, 0.4]
pnl_mu = {m: hedging.hedge_pnl(S0, K, T, r, sigma, m, 252, 20_000, rng) for m in mus}
R["hedging"] = dict(by_frequency={str(f): dict(mean=float(p.mean()), std=float(p.std())) for f, p in pnl.items()},
                    by_mu={str(m): dict(mean=float(p.mean()), std=float(p.std())) for m, p in pnl_mu.items()})

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for f in (12, 52, 252):
    ax[0].hist(pnl[f], bins=100, range=(-8, 8), alpha=.5, density=True, label=f"{f} rebalances")
ax[0].set(title="(a) Hedging P&L of a short call", xlabel="discounted P&L"); ax[0].legend()
sd = np.array([pnl[f].std() for f in freqs])
ax[1].loglog(freqs, sd, "o-", label="std of P&L")
ax[1].loglog(freqs, sd[0] * np.sqrt(freqs[0] / np.array(freqs)), "k--", label=r"$\propto 1/\sqrt{N}$")
ax[1].set(title="(b) Residual risk shrinks like $1/\\sqrt{N}$", xlabel="rebalances N"); ax[1].legend()
ax[2].bar([str(m) for m in mus], [pnl_mu[m].mean() for m in mus], yerr=[2 * pnl_mu[m].std() / np.sqrt(20000) for m in mus], capsize=5)
ax[2].axhline(0, color="k", lw=1)
ax[2].set(title=r"(c) Mean P&L ≈ 0 for any true $\mu$", xlabel=r"true $\mu$", ylabel="mean P&L (±2 s.e.)")
plt.tight_layout(); plt.savefig(FIG / "fig4_hedging.png"); plt.close()

# ---------------- Part 5: Black-Scholes PDE residual & Gamma-Theta ----------------
Sg = np.linspace(60, 150, 200)
V, D, G, Th = (f(Sg, K, T, r, sigma) for f in (bs.call_price, bs.delta, bs.gamma, bs.theta))
resid = Th + 0.5 * sigma**2 * Sg**2 * G + r * Sg * D - r * V
R["greeks"] = dict(max_abs_pde_residual=float(np.abs(resid).max()))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(Sg, D, label=r"$\Delta$"); ax[0].plot(Sg, G * 20, label=r"$\Gamma\times20$"); ax[0].legend()
ax[0].set(title="(a) Delta and Gamma of a call", xlabel="S")
ax[1].plot(Sg, Th, label=r"$\Theta$"); ax[1].plot(Sg, -0.5 * sigma**2 * Sg**2 * G, "--", label=r"$-\frac{1}{2}\sigma^2S^2\Gamma$")
ax[1].legend(); ax[1].set(title=r"(b) $\Theta\approx-\frac{1}{2}\sigma^2S^2\Gamma$ (gamma-theta trade-off)", xlabel="S")
ax[2].plot(Sg, resid); ax[2].set(title="(c) BS PDE residual of the closed form (≈1e-14)", xlabel="S")
plt.tight_layout(); plt.savefig(FIG / "fig5_greeks.png"); plt.close()

json.dump(R, open("results.json", "w"), indent=2, default=float)
print(json.dumps(R, indent=2, default=float))
