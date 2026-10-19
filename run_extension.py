"""Extension (Section C): market smile + optional hedge-vol supplement.

Primary (C1): ^SPX quotes -> invert IV -> fig6 (link to constant-sigma failure in A2).
Optional (C2): Heston-world BS-delta hedge at chosen sigma_h -> fig7 (reinforces A2/A3).
Run after datapull.py has written data/quotes_SPX_REAL.csv.
"""
import json, sys, pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from itobs import bs, heston, implied_vol as iv, hedging

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25, "font.size": 10})

# 真实微笑（与 datapull.py 保持一致）
REAL_TICKER = "^SPX"
REAL_R = 0.045

S0, r = 100.0, 0.03
H = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
R = {"heston_params": H}


def smile_by_expiry(quotes, rate):
    """按到期日分组反推 IV；每组用 S_eff = F e^{-rT}（等价于含股息的前向框架）。"""
    parts = []
    for T, q in quotes.groupby("maturity"):
        S_eff = float(q.forward.iloc[0]) * np.exp(-rate * T)
        parts.append(iv.smile_from_quotes(q, S_eff, rate))
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["iv"]).reset_index(drop=True)


def plot_real_smile(sm, ticker, as_of):
    """fig6：SPY 等真实 OTM 报价反推的隐含波动率微笑。"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for e, q in sm.groupby("expiry"):
        lab = f"{e} (T={q.maturity.iloc[0]:.2f})"
        ax[0].plot(q.strike, 100 * q.iv, "o-", ms=3, label=lab)
        ax[1].plot(np.log(q.strike / q.forward), 100 * q.iv, "o-", ms=3, label=lab)
    ax[0].set(
        title=f"(a) {ticker} implied vol, OTM quotes ({as_of:%Y-%m-%d})",
        xlabel="strike K",
        ylabel="implied vol (%)",
    )
    ax[1].axvline(0, color="k", lw=1, ls=":")
    ax[1].set(
        title="(b) Implied vol vs log-moneyness ln(K/F)",
        xlabel="ln(K/F)",
        ylabel="implied vol (%)",
    )
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig6_smile.png")
    plt.close()


def atm_iv_by_expiry(sm):
    """每个到期日 ln(K/F) 最接近 0 的合约 IV 作为 ATM 代理。"""
    out = {}
    for e, q in sm.groupby("expiry"):
        lm = np.log(q.strike / q.forward).to_numpy()
        out[str(e)] = float(q.iv.iloc[int(np.argmin(np.abs(lm)))])
    return out


def skew_samples(sm):
    """每个到期日：最低/最高 ln(K/F) 各取一档 IV，用于 README 式 skew 描述。"""
    samples = {}
    for e, q in sm.groupby("expiry"):
        q = q.copy()
        q["lm"] = np.log(q.strike / q.forward)
        lo = q.loc[q.lm.idxmin()]
        hi = q.loc[q.lm.idxmax()]
        samples[str(e)] = dict(
            low_strike=dict(K=float(lo.strike), ln_KF=float(lo.lm), iv=float(lo.iv)),
            high_strike=dict(K=float(hi.strike), ln_KF=float(hi.lm), iv=float(hi.iv)),
        )
    return samples


def heston_implied_vol(K, T):
    """Heston 解析价反推 BS 隐含波动率（C2 对冲用，与真实 SPY 微笑无关）。"""
    price = heston.call_price(S0, K, T, r, **H)
    return float(iv.implied_vol(price, S0, K, T, r))


# --- 0. validate Heston pricer ---
p_small_xi = heston.call_price(S0, 100, 1.0, r, 0.04, 2.0, 0.04, 1e-3, 0.0)
p_bs = bs.call_price(S0, 100, 1.0, r, 0.20)
mc_pay = np.concatenate([
    np.exp(-r) * np.maximum(
        heston.simulate(S0, 1.0, r, **H, n_steps=500, n_paths=100_000, rng=rng)[:, -1] - 100, 0)
    for _ in range(6)
])
R["validation"] = dict(
    heston_xi_to_0=p_small_xi,
    bs_sigma20=float(p_bs),
    heston_semi_analytic=heston.call_price(S0, 100, 1.0, r, **H),
    heston_monte_carlo=float(mc_pay.mean()),
    mc_ci95=float(1.96 * mc_pay.std() / np.sqrt(len(mc_pay))),
)

# --- 1. Real ^SPX smile -> fig6 ---
quotes_csv = DATA / f"quotes_{REAL_TICKER.replace('^', '')}_REAL.csv"
if not quotes_csv.is_file():
    raise FileNotFoundError(
        f"缺少 {quotes_csv}。请先运行: python datapull.py")
quotes_real = pd.read_csv(quotes_csv)
sm_real = smile_by_expiry(quotes_real, REAL_R)
csv_mtime = datetime.fromtimestamp(quotes_csv.stat().st_mtime, tz=timezone.utc)
plot_real_smile(sm_real, REAL_TICKER, csv_mtime)

s_eff = [
    g.forward.iloc[0] * np.exp(-REAL_R * g.maturity.iloc[0])
    for _, g in quotes_real.groupby("expiry")
]
spot_eff = float(np.median(s_eff))
R["real_smile"] = dict(
    ticker=REAL_TICKER,
    quotes_csv=str(quotes_csv.relative_to(ROOT)).replace("\\", "/"),
    r=REAL_R,
    spot_eff=spot_eff,
    quotes_csv_mtime=csv_mtime.isoformat(),
    computed_at=datetime.now(timezone.utc).isoformat(),
    n_quotes=int(len(quotes_real)),
    n_iv=int(len(sm_real)),
    forwards={str(e): float(q.forward.iloc[0]) for e, q in quotes_real.groupby("expiry")},
    atm_iv=atm_iv_by_expiry(sm_real),
    skew_sample=skew_samples(sm_real),
)

# --- 2. Hedge in a Heston world with the BS delta ---
def hedge_heston(K, T, n_rebal, n_paths, sigma_hedge, mu=r):
    """Sell the call at its Heston price, hedge with BS delta at sigma_hedge. Returns discounted P&L."""
    price0 = heston.call_price(S0, K, T, r, **H)
    S = heston.simulate(S0, T, mu, **H, n_steps=n_rebal, n_paths=n_paths, rng=rng)
    dt = T / n_rebal
    delta = bs.delta(S[:, 0], K, T, r, sigma_hedge)
    cash = price0 - delta * S[:, 0]
    for k in range(1, n_rebal + 1):
        cash *= np.exp(r * dt)
        if k < n_rebal:
            new = bs.delta(S[:, k], K, T - k * dt, r, sigma_hedge)
            cash -= (new - delta) * S[:, k]
            delta = new
    return np.exp(-r * T) * (cash + delta * S[:, -1] - np.maximum(S[:, -1] - K, 0))


def stats(p):
    return dict(
        mean=float(p.mean()),
        std=float(p.std()),
        q05=float(np.quantile(p, 0.05)),
        skew=float(((p - p.mean()) ** 3).mean() / p.std() ** 3),
    )


T_hedge, n_rebal, n_paths = 1.0, 252, 20_000
res = {}
sig_atm = heston_implied_vol(100, T_hedge)
for K in (100, 120):
    sig_K = heston_implied_vol(K, T_hedge)
    res[f"K{K}_sigma_ATM"] = stats(hedge_heston(K, T_hedge, n_rebal, n_paths, sig_atm))
    res[f"K{K}_sigma_own_IV"] = stats(hedge_heston(K, T_hedge, n_rebal, n_paths, sig_K))
res["BS_world_K100"] = stats(hedging.hedge_pnl(S0, 100, T_hedge, r, sig_atm, r, n_rebal, n_paths, rng))
R["heston_hedging"] = res

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
bsw = hedging.hedge_pnl(S0, 100, T_hedge, r, sig_atm, r, n_rebal, n_paths, rng)
hw = hedge_heston(100, T_hedge, n_rebal, n_paths, sig_atm)
ax[0].hist(bsw, bins=100, range=(-5, 5), alpha=0.5, density=True, label=f"BS world (std {bsw.std():.2f})")
ax[0].hist(hw, bins=100, range=(-5, 5), alpha=0.5, density=True, label=f"Heston world (std {hw.std():.2f})")
ax[0].set(title="(a) ATM call, same BS-delta hedge, 252 rebalances", xlabel="discounted P&L")
ax[0].legend()
h120 = hedge_heston(120, T_hedge, n_rebal, n_paths, sig_atm)
h120b = hedge_heston(120, T_hedge, n_rebal, n_paths, heston_implied_vol(120, T_hedge))
ax[1].hist(h120, bins=100, range=(-5, 5), alpha=0.5, density=True, label=f"hedge vol = ATM IV (mean {h120.mean():+.2f})")
ax[1].hist(
    h120b, bins=100, range=(-5, 5), alpha=0.5, density=True,
    label=f"hedge vol = own-strike IV (mean {h120b.mean():+.2f})",
)
ax[1].set(title="(b) OTM call K=120: which vol to hedge with?", xlabel="discounted P&L")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig7_heston_hedging.png")
plt.close()

with open(ROOT / "results_extension.json", "w", encoding="utf-8") as f:
    json.dump(R, f, indent=2, ensure_ascii=False, default=float)
print(json.dumps(R, indent=2, ensure_ascii=False, default=float))
