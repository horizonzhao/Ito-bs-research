"""Main research question:
   Where does delta-hedging error come from -- discretisation (Ito second-order term) or model misspecification?
   E1  BS world: error ~ 1/sqrt(N); size predicted from theory; gamma-term proxy explains the P&L.
   E2  Heston world, BS-delta hedge: error does NOT vanish with N.
   E3  Heston world: the MEAN P&L is fixed by the premium you charge; predicted exactly, tested.
"""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from itobs import bs, heston, hedging, theory

ROOT = pathlib.Path(__file__).parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25, "font.size": 10})

S0, K, T, r = 100.0, 100.0, 1.0, 0.03
SIG = 0.20                                    # BS-world vol = sqrt(theta) of the Heston world below
H = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
R = {"params": dict(S0=S0, K=K, T=T, r=r, sigma_bs=SIG, heston=H)}

def moments(p):
    z = (p - p.mean()) / p.std()
    return dict(mean=float(p.mean()), std=float(p.std()), skew=float((z**3).mean()),
                ex_kurt=float((z**4).mean() - 3), q01=float(np.quantile(p, .01)), q05=float(np.quantile(p, .05)))

# ------------------------------------------------------------ E1: BS world
Ns = [8, 16, 32, 64, 128, 256, 512, 1024]
n_paths = 20_000
e1 = {}
for N in Ns:
    p, px = hedging.hedge_pnl_with_proxy(S0, K, T, r, SIG, r, N, n_paths, rng)
    e1[N] = dict(std=float(p.std()), mean=float(p.mean()), se_mean=float(p.std() / np.sqrt(n_paths)),
                 pred_std=theory.predicted_hedge_std(S0, K, T, r, SIG, N),
                 corr_proxy=float(np.corrcoef(p, px)[0, 1]),
                 _p=p, _px=px)
slope = float(np.polyfit(np.log(Ns), np.log([e1[N]["std"] for N in Ns]), 1)[0])
R["E1_bs_world"] = dict(loglog_slope=slope,
                        by_N={str(N): {k: v for k, v in e1[N].items() if not k.startswith("_")} for N in Ns})

# ------------------------------------------------------------ E2: Heston world, same BS-delta hedge
Ns2 = [16, 64, 256, 1000]
n2 = 15_000
prem_bs = float(bs.call_price(S0, K, T, r, SIG))
c_heston = heston.call_price(S0, K, T, r, **H)
e2 = {}
for N in Ns2:
    paths = heston.simulate(S0, T, r, **H, n_steps=N, n_paths=n2, rng=rng)
    pH = hedging.hedge_on_paths(paths, K, T, r, SIG, prem_bs)
    pG, _ = hedging.hedge_pnl_with_proxy(S0, K, T, r, SIG, r, N, n2, rng)
    e2[N] = dict(heston=moments(pH - pH.mean()) | dict(mean=float(pH.mean())), bs=moments(pG),
                 _pH=pH, _pG=pG)
R["E2_heston_world"] = dict(premium_bs=prem_bs, heston_price=float(c_heston),
                            by_N={str(N): {"heston": e2[N]["heston"], "bs_world": e2[N]["bs"]} for N in Ns2})

# ------------------------------------------------------------ E3: mean P&L is set by premium fairness
N3, n3 = 252, 15_000
rhos = [-0.7, 0.0, 0.7]
sig_hs = [0.15, 0.20, 0.25]
e3 = []
for rho in rhos:
    Hp = dict(H, rho=rho)
    ch = heston.call_price(S0, K, T, r, **Hp)
    paths = heston.simulate(S0, T, r, **Hp, n_steps=N3, n_paths=n3, rng=rng)
    for sh in sig_hs:
        prem = float(bs.call_price(S0, K, T, r, sh))
        p = hedging.hedge_on_paths(paths, K, T, r, sh, prem)
        e3.append(dict(rho=rho, sigma_h=sh, predicted=prem - float(ch), simulated=float(p.mean()),
                       se=float(p.std() / np.sqrt(n3)), std=float(p.std()), q01=float(np.quantile(p, .01))))
for row in e3:
    row["z"] = (row["simulated"] - row["predicted"]) / row["se"]
R["E3_mean_prediction"] = e3

# ------------------------------------------------------------ Figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
sd = np.array([e1[N]["std"] for N in Ns]); pr = np.array([e1[N]["pred_std"] for N in Ns])
ax[0].loglog(Ns, sd, "o-", label="simulated std")
ax[0].loglog(Ns, pr, "k--", label="theory (Itô 2nd-order term)")
ax[0].set(title=f"(a) BS world: std ∝ N^{slope:.3f} (theory −0.5)", xlabel="rebalances N", ylabel="std of P&L"); ax[0].legend()
ax[1].semilogx(Ns2, [e2[N]["heston"]["std"] for N in Ns2], "o-", color="tab:red", label="Heston world (BS delta)")
ax[1].semilogx(Ns2, [e2[N]["bs"]["std"] for N in Ns2], "o-", color="tab:blue", label="BS world")
ax[1].set(title="(b) Misspecification: error does not vanish", xlabel="rebalances N", ylabel="std of P&L"); ax[1].legend()
ax[1].set_ylim(bottom=0)
p, px = e1[128]["_p"], e1[128]["_px"]
ax[2].scatter(px[:3000], p[:3000], s=3, alpha=.4)
lim = np.quantile(np.abs(p), .995)
ax[2].plot([-lim, lim], [-lim, lim], "k--", lw=1)
ax[2].set(title=f"(c) N=128: P&L vs Γ-term (corr {e1[128]['corr_proxy']:.3f})", xlabel="Σ −½ΓS²σ²Δt(Z²−1)", ylabel="actual P&L")
ax[2].set_xlim(-lim, lim); ax[2].set_ylim(-lim, lim)
plt.tight_layout(); plt.savefig(FIG / "fig8_error_sources.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for N, c in zip(Ns2, ["#bbb", "#88a", "#44a", "#117"]):
    ax[0].hist(e2[N]["_pH"] - e2[N]["_pH"].mean(), bins=120, range=(-12, 8), density=True, histtype="step", color=c, lw=1.6, label=f"N={N}")
ax[0].set(title="(a) Heston world: centred P&L distribution", xlabel="P&L − mean"); ax[0].legend(); ax[0].set_ylim(bottom=0)
x = np.arange(len(e3)); lab = [f"ρ={d['rho']:+.1f}\nσ_h={d['sigma_h']:.2f}" for d in e3]
ax[1].errorbar(x, [d["simulated"] for d in e3], yerr=[2 * d["se"] for d in e3], fmt="o", capsize=3, label="simulated ±2 s.e.")
ax[1].plot(x, [d["predicted"] for d in e3], "k_", ms=14, mew=2, label="predicted  V_BS(σ_h) − C_Heston")
ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=6.5)
ax[1].set(title="(b) Mean P&L = premium − fair value", ylabel="mean discounted P&L"); ax[1].legend(fontsize=8, loc="center right")
plt.tight_layout(); plt.savefig(FIG / "fig9_misspecification_mean.png"); plt.close()

# strip private arrays and dump
json.dump(R, open(ROOT / "results_research.json", "w"), indent=2, default=float)
print(json.dumps(R, indent=1, default=float))
