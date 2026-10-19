# Where Does Delta-Hedging Error Come From?
### Itô calculus, martingales and Black–Scholes, tested against stochastic volatility

## Project summary

Decompose **delta-hedging residual P&L** into **(i) discrete rebalancing** (Itô second-order / gamma) vs **(ii) model misspecification** (Heston truth, Black–Scholes delta in stock only). Theory in [`docs/derivations.md`](docs/derivations.md); validation by simulation and **35 pytest checks**. Market link: **^SPX implied-vol smile** (constant $\sigma$ fails). Code: Python, `src/itobs/`.

**Headline results:** BS-world hedge-error std is **predictable** (~3–7% vs simulation) and scales as **$N^{-1/2}$**; Heston-world std **floors near ~2.2** despite more rebalances; under misspricing the **mean P&L** matches **premium − fair value** but **tails stay large**.

---

## How the repo fits together

Read or run in this order — each step feeds the next:

| Step | Script | Question answered |
|------|--------|-------------------|
| **B. Foundations** | `run_all.py` | Are Itô, martingales, and BS pricing implemented correctly? (Trust the simulator.) |
| **A. Research** | `run_research.py` | **Main question:** discretisation vs wrong model; what can be predicted? |
| **C. Extension** | `datapull.py` + `run_extension.py` | Why does a **single** BS vol fail in real quotes? Optional sim: hedge-vol choice (fig7). |

```
  B (math sanity)  ──►  A (hedge P&L decomposition)  ──►  C (market smile + optional supplement)
```

**Figures for a quick review:** fig8–9 (core), fig6 (market), fig1–3 (foundations). fig7 = optional reinforcement of A2/A3.

---

## Research question

> A trader sells an option and hedges it with the Black–Scholes delta. The hedge leaves a residual profit-and-loss (P&L).
> **How much of that residual comes from (i) hedging only at discrete times, and how much from (ii) the model being wrong? Can each part be predicted in advance?**

---

## Conclusions

### Direct answers

**(i) Discrete rebalancing.** When paths **are** Black–Scholes, residual P&L is mostly the **second-order Itô / gamma** term from finite rebalancing. Its std is **predictable** (`theory.predicted_hedge_std`), scales **$\propto 1/\sqrt{N}$**, and tracks the summed $\Gamma$-terms (corr **0.96–0.99**).

**(ii) Wrong model.** When paths are **Heston** but you hedge with **BS delta in stock only**, std **flattens near ~2.2** ($N=16\to1000$: 2.85→2.23) while BS-world std at the same settings can fall to **~0.22** — **incompleteness / misspecification**, not “too few rebalances.” Left skew and fat tails remain (1% quantile **−7.2** vs **−0.56** in BS world at $N=1000$).

**Predictability.** Yes for **BS discretisation std** and for **mean P&L** when you sell at a BS price: $\mathbb{E}[\text{P\&L}] = V_{\mathrm{BS}}(\sigma_h) - C_{\mathrm{Heston}}$ (9 cases, all **|z| ≤ 1.2**). No closed-form floor formula here — measured by simulation.

**Scope.** Truth is **fixed by experiment** (BS paths vs Heston paths), not “who forecasts SPX best.” Section **C1** only shows **non-flat** IV on ^SPX; Heston is **not** calibrated to those quotes.

**Parameters:** `S0=K=100, T=1, r=3%`; BS $\sigma=20%$; Heston `v0=θ=0.04, κ=2, ξ=0.5, ρ=−0.7`. Numbers: `results_research.json`, `results_extension.json`.

### Summary table

| # | Where | Takeaway |
|---|--------|----------|
| 1 | A1, fig8 | $\Gamma$-term story; pred vs sim **~3–7%**; slope **−0.491** (theory **−0.5**). |
| 2 | A2, fig8 | Heston std **→ ~2.2**; BS-world std **→ ~0.22** at $N=1000$. |
| 3 | A3, fig9 | Mean = **premium − Heston fair**; predictable. |
| 4 | A3 | **Mean ≠ safety:** std **2.2–3.2**; 1% q **−3.7 to −12.9**. |
| 5 | C1, fig6 | Real IV **smiles / skews**; constant $\sigma$ fails. |
| 6 | A3, C2† | Hedge vol $\sigma_h$ moves **mean** (BS-priced) or **shape** (Heston-priced); **does not remove** the ~2 std floor. |

†C2 is an **optional** simulation (fig7); same message as rows 2–4 with fair Heston premium.

### Detail (sections A–C)

**1 — Discretisation (A1).** Per-step error $e_k = -\frac{1}{2}\Gamma_k S_k^2 \sigma^2 \Delta t\,(Z_k^2-1)$; variance integral in derivations §10. Sim std **2.27** vs pred **2.43** at $N=8$; **≤~3%** gap for $N\ge32$.

**2 — Error floor (A2).** Same BS delta at $\sigma_h=20%$; sell at $V_{\mathrm{BS}}$ vs Heston paths. Std ratio Heston/BS at $N=1000$: **2.23 / 0.22**.

**3 — Mean under misspec (A3).** $\sigma_h \in \{15\%,20\%,25\%\}$, $\rho \in \{-0.7,0,0.7\}$: hedge does not create edge; **mispricing vs Heston fair** sets the mean.

**4 — Tails (A3).** Premium sets the centre; **spread and left tail** unchanged in order by $\sigma_h$.

**5 — Hedge vol (A3 + optional C2).** Always **BS delta**; $\sigma_h$ is the input to $\Delta(S,t,\sigma_h)$. Changing $\sigma_h$ does not eliminate the floor; at $K=120$, own-strike IV vs ATM trades **std vs skew** (fig7, `results_extension.json`).

**6 — Market (C1).** Bundled ^SPX: downward skew (e.g. wing IV **~44%** vs **~23%** on 2026-10-30 sample). IV as **quote coordinate**, not one forecast $\sigma$.

**7 — Foundations (B).** fig1–3, 5 verify Itô, martingales, $\mathbb{Q}$ pricing, PDE; support A/C simulators (e.g. $\mathbb{P}$ MC **~55%** too high vs replication price).

---

## Reproduce

```bash
pip install -r requirements.txt
pytest -q                      # 35 tests (~5 s)

python run_all.py              # B -> figures/fig1-5, results.json
python run_research.py         # A -> figures/fig8-9, results_research.json
python datapull.py             # C1 data (network) -> data/quotes_SPX_REAL.csv
python run_extension.py        # C -> fig6; optional fig7 + results_extension.json
```

**Layout:** `src/itobs/` (`bs`, `ito`, `martingale`, `pricing`, `hedging`, `heston`, `implied_vol`, `theory`), `tests/`, `docs/derivations.md`, `data/`, `run_*.py`, `datapull.py`.

---

## B. Foundations — trust the simulator

Derivations: [`docs/derivations.md`](docs/derivations.md). These experiments are **prerequisites for A**, not separate trading claims.

### B1. Itô lemma

```math
d\ln S = (\mu - \tfrac{1}{2}\sigma^2)\,dt + \sigma\,dW
```

![Itô lemma](figures/fig1_ito_lemma.png)

Log-return drifts at $\mu-\sigma^2/2$; $\mathbb{E}[S_T]$ still $S_0 e^{\mu T}$. Itô (left) integration keeps trading integrals fair vs right-endpoint bias.

### B2. Martingales

![Martingale](figures/fig2_martingale.png)

$W_t^2-t$ is a martingale; discounted $S$ is a martingale **only** under $\mathbb{Q}$.

### B3. Black–Scholes pricing

![Pricing](figures/fig3_pricing.png)

Closed form, $\mathbb{Q}$ MC, PDE, binomial agree; **$\mathbb{P}$ MC is wrong** for the replication price.

### B4. Greeks and PDE

![Greeks](figures/fig5_greeks.png)

Closed-form Greeks satisfy the BS PDE; near delta-neutral, $\Theta \approx -\frac{1}{2}\sigma^2 S^2 \Gamma$.

---

## A. Research — hedge error decomposition

### A1. Discretisation in a BS world (fig8)

Between rebalances, delta removes first-order risk; remainder is the $\Gamma$ / Itô term above.  
$\mathrm{Var}(\text{P&L}) \approx \frac{\sigma^4 \Delta t}{2}\int_0^T e^{-2rt}\,\mathbb{E}[\Gamma_t^2 S_t^4]\,dt$ → std $\propto 1/\sqrt{N}$.

| N | 8 | 32 | 128 | 512 | 1024 |
|---|---|---|---|---|---|
| Sim / pred std | 2.27 / 2.43 | 1.18 / 1.21 | 0.59 / 0.61 | 0.30 / 0.30 | 0.21 / 0.21 |
| Corr(P&L, Σ Γ-terms) | 0.957 | 0.976 | 0.988 | 0.994 | 0.996 |

![Error sources](figures/fig8_error_sources.png)

Mean P&L ≈ 0 for all $N$; discretisation risk is **sizable and predictable**.

### A2. Misspecification floor (fig8 panel b)

Heston paths, BS delta at $\sigma_h=20%$, sold at $V_{\mathrm{BS}}$:

| | N=16 | N=1000 |
|---|---|---|
| Std (Heston / BS paths) | 2.85 / 1.65 | **2.23 / 0.22** |
| 1% q (Heston / BS) | −9.53 / −4.61 | −7.16 / −0.56 |

More rebalances **do not** remove the Heston floor — stochastic vol with **stock-only** delta.

### A3. Mean and tails (fig9)

```math
\mathbb{E}[\text{P\&L}] = V_{\mathrm{BS}}(\sigma_h) - C_{\mathrm{Heston}}
```

![Mean prediction](figures/fig9_misspecification_mean.png)

Theory matches simulation in all 9 cases; std **2.2–3.2** and heavy left tails regardless of $\sigma_h$.

---

## C. Extension — market and optional supplement

### C1. ^SPX implied volatility (main market link)

`datapull.py` fetches chains (bid/ask stored; smile uses **mid**). `run_extension.py` inverts BS per strike with $S_{\mathrm{eff}}=Fe^{-rT}$. Align **r** in `datapull.py` and `REAL_R` in `run_extension.py` when refreshing.

![Smile](figures/fig6_smile.png)

**Link to A:** practice uses a **surface** of IVs; A2 shows that even with a chosen $\sigma_h$, **Heston dynamics** leave a hedge P&L floor when you only trade stock.

### C2. Hedge-vol supplement (optional, fig7)

Same message as A3/A2 with **Heston fair premium** (mean ≈ 0): P&L std **~2.3** vs **~0.4** in BS-world reference at 252 rebalances; swapping ATM vs own-strike IV at $K=120$ trades std vs skew — details in `results_extension.json`.

![Heston hedging](figures/fig7_heston_hedging.png)

Not required to reproduce the main research question; kept for completeness and fig7 in applications packet.

---

## Tests

`pytest -q` — BS parity/PDE/Greeks; Itô and martingales; $\mathbb{Q}$ vs $\mathbb{P}$ pricing; Heston↔BS and IV; hedge std slope, prediction match, $\Gamma$ correlation, misspec mean. Fixed seeds.

## Limitations

- ^SPX snapshot (Yahoo, flat **r**); descriptive smile only — no Heston calibration.
- Heston floor not decomposed (Vega shocks); no second instrument.
- A1 first-order in $\Delta t$; Heston paths: full-truncation Euler.
- Not covered: transaction costs, jumps, local vol. `fig4_hedging.png`: small $O(\Delta t)$ mean bias at extreme drift.

## References

- Shreve, *Stochastic Calculus for Finance II*
- Hull, *Options, Futures, and Other Derivatives*
- Black & Scholes (1973), Merton (1973)
- Heston (1993)
- Lewis (2001); Gatheral, *The Volatility Surface*

MIT License.
