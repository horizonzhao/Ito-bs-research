# Where Does Delta-Hedging Error Come From?

### Itô calculus, martingales and Black–Scholes, tested against stochastic volatility

---

## Abstract

This repository studies the **residual profit-and-loss (P&L)** left after a trader sells an option and hedges it with the Black–Scholes delta in the underlying stock alone. Two mechanisms compete for the same residual: **(i) discrete rebalancing**, which exposes the second-order Itô (gamma) term that continuous-time theory absorbs; and **(ii) model misspecification**, when the true dynamics are richer than geometric Brownian motion—for example Heston stochastic volatility—while the hedge still uses a constant-volatility delta. The theoretical decomposition is written out in [`docs/derivations.md`](docs/derivations.md); empirical claims are checked by Monte Carlo simulation and **35 pytest** regression tests. A market-facing extension documents why a **single** Black–Scholes volatility cannot reconcile **SPX** implied-volatility smiles. Implementation is in Python under `src/itobs/`.

The central quantitative findings are as follows. In a Black–Scholes world, the standard deviation of hedge error is **predictable** in advance (simulation tracks theory within roughly **3–7%**) and scales with the number of rebalances $N$ as $N^{-1/2}$, consistent with a summed gamma–variance story. When paths are generated from **Heston** but the hedge remains **stock-only BS delta**, the P&L standard deviation **does not** vanish with $N$: it **floors near ~2.2** even as $N$ grows from 16 to 1000, while the BS-world std at comparable settings can fall to **~0.22**. Under intentional mispricing (sell at a BS premium, fair value from Heston), the **mean** P&L is predictable as **premium minus fair value**, but **tail risk stays large**—a distinction that matters for risk management as much as for expected edge.

---

## Research question and scope

A standard delta-hedging program removes first-order exposure to small moves in $S$ between rebalancing dates. Whatever remains in the hedging account is the object of this study. The guiding question is:

> **How much of the residual P&L comes from (i) hedging only at discrete times, and how much from (ii) using the wrong model? Can each component be forecast before trading?**

**Scope.** “Truth” is fixed **by the simulation design**, not by which model best forecasts live **SPX** returns. Experiments compare paths drawn under Black–Scholes versus Heston; the hedge is always the **Black–Scholes delta in stock only** (no variance swap, no vega hedge). Section **C1** uses a bundled **SPX** option chain only to show that constant $\sigma$ is inconsistent with cross-sectional quotes; **Heston is not calibrated** to those prices. Baseline parameters are $`S_0=K=100`$, $`T=1`$, $`r=3\%`$; Black–Scholes $`\sigma=20\%`$; Heston $`v_0=\theta=0.04`$, $\kappa=2$, $\xi=0.5$, $\rho=-0.7$. Numeric outputs are archived in `results_research.json` and `results_extension.json`.

---

## Conclusions

### Discrete rebalancing in a correctly specified model

When simulated paths **are** Black–Scholes, residual P&L is dominated by the **second-order Itô / gamma** contribution from rebalancing finitely often. Per-step error takes the form $`e_k = -\frac{1}{2}\Gamma_k S_k^2 \sigma^2 \Delta t\,(Z_k^2-1)`$; aggregating variance yields the integral formula in derivations §10, hence std $\propto 1/\sqrt{N}$. The module `theory.predicted_hedge_std` implements that prediction. Simulated std at $N=8$ is **2.27** versus predicted **2.43**; for $N \ge 32$ the gap is **≤~3%**. The correlation between total P&L and the sum of gamma terms ranges **0.96–0.99** across $N$, and the log–log slope of std versus $N$ is **−0.491** (theory **−0.5**). Mean P&L stays near zero: discretisation risk is **material but forecastable**, not a hidden drift.

### Misspecification when volatility is stochastic

When paths are **Heston** but the trader still computes $`\Delta(S,t,\sigma_h)`$ from Black–Scholes and trades **stock only**, residual risk reflects **market incompleteness** under the chosen hedge—not merely “too few” rebalance steps. At fixed $`\sigma_h=20\%`$ and sale at $`V_{\mathrm{BS}}`$, simulated std falls from **2.85** to **2.23** as $N$ goes from 16 to 1000 on Heston paths, whereas on BS paths at the same settings it falls to **~0.22**. The ratio **2.23 / 0.22** at $N=1000$ isolates the **floor** from the $N^{-1/2}$ discretisation decay. Left skew and fat tails persist: the 1% quantile is **−7.2** (Heston) versus **−0.56** (BS) at $N=1000$. No closed-form expression for the floor is claimed here; the level **~2.2** is **measured** by simulation.

### Predictability of mean versus tails

If the option is sold at a Black–Scholes price with input $`\sigma_h`$, theory gives $`\mathbb{E}[\operatorname{PnL}] = V_{\mathrm{BS}}(\sigma_h) - C_{\mathrm{Heston}}`$ across nine cases ($`\sigma_h \in \{15\%,\,20\%,\,25\%\}`$, $\rho \in \{-0.7,\,0,\,0.7\}$); all simulation means match within $|z| \le 1.2$. **Mispricing relative to Heston fair value sets the mean**; the delta hedge does not manufacture alpha. Changing $`\sigma_h`$ moves the **mean** when the premium is BS-based, but **does not remove** the ~2 std floor or the heavy left tail (std **2.2–3.2**, 1% quantiles **−3.7 to −12.9**). Premium sets the centre of the distribution; **spread and tail behaviour** are largely unchanged in order of magnitude across $`\sigma_h`$. Optional extension **C2** (fig7, Heston fair premium so mean ≈ 0) reinforces the same separation: P&L std **~2.3** versus **~0.4** in a BS-world reference at 252 rebalances; swapping ATM versus own-strike implied vol at $K=120$ trades **std versus skew** without eliminating the floor (`results_extension.json`).

### Market link

Bundled **SPX** data show **downward skew** in implied volatility (e.g. wing IV **~44%** versus **~23%** on a 2026-10-30 sample). In practice, traders quote options in an **IV surface**; section **A2** shows that even after picking a hedge vol $`\sigma_h`$, **Heston dynamics** with **stock-only** delta leave a **non-vanishing** hedge P&L scale. IV is used as a **quote coordinate**, not as a single forecast $\sigma$ for the entire economy.

### Empirical summary

| Theme | Location | Result |
|--------|----------|--------|
| Discretisation | A1, fig8 | $\Gamma$ term narrative; pred vs sim **~3–7%**; slope **−0.491** (theory **−0.5**) |
| Misspec floor | A2, fig8 | Heston std **→ ~2.2**; BS-world std **→ ~0.22** at $N=1000$ |
| Mean under misspec | A3, fig9 | Mean = **premium − Heston fair**; predictable in all 9 cases |
| Tails | A3 | **Mean ≠ safety:** std **2.2–3.2**; 1% q **−3.7 to −12.9** |
| Market smile | C1, fig6 | Real IV **smiles / skews**; constant $\sigma$ fails |
| Hedge vol | A3, C2† | $`\sigma_h`$ shifts mean (BS-priced) or shape (Heston-priced); **floor ~2 std remains** |

†C2 is optional (fig7); same message as A2–A3 with Heston fair premium.

---

## Repository workflow

The project is organised in three layers that should be read or executed **in order**; each layer supplies trust for the next. **B (Foundations)** runs `run_all.py` and asks whether Itô calculus, martingales, and Black–Scholes pricing are implemented consistently—this is the **sanity check on the simulator**. **A (Research)** runs `run_research.py` and addresses the **main decomposition** of hedge P&L. **C (Extension)** uses `datapull.py` and `run_extension.py` for the **SPX** smile and an optional hedge-vol supplement (fig7). For a quick visual pass: **fig8–9** (core research), **fig6** (market), **fig1–3** (foundations); **fig7** optionally reinforces A2/A3.

```
  B (math sanity)  ──►  A (hedge P&L decomposition)  ──►  C (market smile + optional supplement)
```

| Step | Script | Role |
|------|--------|------|
| **B. Foundations** | `run_all.py` | Verify Itô, martingales, BS pricing (trust the simulator) |
| **A. Research** | `run_research.py` | Discretisation vs wrong model; predictability |
| **C. Extension** | `datapull.py`, `run_extension.py` | SPX smile; optional hedge-vol study (fig7) |

**Reproduce:**

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

## B. Foundations — trusting the simulator

Full derivations appear in [`docs/derivations.md`](docs/derivations.md). The experiments below are **prerequisites for A**, not standalone trading claims.

### B1. Itô lemma

Under $dS = \mu S\,dt + \sigma S\,dW$, Itô’s lemma gives

```math
d\ln S = (\mu - \tfrac{1}{2}\sigma^2)\,dt + \sigma\,dW.
```

![Itô lemma](figures/fig1_ito_lemma.png)

Log-return drift is $\mu - \tfrac{1}{2}\sigma^2$, while $`\mathbb{E}[S_T]=S_0 e^{\mu T}`$. **Left-endpoint** (Itô) integration aligns simulated trading integrals with the theory; right-endpoint rules introduce a systematic bias in $\int W\,dW$ and related objects.

### B2. Martingales

![Martingale](figures/fig2_martingale.png)

Brownian motion has independent increments of mean zero; $`W_t^2 - t`$ is a martingale. Discounted stock is a martingale under the pricing measure $\mathbb{Q}$, not under every physical $\mathbb{P}$—a distinction reused when separating replication prices from naive physical Monte Carlo.

### B3. Black–Scholes pricing

![Pricing](figures/fig3_pricing.png)

Closed-form values agree with $\mathbb{Q}$ Monte Carlo, PDE finite-difference, and binomial trees; $`\mathbb{P}`$-measure Monte Carlo is **systematically wrong** for the replication price (e.g. ~55% too high in the bundled check), illustrating that hedging and pricing are tied to $\mathbb{Q}$.

### B4. Greeks and PDE

![Greeks](figures/fig5_greeks.png)

Analytic Greeks satisfy the Black–Scholes PDE. Near delta-neutrality, $\Theta \approx -\frac{1}{2}\sigma^2 S^2 \Gamma$, linking time decay to the same second-order spatial sensitivity that drives discrete hedge error in section A.

---

## A. Research — hedge error decomposition

### A1. Discretisation in a Black–Scholes world (fig8)

Between rebalances, delta hedging cancels the first-order $dS$ term in the option’s Itô expansion; what remains is the **gamma /** $(dS)^2$ channel. Theory gives

```math
\operatorname{Var}(\operatorname{PnL}) \approx \frac{\sigma^4 \Delta t}{2}\int_0^T e^{-2rt}\,\mathbb{E}[\Gamma_t^2 S_t^4]\,dt
```

so standard deviation scales as $N^{-1/2}$ for equal time steps. Simulation versus prediction:

| $N$ | 8 | 32 | 128 | 512 | 1024 |
|-----|---|---|---|---|---|
| Sim / pred std | 2.27 / 2.43 | 1.18 / 1.21 | 0.59 / 0.61 | 0.30 / 0.30 | 0.21 / 0.21 |
| $\operatorname{Corr}(\operatorname{PnL}, \sum \Gamma)$ | 0.957 | 0.976 | 0.988 | 0.994 | 0.996 |

![Error sources](figures/fig8_error_sources.png)

Mean P&L is negligible at all $N$: the object to manage is **volatility of error**, and that volatility is **forecastable**.

### A2. Misspecification floor (fig8, panel b)

Paths are **Heston**; the trader sells at $`V_{\mathrm{BS}}`$ and hedges with BS delta at $`\sigma_h=20\%`$:

| | $N=16$ | $N=1000$ |
|---|--------|----------|
| Std (Heston / BS paths) | 2.85 / 1.65 | **2.23 / 0.22** |
| 1% quantile (Heston / BS) | −9.53 / −4.61 | −7.16 / −0.56 |

Increasing $N$ **compresses** error in the BS world but **does not remove** the Heston-world scale—a signature of **stochastic volatility hedged with stock only**.

### A3. Mean and tails under mispricing (fig9)

When the sale price is $`V_{\mathrm{BS}}(\sigma_h)`$ and fair value is $`C_{\mathrm{Heston}}`$,

```math
\mathbb{E}[\operatorname{PnL}] = V_{\mathrm{BS}}(\sigma_h) - C_{\mathrm{Heston}}.
```

![Mean prediction](figures/fig9_misspecification_mean.png)

Simulation means match this identity in all nine parameter cells; **standard deviation and left-tail quantiles** remain large regardless of $`\sigma_h`$, so **expected P&L is not a sufficient risk statistic**.

---

## C. Extension — market data and optional supplement

### C1. SPX implied volatility

`datapull.py` downloads option chains (bid and ask stored; the smile plot uses **mid**). `run_extension.py` inverts Black–Scholes per strike with $`S_{\mathrm{eff}}=Fe^{-rT}`$. When refreshing data, align the rate in `datapull.py` with `REAL_R` in `run_extension.py`.

![Smile](figures/fig6_smile.png)

The figure is **descriptive**: it motivates why practitioners work with a **surface** of implied volatilities. Section **A2** then shows that, even after fixing a hedge input $`\sigma_h`$, **Heston truth** with **stock-only** delta leaves a **persistent** residual P&L scale that more frequent rebalancing cannot erase.

### C2. Hedge-vol supplement (optional, fig7)

This block repeats the A3/A2 logic with **Heston fair premium** (mean ≈ 0) and compares hedge-vol choices (ATM versus own-strike IV at $K=120$). It is **not required** to answer the main research question; it is retained for completeness and for fig7 in application packets.

![Heston hedging](figures/fig7_heston_hedging.png)

---

## Validation and limitations

**Tests.** `pytest -q` covers Black–Scholes put–call parity, PDE, and Greeks; Itô and martingale identities; $\mathbb{Q}$ versus $\mathbb{P}$ pricing; Heston–BS consistency and implied vol; hedge-error std slope, prediction match, gamma correlation, and misspecification mean. Seeds are fixed for reproducibility.

**Limitations.** The **SPX** slice is a single Yahoo snapshot with flat **r**; smiles are not fit to Heston. The Heston **floor** is not decomposed into vega or other channels, and no second hedging instrument is used. A1 theory is first-order in $\Delta t$; Heston paths use full-truncation Euler. Transaction costs, jumps, and local volatility are out of scope. `fig4_hedging.png` documents a small $O(\Delta t)$ mean bias under extreme drift.

---

## References

- Shreve, *Stochastic Calculus for Finance II*
- Hull, *Options, Futures, and Other Derivatives*
- Black & Scholes (1973); Merton (1973)
- Heston (1993)
- Lewis (2001); Gatheral, *The Volatility Surface*

MIT License.
