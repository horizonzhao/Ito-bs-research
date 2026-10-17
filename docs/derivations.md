# Derivations
Notation: $dS = \mu S dt + \sigma S dW$. Itô multiplication table: $dt\cdot dt = 0$, $dt\cdot dW = 0$, $dW\cdot dW = dt$.

## 1. Itô's lemma for $f(S) = \ln S$
Second‑order expansion (second order is kept because $(dW)^2$ is of order $dt$):
$$
df = f'(S) dS + \tfrac12 f''(S) (dS)^2
$$
With $f' = 1/S$, $f'' = -1/S^2$, and $(dS)^2 = \sigma^2 S^2 dt$:
$$
\begin{aligned}
d \ln S &= \frac{1}{S}(\mu S dt + \sigma S dW) - \tfrac12 \frac{1}{S^2} \sigma^2 S^2 dt \\
&= (\mu - \tfrac12\sigma^2) dt + \sigma dW.
\end{aligned}
$$
Integrating: $S_T = S_0 \exp\big[(\mu - \tfrac12\sigma^2)T + \sigma W_T\big]$, so $\ln S_T \sim \mathcal{N}\big(\ln S_0 + (\mu - \tfrac12\sigma^2)T,\; \sigma^2 T\big)$.

Consequences: $\mathbb{E}[S_T] = S_0 e^{\mu T}$ (using $\mathbb{E}[e^{\sigma W_T}] = e^{\sigma^2 T/2}$), while the median is $S_0 e^{(\mu-\sigma^2/2)T}$.

## 2. The integral $\int_0^t W\,dW$
Apply Itô to $f(W) = W^2/2$:
$$
d(W^2/2) = W\,dW + \tfrac12 dt.
$$
Hence
$$
\int_0^t W\,dW = \tfrac12 W_t^2 - \tfrac12 t.
$$

Check on partition sums (left endpoints):
$$
\sum W_{t_k}(W_{t_{k+1}} - W_{t_k}) = \tfrac12 W_t^2 - \tfrac12 \sum (\Delta W_k)^2,
$$
and $\sum (\Delta W_k)^2 \to t$ (quadratic variation).

With right endpoints the sum is $\tfrac12 W_t^2 + \tfrac12 \sum (\Delta W_k)^2$, which converges to $\tfrac12 W_t^2 + t/2$; the two differ by exactly $t$.

## 3. Martingale properties
- $\mathbb{E}[W_t \mid \mathcal{F}_s] = W_s$ since increments are independent with mean 0.
- $W_t^2 - t$: write $W_t = W_s + (W_t - W_s)$. Then
$$
\mathbb{E}[W_t^2 \mid \mathcal{F}_s] = W_s^2 + 2W_s\cdot 0 + (t-s),
$$
so $\mathbb{E}[W_t^2 - t \mid \mathcal{F}_s] = W_s^2 - s$.

- Itô integral $I_t = \int_0^t H\,dW$ with $H$ adapted: each term $H_{t_k}(\Delta W_k)$ has $\mathbb{E}[\cdot \mid \mathcal{F}_{t_k}] = H_{t_k}\cdot \mathbb{E}[\Delta W_k] = 0$, so $I_t$ is a martingale and $\mathbb{E}[I_t] = 0$. This is exactly where "left endpoint" (adaptedness) is used.

## 4. Black–Scholes PDE by delta hedging
Let $V(S,t)$ be the option price and $\Pi = V - \Delta S$ with $\Delta$ held fixed over $dt$.

Itô on $V$:
$$
dV = \big(V_t + \mu S V_S + \tfrac12\sigma^2 S^2 V_{SS}\big) dt + \sigma S V_S\,dW.
$$

Then
$$
d\Pi = dV - \Delta\,dS
= \big(V_t + \mu S V_S + \tfrac12\sigma^2 S^2 V_{SS} - \Delta\mu S\big) dt
+ \sigma S (V_S - \Delta)\,dW.
$$

Choose $\Delta = V_S$: the $dW$ term vanishes and the $\mu$ terms cancel:
$$
d\Pi = \big(V_t + \tfrac12\sigma^2 S^2 V_{SS}\big) dt.
$$

No randomness $\Rightarrow$ no arbitrage requires $d\Pi = r\Pi\,dt = r(V - S V_S) dt$. Therefore
$$
V_t + \tfrac12\sigma^2 S^2 V_{SS} + rS V_S - rV = 0,\quad V(S,T) = (S-K)^+.
$$
$\mu$ does not appear.

## 5. Risk‑neutral representation (Feynman–Kac)
Feynman–Kac: the solution of the PDE above with terminal payoff $g$ is
$$
V(S,t) = e^{-r(T-t)} \mathbb{E}^\mathbb{Q}\big[\,g(S_T) \,\big|\, S_t = S\,\big],
$$
where $dS = rS\,dt + \sigma S\,dW^\mathbb{Q}$.

Equivalently: under $\mathbb{Q}$, $e^{-rt}V(S_t,t)$ is a martingale. Check by Itô:
$$
d(e^{-rt}V)
= e^{-rt}\big(V_t + rS V_S + \tfrac12\sigma^2 S^2 V_{SS} - rV\big)dt
+ (\sigma\text{-term})\,dW^\mathbb{Q},
$$
and the $dt$ coefficient is the PDE, i.e. zero.

## 6. Closed form
Under $\mathbb{Q}$, $\ln S_T \sim \mathcal{N}\big(\ln S_0 + (r - \sigma^2/2)T,\; \sigma^2 T\big)$. Computing $e^{-rT}\mathbb{E}^\mathbb{Q}\big[(S_T-K)^+\big]$ gives
$$
C = S_0\,N(d_1) - K e^{-rT} N(d_2),
$$
$$
d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}},\quad d_2 = d_1 - \sigma\sqrt{T}.
$$

## 7. Greeks identity
Substituting $\Theta = V_t$, $\Delta = V_S$, $\Gamma = V_{SS}$ into the PDE:
$$
\Theta + \tfrac12\sigma^2 S^2 \Gamma + rS\Delta - rV = 0.
$$

For a delta‑hedged position $(V - \Delta S)$ this reads $\Theta_\Pi + \tfrac12\sigma^2 S^2 \Gamma = r\Pi$, so for small $r$, $\Theta \approx -\tfrac12\sigma^2 S^2 \Gamma$.

## 8. Implied volatility and why a smile contradicts BS
For a market price $C_{\text{mkt}}(K,T)$, the implied vol $\sigma_{\text{imp}}(K,T)$ solves
$$
C_{\text{BS}}(S,K,T,r,\sigma_{\text{imp}}) = C_{\text{mkt}}.
$$
Since $C_{\text{BS}}$ is strictly increasing in $\sigma$ (Vega $= S\,\phi(d_1)\sqrt{T} > 0$), the solution is unique whenever $C_{\text{mkt}}$ lies in the no‑arbitrage interval $\big(\max(S - Ke^{-rT},0),\; S\big)$, so root‑finding is safe. If the BS assumption (constant $\sigma$) held, $\sigma_{\text{imp}}$ would be the same for every $(K,T)$; a non‑flat surface therefore falsifies it.

## 9. Heston model (what is used, not re‑derived)
Under $\mathbb{Q}$:
$$
\begin{cases}
dS = rS\,dt + \sqrt{v}\,S\,dW_1,\\
dv = \kappa(\theta - v)dt + \xi\sqrt{v}\,dW_2,\\
d\langle W_1,W_2\rangle = \rho\,dt.
\end{cases}
$$

The log‑price has an affine characteristic function
$$
\mathbb{E}\big[e^{iuX}\big] = \exp\big(C(u,T) + D(u,T)\,v_0\big),
$$
with $C,D$ solving Riccati ODEs (closed forms in `heston.py`). The call price follows from Fourier inversion (Lewis 2001). Intuition for the skew: $\rho<0$ ties price drops to vol increases, making the left tail of returns heavier than lognormal, so low‑strike options cost more than flat‑vol BS predicts. Because there are two random drivers ($W_1,W_2$) but one tradable stock, the delta hedge of Section 4 cannot remove all risk: the market is incomplete and the price is no longer determined by hedging with the stock alone.

## 10. Predicting the discrete delta‑hedging error (used in README §A1)
Sell one call, hold $\Delta_k = V_S(S_{t_k},t_k)$ shares over $[t_k,t_{k+1}]$, $\Delta t = T/N$. By Itô/Taylor, the hedged portfolio's change over one step is the residual of a second‑order expansion:
$$
\begin{aligned}
\Delta\Pi_k
&= V_t\,\Delta t + V_S\,\Delta S + \tfrac12 V_{SS}\,(\Delta S)^2 - \Delta_k\,\Delta S
- r(V - \Delta_k S)\Delta t + (\text{higher order})\\
&= \tfrac12 \Gamma_k \big[(\Delta S)^2 - \sigma^2 S_k^2 \Delta t\big] + O(\Delta t^{3/2})
\end{aligned}
$$
using the Black–Scholes PDE to cancel the terms of order $\Delta t$. With $\Delta S \approx \sigma S_k \sqrt{\Delta t}\cdot Z_k$, $Z_k \sim \mathcal{N}(0,1)$:
$$
e_k \approx \tfrac12 \Gamma_k S_k^2 \sigma^2 \Delta t\,(Z_k^2 - 1)
$$
(sign: the short‑option holder earns $-e_k$).

Key properties: $\mathbb{E}[e_k \mid \mathcal{F}_{t_k}] = 0$ (martingale difference, because $\Gamma_k S_k$ is known at $t_k$: the left‑endpoint rule again) and $\mathrm{Var}(Z^2-1) = 2$. Hence, after discounting,
$$
\mathrm{Var}(\text{P\&L})
= \sum_k e^{-2r t_{k+1}}\, \mathbb{E}\big[\big(\tfrac12\Gamma_k S_k^2 \sigma^2 \Delta t\big)^2\big] \cdot 2
\;\approx\; \frac{\sigma^4 \Delta t}{2}\int_0^T e^{-2rt}\,\mathbb{E}\big[\Gamma_t^2 S_t^4\big] dt,
$$
which is proportional to $\Delta t = T/N$, so $\text{std} \propto N^{-1/2}$. `theory.predicted_hedge_std` evaluates the integral with a lognormal law for $S_t$. Near expiry $\Gamma^2 S^4 \sim 1/\tau$ for $S$ near $K$, but averaging over $S_t$ gives an integrable $1/\sqrt{\tau}$ singularity, handled with the substitution $\tau = T u^2$.

Note this treats the errors as uncorrelated across steps (martingale differences) and uses the continuous‑time $\Gamma$; it is a first‑order approximation in $\Delta t$.

## 11. Mean P&L under misspecification (README §A3)
Let the true dynamics be any model with risk‑neutral measure $\mathbb{Q}$, and let the hedger hold shares $\Delta_k$ (any adapted choice) in a self‑financing strategy started with wealth equal to the premium $p$. The discounted wealth is a $\mathbb{Q}$‑martingale: $\mathbb{E}^\mathbb{Q}\big[e^{-rT} W_T\big] = p$. The short option costs $\mathbb{E}^\mathbb{Q}\big[e^{-rT}(S_T-K)^+\big] = C_{\text{true}}$. Therefore
$$
\mathbb{E}^\mathbb{Q}[\text{discounted P\&L}] = p - C_{\text{true}}.
$$

With $p = V_{\text{BS}}(\sigma_h)$ and $C_{\text{true}} = C_{\text{Heston}}$ this is the prediction tested in §A3. Two consequences:
(i) the mean is independent of the hedge ratios used; they only affect the variance and higher moments;
(ii) selling at the true model price makes the mean zero, which is why the mean P&L in §A1 and Section C is $\approx 0$.
