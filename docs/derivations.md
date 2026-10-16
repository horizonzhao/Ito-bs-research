# Derivations

Notation: `dS = μS dt + σS dW`. Itô multiplication table: `dt·dt = 0`, `dt·dW = 0`, `dW·dW = dt`.

## 1. Itô's lemma for f(S) = ln S

Second-order expansion (second order is kept because `(dW)²` is of order `dt`):

    df = f'(S) dS + ½ f''(S) (dS)²

With `f' = 1/S`, `f'' = −1/S²`, and `(dS)² = σ²S² dt`:

    d ln S = (1/S)(μS dt + σS dW) − ½ (1/S²) σ²S² dt
           = (μ − ½σ²) dt + σ dW.

Integrating: `S_T = S₀ exp[(μ − ½σ²)T + σW_T]`, so `ln S_T ~ N(ln S₀ + (μ − ½σ²)T, σ²T)`.
Consequences: `E[S_T] = S₀e^{μT}` (using `E[e^{σW_T}] = e^{σ²T/2}`), while the median is `S₀e^{(μ−σ²/2)T}`.

## 2. The integral ∫₀ᵗ W dW

Apply Itô to `f(W) = W²/2`: `d(W²/2) = W dW + ½ dt`. Hence

    ∫₀ᵗ W dW = ½ W_t² − ½ t.

Check on partition sums (left endpoints):
`Σ W_{t_k}(W_{t_{k+1}} − W_{t_k}) = ½ W_t² − ½ Σ (ΔW_k)²`, and `Σ (ΔW_k)² → t` (quadratic variation).
With right endpoints the sum is `½ W_t² + ½ Σ (ΔW_k)²`, which converges to `½W_t² + t/2`; the two differ by exactly `t`.

## 3. Martingale properties

- `E[W_t | F_s] = W_s` since increments are independent with mean 0.
- `W_t² − t`: write `W_t = W_s + (W_t − W_s)`. Then
  `E[W_t² | F_s] = W_s² + 2W_s·0 + (t − s)`, so `E[W_t² − t | F_s] = W_s² − s`.
- Itô integral `I_t = ∫₀ᵗ H dW` with `H` adapted: each term `H_{t_k}(ΔW_k)` has `E[· | F_{t_k}] = H_{t_k}·E[ΔW_k] = 0`, so `I_t` is a martingale and `E[I_t] = 0`. This is exactly where "left endpoint" (adaptedness) is used.

## 4. Black–Scholes PDE by delta hedging

Let `V(S,t)` be the option price and `Π = V − ΔS` with Δ held fixed over `dt`.

Itô on V:

    dV = (V_t + μS V_S + ½σ²S²V_SS) dt + σS V_S dW.

Then

    dΠ = dV − Δ dS = (V_t + μS V_S + ½σ²S²V_SS − ΔμS) dt + σS (V_S − Δ) dW.

Choose `Δ = V_S`: the `dW` term vanishes and the `μ` terms cancel:

    dΠ = (V_t + ½σ²S²V_SS) dt.

No randomness ⇒ no arbitrage requires `dΠ = rΠ dt = r(V − S V_S) dt`. Therefore

    V_t + ½σ²S²V_SS + rS V_S − rV = 0,   V(S,T) = (S − K)⁺.

μ does not appear.

## 5. Risk-neutral representation (Feynman–Kac)

Feynman–Kac: the solution of the PDE above with terminal payoff `g` is

    V(S,t) = e^{−r(T−t)} E^Q[ g(S_T) | S_t = S ],  where dS = rS dt + σS dW^Q.

Equivalently: under Q, `e^{−rt}V(S_t,t)` is a martingale. Check by Itô: `d(e^{−rt}V) = e^{−rt}(V_t + rSV_S + ½σ²S²V_SS − rV)dt + (σ-term)dW^Q`, and the `dt` coefficient is the PDE, i.e. zero.

## 6. Closed form

With `ln S_T ~ N(ln S₀ + (r − σ²/2)T, σ²T)` under Q, computing `e^{−rT}E^Q[(S_T − K)⁺]` gives

    C = S₀ N(d₁) − K e^{−rT} N(d₂),
    d₁ = [ln(S₀/K) + (r + σ²/2)T]/(σ√T),  d₂ = d₁ − σ√T.

## 7. Greeks identity

Substituting `Θ = V_t`, `Δ = V_S`, `Γ = V_SS` into the PDE:

    Θ + ½σ²S²Γ + rSΔ − rV = 0.

For a delta-hedged position (`V − ΔS`) this reads `Θ_Π + ½σ²S²Γ = rΠ`, so for small r, `Θ ≈ −½σ²S²Γ`.

## 8. Implied volatility and why a smile contradicts BS

For a market price `C_mkt(K,T)`, the implied vol `σ_imp(K,T)` solves `C_BS(S,K,T,r,σ_imp) = C_mkt`. Since `C_BS` is strictly increasing in σ (Vega = `S φ(d₁)√T > 0`), the solution is unique whenever `C_mkt` lies in the no-arbitrage interval `(max(S − Ke^{−rT}, 0), S)`, so root-finding is safe. If the BS assumption (constant σ) held, `σ_imp` would be the same for every `(K,T)`; a non-flat surface therefore falsifies it.

## 9. Heston model (what is used, not re-derived)

Under Q: `dS = rS dt + √v S dW₁`, `dv = κ(θ − v)dt + ξ√v dW₂`, `d⟨W₁,W₂⟩ = ρ dt`. The log-price has an affine characteristic function `E[e^{iuX}] = exp(C(u,T) + D(u,T)v₀)`, with `C, D` solving Riccati ODEs (closed forms in `heston.py`). The call price follows from Fourier inversion (Lewis 2001). Intuition for the skew: ρ<0 ties price drops to vol increases, making the left tail of returns heavier than lognormal, so low-strike options cost more than flat-vol BS predicts. Because there are two random drivers (W₁, W₂) but one tradable stock, the delta hedge of Section 4 cannot remove all risk: the market is incomplete and the price is no longer determined by hedging with the stock alone.

## 10. Predicting the discrete delta-hedging error (used in README §A1)

Sell one call, hold `Δ_k = V_S(S_{t_k}, t_k)` shares over `[t_k, t_{k+1}]`, `Δt = T/N`. By Itô/Taylor, the hedged portfolio's change over one step is the residual of a second-order expansion:

    ΔΠ_k = V_t Δt + V_S ΔS + ½ V_SS (ΔS)² − Δ_k ΔS − r(V − Δ_k S)Δt  + (higher order)
         = ½ Γ_k [ (ΔS)² − σ²S_k² Δt ] + O(Δt^{3/2})

using the Black–Scholes PDE to cancel the terms of order Δt. With `ΔS ≈ σ S_k √Δt · Z_k`, `Z_k ~ N(0,1)`:

    e_k ≈ ½ Γ_k S_k² σ² Δt (Z_k² − 1)     (sign: the short-option holder earns −e_k)

Key properties: `E[e_k | F_{t_k}] = 0` (martingale difference, because Γ_k S_k is known at t_k: the left-endpoint rule again) and `Var(Z²−1) = 2`. Hence, after discounting,

    Var(P&L) = Σ_k e^{−2r t_{k+1}} E[(½Γ_k S_k² σ² Δt)²] · 2
             ≈ (σ⁴ Δt / 2) ∫₀ᵀ e^{−2rt} E[Γ_t² S_t⁴] dt,

which is proportional to Δt = T/N, so std ∝ N^{−1/2}. `theory.predicted_hedge_std` evaluates the integral with a lognormal law for S_t. Near expiry `Γ² S⁴ ∼ 1/τ` for S near K, but averaging over S_t gives an integrable `1/√τ` singularity, handled with the substitution `τ = T u²`.

Note this treats the errors as uncorrelated across steps (martingale differences) and uses the continuous-time Γ; it is a first-order approximation in Δt.

## 11. Mean P&L under misspecification (README §A3)

Let the true dynamics be any model with risk-neutral measure Q, and let the hedger hold shares `Δ_k` (any adapted choice) in a self-financing strategy started with wealth equal to the premium `p`. The discounted wealth is a Q-martingale: `E^Q[e^{−rT} W_T] = p`. The short option costs `E^Q[e^{−rT}(S_T−K)⁺] = C_true`. Therefore

    E^Q[discounted P&L] = p − C_true.

With `p = V_BS(σ_h)` and `C_true = C_Heston` this is the prediction tested in §A3. Two consequences: (i) the mean is independent of the hedge ratios used; they only affect the variance and higher moments; (ii) selling at the true model price makes the mean zero, which is why the mean P&L in §A1 and Section C is ≈ 0.
