"""Pull real option quotes from Yahoo Finance into data/quotes_<symbol>_REAL.csv.

Default symbol: ^SPX (S&P 500 index European-style options, CBOE cash settlement).

Pipeline:
1. Use yfinance to fetch spot and option chains per expiry (calls + puts)
2. Prefer bid/ask mid; when bid/ask are 0 off-hours, use lastPrice (still usable for liquid contracts)
3. Per expiry, infer forward F from put-call parity (index options: F suffices; no separate dividend model)
4. Keep OTM only: calls for K >= F, puts for K < F (consistent with typical smile construction)
5. Write CSV with strike, maturity, price, type, plus expiry, forward, etc.

Plotting and JSON: see run_extension.py.
"""
import time
import pathlib

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = None

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# ---------------- Params (risk-free R must match REAL_R in run_extension.py) ----------------
TICKER = "^SPX"
R = 0.045
MIN_DAYS = 7
MAX_EXPIRIES = 6
SCAN_EXPIRIES = 18
MONEYNESS = (0.85, 1.15)
MAX_REL_SPREAD = 0.5
MIN_OPEN_INTEREST = 0
USE_LAST_PRICE = True
SLEEP_SEC = 1.2
RETRY_MAX = 8
RETRY_BASE_SEC = 15.0
# --------------------------------------


def _is_rate_limit(exc):
    if YFRateLimitError is not None and isinstance(exc, YFRateLimitError):
        return True
    return exc.__class__.__name__ == "YFRateLimitError"


def yahoo_retry(label, fn):
    """On Yahoo 429 rate limit: exponential backoff and retry."""
    last = None
    for attempt in range(RETRY_MAX):
        try:
            return fn()
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise
            last = exc
            wait = RETRY_BASE_SEC * (2 ** attempt)
            print(f"Yahoo rate limit [{label}], retry in {wait:.0f}s ({attempt + 1}/{RETRY_MAX})...")
            time.sleep(wait)
    raise last


def fetch_spot(tk):
    """Index spot: fast_info (one fewer request than history)."""
    def _load():
        px = float(tk.fast_info["last_price"])
        if not np.isfinite(px) or px <= 0:
            raise RuntimeError("fast_info has no valid last_price")
        return px

    return yahoo_retry("spot", _load)


def list_expiries(tk):
    """List of available expiry dates."""
    return yahoo_retry("expiries", lambda: list(tk.options))


def fetch_option_chain(tk, expiry):
    """Option chain for a single expiry."""
    return yahoo_retry(f"chain {expiry}", lambda: tk.option_chain(expiry))


def year_frac(expiry, now):
    """Year fraction from now to expiry (expiry counted at 16:00 close that day)."""
    return (pd.Timestamp(expiry) + pd.Timedelta(hours=16) - now).total_seconds() / (365.0 * 86400.0)


def attach_mid(df, kind):
    """Option mid price: bid/ask mid when available; else lastPrice if USE_LAST_PRICE."""
    need = ["strike", "bid", "ask", "volume", "openInterest"]
    if "lastPrice" not in df.columns:
        df = df.copy()
        df["lastPrice"] = np.nan
    out = df[need + ["lastPrice"]].copy()
    out["volume"] = out["volume"].fillna(0)
    out["openInterest"] = out["openInterest"].fillna(0)
    bid, ask = out["bid"].fillna(0), out["ask"].fillna(0)
    last = out["lastPrice"].fillna(0)
    has_ba = (bid > 0) & (ask > 0) & (ask >= bid)
    mid = np.where(has_ba, 0.5 * (bid + ask), np.nan)
    if USE_LAST_PRICE:
        mid = np.where(np.isnan(mid) & (last > 0), last, mid)
    out["mid"] = mid
    out["has_bidask"] = has_ba
    out = out[out["mid"].notna() & (out["mid"] > 0)]
    out["type"] = kind
    return out


def quoted_side(df, kind):
    """Contracts with valid mid, for inferring forward (no open-interest filter)."""
    return attach_mid(df, kind)


def clean_side(df, kind):
    """Contracts for output: spread filter (two-sided quotes only) + open interest."""
    out = quoted_side(df, kind)
    spread_ok = (~out["has_bidask"]) | ((out["ask"] - out["bid"]) / out["mid"] <= MAX_REL_SPREAD)
    out = out[spread_ok]
    out = out[out.openInterest >= MIN_OPEN_INTEREST]
    return out.drop(columns=["has_bidask"], errors="ignore")


def overlap_strikes(calls, puts):
    """Count strikes with both call and put quotes."""
    if calls.empty or puts.empty:
        return 0
    return len(set(calls.strike) & set(puts.strike))


def implied_forward(calls, puts, T, r, spot):
    """Put-call parity C - P = e^{-rT}(F - K) for F; median over strikes nearest spot (up to 5)."""
    both = calls[["strike", "mid"]].merge(puts[["strike", "mid"]], on="strike", suffixes=("_c", "_p"))
    if both.empty:
        return None
    both["F"] = both.strike + np.exp(r * T) * (both.mid_c - both.mid_p)
    n = min(5, len(both))
    near = both.iloc[np.argsort((both.strike - spot).abs().to_numpy())[:n]]
    return float(near.F.median())


def forward_fallback(spot, T, r):
    """When strike pairing is thin: F approx S e^{rT} (no dividend on spot; ^SPX should prefer parity F)."""
    return float(spot * np.exp(r * T))


def pick_expiries(tk, spot, now, expiries):
    """Scan expiries; pick those with most strike overlap and OTM rows left after clean; cache chains."""
    candidates = [e for e in expiries if year_frac(e, now) * 365.0 >= MIN_DAYS][:SCAN_EXPIRIES]
    scored = []
    cached = {}
    for e in candidates:
        time.sleep(SLEEP_SEC)
        chain = fetch_option_chain(tk, e)
        cached[e] = chain
        T = year_frac(e, now)
        qc, qp = quoted_side(chain.calls, "C"), quoted_side(chain.puts, "P")
        ov = overlap_strikes(qc, qp)
        F = implied_forward(qc, qp, T, R, spot)
        if F is None:
            F = forward_fallback(spot, T, R)
            ov_eff = 0
        else:
            ov_eff = ov
        calls, puts = clean_side(chain.calls, "C"), clean_side(chain.puts, "P")
        otm_n = len(calls[calls.strike >= F]) + len(puts[puts.strike < F])
        scored.append(dict(expiry=e, T=T, overlap=ov_eff, otm=otm_n, F=F))
    scored.sort(key=lambda x: (x["otm"], x["overlap"], x["T"]), reverse=True)
    chosen = [s["expiry"] for s in scored if s["otm"] >= 3][:MAX_EXPIRIES]
    if len(chosen) < MAX_EXPIRIES:
        for s in scored:
            if s["expiry"] not in chosen and s["overlap"] >= 1:
                chosen.append(s["expiry"])
            if len(chosen) >= MAX_EXPIRIES:
                break
    return chosen, scored, cached


def pull_quotes(ticker):
    """Fetch spot and chains; return (spot, pull time, cleaned quotes DataFrame)."""
    tk = yf.Ticker(ticker)
    spot = fetch_spot(tk)
    expiries = list_expiries(tk)
    if len(expiries) == 0:
        raise RuntimeError(
            f"Yahoo Finance has no option chain for {ticker}. "
            f"Use ^SPX for index European options; ETFs: SPY, QQQ, etc.")

    now = pd.Timestamp.now()
    use, scan, cached = pick_expiries(tk, spot, now, expiries)
    print(f"{ticker} spot {spot:.2f}, scanned {len(scan)} expiries, using {use}")

    rows = []
    for e in use:
        chain = cached[e]
        T = year_frac(e, now)
        qc, qp = quoted_side(chain.calls, "C"), quoted_side(chain.puts, "P")
        calls, puts = clean_side(chain.calls, "C"), clean_side(chain.puts, "P")
        ov = overlap_strikes(qc, qp)
        F = implied_forward(qc, qp, T, R, spot)
        f_src = "parity"
        if F is None:
            F = forward_fallback(spot, T, R)
            f_src = "Spot*exp(rT)"
        print(f"  expiry {e}: strike pairs {ov}, forward F={F:.2f} ({f_src}), "
              f"cleaned calls/puts {len(calls)}/{len(puts)}")
        otm = pd.concat([calls[calls.strike >= F], puts[puts.strike < F]])
        otm = otm[(otm.strike / F >= MONEYNESS[0]) & (otm.strike / F <= MONEYNESS[1])]
        otm["maturity"] = T
        otm["expiry"] = e
        otm["forward"] = F
        rows.append(otm)
        print(f"    kept {len(otm)} OTM contracts (moneyness {MONEYNESS})")

    if len(rows) == 0 or sum(len(x) for x in rows) == 0:
        raise RuntimeError(
            f"No usable quotes for {ticker}. Default is {TICKER!r}; if it still fails, check network/proxy, "
            f"wait a few minutes to avoid Yahoo rate limits, or run during US session (~21:30-04:00 Beijing).")

    quotes = pd.concat(rows, ignore_index=True).rename(columns={"mid": "price"})
    cols = ["strike", "maturity", "price", "type", "expiry", "forward", "bid", "ask", "volume", "openInterest"]
    return spot, now, quotes[cols].sort_values(["maturity", "strike"]).reset_index(drop=True)


if __name__ == "__main__":
    spot, now, quotes = pull_quotes(TICKER)
    name = TICKER.replace("^", "")
    csv_path = DATA / f"quotes_{name}_REAL.csv"
    quotes.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved {len(quotes)} quotes -> {csv_path} (pulled at {now}, spot {spot:.2f})")
