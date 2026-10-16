"""从 Yahoo Finance 拉取真实期权报价，写入 data/quotes_<标的>_REAL.csv。

流程：
1. 用 yfinance 拉取标的现价和各到期日的期权链（看涨 + 看跌）
2. 优先 bid/ask 中间价；盘外 bid/ask 常为 0 时用 lastPrice（美股高流动性标的仍可用）
3. 每个到期日用看涨-看跌平价反推远期价格 F（自动吸收股息/分红的影响）
4. 只保留虚值合约：K >= F 用看涨，K < F 用看跌
5. 输出 CSV（含 strike, maturity, price, type 及 expiry, forward 等辅助列）

画图与 JSON 见 run_extension.py。
"""
import time
import pathlib

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# ---------------- 参数（无风险利率需与 run_extension.py 中 REAL_R 保持一致）----------------
TICKER = "SPY"
R = 0.045
MIN_DAYS = 7
MAX_EXPIRIES = 6
SCAN_EXPIRIES = 18
MONEYNESS = (0.85, 1.15)
MAX_REL_SPREAD = 0.5
MIN_OPEN_INTEREST = 0
USE_LAST_PRICE = True
SLEEP_SEC = 0.6
# --------------------------------------


def year_frac(expiry, now):
    """到期日（按当天 16:00 收盘计）距离现在的年化期限。"""
    return (pd.Timestamp(expiry) + pd.Timedelta(hours=16) - now).total_seconds() / (365.0 * 86400.0)


def attach_mid(df, kind):
    """计算期权价格 mid：优先买卖中间价，否则在 USE_LAST_PRICE 时用 lastPrice。"""
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
    """有有效 mid 的合约，用于反推远期价格（不施加未平仓量过滤）。"""
    return attach_mid(df, kind)


def clean_side(df, kind):
    """用于输出的合约：价差过滤（仅双边报价）+ 未平仓量。"""
    out = quoted_side(df, kind)
    spread_ok = (~out["has_bidask"]) | ((out["ask"] - out["bid"]) / out["mid"] <= MAX_REL_SPREAD)
    out = out[spread_ok]
    out = out[out.openInterest >= MIN_OPEN_INTEREST]
    return out.drop(columns=["has_bidask"], errors="ignore")


def overlap_strikes(calls, puts):
    """同 strike 同时有看涨、看跌报价的数量。"""
    if calls.empty or puts.empty:
        return 0
    return len(set(calls.strike) & set(puts.strike))


def implied_forward(calls, puts, T, r, spot):
    """看涨-看跌平价 C - P = e^{-rT}(F - K) 反推远期价格；取最接近现价的若干个行权价的中位数。"""
    both = calls[["strike", "mid"]].merge(puts[["strike", "mid"]], on="strike", suffixes=("_c", "_p"))
    if both.empty:
        return None
    both["F"] = both.strike + np.exp(r * T) * (both.mid_c - both.mid_p)
    n = min(5, len(both))
    near = both.iloc[np.argsort((both.strike - spot).abs().to_numpy())[:n]]
    return float(near.F.median())


def forward_fallback(spot, T, r):
    """同 strike 配对不足时，用 F ≈ S e^{rT} 近似（忽略股息；SPY 有分红时会略偏）。"""
    return float(spot * np.exp(r * T))


def pick_expiries(tk, spot, now):
    """扫描若干到期日，选同 strike 配对最多、且保留虚值后仍有数据的到期日。"""
    expiries = tk.options
    candidates = [e for e in expiries if year_frac(e, now) * 365.0 >= MIN_DAYS][:SCAN_EXPIRIES]
    scored = []
    for e in candidates:
        time.sleep(SLEEP_SEC)
        chain = tk.option_chain(e)
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
    return chosen, scored


def pull_quotes(ticker):
    """拉取现价和期权链，返回 (现价, 拉取时间, 清洗后的报价表)。"""
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    if len(tk.options) == 0:
        raise RuntimeError(
            f"Yahoo Finance 没有 {ticker} 的期权链。请使用美股期权标的，例如 SPY、AAPL、QQQ。")

    now = pd.Timestamp.now()
    use, scan = pick_expiries(tk, spot, now)
    print(f"{ticker} 现价 {spot:.2f}，扫描 {len(scan)} 个到期日，选用 {use}")

    rows = []
    for e in use:
        time.sleep(SLEEP_SEC)
        chain = tk.option_chain(e)
        T = year_frac(e, now)
        qc, qp = quoted_side(chain.calls, "C"), quoted_side(chain.puts, "P")
        calls, puts = clean_side(chain.calls, "C"), clean_side(chain.puts, "P")
        ov = overlap_strikes(qc, qp)
        F = implied_forward(qc, qp, T, R, spot)
        f_src = "parity"
        if F is None:
            F = forward_fallback(spot, T, R)
            f_src = "Spot*exp(rT)"
        print(f"  到期 {e}: 同 strike 配对 {ov}，远期 F={F:.2f} ({f_src})，"
              f"清洗后 看涨/看跌 {len(calls)}/{len(puts)}")
        otm = pd.concat([calls[calls.strike >= F], puts[puts.strike < F]])
        otm = otm[(otm.strike / F >= MONEYNESS[0]) & (otm.strike / F <= MONEYNESS[1])]
        otm["maturity"] = T
        otm["expiry"] = e
        otm["forward"] = F
        rows.append(otm)
        print(f"    保留 {len(otm)} 个虚值合约 (moneyness {MONEYNESS})")

    if len(rows) == 0 or sum(len(x) for x in rows) == 0:
        raise RuntimeError(
            f"{ticker} 没有可用报价。当前默认 {TICKER!r}；若仍失败请确认网络/代理，"
            f"或在美股交易时段（北京时间约 21:30-04:00）再跑。")

    quotes = pd.concat(rows, ignore_index=True).rename(columns={"mid": "price"})
    cols = ["strike", "maturity", "price", "type", "expiry", "forward", "bid", "ask", "volume", "openInterest"]
    return spot, now, quotes[cols].sort_values(["maturity", "strike"]).reset_index(drop=True)


if __name__ == "__main__":
    spot, now, quotes = pull_quotes(TICKER)
    name = TICKER.replace("^", "")
    csv_path = DATA / f"quotes_{name}_REAL.csv"
    quotes.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"已保存 {len(quotes)} 条报价 -> {csv_path}（拉取时间 {now}，现价 {spot:.2f}）")
