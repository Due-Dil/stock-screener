"""
Backtest technical signals on a large US sample (~full S&P 500), since 2004.

Signals compared, all against the honest benchmark of a RANDOM entry
(because US stocks drift up, raw returns/win-rates are misleading):

  A. Golden Cross (MA50 crosses above MA200)        — timing signal
  B. Trend regime (price above MA200)               — time-series filter
  C. Cross-sectional momentum (12-1 month rank)     — relative strength
  D. Momentum AND in up-trend (combo)

Caveats:
  - Current index constituents => survivorship bias (optimistic).
  - auto_adjust=True => total return (dividends + splits).
  - Equal-weight, no transaction costs, no slippage.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from indices import get_index_tickers
from screener import _max_drawup

START = "2004-01-01"
HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "1y": 252, "2y": 504}
CACHE = os.path.join(os.path.dirname(__file__), "history_cache.pkl")


# ── Data (with disk cache) ───────────────────────────────────────────
def download(tickers: list[str], batch: int = 40) -> dict[str, pd.Series]:
    cache: dict[str, pd.Series] = {}
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
        print(f"  cache: {len(cache)} tickers already downloaded")

    missing = [t for t in tickers if t not in cache]
    for i in range(0, len(missing), batch):
        chunk = missing[i : i + batch]
        raw = yf.download(chunk, start=START, auto_adjust=True, progress=False, group_by="ticker")
        for t in chunk:
            try:
                s = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
                s = s.dropna()
                if len(s) > 260:
                    cache[t] = s
            except Exception:
                continue
        print(f"  downloaded {min(i+batch, len(missing))}/{len(missing)} new tickers", flush=True)
        with open(CACHE, "wb") as f:
            pickle.dump(cache, f)

    return {t: cache[t] for t in tickers if t in cache}


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def fwd(closes, i, h):
    return closes.iloc[i + h] / closes.iloc[i] - 1.0 if i + h < len(closes) else None


def pct(x):
    return f"{x*100:+.1f}%" if np.isfinite(x) else "  n/a"


# ── A+B: per-ticker signals (golden cross, trend regime) vs random ────
def analyse_timeseries(history):
    gc = {h: [] for h in HORIZONS}          # after a golden cross
    above = {h: [] for h in HORIZONS}       # any day while price > MA200
    base = {h: [] for h in HORIZONS}        # random day
    trades = []                              # golden->death strategy
    rng = np.random.default_rng(42)

    for closes in history.values():
        ma50 = closes.rolling(50).mean()
        ma200 = closes.rolling(200).mean()
        diff = (ma50 - ma200).values
        cv, pv = closes.values, ma200.values

        for i in range(1, len(diff)):
            if not (np.isfinite(diff[i - 1]) and np.isfinite(diff[i])):
                continue
            if diff[i - 1] <= 0 < diff[i]:
                for h, d in HORIZONS.items():
                    r = fwd(closes, i, d)
                    if r is not None:
                        gc[h].append(r)

        # trend regime: sample days where price > MA200
        idx_above = [i for i in range(200, len(closes)) if np.isfinite(pv[i]) and cv[i] > pv[i]]
        if idx_above:
            for i in rng.choice(idx_above, size=min(40, len(idx_above)), replace=False):
                for h, d in HORIZONS.items():
                    r = fwd(closes, int(i), d)
                    if r is not None:
                        above[h].append(r)

        # random baseline
        valid = np.arange(200, len(closes))
        for i in rng.choice(valid, size=min(40, len(valid)), replace=False):
            for h, d in HORIZONS.items():
                r = fwd(closes, int(i), d)
                if r is not None:
                    base[h].append(r)

        # strategy golden->death
        in_pos, entry = False, 0
        for i in range(1, len(diff)):
            if not (np.isfinite(diff[i - 1]) and np.isfinite(diff[i])):
                continue
            if not in_pos and diff[i - 1] <= 0 < diff[i]:
                in_pos, entry = True, i
            elif in_pos and diff[i - 1] >= 0 > diff[i]:
                trades.append((cv[i] / cv[entry] - 1.0, i - entry))
                in_pos = False
        if in_pos:
            trades.append((cv[-1] / cv[entry] - 1.0, len(closes) - 1 - entry))

    return gc, above, base, trades


# ── C+D: cross-sectional momentum (relative strength) ─────────────────
def analyse_momentum(history, horizons=(21, 63, 126)):
    panel = pd.DataFrame(history).sort_index()
    # 12-1 momentum: return from t-252 to t-21 (skip most recent month)
    mom = panel.shift(21) / panel.shift(252) - 1.0
    ma200 = panel.rolling(200).mean()
    uptrend = panel > ma200

    dates = panel.index
    # monthly rebalance points with enough history and forward room
    rebal = list(range(252, len(dates) - max(horizons), 21))

    # results[h] = {"Q5":[], "Q1":[], "all":[], "Q5_up":[]}
    res = {h: {"Q5": [], "Q1": [], "all": [], "Q5_up": []} for h in horizons}

    for i in rebal:
        row_mom = mom.iloc[i].dropna()
        if len(row_mom) < 20:
            continue
        q = row_mom.quantile([0.2, 0.8])
        winners = row_mom[row_mom >= q[0.8]].index
        losers = row_mom[row_mom <= q[0.2]].index
        up_now = uptrend.iloc[i]
        winners_up = [t for t in winners if bool(up_now.get(t, False))]

        for h in horizons:
            fwd_ret = panel.iloc[i + h] / panel.iloc[i] - 1.0
            res[h]["Q5"].append(fwd_ret[winners].mean())
            res[h]["Q1"].append(fwd_ret[losers].mean())
            res[h]["all"].append(fwd_ret.dropna().mean())
            if winners_up:
                res[h]["Q5_up"].append(fwd_ret[winners_up].mean())
    return res


def analyse_surge(history, window_months=3, buffer_days=21, horizons=(63, 126, 252)):
    """Does a bigger surge around the golden cross predict higher forward returns?
    No look-ahead: surge is measured up to a decision point ~1 month after the
    cross, and forward returns start FROM that decision point."""
    bands = [(0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 99)]
    res = {h: {b: [] for b in bands} for h in horizons}
    base = {h: [] for h in horizons}
    W = int(window_months * 21)
    rng = np.random.default_rng(7)

    for closes in history.values():
        ma50 = closes.rolling(50).mean()
        ma200 = closes.rolling(200).mean()
        diff = (ma50 - ma200).values
        cv = closes.to_numpy()

        for i in range(1, len(diff)):
            if not (np.isfinite(diff[i - 1]) and np.isfinite(diff[i])):
                continue
            if diff[i - 1] <= 0 < diff[i]:
                dec = i + buffer_days            # decision point (~1m after cross)
                if dec + max(horizons) >= len(cv):
                    continue
                surge = _max_drawup(cv[max(0, i - W):dec + 1])   # past data only
                for h in horizons:
                    r = cv[dec + h] / cv[dec] - 1.0
                    for b in bands:
                        if b[0] <= surge < b[1]:
                            res[h][b].append(r)

        valid = np.arange(200, len(cv) - max(horizons))
        if len(valid):
            for i in rng.choice(valid, size=min(40, len(valid)), replace=False):
                for h in horizons:
                    base[h].append(cv[int(i) + h] / cv[int(i)] - 1.0)

    return res, base, bands


def analyse_surge_momentum(history, surge_thr=0.40, mom_top=0.67,
                           buffer_days=21, window_months=3, horizon=252):
    """Combine a strong surge with strong RELATIVE momentum at the decision point.
    Relative momentum = percentile of the stock's 12-1m return vs all stocks that
    day. 2x2 buckets, forward return over `horizon`. No look-ahead."""
    panel = pd.DataFrame(history).sort_index()
    mom = panel.shift(21) / panel.shift(252) - 1.0
    W = int(window_months * 21)

    keys = ["low surge / low mom", "low surge / HIGH mom",
            "HIGH surge / low mom", "HIGH surge / HIGH mom"]
    buckets = {k: [] for k in keys}
    base = []
    rng = np.random.default_rng(11)

    for ticker, closes in history.items():
        cv = closes.to_numpy()
        ma50 = closes.rolling(50).mean().to_numpy()
        ma200 = closes.rolling(200).mean().to_numpy()
        diff = ma50 - ma200
        idx = closes.index

        for i in range(1, len(diff)):
            if not (np.isfinite(diff[i - 1]) and np.isfinite(diff[i])):
                continue
            if diff[i - 1] <= 0 < diff[i]:
                dec = i + buffer_days
                if dec + horizon >= len(cv):
                    continue
                surge = _max_drawup(cv[max(0, i - W):dec + 1])
                try:
                    row = mom.loc[idx[dec]].dropna()
                except KeyError:
                    continue
                if ticker not in row.index or len(row) < 20:
                    continue
                mom_pct = float((row < row[ticker]).mean())
                fr = cv[dec + horizon] / cv[dec] - 1.0
                hs = "HIGH surge" if surge >= surge_thr else "low surge"
                hm = "HIGH mom" if mom_pct >= mom_top else "low mom"
                buckets[f"{hs} / {hm}"].append(fr)

        valid = np.arange(252, len(cv) - horizon)
        if len(valid):
            for i in rng.choice(valid, size=min(30, len(valid)), replace=False):
                base.append(cv[int(i) + horizon] / cv[int(i)] - 1.0)

    return buckets, base


def report(history):
    n = len(history)
    gc, above, base, trades = analyse_timeseries(history)

    print("\n" + "=" * 70)
    print(f"TECHNICAL SIGNAL BACKTEST — {n} US stocks, since {START}")
    print("=" * 70)

    print("\n[A] GOLDEN CROSS vs [B] TREND REGIME (price>MA200) vs RANDOM entry")
    print(f"{'Horizon':<8}{'GoldenX':>10}{'Trend>MA200':>13}{'Random':>10}"
          f"{'GC edge':>10}{'Trend edge':>12}")
    print("-" * 70)
    for h in HORIZONS:
        g, a, b = np.array(gc[h]), np.array(above[h]), np.array(base[h])
        if not len(g):
            continue
        print(f"{h:<8}{pct(g.mean()):>10}{pct(a.mean()):>13}{pct(b.mean()):>10}"
              f"{pct(g.mean()-b.mean()):>10}{pct(a.mean()-b.mean()):>12}")
    print(f"\n  (golden-cross signals: {len(gc['1y'])} | random samples: {len(base['1y'])})")

    print("\n[C] CROSS-SECTIONAL MOMENTUM (12-1m rank) — winners vs losers vs all")
    mom = analyse_momentum(history)
    print(f"{'Hold':<8}{'Winners Q5':>12}{'Losers Q1':>12}{'All stocks':>12}"
          f"{'Q5-Q1 spread':>14}{'Q5+uptrend':>12}")
    print("-" * 70)
    hmap = {21: "1m", 63: "3m", 126: "6m"}
    for h, d in mom.items():
        q5, q1, al = np.mean(d["Q5"]), np.mean(d["Q1"]), np.mean(d["all"])
        q5up = np.mean(d["Q5_up"]) if d["Q5_up"] else float("nan")
        print(f"{hmap[h]:<8}{pct(q5):>12}{pct(q1):>12}{pct(al):>12}"
              f"{pct(q5-q1):>14}{pct(q5up):>12}")
    print("\n  Edge = Winners minus 'All stocks' (beating a random pick over the same window).")

    if trades:
        r = np.array([x for x, _ in trades]); hold = np.array([h for _, h in trades])
        print("\n[strategy] buy golden cross → sell death cross:")
        print(f"  {len(trades)} trades | avg {pct(r.mean())} | median {pct(np.median(r))}"
              f" | win {(r>0).mean()*100:.0f}% | avg hold {hold.mean()/21:.1f} months")

    print("\n[E] SURGE around the golden cross — does a bigger run-up predict more?")
    print("(surge measured up to ~1m after the cross; returns start from there — no look-ahead)")
    sres, sbase, bands = analyse_surge(history)
    band_lbl = {(0, 0.10): "surge <10%", (0.10, 0.25): "10-25%",
                (0.25, 0.50): "25-50%", (0.50, 99): ">50%"}
    hmap = {63: "3m", 126: "6m", 252: "1y"}
    print(f"{'Surge band':<14}" + "".join(f"{hmap[h]+' fwd':>12}" for h in sres) + f"{'  n (1y)':>10}")
    print("-" * 62)
    for b in bands:
        row = f"{band_lbl[b]:<14}"
        for h in sres:
            arr = np.array(sres[h][b])
            row += f"{pct(arr.mean()) if len(arr) else '  n/a':>12}"
        row += f"{len(sres[252][b]):>10}"
        print(row)
    base_row = f"{'random entry':<14}" + "".join(f"{pct(np.mean(sbase[h])):>12}" for h in sres)
    print(base_row)
    print("\n  Read: compare each surge band's forward return to 'random entry'.")

    print("\n[F] SURGE (>=40%) x RELATIVE MOMENTUM (top third) — 1-year forward return")
    buckets, sm_base = analyse_surge_momentum(history)
    print(f"{'Bucket':<26}{'1y fwd':>10}{'n':>8}")
    print("-" * 46)
    for k, arr in buckets.items():
        a = np.array(arr)
        print(f"{k:<26}{pct(a.mean()) if len(a) else '  n/a':>10}{len(a):>8}")
    print(f"{'random entry':<26}{pct(np.mean(sm_base)):>10}")
    print("\n  The interesting cell is 'HIGH surge / HIGH mom' vs random entry.")
    print("=" * 70)


def main(n=None):
    print("Fetching constituents of S&P 500 + 400 (mid) + 600 (small)...")
    tickers = []
    for idx in ("S&P 500", "S&P 400", "S&P 600"):
        tickers += get_index_tickers(idx)
    tickers = sorted(set(tickers))
    if n:
        tickers = tickers[:n]
    print(f"Loading {len(tickers)} tickers since {START} (first run downloads; then cached)...")
    history = download(tickers)
    print(f"\nUsable tickers: {len(history)}")
    report(history)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
