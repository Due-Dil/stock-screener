from __future__ import annotations

import numpy as np
import yfinance as yf
import pandas as pd


def find_crossovers(
    tickers: list[str],
    ma_fast: int = 20,
    ma_slow: int = 50,
    crossover_within: int = 5,
    surge_window_months: float = 3,
    batch_size: int = 20,
) -> list[dict]:
    if not tickers:
        return []

    # Enough history for the slow MA + the crossover window + the surge window
    # before the crossover. ~1.5x converts trading days to calendar days.
    surge_days = int(surge_window_months * 21)
    trading_needed = ma_slow + crossover_within + surge_days + 10
    period = f"{int(trading_needed * 1.5)}d"

    results = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_results = _scan_batch(batch, period, ma_fast, ma_slow, crossover_within, surge_days)
        results.extend(batch_results)

    return sorted(results, key=lambda x: x["crossover_date"], reverse=True)


def _max_drawup(prices: np.ndarray) -> float:
    """Largest gain from a trough to a later peak in the series, as a fraction."""
    if len(prices) < 2:
        return 0.0
    running_min = prices[0]
    best = 0.0
    for p in prices:
        if p < running_min:
            running_min = p
        elif running_min > 0:
            best = max(best, p / running_min - 1.0)
    return best


def _scan_batch(
    tickers: list[str],
    period: str,
    ma_fast: int,
    ma_slow: int,
    crossover_within: int,
    surge_days: int,
) -> list[dict]:
    try:
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return []

    results = []
    for ticker in tickers:
        try:
            closes = _get_closes(raw, ticker, len(tickers))
            if closes is None or len(closes) < ma_slow:
                continue

            fast = closes.rolling(ma_fast).mean()
            slow = closes.rolling(ma_slow).mean()
            diff = (fast - slow).dropna()

            if len(diff) < crossover_within + 1:
                continue

            # MA Fast must currently be above MA Slow
            if diff.iloc[-1] <= 0:
                continue

            # Find the most recent bullish crossover within the window
            crossed_date = None
            for j in range(len(diff) - 1, len(diff) - crossover_within - 1, -1):
                if diff.iloc[j - 1] < 0 and diff.iloc[j] > 0:
                    crossed_date = diff.index[j].date()
                    break

            if crossed_date is None:
                continue

            # Surge: largest run-up within +/- surge_days around the crossover
            cross_ts = diff.index[j]
            cross_pos = closes.index.get_indexer([cross_ts])[0]
            start = max(0, cross_pos - surge_days)
            end = min(len(closes), cross_pos + surge_days + 1)
            surge = _max_drawup(closes.iloc[start:end].to_numpy())

            results.append({
                "ticker": ticker,
                "crossover_date": crossed_date,
                "price": round(float(closes.iloc[-1]), 2),
                "surge": round(surge * 100, 1),
                f"MA{ma_fast}": round(float(fast.iloc[-1]), 2),
                f"MA{ma_slow}": round(float(slow.iloc[-1]), 2),
            })
        except Exception:
            continue

    return results


def _get_closes(raw: pd.DataFrame, ticker: str, n_tickers: int) -> pd.Series | None:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"][ticker].dropna()
        return raw["Close"].dropna()
    except (KeyError, TypeError):
        return None
