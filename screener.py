from __future__ import annotations

import yfinance as yf
import pandas as pd


def find_crossovers(
    tickers: list[str],
    ma_fast: int = 20,
    ma_slow: int = 50,
    crossover_within: int = 5,
    batch_size: int = 20,
) -> list[dict]:
    if not tickers:
        return []

    # ~1.5x multiplier to convert trading days to calendar days
    period = f"{int((ma_slow + crossover_within + 10) * 1.5)}d"

    results = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_results = _scan_batch(batch, period, ma_fast, ma_slow, crossover_within)
        results.extend(batch_results)

    return sorted(results, key=lambda x: x["crossover_date"], reverse=True)


def _scan_batch(
    tickers: list[str],
    period: str,
    ma_fast: int,
    ma_slow: int,
    crossover_within: int,
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

            results.append({
                "ticker": ticker,
                "crossover_date": crossed_date,
                "price": round(float(closes.iloc[-1]), 2),
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
