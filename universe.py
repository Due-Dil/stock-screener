from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import financedatabase as fd
import yfinance as yf

_equities = None

# Cache real market caps (in USD) across requests: {ticker: cap or None}
_cap_cache: dict[str, float | None] = {}


def _load() -> "pd.DataFrame":
    global _equities
    if _equities is None:
        _equities = fd.Equities().select()
    return _equities


def get_tickers(
    exchange: str | None = None,
    industry_group: str | None = None,
) -> list[str]:
    df = _load().copy()

    if exchange:
        df = df[df["exchange"] == exchange]
    if industry_group:
        df = df[df["industry_group"] == industry_group]

    return df.index.dropna().tolist()


def _fetch_cap(ticker: str) -> float | None:
    """Fetch a single ticker's market cap in USD, cached."""
    if ticker in _cap_cache:
        return _cap_cache[ticker]
    cap = None
    try:
        # Use the attribute accessor — the dict key is camelCase ("marketCap"),
        # but .market_cap resolves correctly.
        cap = getattr(yf.Ticker(ticker).fast_info, "market_cap", None)
    except Exception:
        cap = None
    _cap_cache[ticker] = cap
    return cap


def get_market_caps(tickers: list[str], max_workers: int = 20) -> dict[str, float | None]:
    """Return {ticker: market_cap_usd} for the given tickers, fetched in parallel."""
    to_fetch = [t for t in tickers if t not in _cap_cache]
    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(_fetch_cap, to_fetch))
    return {t: _cap_cache.get(t) for t in tickers}


def filter_by_cap(
    tickers: list[str],
    min_cap_b: float | None = None,
    max_cap_b: float | None = None,
) -> list[str]:
    """Filter tickers by real market cap, with min/max expressed in $ billions."""
    if min_cap_b is None and max_cap_b is None:
        return tickers

    caps = get_market_caps(tickers)
    lo = (min_cap_b or 0) * 1e9
    hi = (max_cap_b * 1e9) if max_cap_b is not None else float("inf")

    result = []
    for t in tickers:
        cap = caps.get(t)
        if cap is None:
            continue
        if lo <= cap <= hi:
            result.append(t)
    return result


def list_exchanges() -> list[str]:
    return sorted(_load()["exchange"].dropna().unique().tolist())


def list_industries() -> list[str]:
    return sorted(_load()["industry_group"].dropna().unique().tolist())
