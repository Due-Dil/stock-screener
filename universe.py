from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import financedatabase as fd
import yfinance as yf

_equities = None
_cap_category_map = None

# Cache real market caps (in USD) across requests: {ticker: cap or None}
_cap_cache: dict[str, float | None] = {}

# financedatabase market-cap categories and their $B bounds
_CAP_BOUNDS = [
    ("Nano Cap", 0, 0.05), ("Micro Cap", 0.05, 0.3), ("Small Cap", 0.3, 2),
    ("Mid Cap", 2, 10), ("Large Cap", 10, 200), ("Mega Cap", 200, float("inf")),
]


def _load() -> "pd.DataFrame":
    global _equities
    if _equities is None:
        # exclude_exchanges=False keeps non-US listings (Paris, London, Xetra, ...)
        # so European stocks are available, not just the US-deduplicated set.
        _equities = fd.Equities().select(exclude_exchanges=False)
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


def _cap_category(ticker: str) -> str | None:
    """financedatabase market-cap category for a ticker (handles duplicate index)."""
    global _cap_category_map
    if _cap_category_map is None:
        _cap_category_map = _load()["market_cap"].groupby(level=0).first().to_dict()
    return _cap_category_map.get(ticker)


def cap_candidates(tickers: list[str], min_cap_b: float | None, max_cap_b: float | None) -> list[str]:
    """Cheaply narrow tickers to those whose cap CATEGORY overlaps the requested
    $B range, using financedatabase (no network). Unknown categories are kept."""
    lo = min_cap_b if min_cap_b is not None else 0.0
    hi = max_cap_b if max_cap_b is not None else float("inf")
    cats = {name for name, clo, chi in _CAP_BOUNDS if clo < hi and chi > lo}
    out = []
    for t in tickers:
        c = _cap_category(t)
        if c is None or c in cats:   # keep unknowns so we don't wrongly drop them
            out.append(t)
    return out


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
