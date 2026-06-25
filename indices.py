from __future__ import annotations

import io
import requests
import pandas as pd

_cache: dict[str, list[str]] = {}

INDICES = {
    "S&P 500":    ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol"]),
    "NASDAQ 100": ("https://en.wikipedia.org/wiki/Nasdaq-100", ["Ticker", "Ticker symbol"]),
    "Dow Jones":  ("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", ["Symbol", "Ticker"]),
    "S&P 400":    ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", ["Symbol", "Ticker"]),
    "S&P 600":    ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", ["Symbol", "Ticker"]),
}

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StockScreener/1.0)"}


def get_index_tickers(name: str) -> list[str]:
    if name in _cache:
        return _cache[name]

    entry = INDICES.get(name)
    if entry is None:
        return []

    url, col_hints = entry
    try:
        html = requests.get(url, headers=_HEADERS, timeout=10).text
        tables = pd.read_html(io.StringIO(html))
        for df in tables:
            for col in col_hints:
                if col in df.columns:
                    tickers = df[col].dropna().astype(str).tolist()
                    tickers = [t.split("[")[0].strip().replace(".", "-") for t in tickers
                               if t and not t.startswith("(")]
                    tickers = [t for t in tickers if 1 <= len(t) <= 6 and t.replace("-", "").isalpha()]
                    if len(tickers) > 10:
                        _cache[name] = tickers
                        return tickers
    except Exception:
        pass
    return []


def list_indices() -> list[str]:
    return list(INDICES.keys())
