from __future__ import annotations

import io
import requests
import pandas as pd

_cache: dict[str, list[str]] = {}

# Each entry: (url, column hints, mode)
#   mode "us"       -> US tickers; turn class-share dots into dashes (BRK.B -> BRK-B)
#   mode "verbatim" -> ticker already carries an exchange suffix (AC.PA, ADS.DE)
#   mode ".X"       -> bare ticker; append this yfinance suffix (FTSE: III -> III.L)
INDICES = {
    # — United States —
    "S&P 500":     ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["Symbol"], "us"),
    "NASDAQ 100":  ("https://en.wikipedia.org/wiki/Nasdaq-100", ["Ticker", "Ticker symbol"], "us"),
    "Dow Jones":   ("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", ["Symbol", "Ticker"], "us"),
    "S&P 400":     ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", ["Symbol", "Ticker"], "us"),
    "S&P 600":     ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", ["Symbol", "Ticker"], "us"),
    # — Europe —
    "CAC 40":      ("https://en.wikipedia.org/wiki/CAC_40", ["Ticker"], "verbatim"),
    "DAX":         ("https://en.wikipedia.org/wiki/DAX", ["Ticker"], "verbatim"),
    "Euro Stoxx 50": ("https://en.wikipedia.org/wiki/EURO_STOXX_50", ["Ticker"], "verbatim"),
    "FTSE 100":    ("https://en.wikipedia.org/wiki/FTSE_100_Index", ["Ticker", "EPIC"], ".L"),
}

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StockScreener/1.0)"}


def _clean(raw: list[str], mode: str) -> list[str]:
    out = []
    for t in raw:
        t = t.split("[")[0].strip()
        if not t or t.startswith("("):
            continue
        if mode == "us":
            t = t.replace(".", "-")
            if 1 <= len(t) <= 6 and t.replace("-", "").isalpha():
                out.append(t)
        elif mode == "verbatim":
            # already includes an exchange suffix, e.g. AC.PA / ADS.DE
            if "." in t and 2 <= len(t) <= 14:
                out.append(t)
        else:  # mode is a suffix to append, e.g. ".L"
            t = t.replace(".", "-")
            if 1 <= len(t) <= 8 and t.replace("-", "").isalpha():
                out.append(t + mode)
    return out


def get_index_tickers(name: str) -> list[str]:
    if name in _cache:
        return _cache[name]

    entry = INDICES.get(name)
    if entry is None:
        return []

    url, col_hints, mode = entry
    try:
        html = requests.get(url, headers=_HEADERS, timeout=10).text
        tables = pd.read_html(io.StringIO(html))
        for df in tables:
            df.columns = [str(c) for c in df.columns]
            for col in col_hints:
                if col in df.columns:
                    tickers = _clean(df[col].dropna().astype(str).tolist(), mode)
                    if len(tickers) > 10:
                        _cache[name] = tickers
                        return tickers
    except Exception:
        pass
    return []


def list_indices() -> list[str]:
    return list(INDICES.keys())
