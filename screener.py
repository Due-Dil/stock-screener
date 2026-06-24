import yfinance as yf


def get_stock_info(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "ticker": ticker,
        "name": info.get("longName", "N/A"),
        "price": info.get("currentPrice", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
    }


def screen_stocks(tickers: list[str], max_pe: float | None = None) -> list[dict]:
    results = []
    for ticker in tickers:
        try:
            info = get_stock_info(ticker)
            if max_pe is not None:
                pe = info.get("pe_ratio")
                if pe == "N/A" or (isinstance(pe, float) and pe > max_pe):
                    continue
            results.append(info)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    return results


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    stocks = screen_stocks(tickers, max_pe=30)
    for s in stocks:
        print(f"{s['ticker']} | {s['name']} | Price: {s['price']} | P/E: {s['pe_ratio']}")
