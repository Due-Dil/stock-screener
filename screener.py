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


def screen_stocks(tickers: list[str]) -> list[dict]:
    results = []
    for ticker in tickers:
        try:
            results.append(get_stock_info(ticker))
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    return results


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    stocks = screen_stocks(tickers)
    for s in stocks:
        print(f"{s['ticker']} | {s['name']} | Price: {s['price']} | P/E: {s['pe_ratio']}")
