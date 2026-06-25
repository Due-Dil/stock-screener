from universe import get_tickers
from screener import find_crossovers

# ── Configuration ──────────────────────────────────────────────────
EXCHANGE = "NMS"              # "NMS" (NASDAQ), "NYQ" (NYSE), "PNK" (OTC), ...
INDUSTRY_GROUP = "Software & Services"  # see list_industries() for all values
MIN_CAP = "Mid Cap"           # None, or one of: Nano Cap, Micro Cap, Small Cap, Mid Cap, Large Cap, Mega Cap
MAX_CAP = None                # same scale — filters out above this size

MA_FAST = 20              # fast moving average (days)
MA_SLOW = 50              # slow moving average (days)
CROSSOVER_WITHIN = 5      # crossover must have occurred within this many days
# ───────────────────────────────────────────────────────────────────


def main():
    print(f"Fetching universe: exchange={EXCHANGE}, industry={INDUSTRY_GROUP}...")
    tickers = get_tickers(exchange=EXCHANGE, industry_group=INDUSTRY_GROUP, min_cap=MIN_CAP, max_cap=MAX_CAP)
    print(f"  {len(tickers)} tickers found")

    if not tickers:
        print("No tickers match your filters.")
        return

    print(f"\nScanning for MA{MA_FAST}/MA{MA_SLOW} crossovers in the last {CROSSOVER_WITHIN} days...")
    hits = find_crossovers(tickers, ma_fast=MA_FAST, ma_slow=MA_SLOW, crossover_within=CROSSOVER_WITHIN)

    if not hits:
        print("No crossovers found.")
        return

    print(f"\n{len(hits)} stock(s) found:\n")
    header = f"{'Ticker':<10} {'Date':<14} {'Price':>8} {f'MA{MA_FAST}':>8} {f'MA{MA_SLOW}':>8}"
    print(header)
    print("-" * len(header))
    for h in hits:
        print(
            f"{h['ticker']:<10} {str(h['crossover_date']):<14}"
            f" {h['price']:>8.2f} {h[f'MA{MA_FAST}']:>8.2f} {h[f'MA{MA_SLOW}']:>8.2f}"
        )


if __name__ == "__main__":
    main()
