from __future__ import annotations

import webbrowser
import os
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from universe import get_tickers
from screener import find_crossovers


# ── Configuration ──────────────────────────────────────────────────
EXCHANGE = "NMS"
INDUSTRY_GROUP = "Software & Services"
MIN_CAP = "Mid Cap"
MAX_CAP = None

MA_FAST = 20
MA_SLOW = 50
CROSSOVER_WITHIN = 20   # wider window to catch more results
# ───────────────────────────────────────────────────────────────────

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "results.html")


def build_chart(ticker: str, ma_fast: int, ma_slow: int, crossover_date) -> go.Figure:
    raw = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
    closes = raw["Close"].dropna()

    fast = closes.rolling(ma_fast).mean()
    slow = closes.rolling(ma_slow).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=closes.index, y=closes,
        name="Price", line=dict(color="#4f8ef7", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=fast.index, y=fast,
        name=f"MA{ma_fast}", line=dict(color="#f5a623", width=1.5, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=slow.index, y=slow,
        name=f"MA{ma_slow}", line=dict(color="#e05c5c", width=1.5, dash="dash")
    ))

    # Mark crossover
    fig.add_vline(
        x=str(crossover_date),
        line_dash="solid", line_color="rgba(80,200,120,0.8)", line_width=2,
        annotation_text="Crossover", annotation_position="top right",
        annotation_font_color="rgba(80,200,120,1)"
    )

    fig.update_layout(
        title=ticker,
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font=dict(color="#eaeaea"),
        xaxis=dict(gridcolor="#2a2a4a"),
        yaxis=dict(gridcolor="#2a2a4a"),
    )
    return fig


def render_html(hits: list[dict], ma_fast: int, ma_slow: int, filters: dict) -> str:
    charts_html = ""
    for h in hits:
        fig = build_chart(h["ticker"], ma_fast, ma_slow, h["crossover_date"])
        charts_html += fig.to_html(full_html=False, include_plotlyjs=False)

    rows = "".join(
        f"<tr>"
        f"<td>{h['ticker']}</td>"
        f"<td>{h['crossover_date']}</td>"
        f"<td>{h['price']:.2f}</td>"
        f"<td>{h[f'MA{ma_fast}']:.2f}</td>"
        f"<td>{h[f'MA{ma_slow}']:.2f}</td>"
        f"</tr>"
        for h in hits
    )

    filter_tags = " ".join(
        f'<span class="tag">{k}: {v}</span>'
        for k, v in filters.items() if v
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stock Screener Results</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0f0f1a; color: #eaeaea; font-family: -apple-system, sans-serif; padding: 32px; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 8px; }}
    .subtitle {{ color: #888; margin-bottom: 24px; font-size: 0.9rem; }}
    .tags {{ margin-bottom: 24px; display: flex; gap: 8px; flex-wrap: wrap; }}
    .tag {{ background: #1e2a4a; border: 1px solid #2e4080; padding: 4px 12px;
            border-radius: 20px; font-size: 0.8rem; color: #aac4ff; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 40px; }}
    th {{ background: #1e2a4a; padding: 10px 16px; text-align: left;
          font-size: 0.85rem; color: #aac4ff; letter-spacing: 0.05em; }}
    td {{ padding: 10px 16px; border-bottom: 1px solid #1e1e30; font-size: 0.9rem; }}
    tr:hover td {{ background: #1a1a2e; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 16px; }}
    .chart-wrap {{ background: #16213e; border-radius: 10px; overflow: hidden; padding: 8px; }}
    .empty {{ text-align: center; padding: 60px; color: #555; font-size: 1.1rem; }}
  </style>
</head>
<body>
  <h1>Stock Screener</h1>
  <p class="subtitle">MA{ma_fast} crossed above MA{ma_slow} — {len(hits)} result(s)</p>
  <div class="tags">{filter_tags}</div>

  {"" if hits else '<div class="empty">No crossovers found with current filters.</div>'}

  {"<table><thead><tr><th>Ticker</th><th>Crossover Date</th><th>Price</th><th>MA" + str(ma_fast) + "</th><th>MA" + str(ma_slow) + "</th></tr></thead><tbody>" + rows + "</tbody></table>" if hits else ""}

  <div class="charts">
    {"".join(f'<div class="chart-wrap">{fig}</div>' for fig in [charts_html]) if hits else ""}
  </div>
</body>
</html>"""


def main():
    print(f"Fetching universe: exchange={EXCHANGE}, industry={INDUSTRY_GROUP}...")
    tickers = get_tickers(exchange=EXCHANGE, industry_group=INDUSTRY_GROUP, min_cap=MIN_CAP, max_cap=MAX_CAP)
    print(f"  {len(tickers)} tickers found")

    print(f"Scanning for MA{MA_FAST}/MA{MA_SLOW} crossovers in the last {CROSSOVER_WITHIN} days...")
    hits = find_crossovers(tickers, ma_fast=MA_FAST, ma_slow=MA_SLOW, crossover_within=CROSSOVER_WITHIN)
    print(f"  {len(hits)} crossover(s) found")

    filters = {
        "Exchange": EXCHANGE,
        "Industry": INDUSTRY_GROUP,
        "Min Cap": MIN_CAP,
        "Max Cap": MAX_CAP,
        f"MA Fast": MA_FAST,
        f"MA Slow": MA_SLOW,
        "Window": f"{CROSSOVER_WITHIN}d",
    }

    html = render_html(hits, MA_FAST, MA_SLOW, filters)
    with open(OUTPUT_FILE, "w") as f:
        f.write(html)

    print(f"\nResults saved to {OUTPUT_FILE}")
    webbrowser.open(f"file://{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
