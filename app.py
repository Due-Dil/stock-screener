from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os

from flask import Flask, render_template_string, request, jsonify
import yfinance as yf
import pandas as pd

from universe import get_tickers, list_industries, filter_by_cap
from screener import find_crossovers
from indices import get_index_tickers, list_indices

app = Flask(__name__)

# ── Watchlist persistence (simple JSON file next to this module) ──────
WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")


def load_watchlist() -> list[dict]:
    try:
        with open(WATCHLIST_FILE) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_watchlist(items: list[dict]) -> None:
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(items, f, indent=2)


@app.after_request
def add_no_cache_headers(response):
    """Prevent the browser from serving a stale cached version of the app."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


INDUSTRIES = list_industries()
INDICES = list_indices()

# Human-readable exchange names → financedatabase code (verified to exist in the dataset)
EXCHANGES = {
    # — United States —
    "NASDAQ": "NMS",
    "NYSE": "NYQ",
    "NYSE American": "ASE",
    "OTC Markets": "PNK",
    # — Europe —
    "Euronext Paris": "PAR",
    "Euronext Amsterdam": "AMS",
    "Euronext Lisbon": "LIS",
    "London Stock Exchange": "LSE",
    "Xetra (Germany)": "GER",
    "Borsa Italiana (Milan)": "MIL",
    "Bolsa de Madrid": "MCE",
    "Nasdaq Stockholm": "STO",
    "Nasdaq Helsinki": "HEL",
    "Nasdaq Copenhagen": "CPH",
    "Oslo Børs": "OSL",
}

HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stock Screener</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f0f1a; color: #eaeaea; font-family: -apple-system, BlinkMacSystemFont, sans-serif; display: flex; height: 100vh; overflow: hidden; }

    /* ── Sidebar ── */
    #sidebar {
      width: 290px; min-width: 290px; background: #13132a; border-right: 1px solid #1e1e40;
      padding: 24px 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px;
    }
    #sidebar h2 { font-size: 1.1rem; color: #aac4ff; letter-spacing: 0.05em; margin-bottom: 2px; }

    .field { display: flex; flex-direction: column; gap: 5px; }
    .field label { font-size: 0.72rem; color: #888; letter-spacing: 0.06em; text-transform: uppercase; }
    .field select, .field input {
      background: #1a1a35; border: 1px solid #2a2a55; color: #eaeaea;
      padding: 8px 10px; border-radius: 6px; font-size: 0.88rem; width: 100%;
      appearance: none;
    }
    .field select:focus, .field input:focus { outline: none; border-color: #4f8ef7; }
    .field .hint { font-size: 0.7rem; color: #556; margin-top: 2px; }

    .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }

    .divider { border: none; border-top: 1px solid #1e1e40; margin: 4px 0; }

    /* Universe count pill */
    #universe-count {
      display: flex; align-items: center; justify-content: space-between;
      background: #1a1a35; border: 1px solid #2a2a55; border-radius: 8px;
      padding: 9px 14px; font-size: 0.85rem;
    }
    #universe-count span { color: #888; }
    #count-val { color: #50c878; font-weight: 700; font-size: 1rem; }
    #count-spinner { display: none; }
    #count-spinner.active { display: inline-block; }

    button#run {
      padding: 11px; background: #4f8ef7; color: #fff;
      border: none; border-radius: 8px; font-size: 0.95rem; font-weight: 600;
      cursor: pointer; transition: background 0.2s;
    }
    button#run:hover { background: #3a7de8; }
    button#run:disabled { background: #2a2a55; color: #555; cursor: not-allowed; }
    #run-status { font-size: 0.8rem; color: #888; text-align: center; min-height: 18px; }

    /* ── Main ── */
    #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

    /* ── Table ── */
    #table-wrap { flex: 1; overflow-y: auto; padding: 24px; }
    table { width: 100%; border-collapse: collapse; }
    th {
      background: #13132a; padding: 10px 16px; text-align: left;
      font-size: 0.75rem; color: #aac4ff; letter-spacing: 0.06em;
      text-transform: uppercase; position: sticky; top: 0; z-index: 1;
    }
    td { padding: 11px 16px; border-bottom: 1px solid #1a1a30; font-size: 0.88rem; }
    tbody tr { cursor: pointer; transition: background 0.15s; }
    tbody tr:hover td { background: #1a1a35; }
    tbody tr.active td { background: #1e2a50; }
    .badge-up {
      display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;
      background: rgba(80,200,120,0.15); color: #50c878; border: 1px solid rgba(80,200,120,0.3);
    }
    .empty-msg { text-align: center; color: #555; padding: 80px 0; font-size: 1rem; }

    /* ── Chart panel ── */
    #chart-panel {
      height: 0; overflow: hidden; border-top: 1px solid #1e1e40;
      transition: height 0.3s ease; background: #13132a;
    }
    #chart-panel.open { height: 360px; }
    #chart-inner { padding: 12px 24px 16px; height: 100%; display: flex; flex-direction: column; }
    #chart-title {
      font-size: 0.92rem; color: #aac4ff; margin-bottom: 6px;
      display: flex; align-items: center; justify-content: space-between; flex-shrink: 0;
    }
    #chart-name { color: #888; font-size: 0.8rem; margin-left: 8px; }
    #close-chart { background: none; border: none; color: #888; font-size: 1.2rem; cursor: pointer; }
    #close-chart:hover { color: #eaeaea; }
    #plotly-chart { flex: 1; min-height: 0; }

    /* ── Spinner ── */
    .spinner {
      display: inline-block; width: 12px; height: 12px;
      border: 2px solid #2a2a55; border-top-color: #4f8ef7;
      border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 5px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Watchlist (sidebar) ── */
    #watchlist-header { font-size: 0.8rem; color: #f5c542; letter-spacing: 0.04em; margin-bottom: 8px; }
    #watchlist-count { color: #666; font-weight: 400; }
    #watchlist-items { display: flex; flex-direction: column; gap: 5px; }
    .watchlist-empty { font-size: 0.75rem; color: #555; }
    .wl-chip {
      display: flex; align-items: center; justify-content: space-between;
      background: #1a1a35; border: 1px solid #2a2a55; border-radius: 6px;
      padding: 6px 8px 6px 10px; font-size: 0.82rem; cursor: pointer; transition: border-color 0.15s;
    }
    .wl-chip:hover { border-color: #4f8ef7; }
    .wl-chip .wl-tk { font-weight: 600; }
    .wl-chip .wl-nm { color: #888; font-size: 0.72rem; margin-left: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 130px; }
    .wl-remove { background: none; border: none; color: #666; cursor: pointer; font-size: 0.95rem; padding: 0 2px; }
    .wl-remove:hover { color: #e05c5c; }

    /* ── Star toggle (table) ── */
    .star-btn { background: none; border: none; cursor: pointer; font-size: 1rem; color: #444; padding: 0; line-height: 1; }
    .star-btn.on { color: #f5c542; }
    .star-btn:hover { color: #f5c542; }
  </style>
</head>
<body>

<div id="sidebar">
  <h2>Filters</h2>

  <div class="field">
    <label>Index</label>
    <select id="index_filter" onchange="scheduleCount()">
      <option value="">—</option>
      {% for i in indices %}<option value="{{ i }}">{{ i }}</option>{% endfor %}
    </select>
  </div>

  <div class="field">
    <label>Exchange</label>
    <select id="exchange" onchange="scheduleCount()">
      <option value="">All</option>
      {% for label, code in exchanges.items() %}<option value="{{ code }}">{{ label }}</option>{% endfor %}
    </select>
  </div>

  <div class="field">
    <label>Industry Group</label>
    <select id="industry" onchange="scheduleCount()">
      <option value="">All</option>
      {% for i in industries %}<option value="{{ i }}">{{ i }}</option>{% endfor %}
    </select>
  </div>

  <div class="row2">
    <div class="field">
      <label>Min Market Cap</label>
      <input type="number" id="min_cap" placeholder="e.g. 2" min="0" step="0.1" oninput="scheduleCount()">
      <span class="hint">in $ billions</span>
    </div>
    <div class="field">
      <label>Max Market Cap</label>
      <input type="number" id="max_cap" placeholder="e.g. 200" min="0" step="0.1" oninput="scheduleCount()">
      <span class="hint">in $ billions</span>
    </div>
  </div>

  <hr class="divider">

  <div class="row2">
    <div class="field">
      <label>MA Fast</label>
      <input type="number" id="ma_fast" value="50" min="2" max="200">
    </div>
    <div class="field">
      <label>MA Slow</label>
      <input type="number" id="ma_slow" value="200" min="2" max="500">
    </div>
  </div>

  <div class="field">
    <label>Crossover within (days)</label>
    <input type="number" id="crossover_within" value="20" min="1" max="90">
  </div>

  <hr class="divider">

  <div id="universe-count">
    <span>Universe size</span>
    <span>
      <span class="spinner" id="count-spinner"></span>
      <span id="count-val">—</span>
    </span>
  </div>
  <div id="cap-hint" style="display:none; font-size:0.72rem; color:#e0a030; line-height:1.4;">
    ⚠︎ Trop d'entreprises pour le filtre de capitalisation exact. Sélectionne un indice ou un exchange/industry pour le réduire sous 800.
  </div>

  <button id="run" onclick="runScreener()">Run Screener</button>
  <div id="run-status"></div>

  <hr class="divider">

  <div id="watchlist-section">
    <div id="watchlist-header">★ Watchlist <span id="watchlist-count"></span></div>
    <div id="watchlist-items"><span class="watchlist-empty">Aucune action enregistrée</span></div>
  </div>
</div>

<div id="main">
  <div id="table-wrap">
    <div class="empty-msg">Set your filters and click <strong>Run Screener</strong>.</div>
  </div>
  <div id="chart-panel">
    <div id="chart-inner">
      <div id="chart-title">
        <div><strong id="chart-ticker-label"></strong><span id="chart-name"></span><span id="chart-earnings" style="color:#a878ff; font-size:0.78rem; margin-left:10px;"></span></div>
        <button id="close-chart" onclick="closeChart()">✕</button>
      </div>
      <div id="plotly-chart"></div>
    </div>
  </div>
</div>

<script>
let currentMaFast = 20, currentMaSlow = 50;
let countTimer = null;

function getFilters() {
  return {
    index_filter: document.getElementById('index_filter').value,
    exchange: document.getElementById('exchange').value,
    industry: document.getElementById('industry').value,
    min_cap: parseFloat(document.getElementById('min_cap').value) || null,
    max_cap: parseFloat(document.getElementById('max_cap').value) || null,
    ma_fast: parseInt(document.getElementById('ma_fast').value) || 20,
    ma_slow: parseInt(document.getElementById('ma_slow').value) || 50,
    crossover_within: parseInt(document.getElementById('crossover_within').value) || 20,
  };
}

function scheduleCount() {
  clearTimeout(countTimer);
  document.getElementById('count-spinner').classList.add('active');
  document.getElementById('count-val').textContent = '—';
  countTimer = setTimeout(fetchCount, 400);
}

async function fetchCount() {
  const f = getFilters();
  try {
    const res = await fetch('/count', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index_filter: f.index_filter, exchange: f.exchange, industry: f.industry, min_cap: f.min_cap, max_cap: f.max_cap }),
    });
    const data = await res.json();
    document.getElementById('count-val').textContent = data.count;
    const hint = document.getElementById('cap-hint');
    hint.style.display = data.cap_skipped ? 'block' : 'none';
  } catch(e) {
    document.getElementById('count-val').textContent = '?';
  }
  document.getElementById('count-spinner').classList.remove('active');
}

async function runScreener() {
  const btn = document.getElementById('run');
  const status = document.getElementById('run-status');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Scanning…';
  closeChart();

  const params = getFilters();
  currentMaFast = params.ma_fast;
  currentMaSlow = params.ma_slow;

  try {
    const res = await fetch('/screen', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const data = await res.json();
    if (data.too_large) {
      document.getElementById('table-wrap').innerHTML =
        `<div class="empty-msg">Univers trop large (${data.tickers_scanned} titres > ${data.too_large}).<br>Affine avec un indice, un exchange ou une industrie.</div>`;
      status.textContent = `${data.tickers_scanned} titres — trop pour scanner`;
      btn.disabled = false;
      return;
    }
    renderTable(data.results, params.ma_fast, params.ma_slow);
    status.textContent = `${data.results.length} result(s) — ${data.tickers_scanned} tickers scanned`;
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
}

let resultsByTicker = {};

function renderTable(results, maFast, maSlow) {
  const wrap = document.getElementById('table-wrap');
  if (!results.length) {
    wrap.innerHTML = '<div class="empty-msg">No crossovers found with current filters.</div>';
    return;
  }
  resultsByTicker = {};
  results.forEach(r => { resultsByTicker[r.ticker] = r; });

  const rows = results.map(r => {
    const on = watchedSet.has(r.ticker) ? ' on' : '';
    const star = watchedSet.has(r.ticker) ? '★' : '☆';
    return `
    <tr onclick="loadChart('${r.ticker}', resultsByTicker['${r.ticker}'].name)">
      <td><button class="star-btn${on}" onclick="event.stopPropagation(); toggleWatch('${r.ticker}')">${star}</button></td>
      <td><strong>${r.ticker}</strong></td>
      <td style="color:#aaa">${r.name || '—'}</td>
      <td><span class="badge-up">▲ ${r.crossover_date}</span></td>
      <td>${r.price.toFixed(2)}</td>
      <td>${r['MA' + maFast].toFixed(2)}</td>
      <td>${r['MA' + maSlow].toFixed(2)}</td>
    </tr>`;
  }).join('');
  wrap.innerHTML = `<table>
    <thead><tr>
      <th></th><th>Ticker</th><th>Name</th><th>Crossover</th><th>Price</th>
      <th>MA${maFast}</th><th>MA${maSlow}</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

// ── Watchlist ──────────────────────────────────────────────────────
let watchedSet = new Set();

async function loadWatchlist() {
  try {
    const res = await fetch('/watchlist');
    const items = await res.json();
    watchedSet = new Set(items.map(it => it.ticker));
    renderWatchlist(items);
  } catch(e) { /* ignore */ }
}

function renderWatchlist(items) {
  document.getElementById('watchlist-count').textContent = items.length ? `(${items.length})` : '';
  const wrap = document.getElementById('watchlist-items');
  if (!items.length) {
    wrap.innerHTML = '<span class="watchlist-empty">Aucune action enregistrée</span>';
    return;
  }
  wrap.innerHTML = items.map(it => `
    <div class="wl-chip" onclick="loadChart('${it.ticker}', ${JSON.stringify(it.name || '')})">
      <span><span class="wl-tk">${it.ticker}</span><span class="wl-nm">${it.name || ''}</span></span>
      <button class="wl-remove" onclick="event.stopPropagation(); toggleWatch('${it.ticker}')" title="Retirer">✕</button>
    </div>`).join('');
}

async function toggleWatch(ticker) {
  const isOn = watchedSet.has(ticker);
  let items;
  if (isOn) {
    items = await (await fetch('/watchlist/' + ticker, { method: 'DELETE' })).json();
  } else {
    const name = (resultsByTicker[ticker] && resultsByTicker[ticker].name) || '';
    items = await (await fetch('/watchlist', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, name }),
    })).json();
  }
  watchedSet = new Set(items.map(it => it.ticker));
  renderWatchlist(items);
  // Update any star button currently shown in the table for this ticker
  document.querySelectorAll('tbody tr').forEach(tr => {
    const tk = tr.querySelector('td strong')?.textContent;
    if (tk === ticker) {
      const btn = tr.querySelector('.star-btn');
      if (btn) {
        const on = watchedSet.has(tk);
        btn.classList.toggle('on', on);
        btn.textContent = on ? '★' : '☆';
      }
    }
  });
}

async function loadChart(ticker, name) {
  document.querySelectorAll('tbody tr').forEach(r => r.classList.remove('active'));
  const row = [...document.querySelectorAll('tbody tr')].find(r => r.querySelector('td strong')?.textContent === ticker);
  if (row) row.classList.add('active');

  const panel = document.getElementById('chart-panel');
  panel.classList.add('open');
  document.getElementById('chart-ticker-label').textContent = ticker;
  document.getElementById('chart-name').textContent = name ? ' — ' + name : '';
  document.getElementById('chart-earnings').textContent = '';
  document.getElementById('plotly-chart').innerHTML =
    '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#555"><span class="spinner"></span> Loading…</div>';

  const res = await fetch(`/chart/${ticker}?ma_fast=${currentMaFast}&ma_slow=${currentMaSlow}`);
  const data = await res.json();
  Plotly.newPlot('plotly-chart', data.traces, data.layout, { responsive: true, displayModeBar: false });

  const earn = data.earnings || [];
  document.getElementById('chart-earnings').textContent =
    earn.length ? '◦ Earnings (±30j): ' + earn.join(', ') : '◦ Aucun earnings dans ±30j';
}

function closeChart() {
  document.getElementById('chart-panel').classList.remove('open');
  document.querySelectorAll('tbody tr').forEach(r => r.classList.remove('active'));
}

// On page load
fetchCount();
loadWatchlist();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML, exchanges=EXCHANGES, industries=INDUSTRIES, indices=INDICES)


@app.route("/watchlist", methods=["GET"])
def watchlist_get():
    return jsonify(load_watchlist())


@app.route("/watchlist", methods=["POST"])
def watchlist_add():
    body = request.json or {}
    ticker = (body.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    items = load_watchlist()
    if not any(it["ticker"] == ticker for it in items):
        items.append({"ticker": ticker, "name": body.get("name") or ""})
        save_watchlist(items)
    return jsonify(items)


@app.route("/watchlist/<ticker>", methods=["DELETE"])
def watchlist_remove(ticker: str):
    ticker = ticker.strip().upper()
    items = [it for it in load_watchlist() if it["ticker"] != ticker]
    save_watchlist(items)
    return jsonify(items)


# Above this universe size we skip the (slow) exact market-cap fetch
MAX_CAP_FILTER_UNIVERSE = 800

# Above this universe size we refuse to run the (very slow) crossover scan
MAX_SCAN_UNIVERSE = 1500


def _build_tickers(params: dict) -> tuple[list[str], bool]:
    """Return (tickers, cap_skipped). cap_skipped is True when a cap filter was
    requested but the universe was too large to fetch real market caps."""
    tickers = get_tickers(
        exchange=params.get("exchange") or None,
        industry_group=params.get("industry") or None,
    )
    index_name = params.get("index_filter") or ""
    if index_name:
        index_set = set(get_index_tickers(index_name))
        tickers = [t for t in tickers if t in index_set] if tickers else list(index_set)

    min_cap = params.get("min_cap")
    max_cap = params.get("max_cap")
    cap_requested = min_cap is not None or max_cap is not None

    if cap_requested:
        if len(tickers) > MAX_CAP_FILTER_UNIVERSE:
            return tickers, True
        tickers = filter_by_cap(
            tickers,
            min_cap_b=float(min_cap) if min_cap is not None else None,
            max_cap_b=float(max_cap) if max_cap is not None else None,
        )
    return tickers, False


@app.route("/count", methods=["POST"])
def count():
    tickers, cap_skipped = _build_tickers(request.json)
    return jsonify({"count": len(tickers), "cap_skipped": cap_skipped})


@app.route("/screen", methods=["POST"])
def screen():
    params = request.json
    tickers, cap_skipped = _build_tickers(params)

    if len(tickers) > MAX_SCAN_UNIVERSE:
        return jsonify({
            "results": [],
            "tickers_scanned": len(tickers),
            "cap_skipped": cap_skipped,
            "too_large": MAX_SCAN_UNIVERSE,
        })

    hits = find_crossovers(
        tickers,
        ma_fast=int(params.get("ma_fast", 20)),
        ma_slow=int(params.get("ma_slow", 50)),
        crossover_within=int(params.get("crossover_within", 20)),
    )

    enriched = []
    for h in hits:
        try:
            name = yf.Ticker(h["ticker"]).fast_info.display_name or ""
        except Exception:
            name = ""
        enriched.append({**h, "crossover_date": str(h["crossover_date"]), "name": name})

    return jsonify({"results": enriched, "tickers_scanned": len(tickers), "cap_skipped": cap_skipped})


@app.route("/chart/<ticker>")
def chart(ticker: str):
    ma_fast = int(request.args.get("ma_fast", 20))
    ma_slow = int(request.args.get("ma_slow", 50))

    # Fetch enough calendar days to cover ma_slow trading days (~1.5x multiplier) + 6 months of visible history
    calendar_days = max(180, int(ma_slow * 1.5) + 180)
    period = f"{calendar_days}d"
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    # Always extract a clean 1-D Series regardless of MultiIndex structure
    close_col = raw["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
    closes = close_col.dropna()

    fast = closes.rolling(ma_fast).mean()
    slow = closes.rolling(ma_slow).mean()

    diff = (fast - slow).dropna()
    cross_dates = [
        str(diff.index[i].date())
        for i in range(1, len(diff))
        if diff.iloc[i - 1] < 0 and diff.iloc[i] > 0
    ]

    def series_to_xy(s: pd.Series) -> tuple[list, list]:
        s = s.dropna()
        return [str(d.date()) for d in s.index], s.tolist()

    dates, prices = series_to_xy(closes)
    fast_x, fast_y = series_to_xy(fast)
    slow_x, slow_y = series_to_xy(slow)

    traces = [
        {"x": dates, "y": prices, "name": "Price", "type": "scatter", "mode": "lines",
         "line": {"color": "#4f8ef7", "width": 1.5}},
        {"x": fast_x, "y": fast_y, "name": f"MA{ma_fast}", "type": "scatter", "mode": "lines",
         "line": {"color": "#f5a623", "width": 1.5, "dash": "dot"}},
        {"x": slow_x, "y": slow_y, "name": f"MA{ma_slow}", "type": "scatter", "mode": "lines",
         "line": {"color": "#e05c5c", "width": 1.5, "dash": "dash"}},
    ]

    # Crossover markers (green)
    shapes = [{"type": "line", "x0": d, "x1": d, "y0": 0, "y1": 1, "xref": "x", "yref": "paper",
               "line": {"color": "rgba(80,200,120,0.7)", "width": 2}} for d in cross_dates]
    annotations = [{"x": d, "y": 1, "xref": "x", "yref": "paper", "text": "Cross",
                    "showarrow": False, "font": {"color": "rgba(80,200,120,1)", "size": 10},
                    "yanchor": "bottom"} for d in cross_dates]

    # Earnings markers (violet) within ±30 days of today
    earnings_dates = _get_earnings_window(ticker, days=30)
    for d in earnings_dates:
        shapes.append({"type": "line", "x0": d, "x1": d, "y0": 0, "y1": 1, "xref": "x", "yref": "paper",
                       "line": {"color": "rgba(168,120,255,0.8)", "width": 2, "dash": "dot"}})
        annotations.append({"x": d, "y": 0, "xref": "x", "yref": "paper", "text": "E",
                            "showarrow": False, "font": {"color": "rgba(168,120,255,1)", "size": 10},
                            "yanchor": "bottom"})

    # If an upcoming earnings sits beyond the last price point, extend the x-axis to keep it visible
    xaxis = {"gridcolor": "#2a2a4a"}
    if earnings_dates:
        last_date = dates[-1] if dates else None
        max_earn = max(earnings_dates)
        if last_date and max_earn > last_date:
            xaxis["range"] = [dates[0], max_earn]

    layout = {
        "height": 300, "margin": {"l": 50, "r": 20, "t": 10, "b": 40},
        "plot_bgcolor": "#1a1a2e", "paper_bgcolor": "#13132a", "font": {"color": "#eaeaea"},
        "xaxis": xaxis, "yaxis": {"gridcolor": "#2a2a4a"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        "shapes": shapes, "annotations": annotations,
    }

    return jsonify({"traces": traces, "layout": layout, "earnings": earnings_dates})


def _get_earnings_window(ticker: str, days: int = 30) -> list[str]:
    """Return earnings dates (as YYYY-MM-DD strings) within +/- `days` of today."""
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=24)
        if ed is None or ed.empty:
            return []
        today = pd.Timestamp.now(tz=ed.index.tz)
        lo = today - pd.Timedelta(days=days)
        hi = today + pd.Timedelta(days=days)
        in_window = [ts for ts in ed.index if lo <= ts <= hi]
        return sorted({ts.date().isoformat() for ts in in_window})
    except Exception:
        return []


if __name__ == "__main__":
    print("Starting Stock Screener at http://localhost:8080")
    app.run(debug=False, port=8080)
