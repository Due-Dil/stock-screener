"""Data layer: fetch & cache Elon Musk posts (XTracker) and Polymarket brackets (Gamma).

XTracker (https://xtracker.polymarket.com) is Polymarket's *resolution source* for the
"# of tweets" markets, so the post set returned here matches what the market resolves on
(main-feed posts + quote posts + reposts; replies excluded). We verified this: the API
returns exactly 240 posts for the June 19-26 window, which resolved in the 240-259 bracket.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

XTRACKER_API = "https://xtracker.polymarket.com/api"
GAMMA_API = "https://gamma-api.polymarket.com"
DEFAULT_HANDLE = "elonmusk"

_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "cache.db"
_UA = {"User-Agent": "tweet-analyst/1.0 (+research)"}


# --------------------------------------------------------------------------- #
# SQLite cache
# --------------------------------------------------------------------------- #
def _conn(path: Path = _CACHE_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute(
        """CREATE TABLE IF NOT EXISTS posts (
               platform_id TEXT PRIMARY KEY,
               handle      TEXT NOT NULL,
               created_at  TEXT NOT NULL,   -- UTC ISO8601
               content     TEXT,
               is_repost   INTEGER NOT NULL DEFAULT 0
           )"""
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_posts_time ON posts(handle, created_at)")
    con.execute(
        """CREATE TABLE IF NOT EXISTS fetched_ranges (
               handle TEXT NOT NULL,
               start  TEXT NOT NULL,
               end    TEXT NOT NULL
           )"""
    )
    return con


# --------------------------------------------------------------------------- #
# XTracker: posts
# --------------------------------------------------------------------------- #
def _iso(d: dt.datetime) -> str:
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _http_get(url: str, params: dict, retries: int = 3) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=_UA, timeout=45)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def _fetch_posts_remote(handle: str, start: dt.datetime, end: dt.datetime) -> list[dict]:
    """Fetch posts in monthly chunks (defensive against any server-side range limits)."""
    rows: dict[str, dict] = {}
    cursor = start
    step = dt.timedelta(days=30)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        payload = _http_get(
            f"{XTRACKER_API}/users/{handle}/posts",
            {"startDate": _iso(cursor), "endDate": _iso(chunk_end)},
        )
        for p in payload.get("data", []):
            content = p.get("content") or ""
            rows[p["platformId"]] = {
                "platform_id": p["platformId"],
                "handle": handle,
                "created_at": p["createdAt"],
                "content": content,
                "is_repost": int(content.startswith("RT @")),
            }
        cursor = chunk_end
    return list(rows.values())


def refresh_posts(
    handle: str = DEFAULT_HANDLE,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
) -> int:
    """Fetch posts over [start, end] from XTracker and upsert into cache. Returns # upserted."""
    if end is None:
        end = dt.datetime.now(dt.timezone.utc)
    if start is None:
        start = end - dt.timedelta(days=240)
    rows = _fetch_posts_remote(handle, start, end)
    con = _conn()
    with con:
        con.executemany(
            """INSERT INTO posts(platform_id, handle, created_at, content, is_repost)
               VALUES(:platform_id,:handle,:created_at,:content,:is_repost)
               ON CONFLICT(platform_id) DO UPDATE SET
                   created_at=excluded.created_at, content=excluded.content,
                   is_repost=excluded.is_repost""",
            rows,
        )
        con.execute(
            "INSERT INTO fetched_ranges(handle,start,end) VALUES(?,?,?)",
            (handle, _iso(start), _iso(end)),
        )
    con.close()
    return len(rows)


def load_posts(
    handle: str = DEFAULT_HANDLE,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    """Return cached posts as a DataFrame with a UTC-aware ``created_at`` column (sorted)."""
    con = _conn()
    q = "SELECT created_at, content, is_repost FROM posts WHERE handle=?"
    args: list = [handle]
    if start is not None:
        q += " AND created_at>=?"
        args.append(_iso(start))
    if end is not None:
        q += " AND created_at<?"
        args.append(_iso(end))
    q += " ORDER BY created_at"
    df = pd.read_sql_query(q, con, params=args)
    con.close()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["is_repost"] = df["is_repost"].astype(bool)
    return df


def ensure_history(
    handle: str = DEFAULT_HANDLE,
    days: int = 240,
    max_age_hours: float = 0.1,      # ~6 min, aligned with XTracker's ~5-min polling of X
    overlap_hours: float = 12.0,     # re-pull a short tail to catch late/edited captures
) -> pd.DataFrame:
    """Ensure a fresh local history, refreshing **incrementally** when stale.

    Only the tail since the last cached tweet (minus an ``overlap_hours`` buffer) is fetched, not
    the full ``days`` window — so a live refresh pulls a few hours of data, not ~240 days. A full
    backfill happens only when the cache is empty or older than the requested window.
    """
    con = _conn()
    last = con.execute(
        "SELECT MAX(created_at) FROM posts WHERE handle=?", (handle,)
    ).fetchone()[0]
    con.close()
    now = dt.datetime.now(dt.timezone.utc)
    window_start = now - dt.timedelta(days=days)

    if last is None:
        refresh_posts(handle, start=window_start, end=now)  # first run: full backfill
    else:
        last_dt = pd.to_datetime(last, utc=True).to_pydatetime()
        if last_dt < window_start:                          # gap too large -> full backfill
            refresh_posts(handle, start=window_start, end=now)
        elif (now - last_dt) > dt.timedelta(hours=max_age_hours):
            inc_start = max(last_dt - dt.timedelta(hours=overlap_hours), window_start)
            refresh_posts(handle, start=inc_start, end=now)  # incremental tail only
    return load_posts(handle, start=window_start, end=now)


# --------------------------------------------------------------------------- #
# XTracker: trackings (market windows with exact start/end, handles DST)
# --------------------------------------------------------------------------- #
@dataclass
class TrackingWindow:
    title: str
    start: dt.datetime  # UTC-aware
    end: dt.datetime    # UTC-aware
    market_link: str
    is_active: bool


def get_trackings(handle: str = DEFAULT_HANDLE) -> list[TrackingWindow]:
    payload = _http_get(f"{XTRACKER_API}/users/{handle}", {})
    out = []
    for t in payload["data"].get("trackings", []):
        out.append(
            TrackingWindow(
                title=t["title"],
                start=pd.to_datetime(t["startDate"], utc=True).to_pydatetime(),
                end=pd.to_datetime(t["endDate"], utc=True).to_pydatetime(),
                market_link=t.get("marketLink", ""),
                is_active=bool(t.get("isActive")),
            )
        )
    return sorted(out, key=lambda w: w.start)


# --------------------------------------------------------------------------- #
# Polymarket Gamma: brackets + live prices
# --------------------------------------------------------------------------- #
@dataclass
class Bracket:
    label: str        # e.g. "240-259", "<20", "500+"
    low: float        # inclusive
    high: float       # inclusive (inf for "500+")
    yes_price: Optional[float]  # market price to BUY YES (≈ implied P(in bracket))
    no_price: Optional[float] = None  # market price to BUY NO (real, incl. spread; ≈1-yes_price)


@dataclass
class MarketEvent:
    title: str
    slug: str
    start: dt.datetime
    end: dt.datetime
    brackets: list[Bracket]


def _parse_bracket_bounds(label: str) -> tuple[float, float]:
    s = label.strip().replace("–", "-").replace("—", "-")
    if s.startswith("<"):
        return (0.0, float(s[1:]) - 1)
    if s.endswith("+"):
        return (float(s[:-1]), float("inf"))
    lo, _, hi = s.partition("-")
    return (float(lo), float(hi))


def get_market(slug: str) -> MarketEvent:
    data = _http_get(f"{GAMMA_API}/events", {"slug": slug})
    if not data:
        raise ValueError(f"No Polymarket event for slug={slug!r}")
    e = data[0]
    brackets: list[Bracket] = []
    for m in e.get("markets", []):
        label = m.get("groupItemTitle") or m.get("question", "")
        lo, hi = _parse_bracket_bounds(label)
        yes_price = None
        prices = m.get("outcomePrices")
        outcomes = m.get("outcomes")
        if isinstance(prices, str):
            import json

            prices = json.loads(prices)
            outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
        no_price = None
        if prices and outcomes:
            try:
                yi = [o.lower() for o in outcomes].index("yes")
                ni = [o.lower() for o in outcomes].index("no")
                yes_price = float(prices[yi])
                no_price = float(prices[ni])
            except (ValueError, TypeError):
                yes_price = float(prices[0])
                no_price = float(prices[1]) if len(prices) > 1 else None
        brackets.append(Bracket(label=label, low=lo, high=hi, yes_price=yes_price, no_price=no_price))
    brackets.sort(key=lambda b: b.low)
    return MarketEvent(
        title=e.get("title", slug),
        slug=slug,
        start=pd.to_datetime(e["startDate"], utc=True).to_pydatetime(),
        end=pd.to_datetime(e["endDate"], utc=True).to_pydatetime(),
        brackets=brackets,
    )


def resolve_window(
    slug: str, market: MarketEvent, handle: str = DEFAULT_HANDLE
) -> tuple[dt.datetime, dt.datetime]:
    """Return the true tweet-counting window (UTC) for a market.

    Gamma's ``event.startDate`` is the *creation* time, not the counting start, so we
    prefer the XTracker tracking window (exact, DST-correct, variable length). We match the
    tracking whose ``marketLink`` contains the slug; otherwise fall back to Gamma's reliable
    ``end`` minus a length inferred from the title (default 7 days).
    """
    try:
        for tw in get_trackings(handle):
            if slug and slug in (tw.market_link or ""):
                return tw.start, tw.end
    except Exception:  # noqa: BLE001  (offline / API hiccup -> fall through)
        pass
    end = market.end
    # Infer span from the title's two dates if possible, else assume a 7-day market.
    span = dt.timedelta(days=7)
    return end - span, end


def slug_from_url(url: str) -> str:
    """Extract the event slug from a full Polymarket URL or return the input if already a slug."""
    u = url.strip().split("?")[0].rstrip("/")
    if "/event/" in u:
        u = u.split("/event/")[1]
    return u.split("/")[0]
