"""Fetch and cache historical event windows for the static frontend.

This script intentionally runs outside the browser. It fetches each unique
ticker once, builds a T-5 through T+5 trading-session window for every call,
and writes a small sidecar consumed by ``export_frontend_data.py``.

Alpaca credentials are read from a local ``.env`` file or from
ALPACA_API_KEY/ALPACA_API_SECRET (the shorter ALPACA_KEY/ALPACA_SECRET names
are also accepted). The resulting files are local research artifacts; API
credentials are never written to them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MARKET_DIR = ROOT / "artifacts" / "market_data"
CACHE_PATH = MARKET_DIR / "alpaca_daily_bars.json"
WINDOW_PATH = MARKET_DIR / "price_windows.json"
DEFAULT_BENCHMARK = "SPY"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load simple KEY=VALUE entries without adding a dotenv dependency."""
    if not path.exists():
        return
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()


def _json_load(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(value) else value


def _call_key(symbol: str, timestamp: Any) -> str:
    parsed = pd.to_datetime(timestamp, errors="coerce")
    return f"{symbol}|{parsed.isoformat() if pd.notna(parsed) else str(timestamp)}"


def _load_calls() -> list[dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    for table_path in sorted((ROOT / "artifacts").glob("*/feature_table.csv")):
        table = pd.read_csv(table_path)
        if not {"symbol", "call_datetime"}.issubset(table.columns):
            continue
        for _, row in table.iterrows():
            timestamp = pd.to_datetime(row.get("call_datetime"), errors="coerce")
            if pd.isna(timestamp):
                continue
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            key = _call_key(symbol, timestamp)
            phase = str(row.get("call_phase", "") or "").lower()
            calls[key] = {
                "id": key,
                "symbol": symbol,
                "timestamp": timestamp,
                "phase": phase,
            }
    return sorted(calls.values(), key=lambda item: item["timestamp"])


def _alpaca_bars(symbol: str, start: date, end: date, api_key: str, api_secret: str, feed: str) -> list[dict[str, Any]]:
    """Read adjusted daily close bars, following pagination when needed."""
    # paper-api.alpaca.markets is the trading/account endpoint. Historical
    # bars are served from the separate market-data endpoint below.
    base = f"https://data.alpaca.markets/v2/stocks/{urllib.parse.quote(symbol)}/bars"
    bars: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params = {
            "timeframe": "1Day",
            "start": f"{start.isoformat()}T00:00:00Z",
            "end": f"{(end + timedelta(days=1)).isoformat()}T00:00:00Z",
            "adjustment": "all",
            "feed": feed,
            "sort": "asc",
            "limit": "10000",
        }
        if page_token:
            params["page_token"] = page_token
        request = urllib.request.Request(
            f"{base}?{urllib.parse.urlencode(params)}",
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca returned HTTP {error.code} for {symbol}: {body[:300]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach Alpaca for {symbol}: {error.reason}") from error

        page = payload.get("bars", []) if isinstance(payload, dict) else []
        bars.extend(page if isinstance(page, list) else [])
        page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
        if not page_token or not page:
            break
    return bars


def _cache_covers(record: dict[str, Any], start: date, end: date) -> bool:
    try:
        return date.fromisoformat(record["start"]) <= start and date.fromisoformat(record["end"]) >= end
    except (KeyError, TypeError, ValueError):
        return False


def _bar_frame(bars: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for bar in bars:
        timestamp = pd.to_datetime(bar.get("t"), errors="coerce", utc=True)
        close = _number(bar.get("c"))
        if pd.isna(timestamp) or close is None:
            continue
        rows.append({"date": timestamp.date(), "close": close})
    if not rows:
        return pd.DataFrame(columns=["date", "close"])
    return pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def _window(
    call: dict[str, Any],
    stock: pd.DataFrame,
    benchmark: pd.DataFrame,
    radius: int,
) -> dict[str, Any] | None:
    if stock.empty:
        return None

    call_date = call["timestamp"].date()
    reaction_position = int(stock["date"].searchsorted(call_date, side="left"))
    start_position = reaction_position - radius
    end_position = reaction_position + radius
    if reaction_position >= len(stock) or start_position < 0 or end_position >= len(stock):
        return None

    stock_slice = stock.iloc[start_position : end_position + 1].copy()
    dates = stock_slice["date"].tolist()
    stock_values = stock_slice["close"].astype(float).tolist()

    benchmark_values: list[float] | None = None
    if not benchmark.empty and all(day in benchmark.index for day in dates):
        benchmark_values = [float(benchmark.loc[day, "close"]) for day in dates]

    phase = call.get("phase", "")
    after_close = "after" in phase or "close" in phase or "post" in phase
    event_index = radius - 0.5 if after_close else radius
    evaluation_start = radius if after_close else radius

    return {
        "priceSeries": stock_values,
        "benchmarkSeries": benchmark_values,
        "priceDates": [day.isoformat() for day in dates],
        "chartLabels": [f"T{index - radius:+d}" for index in range(len(dates))],
        "eventIndex": event_index,
        "evaluationStartIndex": evaluation_start,
        "evaluationEndIndex": radius + radius,
        "provider": "alpaca",
        "benchmarkSymbol": DEFAULT_BENCHMARK,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Benchmark ticker, default: SPY")
    parser.add_argument("--feed", default=os.getenv("ALPACA_FEED", "iex"), help="Alpaca feed, default: iex")
    parser.add_argument("--radius", type=int, default=5, help="Trading sessions before and after the event, default: 5")
    parser.add_argument("--refresh", action="store_true", help="Refetch cached ticker data")
    args = parser.parse_args()

    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY") or os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        print("Set ALPACA_API_KEY and ALPACA_API_SECRET before running this script.", file=sys.stderr)
        return 2

    calls = _load_calls()
    if not calls:
        print("No artifact calls were found.", file=sys.stderr)
        return 2

    start = calls[0]["timestamp"].date() - timedelta(days=20)
    end = calls[-1]["timestamp"].date() + timedelta(days=20)
    symbols = sorted({call["symbol"] for call in calls} | {args.benchmark.upper()})
    cache = _json_load(CACHE_PATH, {"version": 1, "provider": "alpaca", "symbols": {}})
    cache_symbols = cache.setdefault("symbols", {})

    for index, symbol in enumerate(symbols, start=1):
        cached = cache_symbols.get(symbol)
        if not args.refresh and isinstance(cached, dict) and _cache_covers(cached, start, end):
            print(f"[{index}/{len(symbols)}] {symbol}: using cached bars")
            continue
        print(f"[{index}/{len(symbols)}] {symbol}: fetching {start} to {end}")
        bars = _alpaca_bars(symbol, start, end, api_key, api_secret, args.feed)
        cache_symbols[symbol] = {"start": start.isoformat(), "end": end.isoformat(), "bars": bars}
        _json_dump(CACHE_PATH, cache)

    windows: dict[str, Any] = {}
    bar_frames = {
        symbol: _bar_frame(record.get("bars", []))
        for symbol, record in cache_symbols.items()
        if isinstance(record, dict)
    }
    benchmark_frame = bar_frames.get(args.benchmark.upper(), pd.DataFrame())
    if not benchmark_frame.empty:
        benchmark_frame = benchmark_frame.set_index("date")
    for call in calls:
        result = _window(call, bar_frames.get(call["symbol"], pd.DataFrame()), benchmark_frame, args.radius)
        if result is not None:
            windows[call["id"]] = result

    output = {
        "version": 1,
        "provider": "alpaca",
        "benchmark": args.benchmark.upper(),
        "radius": args.radius,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "windows": windows,
    }
    _json_dump(WINDOW_PATH, output)
    print(f"Wrote {len(windows)} of {len(calls)} call windows to {WINDOW_PATH}")
    print(f"Cached raw bars at {CACHE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
