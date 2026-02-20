"""
Download historical bar data from Alpaca and save as CSV.

Usage:
    python download_data.py RVNC
    python download_data.py AAPL --timeframe 5Min --limit 5000
    python download_data.py TSLA --timeframe 1Min --start 2025-01-01 --end 2025-02-01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from pipeline.alpaca import fetch_stock_bars, fetch_crypto_bars, get_rest, save_bars, _normalize_bars, _parse_timeframe


DATA_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical bar data from Alpaca.")
    parser.add_argument("symbol", help="Ticker symbol (e.g. AAPL, RVNC, TSLA)")
    parser.add_argument("--timeframe", default="1Min", help="Bar timeframe: 1Min, 5Min, 15Min, 1Hour, 1Day (default: 1Min)")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max bars in latest-bars mode. In crypto date-range mode, this is page size (default: 1000, max page: 10000).",
    )
    parser.add_argument("--start", default=None, help="Start date (e.g. 2025-01-01)")
    parser.add_argument("--end", default=None, help="End date (e.g. 2025-02-01)")
    parser.add_argument("--feed", default="iex", help="Data feed: iex or sip (default: iex)")
    parser.add_argument("--asset-class", choices=["stock", "crypto"], default="stock", help="Asset class (default: stock)")
    return parser.parse_args()


def fetch_crypto_bars_range_paged(
    api,
    symbol: str,
    timeframe: str,
    start: str,
    end: Optional[str],
    page_size: int,
):
    if page_size <= 0:
        raise ValueError("--limit/page size must be positive")
    page_size = min(page_size, 10_000)
    timeframe_value = str(_parse_timeframe(timeframe))

    rows = []
    page_token = None

    while True:
        data = {
            "symbols": symbol,
            "timeframe": timeframe_value,
            "start": start,
            "limit": page_size,
        }
        if end:
            data["end"] = end
        if page_token:
            data["page_token"] = page_token

        resp = api.data_get("/crypto/us/bars", data=data, api_version="v1beta3")
        bars_by_symbol = resp.get("bars", {}) if isinstance(resp, dict) else {}
        items = bars_by_symbol.get(symbol, [])
        if not items and bars_by_symbol:
            # Some responses may key by normalized symbol variant.
            first_key = next(iter(bars_by_symbol.keys()))
            items = bars_by_symbol.get(first_key, [])

        for item in items:
            rows.append(
                {
                    "Datetime": item.get("t"),
                    "Open": item.get("o"),
                    "High": item.get("h"),
                    "Low": item.get("l"),
                    "Close": item.get("c"),
                    "Volume": item.get("v"),
                }
            )

        page_token = resp.get("next_page_token") if isinstance(resp, dict) else None
        if not page_token:
            break

    if not rows:
        raise ValueError(f"No crypto bars returned for {symbol} in requested range.")

    import pandas as pd

    df = pd.DataFrame.from_records(rows)
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Datetime", "Close"]).sort_values("Datetime")
    df = df.drop_duplicates(subset=["Datetime"], keep="last")
    return df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()

    if args.asset_class == "crypto":
        if "/" not in symbol:
            symbol = symbol.replace("USD", "/USD")
        if args.start or args.end:
            api = get_rest()
            if not args.start:
                raise ValueError("--start is required for crypto date-range mode.")
            df = fetch_crypto_bars_range_paged(
                api=api,
                symbol=symbol,
                timeframe=args.timeframe,
                start=args.start,
                end=args.end,
                page_size=args.limit,
            )
        else:
            df = fetch_crypto_bars(symbol, timeframe=args.timeframe, limit=args.limit)
    elif args.start or args.end:
        api = get_rest()
        tf = _parse_timeframe(args.timeframe)
        kwargs = {"limit": args.limit, "feed": args.feed}
        if args.start:
            kwargs["start"] = args.start
        if args.end:
            kwargs["end"] = args.end
        bars = api.get_bars(symbol, tf, **kwargs).df
        df = _normalize_bars(bars, symbol)
    else:
        df = fetch_stock_bars(symbol, timeframe=args.timeframe, limit=args.limit, feed=args.feed)

    path = save_bars(df, symbol.replace("/", ""), args.timeframe, args.asset_class)
    print(f"Saved {len(df)} bars to {path}")


if __name__ == "__main__":
    main()
