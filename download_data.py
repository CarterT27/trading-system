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

from pipeline.alpaca import fetch_stock_bars, fetch_crypto_bars, get_rest, save_bars, _normalize_bars, _parse_timeframe


DATA_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical bar data from Alpaca.")
    parser.add_argument("symbol", help="Ticker symbol (e.g. AAPL, RVNC, TSLA)")
    parser.add_argument("--timeframe", default="1Min", help="Bar timeframe: 1Min, 5Min, 15Min, 1Hour, 1Day (default: 1Min)")
    parser.add_argument("--limit", type=int, default=1000, help="Max number of bars (default: 1000, max: 10000)")
    parser.add_argument("--start", default=None, help="Start date (e.g. 2025-01-01)")
    parser.add_argument("--end", default=None, help="End date (e.g. 2025-02-01)")
    parser.add_argument("--feed", default="iex", help="Data feed: iex or sip (default: iex)")
    parser.add_argument("--asset-class", choices=["stock", "crypto"], default="stock", help="Asset class (default: stock)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()

    if args.asset_class == "crypto":
        if "/" not in symbol:
            symbol = symbol.replace("USD", "/USD")
        if args.start or args.end:
            api = get_rest()
            tf = _parse_timeframe(args.timeframe)
            kwargs = {"limit": args.limit}
            if args.start:
                kwargs["start"] = args.start
            if args.end:
                kwargs["end"] = args.end
            bars = api.get_crypto_bars(symbol, tf, **kwargs).df
            df = _normalize_bars(bars, symbol)
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
