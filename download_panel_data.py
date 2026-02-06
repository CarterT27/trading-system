"""
Download multi-symbol stock bar data from Alpaca and save one panel CSV.

Examples:
    python download_panel_data.py --symbols AAPL,MSFT,NVDA,SPY --timeframe 1Min --limit 5000
    python download_panel_data.py --symbols-file data/universe.txt --start 2025-01-01 --end 2025-01-31
    python download_panel_data.py --alpaca-universe --symbols-only --symbols-out data/universe.txt
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import deque
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from pipeline.alpaca import (
    _normalize_bars,
    _parse_timeframe,
    fetch_stock_bars,
    get_rest,
)


class RequestRateLimiter:
    """
    Sliding-window rate limiter.

    Alpaca stock data commonly has a 200 requests/minute limit in standard setups.
    We keep a configurable cap and default to a conservative 190 req/min.
    """

    def __init__(self, max_requests_per_minute: int) -> None:
        if max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive")
        self.max_requests_per_minute = max_requests_per_minute
        self.window_seconds = 60.0
        self.request_times: deque[float] = deque()

    def wait_for_slot(self) -> None:
        now = time.monotonic()
        self._evict_old(now)
        if len(self.request_times) >= self.max_requests_per_minute:
            earliest = self.request_times[0]
            sleep_for = max(0.0, earliest + self.window_seconds - now) + 0.02
            time.sleep(sleep_for)
            now = time.monotonic()
            self._evict_old(now)
        self.request_times.append(time.monotonic())

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a multi-symbol panel CSV from Alpaca."
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols (e.g. AAPL,MSFT,NVDA,SPY).",
    )
    parser.add_argument(
        "--symbols-file", default="", help="Path to symbols file (.txt, .csv)."
    )
    parser.add_argument(
        "--alpaca-universe",
        action="store_true",
        help="Pull active tradable US equity symbols from Alpaca and merge into the universe.",
    )
    parser.add_argument(
        "--universe-exchanges",
        default="NYSE,NASDAQ,ARCA",
        help="Comma-separated exchange filter for Alpaca universe (default: NYSE,NASDAQ,ARCA). Use empty string for all.",
    )
    parser.add_argument(
        "--only-shortable",
        action="store_true",
        help="When using Alpaca universe, keep only shortable symbols.",
    )
    parser.add_argument(
        "--only-easy-to-borrow",
        action="store_true",
        help="When using Alpaca universe, keep only easy-to-borrow symbols.",
    )
    parser.add_argument(
        "--fractionable-only",
        action="store_true",
        help="When using Alpaca universe, keep only fractionable symbols.",
    )
    parser.add_argument(
        "--simple-symbols-only",
        action="store_true",
        help="When using Alpaca universe, keep only simple tickers matching ^[A-Z]{1,5}$.",
    )
    parser.add_argument(
        "--symbol-limit",
        type=int,
        default=0,
        help="Optional cap on number of symbols after filtering (0 = no cap).",
    )
    parser.add_argument(
        "--symbols-out",
        default="",
        help="Optional path to save resolved symbols list (.txt or .csv with symbol column).",
    )
    parser.add_argument(
        "--symbols-only",
        action="store_true",
        help="Resolve/write symbol universe and exit without downloading bars.",
    )
    parser.add_argument(
        "--timeframe", default="1Min", help="Bar timeframe (default: 1Min)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Latest-bars cap in single-symbol mode; page size in ranged mode (default: 5000).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional start date/time (RFC3339 or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end", default=None, help="Optional end date/time (RFC3339 or YYYY-MM-DD)."
    )
    parser.add_argument("--feed", default="iex", help="Stock data feed (iex or sip).")
    parser.add_argument(
        "--exclude-spy",
        action="store_true",
        help="Do not auto-append SPY to the symbol list.",
    )
    parser.add_argument(
        "--max-requests-per-minute",
        type=int,
        default=190,
        help="Rate limiter cap (default: 190, under the 200 req/min limit).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Symbols per bars request when using --start/--end (default: 100).",
    )
    parser.add_argument(
        "--verbose-symbols",
        action="store_true",
        help="Print per-symbol bar counts even in batch mode.",
    )
    parser.add_argument(
        "--output",
        default="data/panels/us_equities_1min_panel.csv",
        help="Output panel CSV path (default: data/panels/us_equities_1min_panel.csv).",
    )
    parser.add_argument(
        "--report-path",
        default="data/panels/us_equities_1min_panel_coverage.csv",
        help="Per-symbol coverage report output path.",
    )
    return parser.parse_args()


def _split_symbols(text: str) -> List[str]:
    if not text:
        return []
    return [s.strip().upper() for s in text.split(",") if s.strip()]


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        key = v.upper().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _split_csv_values(text: str) -> List[str]:
    if not text:
        return []
    return [s.strip().upper() for s in text.split(",") if s.strip()]


def _chunked(values: List[str], size: int) -> Iterable[List[str]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _read_symbols_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".txt":
        symbols = [
            line.strip().upper()
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        return _dedupe_keep_order(symbols)

    if ext == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            normalized = {name.lower().strip(): name for name in reader.fieldnames}
            symbol_col = normalized.get("symbol") or normalized.get("ticker")
            if symbol_col is None:
                raise ValueError(f"CSV must have 'symbol' or 'ticker' column: {path}")

            symbols = []
            for row in reader:
                raw = str(row.get(symbol_col, "")).strip().upper()
                if raw:
                    symbols.append(raw)
            return _dedupe_keep_order(symbols)

    raise ValueError("Unsupported symbols-file type. Use .txt or .csv")


def fetch_alpaca_universe_symbols(
    api,
    exchanges_csv: str,
    only_shortable: bool,
    only_easy_to_borrow: bool,
    fractionable_only: bool,
    simple_symbols_only: bool,
) -> List[str]:
    exchanges = set(_split_csv_values(exchanges_csv))
    assets = api.list_assets(status="active", asset_class="us_equity")
    symbols: List[str] = []

    for asset in assets:
        symbol = str(getattr(asset, "symbol", "")).strip().upper()
        if not symbol:
            continue

        if simple_symbols_only and re.fullmatch(r"[A-Z]{1,5}", symbol) is None:
            continue

        if not bool(getattr(asset, "tradable", False)):
            continue

        if exchanges:
            exchange = str(getattr(asset, "exchange", "")).strip().upper()
            if exchange not in exchanges:
                continue

        if only_shortable and not bool(getattr(asset, "shortable", False)):
            continue

        if only_easy_to_borrow and not bool(getattr(asset, "easy_to_borrow", False)):
            continue

        if fractionable_only and not bool(getattr(asset, "fractionable", False)):
            continue

        symbols.append(symbol)

    return _dedupe_keep_order(sorted(symbols))


def write_symbols_file(symbols: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        pd.DataFrame({"symbol": symbols}).to_csv(path, index=False)
        return

    if path.suffix.lower() in {"", ".txt"}:
        path.write_text("\n".join(symbols) + "\n")
        return

    raise ValueError("Unsupported --symbols-out extension. Use .txt or .csv")


def resolve_symbols(
    symbols_arg: str,
    symbols_file_arg: str,
    include_spy: bool,
    api,
    use_alpaca_universe: bool,
    universe_exchanges: str,
    only_shortable: bool,
    only_easy_to_borrow: bool,
    fractionable_only: bool,
    simple_symbols_only: bool,
    symbol_limit: int,
) -> List[str]:
    symbols = _split_symbols(symbols_arg)
    if symbols_file_arg:
        symbols.extend(_read_symbols_file(Path(symbols_file_arg)))

    if use_alpaca_universe:
        symbols.extend(
            fetch_alpaca_universe_symbols(
                api=api,
                exchanges_csv=universe_exchanges,
                only_shortable=only_shortable,
                only_easy_to_borrow=only_easy_to_borrow,
                fractionable_only=fractionable_only,
                simple_symbols_only=simple_symbols_only,
            )
        )

    symbols = _dedupe_keep_order(symbols)

    if include_spy and "SPY" not in symbols:
        symbols.append("SPY")

    if symbol_limit < 0:
        raise ValueError("--symbol-limit must be >= 0")
    if symbol_limit > 0 and len(symbols) > symbol_limit:
        if include_spy and "SPY" in symbols and "SPY" not in symbols[:symbol_limit]:
            trimmed = symbols[: max(0, symbol_limit - 1)]
            trimmed.append("SPY")
            symbols = _dedupe_keep_order(trimmed)
        else:
            symbols = symbols[:symbol_limit]

    if not symbols:
        raise ValueError(
            "No symbols resolved. Use --symbols, --symbols-file, or --alpaca-universe."
        )
    return symbols


def fetch_symbol_bars(
    symbol: str,
    timeframe: str,
    limit: int,
    start: str | None,
    end: str | None,
    feed: str,
    api,
) -> pd.DataFrame:
    if start or end:
        tf = _parse_timeframe(timeframe)
        kwargs = {"limit": limit, "feed": feed}
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
        bars = api.get_bars(symbol, tf, **kwargs).df
        return _normalize_bars(bars, symbol)

    return fetch_stock_bars(
        symbol, timeframe=timeframe, limit=limit, feed=feed, api=api
    )


def fetch_batch_bars(
    symbols: List[str],
    timeframe: str,
    limit: int,
    start: str | None,
    end: str | None,
    feed: str,
    api,
    limiter: RequestRateLimiter,
) -> tuple[pd.DataFrame, int]:
    if not symbols:
        empty = pd.DataFrame(
            columns=["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
        )
        return empty, 0

    page_limit = 10_000
    if limit and limit > 0:
        page_limit = min(limit, 10_000)

    timeframe_str = str(_parse_timeframe(timeframe))
    page_token: str | None = None
    pages = 0
    chunks: List[pd.DataFrame] = []

    while True:
        limiter.wait_for_slot()
        data: dict[str, object] = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe_str,
            "limit": page_limit,
        }
        if start:
            data["start"] = start
        if end:
            data["end"] = end
        if page_token:
            data["page_token"] = page_token

        resp = api.data_get("/stocks/bars", data=data, feed=feed, api_version="v2")
        pages += 1

        bars_by_symbol = resp.get("bars", {}) if isinstance(resp, dict) else {}
        rows: List[dict] = []
        for symbol in symbols:
            items = bars_by_symbol.get(symbol, [])
            for item in items:
                rows.append(
                    {
                        "Datetime": item.get("t"),
                        "symbol": symbol,
                        "Open": item.get("o"),
                        "High": item.get("h"),
                        "Low": item.get("l"),
                        "Close": item.get("c"),
                        "Volume": item.get("v"),
                    }
                )

        if rows:
            chunk_df = pd.DataFrame.from_records(rows)
            chunk_df["Datetime"] = pd.to_datetime(
                chunk_df["Datetime"], utc=True, errors="coerce"
            )
            chunk_df["symbol"] = chunk_df["symbol"].astype(str).str.upper()
            chunk_df = chunk_df.dropna(subset=["Datetime", "symbol", "Close", "Volume"])
            if not chunk_df.empty:
                chunks.append(chunk_df)

        page_token = resp.get("next_page_token") if isinstance(resp, dict) else None
        if not page_token:
            break

    if not chunks:
        empty = pd.DataFrame(
            columns=["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
        )
        return empty, pages

    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values(["Datetime", "symbol"]).drop_duplicates(
        ["Datetime", "symbol"], keep="last"
    )
    return out, pages


def main() -> None:
    args = parse_args()
    api = get_rest()
    symbols = resolve_symbols(
        args.symbols,
        args.symbols_file,
        include_spy=(not args.exclude_spy),
        api=api,
        use_alpaca_universe=args.alpaca_universe,
        universe_exchanges=args.universe_exchanges,
        only_shortable=args.only_shortable,
        only_easy_to_borrow=args.only_easy_to_borrow,
        fractionable_only=args.fractionable_only,
        simple_symbols_only=args.simple_symbols_only,
        symbol_limit=args.symbol_limit,
    )

    if args.symbols_out:
        symbols_out_path = Path(args.symbols_out)
        write_symbols_file(symbols, symbols_out_path)
        print(f"Resolved symbols file: {symbols_out_path} ({len(symbols)} symbols)")

    if args.symbols_only:
        print(f"Resolved {len(symbols)} symbols")
        return

    limiter = RequestRateLimiter(max_requests_per_minute=args.max_requests_per_minute)

    frames: List[pd.DataFrame] = []
    failures: List[dict] = []

    print(
        f"Downloading {len(symbols)} symbols | timeframe={args.timeframe} "
        f"| rate_limit={args.max_requests_per_minute} req/min"
    )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    use_batch_mode = args.batch_size > 1 and bool(args.start or args.end)
    if args.batch_size > 1 and not use_batch_mode:
        print(
            "Batch requests require --start/--end for reliable multi-symbol coverage; "
            "falling back to single-symbol mode."
        )

    if use_batch_mode:
        chunks = list(_chunked(symbols, args.batch_size))
        print(
            f"Using batched requests: {len(chunks)} requests (batch_size={args.batch_size})"
        )

        for idx, chunk in enumerate(chunks, start=1):
            try:
                batch_df, pages = fetch_batch_bars(
                    symbols=chunk,
                    timeframe=args.timeframe,
                    limit=args.limit,
                    start=args.start,
                    end=args.end,
                    feed=args.feed,
                    api=api,
                    limiter=limiter,
                )
            except Exception as exc:
                print(
                    f"[{idx}/{len(chunks)}] batch failed, fallback to singles ({exc})"
                )
                for symbol in chunk:
                    try:
                        if args.start or args.end:
                            df, _ = fetch_batch_bars(
                                symbols=[symbol],
                                timeframe=args.timeframe,
                                limit=args.limit,
                                start=args.start,
                                end=args.end,
                                feed=args.feed,
                                api=api,
                                limiter=limiter,
                            )
                        else:
                            limiter.wait_for_slot()
                            df = fetch_symbol_bars(
                                symbol=symbol,
                                timeframe=args.timeframe,
                                limit=args.limit,
                                start=args.start,
                                end=args.end,
                                feed=args.feed,
                                api=api,
                            )
                        if df.empty:
                            failures.append(
                                {"symbol": symbol, "error": "No data returned"}
                            )
                            if args.verbose_symbols:
                                print(f"  - {symbol}: no data")
                            continue

                        local = df.copy()
                        local["symbol"] = symbol
                        local = local.loc[
                            :,
                            [
                                "Datetime",
                                "symbol",
                                "Open",
                                "High",
                                "Low",
                                "Close",
                                "Volume",
                            ],
                        ].copy()
                        frames.append(local)
                        if args.verbose_symbols:
                            print(f"  - {symbol}: {len(local)} bars")
                    except Exception as single_exc:
                        failures.append({"symbol": symbol, "error": str(single_exc)})
                        if args.verbose_symbols:
                            print(f"  - {symbol}: failed ({single_exc})")
                continue

            counts = (
                batch_df.groupby("symbol").size().to_dict()
                if not batch_df.empty
                else {}
            )
            missing_symbols = [s for s in chunk if s not in counts]
            for symbol in missing_symbols:
                failures.append({"symbol": symbol, "error": "No data returned"})

            if not batch_df.empty:
                frames.append(batch_df)

            if args.verbose_symbols:
                for symbol in chunk:
                    cnt = int(counts.get(symbol, 0))
                    status = f"{cnt} bars" if cnt > 0 else "no data"
                    print(f"[{idx}/{len(chunks)}] {symbol}: {status}")
            else:
                print(
                    f"[{idx}/{len(chunks)}] batch symbols={len(chunk)} "
                    f"hits={len(counts)} bars={len(batch_df)} pages={pages}"
                )

    else:
        for idx, symbol in enumerate(symbols, start=1):
            try:
                if args.start or args.end:
                    df, pages = fetch_batch_bars(
                        symbols=[symbol],
                        timeframe=args.timeframe,
                        limit=args.limit,
                        start=args.start,
                        end=args.end,
                        feed=args.feed,
                        api=api,
                        limiter=limiter,
                    )
                else:
                    limiter.wait_for_slot()
                    df = fetch_symbol_bars(
                        symbol=symbol,
                        timeframe=args.timeframe,
                        limit=args.limit,
                        start=args.start,
                        end=args.end,
                        feed=args.feed,
                        api=api,
                    )
                    pages = 1
                if df.empty:
                    failures.append({"symbol": symbol, "error": "No data returned"})
                    print(f"[{idx}/{len(symbols)}] {symbol}: no data")
                    continue

                local = df.copy()
                local["symbol"] = symbol
                local = local.loc[
                    :, ["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
                ].copy()
                frames.append(local)
                print(
                    f"[{idx}/{len(symbols)}] {symbol}: {len(local)} bars (pages={pages})"
                )
            except Exception as exc:
                failures.append({"symbol": symbol, "error": str(exc)})
                print(f"[{idx}/{len(symbols)}] {symbol}: failed ({exc})")

    if not frames:
        raise RuntimeError("No symbols were downloaded successfully.")

    panel = pd.concat(frames, ignore_index=True)
    panel["Datetime"] = pd.to_datetime(panel["Datetime"], utc=True, errors="coerce")
    panel = panel.dropna(subset=["Datetime", "symbol", "Close", "Volume"])
    panel = panel.sort_values(["Datetime", "symbol"]).drop_duplicates(
        ["Datetime", "symbol"], keep="last"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)

    coverage = (
        panel.groupby("symbol")
        .agg(
            bars=("Close", "size"),
            first_timestamp=("Datetime", "min"),
            last_timestamp=("Datetime", "max"),
            avg_volume=("Volume", "mean"),
        )
        .reset_index()
        .sort_values("bars", ascending=False)
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(report_path, index=False)

    failure_path = output_path.with_name(output_path.stem + "_failures.csv")
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)

    print("\nPanel download complete")
    print(f"Successful symbols: {coverage['symbol'].nunique()} / {len(symbols)}")
    print(f"Rows: {len(panel)}")
    print(f"Panel CSV: {output_path}")
    print(f"Coverage report: {report_path}")
    if failures:
        print(f"Failures: {len(failures)} (see {failure_path})")


if __name__ == "__main__":
    main()
