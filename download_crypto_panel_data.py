"""
Download multi-symbol crypto bar data from Alpaca and save one panel CSV.

Examples:
    python download_crypto_panel_data.py --symbols BTCUSD,ETHUSD,SOLUSD --timeframe 2Min --start 2025-12-01 --end 2026-01-01
    python download_crypto_panel_data.py --symbols-file data/universe_crypto.txt --timeframe 2Min --start 2025-12-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from core.alpaca_trader import normalize_crypto_symbols
from pipeline.alpaca import _parse_timeframe, get_rest


class RequestRateLimiter:
    """Sliding-window request limiter."""

    def __init__(self, max_requests_per_minute: int) -> None:
        if max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive")
        self.max_requests_per_minute = int(max_requests_per_minute)
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
        description="Download a multi-symbol crypto panel CSV from Alpaca."
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols (e.g. BTCUSD,ETHUSD,SOLUSD or BTC/USD,ETH/USD).",
    )
    parser.add_argument(
        "--symbols-file",
        default="",
        help="Path to symbols file (.txt or .csv with symbol/ticker column).",
    )
    parser.add_argument(
        "--symbol-limit",
        type=int,
        default=0,
        help="Optional cap on symbol count after normalization (0 = no cap).",
    )
    parser.add_argument(
        "--symbols-out",
        default="",
        help="Optional path to save resolved symbols list (.txt or .csv).",
    )
    parser.add_argument(
        "--symbols-only",
        action="store_true",
        help="Resolve/write symbol universe and exit without downloading bars.",
    )
    parser.add_argument(
        "--timeframe",
        default="2Min",
        help="Bar timeframe (default: 2Min).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Page size for ranged mode (default: 10000, max: 10000).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional start timestamp (RFC3339 or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Optional end timestamp (RFC3339 or YYYY-MM-DD). "
            "Date-only end is treated as inclusive (expanded by +1 day)."
        ),
    )
    parser.add_argument(
        "--max-requests-per-minute",
        type=int,
        default=190,
        help="Rate limiter cap (default: 190).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Symbols per request in ranged mode (default: 50).",
    )
    parser.add_argument(
        "--panel-symbol-format",
        choices=("dash", "slash", "trade"),
        default="dash",
        help="Output symbol format in panel CSV (default: dash; e.g., BTC-USD).",
    )
    parser.add_argument(
        "--verbose-symbols",
        action="store_true",
        help="Print per-symbol bar counts.",
    )
    parser.add_argument(
        "--output",
        default="data/panels/crypto_us_2min_panel.csv",
        help="Output panel CSV path.",
    )
    parser.add_argument(
        "--report-path",
        default="data/panels/crypto_us_2min_panel_coverage.csv",
        help="Per-symbol coverage report path.",
    )
    return parser.parse_args()


def _split_symbols(text: str) -> List[str]:
    if not text:
        return []
    return [s.strip().upper() for s in text.split(",") if s.strip()]


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        key = value.strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


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
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
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


def _normalize_range_bounds(
    start: str | None,
    end: str | None,
) -> tuple[str | None, str | None]:
    """Normalize date-only range bounds to UTC RFC3339 strings."""
    start_out = start
    end_out = end

    if start_out and len(start_out) == 10:
        ts = pd.Timestamp(start_out, tz="UTC")
        start_out = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    if end_out and len(end_out) == 10:
        ts = pd.Timestamp(end_out, tz="UTC") + pd.Timedelta(days=1)
        end_out = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    return start_out, end_out


def resolve_symbols(
    symbols_arg: str,
    symbols_file_arg: str,
    symbol_limit: int,
) -> tuple[List[str], dict[str, str]]:
    requested = _split_symbols(symbols_arg)
    if symbols_file_arg:
        requested.extend(_read_symbols_file(Path(symbols_file_arg)))
    requested = _dedupe_keep_order(requested)
    if not requested:
        raise ValueError("No symbols resolved. Use --symbols or --symbols-file.")

    data_symbols: List[str] = []
    trade_by_data: dict[str, str] = {}
    for symbol in requested:
        trade_symbol, data_symbol = normalize_crypto_symbols(symbol)
        trade_symbol = trade_symbol.upper()
        data_symbol = data_symbol.upper()
        if "/" not in data_symbol:
            raise ValueError(
                f"Invalid crypto symbol '{symbol}'. Use quote pairs like BTCUSD or BTC/USD."
            )
        if data_symbol in trade_by_data:
            continue
        data_symbols.append(data_symbol)
        trade_by_data[data_symbol] = trade_symbol

    if symbol_limit < 0:
        raise ValueError("--symbol-limit must be >= 0")
    if symbol_limit > 0:
        data_symbols = data_symbols[:symbol_limit]
        trade_by_data = {s: trade_by_data[s] for s in data_symbols}

    if not data_symbols:
        raise ValueError("No usable symbols after normalization.")
    return data_symbols, trade_by_data


def _format_panel_symbol(data_symbol: str, trade_symbol: str, fmt: str) -> str:
    if fmt == "slash":
        return data_symbol
    if fmt == "trade":
        return trade_symbol
    return data_symbol.replace("/", "-")


def write_symbols_file(
    data_symbols: List[str],
    trade_by_data: dict[str, str],
    output_format: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        rows = [
            {
                "symbol": _format_panel_symbol(s, trade_by_data[s], output_format),
                "data_symbol": s,
                "trade_symbol": trade_by_data[s],
            }
            for s in data_symbols
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        return

    if path.suffix.lower() in {"", ".txt"}:
        lines = [
            _format_panel_symbol(s, trade_by_data[s], output_format)
            for s in data_symbols
        ]
        path.write_text("\n".join(lines) + "\n")
        return

    raise ValueError("Unsupported --symbols-out extension. Use .txt or .csv")


def fetch_batch_bars(
    symbols: List[str],
    timeframe: str,
    limit: int,
    start: str | None,
    end: str | None,
    api,
    limiter: RequestRateLimiter,
) -> tuple[pd.DataFrame, int]:
    if not symbols:
        empty = pd.DataFrame(
            columns=["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
        )
        return empty, 0

    page_limit = min(max(int(limit), 1), 10_000)
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

        resp = api.data_get("/crypto/us/bars", data=data, api_version="v1beta3")
        pages += 1

        bars_by_symbol = resp.get("bars", {}) if isinstance(resp, dict) else {}
        normalized_map: dict[str, list] = {}
        for key, values in bars_by_symbol.items():
            _, data_symbol = normalize_crypto_symbols(str(key))
            normalized_map[data_symbol.upper()] = values

        rows: List[dict] = []
        for symbol in symbols:
            items = normalized_map.get(symbol, [])
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
            for col in ("Open", "High", "Low", "Close", "Volume"):
                chunk_df[col] = pd.to_numeric(chunk_df[col], errors="coerce")
            chunk_df = chunk_df.dropna(
                subset=["Datetime", "symbol", "Close", "Volume"]
            )
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

    start, end = _normalize_range_bounds(args.start, args.end)
    data_symbols, trade_by_data = resolve_symbols(
        symbols_arg=args.symbols,
        symbols_file_arg=args.symbols_file,
        symbol_limit=args.symbol_limit,
    )

    if args.symbols_out:
        symbols_out_path = Path(args.symbols_out)
        write_symbols_file(
            data_symbols=data_symbols,
            trade_by_data=trade_by_data,
            output_format=args.panel_symbol_format,
            path=symbols_out_path,
        )
        print(
            f"Resolved symbols file: {symbols_out_path} ({len(data_symbols)} symbols)"
        )

    if args.symbols_only:
        print(f"Resolved {len(data_symbols)} symbols")
        return

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    limiter = RequestRateLimiter(max_requests_per_minute=args.max_requests_per_minute)
    failures: List[dict] = []
    frames: List[pd.DataFrame] = []

    print(
        f"Downloading {len(data_symbols)} symbols | timeframe={args.timeframe} "
        f"| rate_limit={args.max_requests_per_minute} req/min"
    )
    chunks = list(_chunked(data_symbols, args.batch_size))
    print(f"Using batched requests: {len(chunks)} requests (batch_size={args.batch_size})")

    for idx, chunk in enumerate(chunks, start=1):
        try:
            batch_df, pages = fetch_batch_bars(
                symbols=chunk,
                timeframe=args.timeframe,
                limit=args.limit,
                start=start,
                end=end,
                api=api,
                limiter=limiter,
            )
        except Exception as exc:
            print(f"[{idx}/{len(chunks)}] batch failed, fallback to singles ({exc})")
            for symbol in chunk:
                try:
                    df, pages = fetch_batch_bars(
                        symbols=[symbol],
                        timeframe=args.timeframe,
                        limit=args.limit,
                        start=start,
                        end=end,
                        api=api,
                        limiter=limiter,
                    )
                except Exception as single_exc:
                    failures.append({"symbol": symbol, "error": str(single_exc)})
                    if args.verbose_symbols:
                        print(f"  - {symbol}: failed ({single_exc})")
                    continue

                if df.empty:
                    failures.append({"symbol": symbol, "error": "No data returned"})
                    if args.verbose_symbols:
                        print(f"  - {symbol}: no data")
                    continue
                frames.append(df)
                if args.verbose_symbols:
                    print(f"  - {symbol}: {len(df)} bars (pages={pages})")
            continue

        counts = batch_df.groupby("symbol").size().to_dict() if not batch_df.empty else {}
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

    if not frames:
        raise RuntimeError("No symbols were downloaded successfully.")

    panel = pd.concat(frames, ignore_index=True)
    panel["Datetime"] = pd.to_datetime(panel["Datetime"], utc=True, errors="coerce")
    panel = panel.dropna(subset=["Datetime", "symbol", "Close", "Volume"])
    panel = panel.sort_values(["Datetime", "symbol"]).drop_duplicates(
        ["Datetime", "symbol"], keep="last"
    )

    if start:
        start_ts = pd.Timestamp(start, tz="UTC")
        panel = panel[panel["Datetime"] >= start_ts]
    if end:
        end_ts = pd.Timestamp(end, tz="UTC")
        panel = panel[panel["Datetime"] < end_ts]

    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    panel["symbol"] = panel["symbol"].apply(
        lambda s: _format_panel_symbol(
            data_symbol=s,
            trade_symbol=trade_by_data.get(s, s.replace("/", "")),
            fmt=args.panel_symbol_format,
        )
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
        mapped_failures = []
        for row in failures:
            raw_symbol = str(row.get("symbol", ""))
            mapped_failures.append(
                {
                    "symbol": _format_panel_symbol(
                        data_symbol=raw_symbol,
                        trade_symbol=trade_by_data.get(raw_symbol, raw_symbol.replace("/", "")),
                        fmt=args.panel_symbol_format,
                    ),
                    "data_symbol": raw_symbol,
                    "error": row.get("error", ""),
                }
            )
        pd.DataFrame(mapped_failures).to_csv(failure_path, index=False)

    print("\nCrypto panel download complete")
    print(f"Successful symbols: {coverage['symbol'].nunique()} / {len(data_symbols)}")
    print(f"Rows: {len(panel)}")
    print(f"Panel CSV: {output_path}")
    print(f"Coverage report: {report_path}")
    if failures:
        print(f"Failures: {len(failures)} (see {failure_path})")


if __name__ == "__main__":
    main()
