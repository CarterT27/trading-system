from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        symbol = str(value).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def parse_symbols_csv(text: str) -> List[str]:
    if not text:
        return []
    return _dedupe_keep_order([s for s in text.split(",") if s.strip()])


def read_symbols_file(path: str | Path) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Symbols file not found: {path}")

    if path.suffix.lower() == ".txt":
        return _dedupe_keep_order(path.read_text().splitlines())

    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            normalized = {name.lower().strip(): name for name in reader.fieldnames}
            symbol_col = normalized.get("symbol") or normalized.get("ticker")
            if symbol_col is None:
                raise ValueError(
                    f"CSV must include 'symbol' or 'ticker' column: {path}"
                )
            symbols = [str(row.get(symbol_col, "")).strip().upper() for row in reader]
        return _dedupe_keep_order(symbols)

    raise ValueError("Unsupported symbols file type. Use .txt or .csv")


def resolve_symbols(
    default_symbol: str,
    symbols_arg: str = "",
    symbols_file_arg: str = "",
    symbol_limit: int = 0,
) -> List[str]:
    symbols = []
    if symbols_arg:
        symbols.extend(parse_symbols_csv(symbols_arg))
    if symbols_file_arg:
        symbols.extend(read_symbols_file(symbols_file_arg))
    if not symbols:
        symbols = [default_symbol]

    symbols = _dedupe_keep_order(symbols)
    if symbol_limit < 0:
        raise ValueError("symbol_limit must be >= 0")
    if symbol_limit > 0:
        symbols = symbols[:symbol_limit]
    return symbols
