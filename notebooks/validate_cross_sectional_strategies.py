from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _infer_symbol(path: Path) -> str:
    return path.stem.split("_")[0].upper()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"datetime", "timestamp", "date", "time", "index"}:
            rename[col] = "Datetime"
        elif key in {"ticker", "symbol"}:
            rename[col] = "symbol"
        elif key == "close":
            rename[col] = "Close"
        elif key == "volume":
            rename[col] = "Volume"
    if rename:
        df = df.rename(columns=rename)
    return df


def _looks_like_legacy_raw(df: pd.DataFrame) -> bool:
    if df.shape[0] < 3 or df.shape[1] < 2:
        return False
    first_col = str(df.columns[0]).strip().lower()
    if first_col not in {"price", "datetime", "index"}:
        return False
    row0 = str(df.iloc[0, 0]).strip().lower()
    row1 = str(df.iloc[1, 0]).strip().lower()
    return row0 == "ticker" and row1 == "datetime"


def _extract_legacy_rows(df: pd.DataFrame, fallback_symbol: str) -> pd.DataFrame:
    symbol = str(df.iloc[0, 1]).strip().upper() if df.shape[1] > 1 else fallback_symbol
    if not symbol or symbol == "NAN":
        symbol = fallback_symbol

    local = pd.DataFrame(
        {
            "Datetime": df.iloc[2:, 0],
            "Close": df.iloc[2:, 1] if df.shape[1] > 1 else None,
            "Volume": df.iloc[2:, 5] if df.shape[1] > 5 else 0.0,
            "symbol": symbol,
        }
    )
    return local


def build_panel_from_symbol_csvs(glob_pattern: str, out_path: Path) -> Path:
    files = sorted(Path().glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")

    rows: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        if _looks_like_legacy_raw(df):
            rows.append(_extract_legacy_rows(df, fallback_symbol=_infer_symbol(path)))
            continue

        df = _standardize_columns(df)
        if "Datetime" not in df.columns or "Close" not in df.columns:
            continue
        symbol = (
            str(df["symbol"].dropna().iloc[0]).upper()
            if "symbol" in df.columns and df["symbol"].notna().any()
            else _infer_symbol(path)
        )
        local = df[["Datetime", "Close"]].copy()
        local["Volume"] = pd.to_numeric(
            df["Volume"], errors="coerce"
        ) if "Volume" in df.columns else 0.0
        local["symbol"] = symbol
        rows.append(local)

    if not rows:
        raise RuntimeError("No usable symbol CSVs for panel build.")

    panel = pd.concat(rows, ignore_index=True)
    panel["Datetime"] = pd.to_datetime(panel["Datetime"], utc=True, errors="coerce")
    panel["Close"] = pd.to_numeric(panel["Close"], errors="coerce")
    panel["Volume"] = pd.to_numeric(panel["Volume"], errors="coerce").fillna(0.0)
    panel = panel.dropna(subset=["Datetime", "symbol", "Close"])
    panel["symbol"] = panel["symbol"].astype(str).str.upper()
    panel = panel.sort_values(["Datetime", "symbol"]).drop_duplicates(
        ["Datetime", "symbol"], keep="last"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)
    return out_path


def maybe_refresh_panel(
    refresh_data: bool,
    symbols: str,
    timeframe: str,
    limit: int,
    feed: str,
    out_path: Path,
) -> Path | None:
    if not refresh_data:
        return None
    cmd = [
        sys.executable,
        "download_panel_data.py",
        "--symbols",
        symbols,
        "--timeframe",
        timeframe,
        "--limit",
        str(limit),
        "--feed",
        feed,
        "--output",
        str(out_path),
        "--report-path",
        str(out_path.with_name(out_path.stem + "_coverage.csv")),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def run_reversal_notebook(
    panel_csv: Path,
    output_dir: Path,
    lookback_grid: str,
    hold_grid: str,
    decile: float,
    top_n: int,
    min_universe: int,
    target_annual_vol: float,
    max_leverage: float,
    liquidity_lookback: int,
    cost_bps: float,
) -> Path:
    cmd = [
        sys.executable,
        "notebooks/exp_intraday_reversal.py",
        "--panel-csv",
        str(panel_csv),
        "--lookback-grid",
        lookback_grid,
        "--hold-grid",
        hold_grid,
        "--decile",
        str(decile),
        "--top-n",
        str(top_n),
        "--min-universe",
        str(min_universe),
        "--liquidity-lookback",
        str(liquidity_lookback),
        "--target-annual-vol",
        str(target_annual_vol),
        "--max-leverage",
        str(max_leverage),
        "--cost-bps",
        str(cost_bps),
        "--cost-grid",
        str(cost_bps),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    return output_dir / "intraday_reversal_best_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorized validation wrapper for cross-sectional reversal ideas."
    )
    parser.add_argument(
        "--panel-csv",
        default="",
        help="Optional existing long-form panel CSV (Datetime,symbol,Close,Volume).",
    )
    parser.add_argument(
        "--stock-glob",
        default="data/legacy/raw_data_stock/*_1m_data.csv",
        help="If --panel-csv is omitted, build panel from this glob.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh panel via download_panel_data.py before validation.",
    )
    parser.add_argument(
        "--symbols",
        default="AAPL,AMD,NVDA,TSLA,NFLX,PLTR,SPY",
        help="Symbols used for refresh mode.",
    )
    parser.add_argument("--timeframe", default="1Min", help="Timeframe for refresh mode.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Bar limit for refresh mode.",
    )
    parser.add_argument(
        "--feed",
        default="iex",
        help="Stock data feed for refresh mode.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        help="Cost bps passed to vectorized reversal script.",
    )
    parser.add_argument(
        "--output-dir",
        default="notebooks/results",
        help="Output root for generated artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.panel_csv:
        panel_path = Path(args.panel_csv)
        if not panel_path.exists():
            raise FileNotFoundError(f"--panel-csv not found: {panel_path}")
    else:
        panel_path = output_root / "stock_panel_1m.csv"
        refreshed = False
        if args.refresh_data:
            try:
                maybe_refresh_panel(
                    refresh_data=True,
                    symbols=args.symbols,
                    timeframe=args.timeframe,
                    limit=args.limit,
                    feed=args.feed,
                    out_path=panel_path,
                )
                refreshed = True
            except Exception as exc:
                print(f"Panel refresh failed; falling back to local files. Error: {exc}")
        if not refreshed:
            panel_path = build_panel_from_symbol_csvs(args.stock_glob, panel_path)

    aggressive_dir = output_root / "cross_sectional_aggressive"
    vol_targeted_dir = output_root / "cross_sectional_vol_targeted"

    aggressive_metrics_path = run_reversal_notebook(
        panel_csv=panel_path,
        output_dir=aggressive_dir,
        lookback_grid="1,2",
        hold_grid="1,2,3",
        decile=0.30,
        top_n=600,
        min_universe=4,
        target_annual_vol=0.0,
        max_leverage=1.0,
        liquidity_lookback=30,
        cost_bps=args.cost_bps,
    )
    vol_targeted_metrics_path = run_reversal_notebook(
        panel_csv=panel_path,
        output_dir=vol_targeted_dir,
        lookback_grid="1,2,3",
        hold_grid="2,3,5",
        decile=0.25,
        top_n=600,
        min_universe=4,
        target_annual_vol=1.20,
        max_leverage=8.0,
        liquidity_lookback=30,
        cost_bps=args.cost_bps,
    )

    aggressive = pd.read_csv(aggressive_metrics_path)
    aggressive.insert(0, "strategy", "cross_sectional_aggressive")
    vol_targeted = pd.read_csv(vol_targeted_metrics_path)
    vol_targeted.insert(0, "strategy", "cross_sectional_vol_targeted")

    summary = pd.concat([aggressive, vol_targeted], ignore_index=True)
    summary_path = output_root / "cross_sectional_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Cross-sectional vectorized validation complete")
    print(f"Panel used: {panel_path}")
    print(f"Summary: {summary_path}")
    print(f"Aggressive artifacts: {aggressive_dir}")
    print(f"Vol-targeted artifacts: {vol_targeted_dir}")


if __name__ == "__main__":
    main()
