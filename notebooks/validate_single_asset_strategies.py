from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ANNUAL_MINUTES_STOCK = 252 * 390
ANNUAL_MINUTES_CRYPTO = 365 * 24 * 60


@dataclass
class SimulationResult:
    net_returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    position: pd.Series
    metrics: Dict[str, float]


def _split_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _infer_symbol(path: Path) -> str:
    stem = path.stem
    return stem.split("_")[0].upper()


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


def _load_legacy_raw_price_series(df: pd.DataFrame, fallback_symbol: str) -> Tuple[str, pd.Series]:
    symbol = str(df.iloc[0, 1]).strip().upper() if df.shape[1] > 1 else fallback_symbol
    if not symbol or symbol == "NAN":
        symbol = fallback_symbol

    local = pd.DataFrame(
        {
            "Datetime": df.iloc[2:, 0],
            "Close": df.iloc[2:, 1] if df.shape[1] > 1 else np.nan,
        }
    )
    local["Datetime"] = pd.to_datetime(local["Datetime"], utc=True, errors="coerce")
    local["Close"] = pd.to_numeric(local["Close"], errors="coerce")
    local = local.dropna(subset=["Datetime", "Close"]).sort_values("Datetime")
    local = local.drop_duplicates(subset=["Datetime"], keep="last")
    return symbol, local.set_index("Datetime")["Close"]


def load_price_series(path: Path) -> Tuple[str, pd.Series]:
    df = pd.read_csv(path)
    if _looks_like_legacy_raw(df):
        return _load_legacy_raw_price_series(df, fallback_symbol=_infer_symbol(path))

    df = _standardize_columns(df)
    if "Datetime" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"{path} must contain Datetime and Close columns.")

    if "symbol" in df.columns and df["symbol"].notna().any():
        symbol = str(df["symbol"].dropna().iloc[0]).upper()
    else:
        symbol = _infer_symbol(path)

    out = df[["Datetime", "Close"]].copy()
    out["Datetime"] = pd.to_datetime(out["Datetime"], utc=True, errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Datetime", "Close"]).sort_values("Datetime")
    out = out.drop_duplicates(subset=["Datetime"], keep="last")

    return symbol, out.set_index("Datetime")["Close"]


def safe_sharpe(returns: pd.Series, bars_per_year: int) -> float:
    std = float(returns.std())
    if std == 0.0:
        return 0.0
    return float(np.sqrt(bars_per_year) * returns.mean() / std)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def compute_metrics(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    turnover: pd.Series,
    bars_per_year: int,
) -> Dict[str, float]:
    gross_returns = gross_returns.fillna(0.0)
    net_returns = net_returns.fillna(0.0)
    turnover = turnover.fillna(0.0)
    gross_pnl = float(gross_returns.sum())
    total_turnover = float(turnover.sum())
    break_even_bps = 0.0
    if total_turnover > 0:
        break_even_bps = float(10_000.0 * gross_pnl / total_turnover)

    return {
        "gross_sharpe": safe_sharpe(gross_returns, bars_per_year),
        "net_sharpe": safe_sharpe(net_returns, bars_per_year),
        "gross_ann_return": float(gross_returns.mean() * bars_per_year),
        "net_ann_return": float(net_returns.mean() * bars_per_year),
        "gross_ann_vol": float(gross_returns.std() * np.sqrt(bars_per_year)),
        "net_ann_vol": float(net_returns.std() * np.sqrt(bars_per_year)),
        "gross_max_drawdown": max_drawdown(gross_returns),
        "net_max_drawdown": max_drawdown(net_returns),
        "hit_rate": float((net_returns > 0.0).mean()),
        "avg_turnover": float(turnover.mean()),
        "total_turnover": total_turnover,
        "break_even_cost_bps": break_even_bps,
    }


def compute_position_metrics(position: pd.Series, turnover: pd.Series) -> Dict[str, float]:
    pos = position.fillna(0.0).astype(float)
    turn = turnover.fillna(0.0).astype(float)
    n = int(len(pos))
    if n == 0:
        return {
            "trade_ticks": 0.0,
            "trade_tick_pct": 0.0,
            "entry_ticks": 0.0,
            "entry_tick_pct": 0.0,
            "exit_ticks": 0.0,
            "exit_tick_pct": 0.0,
            "flip_ticks": 0.0,
            "flip_tick_pct": 0.0,
            "bars_in_market": 0.0,
            "market_exposure_pct": 0.0,
            "avg_abs_position": 0.0,
            "hold_count": 0.0,
            "avg_hold_bars": 0.0,
            "median_hold_bars": 0.0,
            "max_hold_bars": 0.0,
        }

    prev = pos.shift(1).fillna(0.0)
    trade_ticks = int((turn > 0.0).sum())
    entry_ticks = int(((prev == 0.0) & (pos != 0.0)).sum())
    exit_ticks = int(((prev != 0.0) & (pos == 0.0)).sum())
    flip_ticks = int(((prev * pos) < 0.0).sum())
    bars_in_market = int((pos != 0.0).sum())

    # Holding periods measured as contiguous runs of non-zero position.
    hold_lengths: list[int] = []
    run = 0
    for value in (pos != 0.0).astype(int):
        if value == 1:
            run += 1
        elif run > 0:
            hold_lengths.append(run)
            run = 0
    if run > 0:
        hold_lengths.append(run)

    hold_count = len(hold_lengths)
    avg_hold = float(np.mean(hold_lengths)) if hold_lengths else 0.0
    median_hold = float(np.median(hold_lengths)) if hold_lengths else 0.0
    max_hold = float(np.max(hold_lengths)) if hold_lengths else 0.0

    return {
        "trade_ticks": float(trade_ticks),
        "trade_tick_pct": float(trade_ticks / n),
        "entry_ticks": float(entry_ticks),
        "entry_tick_pct": float(entry_ticks / n),
        "exit_ticks": float(exit_ticks),
        "exit_tick_pct": float(exit_ticks / n),
        "flip_ticks": float(flip_ticks),
        "flip_tick_pct": float(flip_ticks / n),
        "bars_in_market": float(bars_in_market),
        "market_exposure_pct": float(bars_in_market / n),
        "avg_abs_position": float(pos.abs().mean()),
        "hold_count": float(hold_count),
        "avg_hold_bars": avg_hold,
        "median_hold_bars": median_hold,
        "max_hold_bars": max_hold,
    }


def simulate_from_position(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float,
    bars_per_year: int,
) -> SimulationResult:
    close = close.astype(float)
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pos = position.reindex(close.index).fillna(0.0).astype(float).clip(-1.0, 1.0)
    bar_pos = pos.shift(1).fillna(0.0)

    gross = bar_pos * ret
    turnover = pos.diff().abs().fillna(pos.abs())
    net = gross - turnover * (cost_bps / 10_000.0)

    metrics = {
        **compute_metrics(
            gross_returns=gross,
            net_returns=net,
            turnover=turnover,
            bars_per_year=bars_per_year,
        ),
        **compute_position_metrics(position=pos, turnover=turnover),
    }
    return SimulationResult(
        net_returns=net,
        gross_returns=gross,
        turnover=turnover,
        position=pos,
        metrics=metrics,
    )


def summarize_portfolio_metrics(
    net_returns: pd.Series,
    turnover: pd.Series,
    bars_per_year: int,
    positions_wide: pd.DataFrame,
    turnover_wide: pd.DataFrame,
) -> Dict[str, float]:
    metrics = compute_metrics(
        gross_returns=net_returns,
        net_returns=net_returns,
        turnover=turnover,
        bars_per_year=bars_per_year,
    )
    if len(net_returns) == 0:
        metrics.update(
            {
                "trade_ticks": 0.0,
                "trade_tick_pct": 0.0,
                "entry_ticks": 0.0,
                "entry_tick_pct": 0.0,
                "exit_ticks": 0.0,
                "exit_tick_pct": 0.0,
                "flip_ticks": 0.0,
                "flip_tick_pct": 0.0,
                "bars_in_market": 0.0,
                "market_exposure_pct": 0.0,
                "avg_abs_position": 0.0,
                "hold_count": 0.0,
                "avg_hold_bars": 0.0,
                "median_hold_bars": 0.0,
                "max_hold_bars": 0.0,
            }
        )
        return metrics

    any_trade = (turnover_wide > 0.0).any(axis=1)
    prev = positions_wide.shift(1).fillna(0.0)
    now = positions_wide.fillna(0.0)
    any_entry = ((prev == 0.0) & (now != 0.0)).any(axis=1)
    any_exit = ((prev != 0.0) & (now == 0.0)).any(axis=1)
    any_flip = ((prev * now) < 0.0).any(axis=1)
    any_in_market = (now != 0.0).any(axis=1)

    hold_lengths: list[int] = []
    run = 0
    for value in any_in_market.astype(int):
        if value == 1:
            run += 1
        elif run > 0:
            hold_lengths.append(run)
            run = 0
    if run > 0:
        hold_lengths.append(run)
    hold_count = len(hold_lengths)
    avg_hold = float(np.mean(hold_lengths)) if hold_lengths else 0.0
    median_hold = float(np.median(hold_lengths)) if hold_lengths else 0.0
    max_hold = float(np.max(hold_lengths)) if hold_lengths else 0.0

    metrics.update(
        {
            "trade_ticks": float(any_trade.sum()),
            "trade_tick_pct": float(any_trade.mean()),
            "entry_ticks": float(any_entry.sum()),
            "entry_tick_pct": float(any_entry.mean()),
            "exit_ticks": float(any_exit.sum()),
            "exit_tick_pct": float(any_exit.mean()),
            "flip_ticks": float(any_flip.sum()),
            "flip_tick_pct": float(any_flip.mean()),
            "bars_in_market": float(any_in_market.sum()),
            "market_exposure_pct": float(any_in_market.mean()),
            "avg_abs_position": float(now.abs().mean(axis=1).mean()),
            "hold_count": float(hold_count),
            "avg_hold_bars": avg_hold,
            "median_hold_bars": median_hold,
            "max_hold_bars": max_hold,
        }
    )
    return metrics


def position_fast_ma(close: pd.Series, short_window: int, long_window: int) -> pd.Series:
    ma_short = close.rolling(short_window, min_periods=1).mean()
    ma_long = close.rolling(long_window, min_periods=1).mean()
    return np.sign(ma_short - ma_long).astype(float)


def position_micro_momentum(
    close: pd.Series,
    lookback: int,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.Series:
    momentum = close.pct_change(lookback).fillna(0.0)
    signal = pd.Series(0.0, index=close.index, dtype=float)
    signal.loc[momentum > buy_threshold] = 1.0
    signal.loc[momentum < sell_threshold] = -1.0
    return signal.replace(0.0, np.nan).ffill().fillna(0.0)


def position_crypto_ema(close: pd.Series, short_window: int, long_window: int) -> pd.Series:
    ema_fast = close.ewm(span=short_window, adjust=False).mean()
    ema_slow = close.ewm(span=long_window, adjust=False).mean()
    return (ema_fast > ema_slow).astype(float)


def position_crypto_regime_trend(
    close: pd.Series,
    short_window: int = 20,
    long_window: int = 120,
    slope_lookback: int = 240,
    volatility_window: int = 1440,
    volatility_quantile: float = 0.70,
) -> pd.Series:
    ema_fast = close.ewm(span=short_window, adjust=False).mean()
    ema_slow = close.ewm(span=long_window, adjust=False).mean()
    slow_slope = ema_slow.pct_change(slope_lookback).fillna(0.0)

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    min_vol_samples = max(20, volatility_window // 3)
    min_quantile_samples = max(50, volatility_window // 2)
    realized_vol = returns.rolling(volatility_window, min_periods=min_vol_samples).std()
    vol_q = realized_vol.rolling(
        volatility_window, min_periods=min_quantile_samples
    ).quantile(volatility_quantile)
    vol_gate = realized_vol <= vol_q.shift(1)

    return ((ema_fast > ema_slow) & (slow_slope > 0.0) & vol_gate.fillna(False)).astype(
        float
    )


def run_download(
    symbol: str,
    timeframe: str,
    limit: int,
    asset_class: str,
    feed: str,
) -> None:
    cmd = [
        sys.executable,
        "download_data.py",
        symbol,
        "--timeframe",
        timeframe,
        "--limit",
        str(limit),
        "--asset-class",
        asset_class,
    ]
    if asset_class == "stock":
        cmd.extend(["--feed", feed])
    subprocess.run(cmd, check=True)


def maybe_refresh_data(
    refresh_data: bool,
    stock_symbols: Iterable[str],
    crypto_symbols: Iterable[str],
    timeframe: str,
    limit: int,
    feed: str,
) -> None:
    if not refresh_data:
        return
    for sym in stock_symbols:
        run_download(
            symbol=sym.upper(),
            timeframe=timeframe,
            limit=limit,
            asset_class="stock",
            feed=feed,
        )
    for sym in crypto_symbols:
        run_download(
            symbol=sym.upper(),
            timeframe=timeframe,
            limit=limit,
            asset_class="crypto",
            feed=feed,
        )


def load_universe_from_glob(glob_pattern: str) -> Dict[str, pd.Series]:
    files = sorted(Path().glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    out: Dict[str, pd.Series] = {}
    for path in files:
        symbol, close = load_price_series(path)
        out[symbol] = close
    return out


def evaluate_strategy(
    strategy_name: str,
    close_by_symbol: Dict[str, pd.Series],
    position_builder,
    cost_bps: float,
    bars_per_year: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    per_symbol_rows: List[Dict[str, float | str | int]] = []
    net_returns: Dict[str, pd.Series] = {}
    turnover: Dict[str, pd.Series] = {}
    positions: Dict[str, pd.Series] = {}

    for symbol, close in close_by_symbol.items():
        pos = position_builder(close)
        result = simulate_from_position(
            close=close,
            position=pos,
            cost_bps=cost_bps,
            bars_per_year=bars_per_year,
        )
        per_symbol_rows.append(
            {
                "strategy": strategy_name,
                "symbol": symbol,
                "bars": int(close.shape[0]),
                **result.metrics,
            }
        )
        net_returns[symbol] = result.net_returns
        turnover[symbol] = result.turnover
        positions[symbol] = result.position

    symbol_df = pd.DataFrame(per_symbol_rows)

    joined_returns = pd.concat(net_returns, axis=1).sort_index().fillna(0.0)
    joined_turnover = pd.concat(turnover, axis=1).sort_index().fillna(0.0)
    joined_positions = pd.concat(positions, axis=1).sort_index().fillna(0.0)
    portfolio_returns = joined_returns.mean(axis=1)
    portfolio_turnover = joined_turnover.mean(axis=1)
    portfolio_metrics = summarize_portfolio_metrics(
        net_returns=portfolio_returns,
        turnover=portfolio_turnover,
        bars_per_year=bars_per_year,
        positions_wide=joined_positions,
        turnover_wide=joined_turnover,
    )
    portfolio_metrics["universe_size"] = int(symbol_df["symbol"].nunique())
    portfolio_metrics["bars"] = int(joined_returns.shape[0])
    return symbol_df, portfolio_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorized notebook-style validation for single-asset strategy ideas."
    )
    parser.add_argument(
        "--stock-glob",
        default="data/legacy/raw_data_stock/*_1m_data.csv",
        help="Glob for stock OHLCV CSV files.",
    )
    parser.add_argument(
        "--crypto-glob",
        default="data/legacy/raw_data_crypto/*_1m_data.csv",
        help="Glob for crypto OHLCV CSV files.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh data by calling download_data.py before running validations.",
    )
    parser.add_argument(
        "--stock-symbols",
        default="AAPL,AMD,NVDA,TSLA,NFLX,PLTR,SPY",
        help="Symbols for refresh mode.",
    )
    parser.add_argument(
        "--crypto-symbols",
        default="BTCUSD,ETHUSD,SOLUSD",
        help="Crypto symbols for refresh mode.",
    )
    parser.add_argument("--timeframe", default="1Min", help="Timeframe for refresh mode.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Limit bars per symbol in refresh mode.",
    )
    parser.add_argument(
        "--feed",
        default="iex",
        help="Alpaca stock feed for refresh mode.",
    )
    parser.add_argument(
        "--stock-cost-bps",
        type=float,
        default=0.0,
        help="Linear transaction cost in bps for stock strategies.",
    )
    parser.add_argument(
        "--crypto-cost-bps",
        type=float,
        default=0.0,
        help="Linear transaction cost in bps for crypto strategy.",
    )
    parser.add_argument(
        "--output-dir",
        default="notebooks/results",
        help="Output directory for CSV artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_symbols = [s.upper() for s in _split_csv(args.stock_symbols)]
    crypto_symbols = [s.upper() for s in _split_csv(args.crypto_symbols)]

    if args.refresh_data:
        try:
            maybe_refresh_data(
                refresh_data=True,
                stock_symbols=stock_symbols,
                crypto_symbols=crypto_symbols,
                timeframe=args.timeframe,
                limit=args.limit,
                feed=args.feed,
            )
        except Exception as exc:
            print(f"Data refresh failed; falling back to local files. Error: {exc}")

    stock_close = load_universe_from_glob(args.stock_glob)
    crypto_close = load_universe_from_glob(args.crypto_glob)

    all_symbol_metrics: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, float | str | int]] = []

    ma_symbol, ma_portfolio = evaluate_strategy(
        strategy_name="fast_ma_churn",
        close_by_symbol=stock_close,
        position_builder=lambda close: position_fast_ma(close, short_window=5, long_window=20),
        cost_bps=args.stock_cost_bps,
        bars_per_year=ANNUAL_MINUTES_STOCK,
    )
    all_symbol_metrics.append(ma_symbol)
    summary_rows.append({"strategy": "fast_ma_churn", "asset_class": "stock", **ma_portfolio})

    mom_symbol, mom_portfolio = evaluate_strategy(
        strategy_name="micro_momentum_flip",
        close_by_symbol=stock_close,
        position_builder=lambda close: position_micro_momentum(
            close, lookback=3, buy_threshold=0.0008, sell_threshold=-0.0008
        ),
        cost_bps=args.stock_cost_bps,
        bars_per_year=ANNUAL_MINUTES_STOCK,
    )
    all_symbol_metrics.append(mom_symbol)
    summary_rows.append(
        {"strategy": "micro_momentum_flip", "asset_class": "stock", **mom_portfolio}
    )

    ema_symbol, ema_portfolio = evaluate_strategy(
        strategy_name="crypto_fast_ema_long_only",
        close_by_symbol=crypto_close,
        position_builder=lambda close: position_crypto_ema(close, short_window=5, long_window=20),
        cost_bps=args.crypto_cost_bps,
        bars_per_year=ANNUAL_MINUTES_CRYPTO,
    )
    all_symbol_metrics.append(ema_symbol)
    summary_rows.append(
        {
            "strategy": "crypto_fast_ema_long_only",
            "asset_class": "crypto",
            **ema_portfolio,
        }
    )

    regime_symbol, regime_portfolio = evaluate_strategy(
        strategy_name="crypto_regime_trend",
        close_by_symbol=crypto_close,
        position_builder=lambda close: position_crypto_regime_trend(
            close=close,
            short_window=20,
            long_window=120,
            slope_lookback=240,
            volatility_window=1440,
            volatility_quantile=0.70,
        ),
        cost_bps=args.crypto_cost_bps,
        bars_per_year=ANNUAL_MINUTES_CRYPTO,
    )
    all_symbol_metrics.append(regime_symbol)
    summary_rows.append(
        {
            "strategy": "crypto_regime_trend",
            "asset_class": "crypto",
            **regime_portfolio,
        }
    )

    per_symbol_df = pd.concat(all_symbol_metrics, ignore_index=True).sort_values(
        ["strategy", "symbol"]
    )
    summary_df = pd.DataFrame(summary_rows).sort_values("net_sharpe", ascending=False)

    per_symbol_path = output_dir / "single_asset_per_symbol_metrics.csv"
    summary_path = output_dir / "single_asset_portfolio_summary.csv"

    per_symbol_df.to_csv(per_symbol_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("Single-asset vectorized validation complete")
    print(f"Stock symbols: {len(stock_close)} | Crypto symbols: {len(crypto_close)}")
    print(f"Portfolio summary: {summary_path}")
    print(f"Per-symbol metrics: {per_symbol_path}")


if __name__ == "__main__":
    main()
