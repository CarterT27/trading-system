from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANNUAL_MINUTES = 252 * 390


@dataclass
class BacktestResult:
    lookback: int
    hold_minutes: int
    cost_bps: float
    equity_net: pd.Series
    equity_gross: pd.Series
    net_returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    metrics: Dict[str, float]


def parse_int_grid(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Grid cannot be empty.")
    return sorted(set(values))


def parse_float_grid(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Grid cannot be empty.")
    return sorted(set(values))


def infer_symbol_from_filename(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_")[0].upper()
    return stem.upper()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c in {"datetime", "timestamp", "date", "time"}:
            rename[col] = "Datetime"
        elif c in {"symbol", "ticker"}:
            rename[col] = "symbol"
        elif c == "close":
            rename[col] = "Close"
        elif c == "volume":
            rename[col] = "Volume"
        elif c == "sector":
            rename[col] = "sector"
    if rename:
        df = df.rename(columns=rename)
    return df


def load_panel_from_single_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize_columns(df)
    required = {"Datetime", "symbol", "Close", "Volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    keep_cols = ["Datetime", "symbol", "Close", "Volume"]
    if "sector" in df.columns:
        keep_cols.append("sector")

    out = df[keep_cols].copy()
    out["Datetime"] = pd.to_datetime(out["Datetime"], utc=True, errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out = out.dropna(subset=["Datetime", "symbol", "Close", "Volume"])
    out = out.sort_values(["Datetime", "symbol"]).drop_duplicates(
        ["Datetime", "symbol"], keep="last"
    )
    return out


def load_panel_from_symbol_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        df = _standardize_columns(df)
        if (
            "Datetime" not in df.columns
            or "Close" not in df.columns
            or "Volume" not in df.columns
        ):
            continue

        symbol = infer_symbol_from_filename(path)
        local = df[["Datetime", "Close", "Volume"]].copy()
        local["symbol"] = symbol
        local["Datetime"] = pd.to_datetime(local["Datetime"], utc=True, errors="coerce")
        local = local.dropna(subset=["Datetime", "Close", "Volume"])
        rows.append(local)

    if not rows:
        raise ValueError(
            "No usable CSVs found. Check --input-glob or provide --panel-csv."
        )

    out = pd.concat(rows, ignore_index=True)
    out = out[["Datetime", "symbol", "Close", "Volume"]]
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out = out.sort_values(["Datetime", "symbol"]).drop_duplicates(
        ["Datetime", "symbol"], keep="last"
    )
    return out


def load_sector_map(path: Path | None, panel: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if "sector" in panel.columns:
        from_panel = panel[["symbol", "sector"]].dropna().drop_duplicates("symbol")
        mapping.update(
            dict(
                zip(
                    from_panel["symbol"].astype(str).str.upper(),
                    from_panel["sector"].astype(str),
                )
            )
        )

    if path is None:
        return mapping

    df = pd.read_csv(path)
    df = _standardize_columns(df)
    required = {"symbol", "sector"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Sector map missing columns: {missing}")
    ext = df[["symbol", "sector"]].dropna().drop_duplicates("symbol")
    mapping.update(
        dict(zip(ext["symbol"].astype(str).str.upper(), ext["sector"].astype(str)))
    )
    return mapping


def neutralize_by_sector(
    scores: pd.DataFrame, sector_map: Dict[str, str]
) -> pd.DataFrame:
    if not sector_map:
        return scores

    out = scores.copy()
    grouped: Dict[str, List[str]] = {}
    for col in out.columns:
        sector = sector_map.get(col)
        if sector:
            grouped.setdefault(sector, []).append(col)

    for cols in grouped.values():
        if len(cols) < 2:
            continue
        out.loc[:, cols] = out[cols].sub(out[cols].mean(axis=1), axis=0)

    return out


def build_entry_weights(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    lookback: int,
    decile: float,
    top_n: int,
    liquidity_lookback: int,
    per_name_cap: float,
    sector_map: Dict[str, str],
    min_universe_size: int,
) -> pd.DataFrame:
    returns_k = close.pct_change(lookback)
    scores = neutralize_by_sector(returns_k, sector_map)

    dollar_volume = close * volume
    adv = dollar_volume.rolling(
        liquidity_lookback, min_periods=max(5, liquidity_lookback // 3)
    ).mean()
    liquid_rank = adv.rank(axis=1, ascending=False, method="first")
    liquid_mask = liquid_rank <= top_n

    rank_pct = scores.rank(axis=1, pct=True, method="average")
    long_mask = (rank_pct <= decile) & liquid_mask & scores.notna()
    short_mask = (rank_pct >= (1.0 - decile)) & liquid_mask & scores.notna()

    universe_size = liquid_mask.sum(axis=1)
    eligible = universe_size >= min_universe_size
    long_mask = long_mask[eligible]
    short_mask = short_mask[eligible]

    long_count = long_mask.sum(axis=1).replace(0, np.nan)
    short_count = short_mask.sum(axis=1).replace(0, np.nan)

    long_w = long_mask.div(long_count, axis=0) * 0.5
    short_w = short_mask.div(short_count, axis=0) * -0.5
    entry = (long_w + short_w).reindex(close.index).fillna(0.0)

    if per_name_cap > 0:
        entry = entry.clip(lower=-per_name_cap, upper=per_name_cap)

    pos = entry.clip(lower=0.0)
    neg = -entry.clip(upper=0.0)
    pos_sum = pos.sum(axis=1).replace(0, np.nan)
    neg_sum = neg.sum(axis=1).replace(0, np.nan)
    entry = (
        pos.div(pos_sum, axis=0).fillna(0.0) * 0.5
        - neg.div(neg_sum, axis=0).fillna(0.0) * 0.5
    )
    return entry


def compute_metrics(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    turnover: pd.Series,
    weights: pd.DataFrame,
) -> Dict[str, float]:
    gross_returns = gross_returns.fillna(0.0)
    net_returns = net_returns.fillna(0.0)
    turnover = turnover.fillna(0.0)

    def _safe_sharpe(x: pd.Series) -> float:
        std = float(x.std())
        if std == 0.0:
            return 0.0
        return float(np.sqrt(ANNUAL_MINUTES) * x.mean() / std)

    def _max_drawdown(x: pd.Series) -> float:
        eq = (1.0 + x).cumprod()
        peak = eq.cummax()
        dd = eq / peak - 1.0
        return float(dd.min())

    gross_pnl = float(gross_returns.sum())
    total_turnover = float(turnover.sum())
    break_even_bps = 0.0
    if total_turnover > 0:
        break_even_bps = 10_000.0 * gross_pnl / total_turnover

    return {
        "gross_sharpe": _safe_sharpe(gross_returns),
        "net_sharpe": _safe_sharpe(net_returns),
        "gross_ann_return": float(gross_returns.mean() * ANNUAL_MINUTES),
        "net_ann_return": float(net_returns.mean() * ANNUAL_MINUTES),
        "gross_ann_vol": float(gross_returns.std() * np.sqrt(ANNUAL_MINUTES)),
        "net_ann_vol": float(net_returns.std() * np.sqrt(ANNUAL_MINUTES)),
        "gross_max_drawdown": _max_drawdown(gross_returns),
        "net_max_drawdown": _max_drawdown(net_returns),
        "hit_rate": float((net_returns > 0).mean()),
        "avg_turnover": float(turnover.mean()),
        "break_even_cost_bps": float(break_even_bps),
        "avg_gross_exposure": float(weights.abs().sum(axis=1).mean()),
        "avg_net_exposure": float(weights.sum(axis=1).mean()),
    }


def run_vectorized_backtest(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    spy_close: pd.Series,
    lookback: int,
    hold_minutes: int,
    decile: float,
    top_n: int,
    liquidity_lookback: int,
    per_name_cap: float,
    beta_window: int,
    cost_bps: float,
    sector_map: Dict[str, str],
    min_universe_size: int,
    target_annual_vol: float,
    max_leverage: float,
) -> BacktestResult:
    entry = build_entry_weights(
        close=close,
        volume=volume,
        lookback=lookback,
        decile=decile,
        top_n=top_n,
        liquidity_lookback=liquidity_lookback,
        per_name_cap=per_name_cap,
        sector_map=sector_map,
        min_universe_size=min_universe_size,
    )

    close_returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    spy_returns = spy_close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    active_weights = (
        entry.shift(1).rolling(hold_minutes, min_periods=1).mean().fillna(0.0)
    )

    beta_cov = close_returns.rolling(
        beta_window, min_periods=max(10, beta_window // 3)
    ).cov(spy_returns)
    spy_var = spy_returns.rolling(
        beta_window, min_periods=max(10, beta_window // 3)
    ).var()
    beta = beta_cov.div(spy_var, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if target_annual_vol > 0:
        base_weights = active_weights.shift(1).fillna(0.0)
        base_asset_ret = (base_weights * close_returns).sum(axis=1)
        base_port_beta = (base_weights * beta).sum(axis=1)
        base_hedge_ret = -base_port_beta * spy_returns
        base_total = base_asset_ret + base_hedge_ret

        target_minute_vol = target_annual_vol / np.sqrt(ANNUAL_MINUTES)
        realized_minute_vol = base_total.rolling(60, min_periods=20).std()
        leverage = (target_minute_vol / realized_minute_vol).replace(
            [np.inf, -np.inf], np.nan
        )
        leverage = leverage.clip(lower=0.0, upper=max_leverage).fillna(1.0)
        scaled_weights = active_weights.mul(leverage, axis=0)
    else:
        scaled_weights = active_weights

    bar_weights = scaled_weights.shift(1).fillna(0.0)
    asset_ret = (bar_weights * close_returns).sum(axis=1)
    portfolio_beta = (bar_weights * beta).sum(axis=1)
    hedge_ret = -portfolio_beta * spy_returns
    gross_returns = asset_ret + hedge_ret

    turnover = (scaled_weights - scaled_weights.shift(1)).abs().sum(axis=1).fillna(0.0)
    net_returns = gross_returns - turnover * (cost_bps / 10_000.0)

    equity_gross = (1.0 + gross_returns).cumprod()
    equity_net = (1.0 + net_returns).cumprod()
    metrics = compute_metrics(gross_returns, net_returns, turnover, scaled_weights)

    return BacktestResult(
        lookback=lookback,
        hold_minutes=hold_minutes,
        cost_bps=cost_bps,
        equity_net=equity_net,
        equity_gross=equity_gross,
        net_returns=net_returns,
        gross_returns=gross_returns,
        turnover=turnover,
        metrics=metrics,
    )


def build_wide_frames(
    panel: pd.DataFrame, spy_symbol: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    wide_close = (
        panel.pivot_table(
            index="Datetime", columns="symbol", values="Close", aggfunc="last"
        )
        .sort_index()
        .astype(float)
    )
    wide_volume = (
        panel.pivot_table(
            index="Datetime", columns="symbol", values="Volume", aggfunc="sum"
        )
        .sort_index()
        .astype(float)
    )

    spy_symbol = spy_symbol.upper()
    if spy_symbol not in wide_close.columns:
        raise ValueError(f"SPY hedge symbol '{spy_symbol}' is missing from input data.")

    spy_close = wide_close[spy_symbol].copy()
    universe_cols = [c for c in wide_close.columns if c != spy_symbol]
    if not universe_cols:
        raise ValueError("No tradable symbols remain after removing SPY hedge symbol.")

    close = wide_close[universe_cols]
    volume = wide_volume[universe_cols].fillna(0.0)

    valid_rows = close.notna().sum(axis=1) > 0
    close = close.loc[valid_rows]
    volume = volume.loc[valid_rows]
    spy_close = spy_close.reindex(close.index).ffill().bfill()
    return close, volume, spy_close


def plot_equity(result: BacktestResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(result.equity_gross.index, result.equity_gross.values, label="Gross")
    plt.plot(
        result.equity_net.index,
        result.equity_net.values,
        label=f"Net ({result.cost_bps:.2f} bps)",
    )
    plt.title(
        f"Intraday Reversal Equity | lookback={result.lookback} hold={result.hold_minutes}m"
    )
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental 1-minute cross-sectional reversal backtest (vectorized pandas)."
    )
    parser.add_argument(
        "--panel-csv",
        type=str,
        default="",
        help="Single long-form CSV with Datetime,symbol,Close,Volume.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="data/*_1Min_stock_alpaca_clean.csv",
        help="Glob for one-file-per-symbol clean CSVs if --panel-csv is not provided.",
    )
    parser.add_argument(
        "--spy-symbol", type=str, default="SPY", help="Benchmark symbol for beta hedge."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Top liquid symbols to trade each minute.",
    )
    parser.add_argument(
        "--liquidity-lookback",
        type=int,
        default=390,
        help="Lookback minutes for rolling dollar-volume liquidity.",
    )
    parser.add_argument(
        "--lookback-grid",
        type=str,
        default="1,2,3,5",
        help="Signal lookback grid in minutes.",
    )
    parser.add_argument(
        "--hold-grid",
        type=str,
        default="1,2,3,5,10",
        help="Holding window grid in minutes.",
    )
    parser.add_argument(
        "--decile",
        type=float,
        default=0.10,
        help="Fraction for long/short tails (default deciles = 0.10).",
    )
    parser.add_argument(
        "--weight-cap", type=float, default=0.01, help="Per-name absolute weight cap."
    )
    parser.add_argument(
        "--min-universe",
        type=int,
        default=40,
        help="Minimum eligible liquid symbols to activate portfolio.",
    )
    parser.add_argument(
        "--beta-window",
        type=int,
        default=60,
        help="Rolling window in minutes for beta estimation.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        help="Linear transaction cost in bps per unit turnover (default: 0 for paper mode).",
    )
    parser.add_argument(
        "--cost-grid",
        type=str,
        default="0",
        help="Cost sensitivity grid (bps) for best parameter pair.",
    )
    parser.add_argument(
        "--target-annual-vol",
        type=float,
        default=0.0,
        help="Optional annualized vol target; 0 disables targeting.",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=3.0,
        help="Max leverage multiplier for volatility targeting.",
    )
    parser.add_argument(
        "--sector-map",
        type=str,
        default="",
        help="Optional CSV with symbol,sector for neutralization.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="notebooks/results",
        help="Directory to save CSV outputs and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lookback_grid = parse_int_grid(args.lookback_grid)
    hold_grid = parse_int_grid(args.hold_grid)
    cost_grid = parse_float_grid(args.cost_grid)

    if args.panel_csv:
        panel = load_panel_from_single_csv(Path(args.panel_csv))
    else:
        files = sorted(Path().glob(args.input_glob))
        panel = load_panel_from_symbol_csvs(files)

    close, volume, spy_close = build_wide_frames(panel, spy_symbol=args.spy_symbol)
    sector_map = (
        load_sector_map(Path(args.sector_map), panel)
        if args.sector_map
        else load_sector_map(None, panel)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows = []
    best_result: BacktestResult | None = None
    best_key = (-np.inf, -np.inf)

    for lookback in lookback_grid:
        for hold_minutes in hold_grid:
            result = run_vectorized_backtest(
                close=close,
                volume=volume,
                spy_close=spy_close,
                lookback=lookback,
                hold_minutes=hold_minutes,
                decile=args.decile,
                top_n=args.top_n,
                liquidity_lookback=args.liquidity_lookback,
                per_name_cap=args.weight_cap,
                beta_window=args.beta_window,
                cost_bps=args.cost_bps,
                sector_map=sector_map,
                min_universe_size=args.min_universe,
                target_annual_vol=args.target_annual_vol,
                max_leverage=args.max_leverage,
            )
            row = {
                "lookback": lookback,
                "hold_minutes": hold_minutes,
                "cost_bps": args.cost_bps,
                **result.metrics,
            }
            sweep_rows.append(row)

            score = (result.metrics["net_sharpe"], result.metrics["net_ann_return"])
            if score > best_key:
                best_key = score
                best_result = result

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        ["net_sharpe", "net_ann_return"], ascending=False
    )
    sweep_path = output_dir / "intraday_reversal_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)

    if best_result is None:
        raise RuntimeError("No backtest result produced.")

    cost_rows = []
    for cost_bps in cost_grid:
        cost_result = run_vectorized_backtest(
            close=close,
            volume=volume,
            spy_close=spy_close,
            lookback=best_result.lookback,
            hold_minutes=best_result.hold_minutes,
            decile=args.decile,
            top_n=args.top_n,
            liquidity_lookback=args.liquidity_lookback,
            per_name_cap=args.weight_cap,
            beta_window=args.beta_window,
            cost_bps=cost_bps,
            sector_map=sector_map,
            min_universe_size=args.min_universe,
            target_annual_vol=args.target_annual_vol,
            max_leverage=args.max_leverage,
        )
        cost_rows.append({"cost_bps": cost_bps, **cost_result.metrics})
        if abs(cost_bps - args.cost_bps) < 1e-12:
            best_result = cost_result

    cost_df = pd.DataFrame(cost_rows).sort_values("cost_bps")
    cost_path = output_dir / "intraday_reversal_cost_sensitivity.csv"
    cost_df.to_csv(cost_path, index=False)

    if best_result is None:
        raise RuntimeError("Failed to build final result.")

    equity_path = output_dir / "intraday_reversal_equity.png"
    plot_equity(best_result, equity_path)

    metrics_path = output_dir / "intraday_reversal_best_metrics.csv"
    pd.DataFrame(
        [
            {
                "lookback": best_result.lookback,
                "hold_minutes": best_result.hold_minutes,
                "cost_bps": best_result.cost_bps,
                **best_result.metrics,
            }
        ]
    ).to_csv(metrics_path, index=False)

    print("Intraday reversal experiment complete")
    print(f"Symbols: {close.shape[1]} | Bars: {close.shape[0]}")
    print(
        f"Best params: lookback={best_result.lookback}, hold={best_result.hold_minutes}m"
    )
    print(f"Net Sharpe: {best_result.metrics['net_sharpe']:.3f}")
    print(f"Break-even cost (bps): {best_result.metrics['break_even_cost_bps']:.2f}")
    print(f"Sweep results: {sweep_path}")
    print(f"Cost sensitivity: {cost_path}")
    print(f"Best metrics: {metrics_path}")
    print(f"Equity plot: {equity_path}")


if __name__ == "__main__":
    main()
