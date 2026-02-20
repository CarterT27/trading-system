from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd

from validate_single_asset_strategies import (
    ANNUAL_MINUTES_CRYPTO,
    load_universe_from_glob,
    position_crypto_ema,
    position_crypto_regime_trend,
    simulate_from_position,
    summarize_portfolio_metrics,
)


@dataclass
class FoldWindow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _parse_grid_int(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Integer grid cannot be empty.")
    return sorted(set(vals))


def _parse_grid_float(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Float grid cannot be empty.")
    return sorted(set(vals))


def _slice_universe(
    close_by_symbol: Dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_bars: int,
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for symbol, close in close_by_symbol.items():
        sliced = close[(close.index >= start) & (close.index < end)]
        sliced = sliced.dropna()
        if len(sliced) >= min_bars:
            out[symbol] = sliced
    return out


def _make_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[FoldWindow]:
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days, test_days, and step_days must be positive.")

    folds: List[FoldWindow] = []
    train_td = pd.Timedelta(days=train_days)
    test_td = pd.Timedelta(days=test_days)
    step_td = pd.Timedelta(days=step_days)

    cursor = start
    fold_id = 1
    while True:
        train_start = cursor
        train_end = train_start + train_td
        test_start = train_end
        test_end = test_start + test_td
        if test_end > end:
            break
        folds.append(
            FoldWindow(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        cursor = cursor + step_td
    return folds


def _simulate_portfolio(
    close_by_symbol: Dict[str, pd.Series],
    position_builder: Callable[[pd.Series], pd.Series],
    cost_bps: float,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    if not close_by_symbol:
        idx = pd.DatetimeIndex([], tz="UTC")
        empty = pd.Series(index=idx, dtype=float)
        empty_df = pd.DataFrame(index=idx)
        metrics = summarize_portfolio_metrics(
            net_returns=empty,
            turnover=empty,
            bars_per_year=ANNUAL_MINUTES_CRYPTO,
            positions_wide=empty_df,
            turnover_wide=empty_df,
        )
        return empty, empty, empty_df, empty_df, metrics

    net_returns: Dict[str, pd.Series] = {}
    turnover: Dict[str, pd.Series] = {}
    positions: Dict[str, pd.Series] = {}

    for symbol, close in close_by_symbol.items():
        pos = position_builder(close)
        result = simulate_from_position(
            close=close,
            position=pos,
            cost_bps=cost_bps,
            bars_per_year=ANNUAL_MINUTES_CRYPTO,
        )
        net_returns[symbol] = result.net_returns
        turnover[symbol] = result.turnover
        positions[symbol] = result.position

    returns_wide = pd.concat(net_returns, axis=1).sort_index().fillna(0.0)
    turnover_wide = pd.concat(turnover, axis=1).sort_index().fillna(0.0)
    positions_wide = pd.concat(positions, axis=1).sort_index().fillna(0.0)

    portfolio_returns = returns_wide.mean(axis=1)
    portfolio_turnover = turnover_wide.mean(axis=1)
    metrics = summarize_portfolio_metrics(
        net_returns=portfolio_returns,
        turnover=portfolio_turnover,
        bars_per_year=ANNUAL_MINUTES_CRYPTO,
        positions_wide=positions_wide,
        turnover_wide=turnover_wide,
    )
    return portfolio_returns, portfolio_turnover, positions_wide, turnover_wide, metrics


def _build_regime_builder(params: dict) -> Callable[[pd.Series], pd.Series]:
    return lambda close: position_crypto_regime_trend(
        close=close,
        short_window=int(params["short_window"]),
        long_window=int(params["long_window"]),
        slope_lookback=int(params["slope_lookback"]),
        volatility_window=int(params["volatility_window"]),
        volatility_quantile=float(params["volatility_quantile"]),
    )


def _pick_best_regime_params(
    train_close: Dict[str, pd.Series],
    candidates: Iterable[dict],
    cost_bps: float,
) -> Tuple[dict, Dict[str, float]]:
    best_params = None
    best_metrics: Dict[str, float] | None = None
    best_key = (-1e9, -1e9)

    for params in candidates:
        _, _, _, _, metrics = _simulate_portfolio(
            close_by_symbol=train_close,
            position_builder=_build_regime_builder(params),
            cost_bps=cost_bps,
        )
        score = (float(metrics["net_sharpe"]), float(metrics["net_ann_return"]))
        if score > best_key:
            best_key = score
            best_params = params
            best_metrics = metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError("No valid parameter candidate found for training fold.")
    return best_params, best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward validation for crypto strategies with regime-parameter tuning."
    )
    parser.add_argument(
        "--crypto-glob",
        default="data/*_1Min_crypto_alpaca_raw.csv",
        help="Glob pattern for crypto OHLCV data files.",
    )
    parser.add_argument("--train-days", type=int, default=14, help="Train window in days.")
    parser.add_argument("--test-days", type=int, default=7, help="Test window in days.")
    parser.add_argument(
        "--step-days",
        type=int,
        default=7,
        help="Step size in days between folds (7 = non-overlapping tests).",
    )
    parser.add_argument(
        "--min-train-bars",
        type=int,
        default=5000,
        help="Minimum per-symbol bars required in training slice.",
    )
    parser.add_argument(
        "--min-test-bars",
        type=int,
        default=1000,
        help="Minimum per-symbol bars required in test slice.",
    )
    parser.add_argument("--cost-bps", type=float, default=0.0, help="Linear cost in bps.")
    parser.add_argument(
        "--short-grid",
        default="20,30",
        help="Grid for regime short EMA window.",
    )
    parser.add_argument(
        "--long-grid",
        default="120,180",
        help="Grid for regime long EMA window.",
    )
    parser.add_argument(
        "--slope-grid",
        default="120,240,720",
        help="Grid for slow EMA slope lookback.",
    )
    parser.add_argument(
        "--vol-window-grid",
        default="720,1440",
        help="Grid for volatility-window bars.",
    )
    parser.add_argument(
        "--vol-quantile-grid",
        default="0.6,0.7,0.8",
        help="Grid for volatility quantile gate.",
    )
    parser.add_argument(
        "--output-dir",
        default="notebooks/results/walkforward_crypto",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    short_grid = _parse_grid_int(args.short_grid)
    long_grid = _parse_grid_int(args.long_grid)
    slope_grid = _parse_grid_int(args.slope_grid)
    vol_window_grid = _parse_grid_int(args.vol_window_grid)
    vol_quantile_grid = _parse_grid_float(args.vol_quantile_grid)

    candidates: List[dict] = []
    for short_w, long_w, slope_lb, vol_w, vol_q in itertools.product(
        short_grid, long_grid, slope_grid, vol_window_grid, vol_quantile_grid
    ):
        if short_w >= long_w:
            continue
        candidates.append(
            {
                "short_window": short_w,
                "long_window": long_w,
                "slope_lookback": slope_lb,
                "volatility_window": vol_w,
                "volatility_quantile": vol_q,
            }
        )
    if not candidates:
        raise ValueError("No valid regime parameter candidates after filtering.")

    close_by_symbol = load_universe_from_glob(args.crypto_glob)
    if not close_by_symbol:
        raise RuntimeError("No crypto data loaded.")

    common_start = max(series.index.min() for series in close_by_symbol.values())
    common_end = min(series.index.max() for series in close_by_symbol.values())
    folds = _make_folds(
        start=common_start,
        end=common_end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    if not folds:
        raise RuntimeError("No folds produced. Reduce train/test window sizes.")

    fold_rows: List[dict] = []
    oos_returns: Dict[str, List[pd.Series]] = {
        "crypto_fast_ema_long_only": [],
        "crypto_regime_trend_tuned": [],
    }

    for fold in folds:
        train_close = _slice_universe(
            close_by_symbol=close_by_symbol,
            start=fold.train_start,
            end=fold.train_end,
            min_bars=args.min_train_bars,
        )
        test_close = _slice_universe(
            close_by_symbol=close_by_symbol,
            start=fold.test_start,
            end=fold.test_end,
            min_bars=args.min_test_bars,
        )
        if not train_close or not test_close:
            continue

        baseline_builder = lambda close: position_crypto_ema(
            close=close, short_window=5, long_window=20
        )
        base_ret, _, _, _, base_metrics = _simulate_portfolio(
            close_by_symbol=test_close,
            position_builder=baseline_builder,
            cost_bps=args.cost_bps,
        )
        oos_returns["crypto_fast_ema_long_only"].append(base_ret.rename(f"fold_{fold.fold_id}"))
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "model": "crypto_fast_ema_long_only",
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "params": "short=5,long=20",
                **base_metrics,
            }
        )

        best_params, train_metrics = _pick_best_regime_params(
            train_close=train_close,
            candidates=candidates,
            cost_bps=args.cost_bps,
        )
        regime_builder = _build_regime_builder(best_params)
        tuned_ret, _, _, _, tuned_metrics = _simulate_portfolio(
            close_by_symbol=test_close,
            position_builder=regime_builder,
            cost_bps=args.cost_bps,
        )
        oos_returns["crypto_regime_trend_tuned"].append(
            tuned_ret.rename(f"fold_{fold.fold_id}")
        )
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "model": "crypto_regime_trend_tuned",
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "params": (
                    f"short={best_params['short_window']},long={best_params['long_window']},"
                    f"slope={best_params['slope_lookback']},vol_w={best_params['volatility_window']},"
                    f"vol_q={best_params['volatility_quantile']:.2f}"
                ),
                "train_net_sharpe": float(train_metrics["net_sharpe"]),
                "train_net_ann_return": float(train_metrics["net_ann_return"]),
                **tuned_metrics,
            }
        )

    if not fold_rows:
        raise RuntimeError("No valid folds evaluated after data-quality filters.")

    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "fold_id"])
    fold_path = out_dir / "crypto_walkforward_fold_results.csv"
    fold_df.to_csv(fold_path, index=False)

    summary_rows: List[dict] = []
    for model, series_list in oos_returns.items():
        model_rows = fold_df[fold_df["model"] == model]
        if not series_list:
            continue
        pooled = pd.concat(series_list).sort_index()
        pooled = pooled[~pooled.index.duplicated(keep="last")]
        pooled_turnover = pd.Series(index=pooled.index, dtype=float).fillna(0.0)
        # Pooled summary focuses on return-path behavior plus fold-level diagnostics.
        pooled_metrics = summarize_portfolio_metrics(
            net_returns=pooled,
            turnover=pooled_turnover,
            bars_per_year=ANNUAL_MINUTES_CRYPTO,
            positions_wide=pd.DataFrame(index=pooled.index),
            turnover_wide=pd.DataFrame(index=pooled.index),
        )
        summary_rows.append(
            {
                "model": model,
                "folds": int(len(model_rows)),
                "mean_fold_net_sharpe": float(model_rows["net_sharpe"].mean()),
                "median_fold_net_sharpe": float(model_rows["net_sharpe"].median()),
                "mean_fold_net_ann_return": float(model_rows["net_ann_return"].mean()),
                "mean_fold_trade_tick_pct": float(model_rows["trade_tick_pct"].mean()),
                "mean_fold_market_exposure_pct": float(
                    model_rows["market_exposure_pct"].mean()
                ),
                "positive_sharpe_fold_pct": float(
                    (model_rows["net_sharpe"] > 0.0).mean()
                ),
                "pooled_net_sharpe": float(pooled_metrics["net_sharpe"]),
                "pooled_net_ann_return": float(pooled_metrics["net_ann_return"]),
                "pooled_net_max_drawdown": float(pooled_metrics["net_max_drawdown"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_fold_net_sharpe", ascending=False)
    summary_path = out_dir / "crypto_walkforward_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("Crypto walk-forward validation complete")
    print(f"Folds evaluated: {fold_df['fold_id'].nunique()}")
    print(f"Fold results: {fold_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
