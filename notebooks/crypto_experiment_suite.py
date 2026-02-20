from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from validate_single_asset_strategies import (
    load_price_series,
    max_drawdown,
    safe_sharpe,
    simulate_from_position,
)

BARS_PER_YEAR_2M = 365 * 24 * 30


@dataclass
class FoldWindow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class RegimeParams:
    short_window: int
    long_window: int
    slope_lookback: int
    volatility_window: int
    volatility_quantile: float

    def to_string(self) -> str:
        return (
            f"short={self.short_window},long={self.long_window},"
            f"slope={self.slope_lookback},vol_w={self.volatility_window},"
            f"vol_q={self.volatility_quantile:.2f}"
        )


@dataclass
class MetaModel:
    feature_cols: Sequence[str]
    estimator: Optional[RandomForestClassifier] = None
    const_prob: Optional[float] = None


def parse_int_grid(text: str) -> List[int]:
    out = sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))
    if not out:
        raise ValueError("Integer grid cannot be empty.")
    return out


def parse_float_grid(text: str) -> List[float]:
    out = sorted(set(float(x.strip()) for x in text.split(",") if x.strip()))
    if not out:
        raise ValueError("Float grid cannot be empty.")
    return out


def load_universe_from_glob(glob_pattern: str) -> Dict[str, pd.Series]:
    files = sorted(Path().glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    out: Dict[str, pd.Series] = {}
    for path in files:
        symbol, close = load_price_series(path)
        out[symbol] = close
    return out


def make_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[FoldWindow]:
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days/test_days/step_days must be positive.")

    train_td = pd.Timedelta(days=train_days)
    test_td = pd.Timedelta(days=test_days)
    step_td = pd.Timedelta(days=step_days)

    folds: List[FoldWindow] = []
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


def slice_universe(
    close_by_symbol: Dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_bars: int,
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for symbol, close in close_by_symbol.items():
        sliced = close[(close.index >= start) & (close.index < end)].dropna()
        if len(sliced) >= min_bars:
            out[symbol] = sliced
    return out


def summarize_returns(
    returns: pd.Series,
    turnover: pd.Series,
    exposure: pd.Series,
    bars_per_year: int,
) -> Dict[str, float]:
    r = returns.fillna(0.0)
    t = turnover.fillna(0.0)
    e = exposure.fillna(0.0)

    ann_return = float(r.mean() * bars_per_year)
    ann_vol = float(r.std() * np.sqrt(bars_per_year))
    net_sharpe = safe_sharpe(r, bars_per_year)
    net_max_dd = max_drawdown(r)
    total_return = float((1.0 + r).prod() - 1.0) if len(r) else 0.0
    calmar = 0.0
    if net_max_dd < 0:
        calmar = float(ann_return / abs(net_max_dd))

    return {
        "total_return": total_return,
        "net_sharpe": float(net_sharpe),
        "net_ann_return": ann_return,
        "net_ann_vol": ann_vol,
        "net_max_drawdown": float(net_max_dd),
        "hit_rate": float((r > 0.0).mean()),
        "avg_turnover": float(t.mean()),
        "total_turnover": float(t.sum()),
        "avg_exposure": float(e.mean()),
        "max_exposure": float(e.max()) if len(e) else 0.0,
        "calmar": calmar,
        "bars": float(len(r)),
    }


def score_key(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        float(metrics["total_return"]),
        float(metrics["net_sharpe"]),
        float(metrics["net_ann_return"]),
    )


def split_inner_train_validation(
    close_by_symbol: Dict[str, pd.Series],
    features_by_symbol: Dict[str, pd.DataFrame],
    val_frac: float = 0.30,
    min_train_bars: int = 1200,
    min_val_bars: int = 350,
) -> Tuple[
    Dict[str, pd.Series],
    Dict[str, pd.Series],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
]:
    core_close: Dict[str, pd.Series] = {}
    val_close: Dict[str, pd.Series] = {}
    core_feat: Dict[str, pd.DataFrame] = {}
    val_feat: Dict[str, pd.DataFrame] = {}

    for symbol, close in close_by_symbol.items():
        feat = features_by_symbol.get(symbol)
        if feat is None or len(close) < (min_train_bars + min_val_bars):
            continue

        cut = int(len(close) * (1.0 - val_frac))
        cut = max(min_train_bars, min(cut, len(close) - min_val_bars))
        if cut <= 0 or cut >= len(close):
            continue

        c_core = close.iloc[:cut]
        c_val = close.iloc[cut:]
        if len(c_core) < min_train_bars or len(c_val) < min_val_bars:
            continue

        f_core = feat.reindex(c_core.index).copy()
        f_val = feat.reindex(c_val.index).copy()
        core_close[symbol] = c_core
        val_close[symbol] = c_val
        core_feat[symbol] = f_core
        val_feat[symbol] = f_val

    return core_close, val_close, core_feat, val_feat


def build_features(close: pd.Series, params: RegimeParams) -> pd.DataFrame:
    df = pd.DataFrame({"Close": close.astype(float)})
    df["ret1"] = df["Close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ret5"] = df["Close"].pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ret30"] = df["Close"].pct_change(30).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["fwd_ret"] = df["Close"].pct_change().shift(-1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["ema_fast"] = df["Close"].ewm(span=params.short_window, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=params.long_window, adjust=False).mean()
    df["slow_slope"] = df["ema_slow"].pct_change(params.slope_lookback).fillna(0.0)

    min_vol_samples = max(20, params.volatility_window // 3)
    min_q_samples = max(50, params.volatility_window // 2)
    df["realized_vol"] = df["ret1"].rolling(
        params.volatility_window, min_periods=min_vol_samples
    ).std()

    df["vol_threshold"] = (
        df["realized_vol"]
        .rolling(params.volatility_window, min_periods=min_q_samples)
        .quantile(params.volatility_quantile)
        .shift(1)
    )
    df["vol_med_threshold"] = (
        df["realized_vol"]
        .rolling(params.volatility_window, min_periods=min_q_samples)
        .quantile(0.5)
        .shift(1)
    )
    df["vol_ok"] = (df["realized_vol"] <= df["vol_threshold"]).fillna(False)
    df["vol_low"] = (df["realized_vol"] <= df["vol_med_threshold"]).fillna(False)

    df["spread"] = ((df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]).replace(
        [np.inf, -np.inf], np.nan
    )
    spread_mean = df["spread"].rolling(720, min_periods=100).mean()
    spread_std = df["spread"].rolling(720, min_periods=100).std()
    df["spread_z"] = ((df["spread"] - spread_mean) / (spread_std + 1e-12)).replace(
        [np.inf, -np.inf], np.nan
    )
    df["trend_tstat"] = (
        (df["ema_fast"] - df["ema_slow"]).abs() / (df["Close"] * df["realized_vol"] + 1e-12)
    ).replace([np.inf, -np.inf], np.nan)
    df["vol_ratio"] = (df["realized_vol"] / (df["vol_threshold"] + 1e-12)).replace(
        [np.inf, -np.inf], np.nan
    )

    df["trend_pre"] = (df["ema_fast"] > df["ema_slow"]).fillna(False)
    df["base_long"] = (df["trend_pre"] & (df["slow_slope"] > 0.0) & df["vol_ok"]).astype(float)
    return df


def build_feature_universe(
    close_by_symbol: Dict[str, pd.Series],
    params: RegimeParams,
) -> Dict[str, pd.DataFrame]:
    return {symbol: build_features(close, params) for symbol, close in close_by_symbol.items()}


def simulate_equal_weight_positions(
    close_by_symbol: Dict[str, pd.Series],
    positions_by_symbol: Dict[str, pd.Series],
    execution_model: str = "ideal",
    execution_seed: int = 42,
    full_fill_prob: float = 0.70,
    partial_fill_prob: float = 0.20,
    partial_fill_min: float = 0.10,
    partial_fill_max: float = 0.90,
) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, pd.Series]]:
    net_returns: Dict[str, pd.Series] = {}
    turnover: Dict[str, pd.Series] = {}
    exposure: Dict[str, pd.Series] = {}

    for i, (symbol, close) in enumerate(close_by_symbol.items()):
        pos = positions_by_symbol[symbol].reindex(close.index).fillna(0.0).clip(-1.0, 1.0)
        sim = simulate_from_position(
            close=close,
            position=pos,
            cost_bps=0.0,
            bars_per_year=BARS_PER_YEAR_2M,
            execution_model=execution_model,
            random_seed=int(execution_seed) + int(i * 9973),
            full_fill_prob=full_fill_prob,
            partial_fill_prob=partial_fill_prob,
            partial_fill_min=partial_fill_min,
            partial_fill_max=partial_fill_max,
        )
        net_returns[symbol] = sim.net_returns
        turnover[symbol] = sim.turnover
        exposure[symbol] = sim.position.shift(1).fillna(0.0).abs()

    ret_wide = pd.concat(net_returns, axis=1).sort_index().fillna(0.0)
    turn_wide = pd.concat(turnover, axis=1).sort_index().fillna(0.0)
    exp_wide = pd.concat(exposure, axis=1).sort_index().fillna(0.0)
    portfolio_ret = ret_wide.mean(axis=1)
    portfolio_turn = turn_wide.mean(axis=1)
    portfolio_exp = exp_wide.mean(axis=1)
    return portfolio_ret, portfolio_turn, portfolio_exp, net_returns


def simulate_weighted_portfolio(
    close_by_symbol: Dict[str, pd.Series],
    weights_wide: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    close_wide = pd.concat(close_by_symbol, axis=1).sort_index().ffill()
    rets = close_wide.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = weights_wide.reindex(rets.index).fillna(0.0)
    w_lag = w.shift(1).fillna(0.0)
    port_ret = (w_lag * rets).sum(axis=1)
    port_turn = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    port_exp = w_lag.abs().sum(axis=1)
    return port_ret, port_turn, port_exp


def _sample_timestamps_for_audit(
    close_by_symbol: Dict[str, pd.Series],
    n_samples: int,
    warmup_bars: int,
) -> List[pd.Timestamp]:
    common_idx: Optional[pd.DatetimeIndex] = None
    for s in close_by_symbol.values():
        idx = pd.DatetimeIndex(s.index)
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    if common_idx is None or len(common_idx) == 0:
        return []
    start = int(min(max(1, warmup_bars), max(1, len(common_idx) - 1)))
    end = len(common_idx) - 1
    if end <= start:
        return []
    picks = np.linspace(start, end - 1, num=max(1, n_samples), dtype=int)
    return [common_idx[i] for i in sorted(set(int(x) for x in picks))]


def _truncated_panel_until(
    close_by_symbol: Dict[str, pd.Series],
    ts: pd.Timestamp,
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for symbol, close in close_by_symbol.items():
        truncated = close[close.index <= ts]
        if len(truncated):
            out[symbol] = truncated
    return out


def assert_causal_positions_builder(
    close_by_symbol: Dict[str, pd.Series],
    builder: Callable[[Dict[str, pd.Series]], Dict[str, pd.Series]],
    n_samples: int = 8,
    warmup_bars: int = 200,
    atol: float = 1e-12,
) -> Tuple[bool, str]:
    full_pos = builder(close_by_symbol)
    timestamps = _sample_timestamps_for_audit(close_by_symbol, n_samples=n_samples, warmup_bars=warmup_bars)
    if not timestamps:
        return True, "insufficient_data_for_audit"

    for ts in timestamps:
        truncated_panel = _truncated_panel_until(close_by_symbol, ts)
        trunc_pos = builder(truncated_panel)
        for symbol in sorted(full_pos.keys()):
            if symbol not in trunc_pos:
                continue
            if ts not in full_pos[symbol].index or ts not in trunc_pos[symbol].index:
                continue
            a = float(full_pos[symbol].loc[ts])
            b = float(trunc_pos[symbol].loc[ts])
            if not np.isclose(a, b, atol=atol, equal_nan=True):
                return False, f"symbol={symbol},ts={ts},full={a},trunc={b}"
    return True, "ok"


def assert_causal_weights_builder(
    close_by_symbol: Dict[str, pd.Series],
    builder: Callable[[Dict[str, pd.Series]], pd.DataFrame],
    n_samples: int = 8,
    warmup_bars: int = 200,
    atol: float = 1e-12,
) -> Tuple[bool, str]:
    full_w = builder(close_by_symbol).sort_index()
    timestamps = _sample_timestamps_for_audit(close_by_symbol, n_samples=n_samples, warmup_bars=warmup_bars)
    if not timestamps:
        return True, "insufficient_data_for_audit"

    for ts in timestamps:
        truncated_panel = _truncated_panel_until(close_by_symbol, ts)
        trunc_w = builder(truncated_panel).sort_index()
        if ts not in full_w.index or ts not in trunc_w.index:
            continue
        full_row = full_w.loc[ts].fillna(0.0)
        trunc_row = trunc_w.loc[ts].reindex(full_row.index).fillna(0.0)
        if not np.allclose(full_row.values.astype(float), trunc_row.values.astype(float), atol=atol, equal_nan=True):
            return False, f"ts={ts}"
    return True, "ok"


def tune_regime_params(
    train_close: Dict[str, pd.Series],
    short_grid: Iterable[int],
    long_grid: Iterable[int],
    slope_grid: Iterable[int],
    vol_window_grid: Iterable[int],
    vol_quantile_grid: Iterable[float],
) -> Tuple[RegimeParams, Dict[str, float]]:
    best_params: Optional[RegimeParams] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_key = (-1e9, -1e9, -1e9)

    for short_w, long_w, slope_lb, vol_w, vol_q in itertools.product(
        short_grid, long_grid, slope_grid, vol_window_grid, vol_quantile_grid
    ):
        if short_w >= long_w:
            continue
        params = RegimeParams(
            short_window=short_w,
            long_window=long_w,
            slope_lookback=slope_lb,
            volatility_window=vol_w,
            volatility_quantile=vol_q,
        )
        features = build_feature_universe(train_close, params)
        pos = {symbol: df["base_long"] for symbol, df in features.items()}
        ret, turn, exp, _ = simulate_equal_weight_positions(train_close, pos)
        metrics = summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)
        key = score_key(metrics)
        if key > best_key:
            best_key = key
            best_params = params
            best_metrics = metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError("No valid regime parameter candidate found.")
    return best_params, best_metrics


def evaluate_base_regime(
    close_by_symbol: Dict[str, pd.Series],
    features_by_symbol: Dict[str, pd.DataFrame],
) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, float]]:
    pos = {symbol: df["base_long"] for symbol, df in features_by_symbol.items()}
    ret, turn, exp, _ = simulate_equal_weight_positions(close_by_symbol, pos)
    return ret, turn, exp, summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)


def fit_regime_decomposition_states(
    train_features: Dict[str, pd.DataFrame],
    min_samples: int,
) -> Dict[str, object]:
    rows: List[pd.DataFrame] = []
    for symbol, df in train_features.items():
        tmp = pd.DataFrame(index=df.index)
        tmp["trend_pre"] = df["trend_pre"].astype(bool)
        tmp["slope_pos"] = (df["slow_slope"] > 0.0).astype(int)
        tmp["vol_low"] = df["vol_low"].astype(int)
        tmp["spread_pos"] = (df["spread_z"] > 0.0).astype(int)
        tmp["state"] = tmp["slope_pos"] * 4 + tmp["vol_low"] * 2 + tmp["spread_pos"]
        tmp["fwd_ret"] = df["fwd_ret"]
        tmp["symbol"] = symbol
        rows.append(tmp)
    all_df = pd.concat(rows).sort_index()
    all_df = all_df[all_df["trend_pre"]]
    grouped = all_df.groupby("state")["fwd_ret"].agg(["mean", "count"]).reset_index()
    keep = grouped[(grouped["count"] >= min_samples) & (grouped["mean"] > 0.0)]
    allowed_states = set(int(x) for x in keep["state"].tolist())
    return {
        "allowed_states": allowed_states,
        "min_samples": int(min_samples),
    }


def apply_regime_decomposition(
    features_by_symbol: Dict[str, pd.DataFrame],
    allowed_states: set[int],
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for symbol, df in features_by_symbol.items():
        state = (
            ((df["slow_slope"] > 0.0).astype(int) * 4)
            + (df["vol_low"].astype(int) * 2)
            + (df["spread_z"] > 0.0).astype(int)
        )
        pos = (df["trend_pre"].astype(bool) & state.isin(allowed_states)).astype(float)
        out[symbol] = pos
    return out


def tune_regime_decomposition(
    train_close: Dict[str, pd.Series],
    train_features: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, object], Dict[str, float]]:
    best_choice: Optional[Dict[str, object]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_key = (-1e9, -1e9, -1e9)
    for min_samples in (50, 100, 200, 400):
        choice = fit_regime_decomposition_states(train_features, min_samples=min_samples)
        allowed_states = choice["allowed_states"]
        if not allowed_states:
            continue
        pos = apply_regime_decomposition(train_features, allowed_states=allowed_states)
        ret, turn, exp, _ = simulate_equal_weight_positions(train_close, pos)
        metrics = summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)
        key = score_key(metrics)
        if key > best_key:
            best_key = key
            best_choice = choice
            best_metrics = metrics
    if best_choice is None or best_metrics is None:
        fallback = {"allowed_states": {6, 7}, "min_samples": -1}
        pos = apply_regime_decomposition(train_features, allowed_states=fallback["allowed_states"])
        ret, turn, exp, _ = simulate_equal_weight_positions(train_close, pos)
        return fallback, summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)
    return best_choice, best_metrics


def apply_quality_filter(
    features_by_symbol: Dict[str, pd.DataFrame],
    spread_z_threshold: float,
    require_ret30_positive: bool,
    trend_tstat_threshold: float = 0.0,
    vol_ratio_max: float = 10.0,
    require_ret5_positive: bool = False,
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for symbol, df in features_by_symbol.items():
        q = df["base_long"] > 0.0
        q = q & (df["spread_z"] > spread_z_threshold).fillna(False)
        q = q & (df["trend_tstat"] > trend_tstat_threshold).fillna(False)
        q = q & (df["vol_ratio"] <= vol_ratio_max).fillna(False)
        if require_ret30_positive:
            q = q & (df["ret30"] > 0.0).fillna(False)
        if require_ret5_positive:
            q = q & (df["ret5"] > 0.0).fillna(False)
        out[symbol] = q.astype(float)
    return out


def tune_quality_filter(
    train_close: Dict[str, pd.Series],
    train_features: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, object], Dict[str, float]]:
    core_close, val_close, core_feat, val_feat = split_inner_train_validation(
        train_close, train_features
    )
    use_validation = len(core_close) >= 2 and len(val_close) >= 2

    best_params: Optional[Dict[str, object]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_key = (-1e9, -1e9, -1e9)
    for spread_thr, mom_flag, tstat_thr, vol_max, ret5_flag in itertools.product(
        (0.0, 0.3, 0.6),
        (False, True),
        (0.0, 0.3),
        (1.2, 1.6),
        (False, True),
    ):
        fit_feat = core_feat if use_validation else train_features
        eval_feat = val_feat if use_validation else train_features
        eval_close = val_close if use_validation else train_close
        pos = apply_quality_filter(
            fit_feat,
            spread_z_threshold=float(spread_thr),
            require_ret30_positive=bool(mom_flag),
            trend_tstat_threshold=float(tstat_thr),
            vol_ratio_max=float(vol_max),
            require_ret5_positive=bool(ret5_flag),
        )
        if use_validation:
            pos = apply_quality_filter(
                eval_feat,
                spread_z_threshold=float(spread_thr),
                require_ret30_positive=bool(mom_flag),
                trend_tstat_threshold=float(tstat_thr),
                vol_ratio_max=float(vol_max),
                require_ret5_positive=bool(ret5_flag),
            )
        ret, turn, exp, _ = simulate_equal_weight_positions(eval_close, pos)
        metrics = summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)
        key = score_key(metrics)
        if key > best_key:
            best_key = key
            best_params = {
                "spread_z_threshold": float(spread_thr),
                "require_ret30_positive": bool(mom_flag),
                "trend_tstat_threshold": float(tstat_thr),
                "vol_ratio_max": float(vol_max),
                "require_ret5_positive": bool(ret5_flag),
            }
            best_metrics = metrics
    if best_params is None or best_metrics is None:
        raise RuntimeError("Failed to tune quality filter.")

    train_pos = apply_quality_filter(
        train_features,
        spread_z_threshold=float(best_params["spread_z_threshold"]),
        require_ret30_positive=bool(best_params["require_ret30_positive"]),
        trend_tstat_threshold=float(best_params["trend_tstat_threshold"]),
        vol_ratio_max=float(best_params["vol_ratio_max"]),
        require_ret5_positive=bool(best_params["require_ret5_positive"]),
    )
    train_ret, train_turn, train_exp, _ = simulate_equal_weight_positions(train_close, train_pos)
    train_metrics = summarize_returns(train_ret, train_turn, train_exp, BARS_PER_YEAR_2M)
    return best_params, train_metrics


def fit_meta_models(
    train_features: Dict[str, pd.DataFrame],
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    max_features: float | str,
) -> Dict[str, MetaModel]:
    feature_cols = ["spread_z", "slow_slope", "vol_ratio", "ret1", "ret5", "ret30", "trend_tstat"]
    out: Dict[str, MetaModel] = {}
    for symbol, df in train_features.items():
        sample = df[df["trend_pre"].astype(bool)].copy()
        sample = sample.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["fwd_ret"])
        if len(sample) < 300:
            out[symbol] = MetaModel(
                feature_cols=feature_cols,
                estimator=None,
                const_prob=0.5,
            )
            continue

        y = (sample["fwd_ret"].values > 0.0).astype(float)
        positive_rate = float(y.mean())
        if positive_rate <= 0.01 or positive_rate >= 0.99:
            out[symbol] = MetaModel(
                feature_cols=feature_cols,
                estimator=None,
                const_prob=positive_rate,
            )
            continue

        X = sample[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)
        X = np.clip(X, -50.0, 50.0)
        estimator = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            random_state=42,
        )
        try:
            estimator.fit(X, y.astype(int))
        except Exception:
            out[symbol] = MetaModel(
                feature_cols=feature_cols,
                estimator=None,
                const_prob=positive_rate,
            )
            continue

        out[symbol] = MetaModel(
            feature_cols=feature_cols,
            estimator=estimator,
            const_prob=None,
        )
    return out


def predict_meta_prob(df: pd.DataFrame, model: MetaModel) -> pd.Series:
    if model.const_prob is not None:
        return pd.Series(model.const_prob, index=df.index, dtype=float)
    if model.estimator is None:
        return pd.Series(0.5, index=df.index, dtype=float)
    X = df[list(model.feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)
    X = np.clip(X, -50.0, 50.0)
    p = model.estimator.predict_proba(X)[:, 1]
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return pd.Series(p, index=df.index, dtype=float)


def apply_meta_model_gate(
    features_by_symbol: Dict[str, pd.DataFrame],
    models: Dict[str, MetaModel],
    threshold: float,
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for symbol, df in features_by_symbol.items():
        model = models[symbol]
        prob = predict_meta_prob(df, model)
        out[symbol] = ((df["base_long"] > 0.0) & (prob > threshold)).astype(float)
    return out


def tune_meta_model(
    train_close: Dict[str, pd.Series],
    train_features: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, object], Dict[str, float], Dict[str, MetaModel]]:
    core_close, val_close, core_feat, val_feat = split_inner_train_validation(
        train_close, train_features
    )
    use_validation = len(core_close) >= 2 and len(val_close) >= 2

    best_params: Optional[Dict[str, object]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_key = (-1e9, -1e9, -1e9)
    hyper_grid = itertools.product(
        (200, 350),
        (4, 6),
        (15, 30),
        ("sqrt",),
        (0.50, 0.55, 0.60, 0.65),
    )
    for n_est, depth, min_leaf, max_feat, threshold in hyper_grid:
        fit_feat = core_feat if use_validation else train_features
        eval_feat = val_feat if use_validation else train_features
        eval_close = val_close if use_validation else train_close

        models = fit_meta_models(
            fit_feat,
            n_estimators=int(n_est),
            max_depth=int(depth),
            min_samples_leaf=int(min_leaf),
            max_features=max_feat,
        )
        pos = apply_meta_model_gate(eval_feat, models=models, threshold=float(threshold))
        ret, turn, exp, _ = simulate_equal_weight_positions(eval_close, pos)
        metrics = summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)
        key = score_key(metrics)
        if key > best_key:
            best_key = key
            best_params = {
                "n_estimators": int(n_est),
                "max_depth": int(depth),
                "min_samples_leaf": int(min_leaf),
                "max_features": max_feat,
                "prob_threshold": float(threshold),
            }
            best_metrics = metrics
    if best_params is None:
        raise RuntimeError("Failed to tune meta model.")

    final_models = fit_meta_models(
        train_features,
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        max_features=best_params["max_features"],
    )
    train_pos = apply_meta_model_gate(
        train_features, models=final_models, threshold=float(best_params["prob_threshold"])
    )
    train_ret, train_turn, train_exp, _ = simulate_equal_weight_positions(train_close, train_pos)
    train_metrics = summarize_returns(train_ret, train_turn, train_exp, BARS_PER_YEAR_2M)
    return best_params, train_metrics, final_models


def apply_dynamic_sizing(
    features_by_symbol: Dict[str, pd.DataFrame],
    scale: float,
    offset: float,
    vol_exponent: float = 1.0,
    smooth_span: int = 1,
    min_size: float = 0.0,
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for symbol, df in features_by_symbol.items():
        strength = (df["spread_z"] - offset).clip(lower=0.0).fillna(0.0)
        vol_ratio = df["vol_ratio"].clip(lower=0.5, upper=5.0).fillna(1.0)
        vol_adj = 1.0 / (vol_ratio**vol_exponent)
        size = (scale * strength * vol_adj).clip(lower=0.0, upper=1.0)
        if smooth_span > 1:
            size = size.ewm(span=int(smooth_span), adjust=False).mean()
        size = size.where(size >= min_size, 0.0)
        pos = ((df["base_long"] > 0.0).astype(float) * size).clip(lower=0.0, upper=1.0)
        out[symbol] = pos
    return out


def tune_dynamic_sizing(
    train_close: Dict[str, pd.Series],
    train_features: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, object], Dict[str, float]]:
    core_close, val_close, core_feat, val_feat = split_inner_train_validation(
        train_close, train_features
    )
    use_validation = len(core_close) >= 2 and len(val_close) >= 2

    best_params: Optional[Dict[str, object]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_key = (-1e9, -1e9, -1e9)
    for scale, offset, vol_exp, smooth_span, min_size in itertools.product(
        (0.5, 1.0, 1.5),
        (0.0, 0.3, 0.6),
        (0.75, 1.25),
        (1, 8),
        (0.0, 0.05),
    ):
        eval_close = val_close if use_validation else train_close
        eval_feat = val_feat if use_validation else train_features
        pos = apply_dynamic_sizing(
            eval_feat,
            scale=float(scale),
            offset=float(offset),
            vol_exponent=float(vol_exp),
            smooth_span=int(smooth_span),
            min_size=float(min_size),
        )
        ret, turn, exp, _ = simulate_equal_weight_positions(eval_close, pos)
        metrics = summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)
        key = score_key(metrics)
        if key > best_key:
            best_key = key
            best_params = {
                "scale": float(scale),
                "offset": float(offset),
                "vol_exponent": float(vol_exp),
                "smooth_span": int(smooth_span),
                "min_size": float(min_size),
            }
            best_metrics = metrics
    if best_params is None or best_metrics is None:
        raise RuntimeError("Failed to tune dynamic sizing.")

    train_pos = apply_dynamic_sizing(
        train_features,
        scale=float(best_params["scale"]),
        offset=float(best_params["offset"]),
        vol_exponent=float(best_params["vol_exponent"]),
        smooth_span=int(best_params["smooth_span"]),
        min_size=float(best_params["min_size"]),
    )
    train_ret, train_turn, train_exp, _ = simulate_equal_weight_positions(train_close, train_pos)
    train_metrics = summarize_returns(train_ret, train_turn, train_exp, BARS_PER_YEAR_2M)
    return best_params, train_metrics


def position_with_exit_rules(
    base_long: pd.Series,
    close: pd.Series,
    mode: str,
    time_bars: Optional[int] = None,
    trail_stop: Optional[float] = None,
    profit_take: Optional[float] = None,
    min_hold_bars: int = 0,
) -> pd.Series:
    sig = base_long.fillna(0.0).astype(float).values
    px = close.astype(float).values
    n = len(base_long)
    out = np.zeros(n, dtype=float)

    in_pos = False
    hold = 0
    peak = 0.0
    entry_price = 0.0
    lock_until_reset = False
    for i in range(n):
        signal_on = sig[i] > 0.5
        price = float(px[i])

        if not signal_on:
            lock_until_reset = False

        if not in_pos:
            if signal_on and not lock_until_reset:
                in_pos = True
                hold = 0
                peak = price
                entry_price = price
        else:
            hold += 1
            if price > peak:
                peak = price
            exit_now = False
            forced_exit = False
            if not signal_on:
                exit_now = True
            else:
                if hold >= max(0, int(min_hold_bars)):
                    if mode in {"time", "combo"} and time_bars is not None and hold >= time_bars:
                        exit_now = True
                        forced_exit = True
                    if mode in {"trail", "combo"} and trail_stop is not None and peak > 0:
                        drawdown = price / peak - 1.0
                        if drawdown <= -trail_stop:
                            exit_now = True
                            forced_exit = True
                    if profit_take is not None and entry_price > 0:
                        pnl = price / entry_price - 1.0
                        if pnl >= float(profit_take):
                            exit_now = True
                            forced_exit = True
            if exit_now:
                in_pos = False
                hold = 0
                peak = 0.0
                entry_price = 0.0
                if forced_exit:
                    lock_until_reset = True
        out[i] = 1.0 if in_pos else 0.0
    return pd.Series(out, index=base_long.index, dtype=float)


def apply_exit_strategy(
    features_by_symbol: Dict[str, pd.DataFrame],
    mode: str,
    time_bars: Optional[int],
    trail_stop: Optional[float],
    profit_take: Optional[float] = None,
    min_hold_bars: int = 0,
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for symbol, df in features_by_symbol.items():
        pos = position_with_exit_rules(
            base_long=df["base_long"],
            close=df["Close"],
            mode=mode,
            time_bars=time_bars,
            trail_stop=trail_stop,
            profit_take=profit_take,
            min_hold_bars=min_hold_bars,
        )
        out[symbol] = pos
    return out


def tune_exit_strategy(
    train_close: Dict[str, pd.Series],
    train_features: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, object], Dict[str, float]]:
    core_close, val_close, core_feat, val_feat = split_inner_train_validation(
        train_close, train_features
    )
    use_validation = len(core_close) >= 2 and len(val_close) >= 2

    candidates: List[Dict[str, object]] = [
        {
            "mode": "baseline",
            "time_bars": None,
            "trail_stop": None,
            "profit_take": None,
            "min_hold_bars": 0,
        }
    ]
    candidates.extend(
        {
            "mode": "time",
            "time_bars": x,
            "trail_stop": None,
            "profit_take": p,
            "min_hold_bars": h,
        }
        for x, p, h in itertools.product((90, 180, 360), (None, 0.007), (0, 10))
    )
    candidates.extend(
        {
            "mode": "trail",
            "time_bars": None,
            "trail_stop": x,
            "profit_take": p,
            "min_hold_bars": h,
        }
        for x, p, h in itertools.product((0.005, 0.008, 0.012), (None, 0.007), (0, 10))
    )
    candidates.extend(
        {
            "mode": "combo",
            "time_bars": t,
            "trail_stop": s,
            "profit_take": p,
            "min_hold_bars": h,
        }
        for t, s, p, h in itertools.product(
            (180, 360), (0.005, 0.008), (None, 0.007), (0, 10)
        )
    )

    best_params: Optional[Dict[str, object]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_key = (-1e9, -1e9, -1e9)
    for params in candidates:
        eval_close = val_close if use_validation else train_close
        eval_feat = val_feat if use_validation else train_features
        if params["mode"] == "baseline":
            pos = {symbol: df["base_long"] for symbol, df in eval_feat.items()}
        else:
            pos = apply_exit_strategy(
                eval_feat,
                mode=str(params["mode"]),
                time_bars=params["time_bars"],
                trail_stop=params["trail_stop"],
                profit_take=params["profit_take"],
                min_hold_bars=int(params["min_hold_bars"]),
            )
        ret, turn, exp, _ = simulate_equal_weight_positions(eval_close, pos)
        metrics = summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)
        key = score_key(metrics)
        if key > best_key:
            best_key = key
            best_params = params
            best_metrics = metrics
    if best_params is None or best_metrics is None:
        raise RuntimeError("Failed to tune exit strategy.")

    if best_params["mode"] == "baseline":
        train_pos = {symbol: df["base_long"] for symbol, df in train_features.items()}
    else:
        train_pos = apply_exit_strategy(
            train_features,
            mode=str(best_params["mode"]),
            time_bars=best_params["time_bars"],
            trail_stop=best_params["trail_stop"],
            profit_take=best_params["profit_take"],
            min_hold_bars=int(best_params["min_hold_bars"]),
        )
    train_ret, train_turn, train_exp, _ = simulate_equal_weight_positions(train_close, train_pos)
    train_metrics = summarize_returns(train_ret, train_turn, train_exp, BARS_PER_YEAR_2M)
    return best_params, train_metrics


def build_overlay_weights(
    features_by_symbol: Dict[str, pd.DataFrame],
    mode: str,
) -> pd.DataFrame:
    signal_wide = pd.concat(
        {symbol: (df["base_long"] > 0.0).astype(float) for symbol, df in features_by_symbol.items()},
        axis=1,
    ).sort_index()
    score_wide = pd.concat(
        {symbol: df["spread_z"].clip(lower=0.0).fillna(0.0) for symbol, df in features_by_symbol.items()},
        axis=1,
    ).sort_index()
    vol_wide = pd.concat(
        {symbol: df["realized_vol"].replace(0.0, np.nan).ffill().bfill() for symbol, df in features_by_symbol.items()},
        axis=1,
    ).sort_index()
    symbols = list(signal_wide.columns)
    w = pd.DataFrame(0.0, index=signal_wide.index, columns=symbols)

    for ts in w.index:
        active = signal_wide.loc[ts] > 0.5
        if int(active.sum()) == 0:
            continue
        active_syms = [s for s in symbols if bool(active[s])]
        if mode == "active_equal":
            wt = 1.0 / len(active_syms)
            for s in active_syms:
                w.at[ts, s] = wt
        elif mode == "strength":
            scores = score_wide.loc[ts, active_syms].astype(float)
            score_sum = float(scores.sum())
            if score_sum <= 1e-12:
                wt = 1.0 / len(active_syms)
                for s in active_syms:
                    w.at[ts, s] = wt
            else:
                for s in active_syms:
                    w.at[ts, s] = float(scores[s] / score_sum)
        elif mode == "inv_vol":
            inv = 1.0 / vol_wide.loc[ts, active_syms].astype(float).clip(lower=1e-12)
            inv_sum = float(inv.sum())
            if inv_sum <= 1e-12:
                wt = 1.0 / len(active_syms)
                for s in active_syms:
                    w.at[ts, s] = wt
            else:
                for s in active_syms:
                    w.at[ts, s] = float(inv[s] / inv_sum)
        elif mode == "top1":
            scores = score_wide.loc[ts, active_syms].astype(float)
            top = str(scores.idxmax())
            w.at[ts, top] = 1.0
        else:
            raise ValueError(f"Unknown overlay mode: {mode}")
    return w


def tune_overlay_mode(
    train_close: Dict[str, pd.Series],
    train_features: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, object], Dict[str, float]]:
    best_params: Optional[Dict[str, object]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_key = (-1e9, -1e9, -1e9)

    for mode in ("active_equal", "strength", "inv_vol", "top1"):
        weights = build_overlay_weights(train_features, mode=mode)
        ret, turn, exp = simulate_weighted_portfolio(train_close, weights)
        metrics = summarize_returns(ret, turn, exp, BARS_PER_YEAR_2M)
        key = score_key(metrics)
        if key > best_key:
            best_key = key
            best_params = {"overlay_mode": mode}
            best_metrics = metrics
    if best_params is None or best_metrics is None:
        raise RuntimeError("Failed to tune overlay mode.")
    return best_params, best_metrics


def evaluate_model(
    model_name: str,
    regime_params: RegimeParams,
    train_close: Dict[str, pd.Series],
    test_close: Dict[str, pd.Series],
    train_features: Dict[str, pd.DataFrame],
    test_features: Dict[str, pd.DataFrame],
) -> Tuple[
    Dict[str, object],
    Dict[str, float],
    Dict[str, float],
    pd.Series,
    pd.Series,
    pd.Series,
    bool,
    str,
]:
    if model_name == "base_regime_tuned":
        train_ret, train_turn, train_exp, train_metrics = evaluate_base_regime(
            train_close, train_features
        )
        test_ret, test_turn, test_exp, test_metrics = evaluate_base_regime(test_close, test_features)
        _ = (train_ret, train_turn, train_exp)
        builder = lambda panel: {sym: df["base_long"] for sym, df in build_feature_universe(panel, regime_params).items()}
        ok, msg = assert_causal_positions_builder(test_close, builder=builder)
        return {}, train_metrics, test_metrics, test_ret, test_turn, test_exp, ok, msg

    if model_name == "regime_decomposition":
        best_params, train_metrics = tune_regime_decomposition(train_close, train_features)
        test_pos = apply_regime_decomposition(
            test_features, allowed_states=set(best_params["allowed_states"])
        )
        test_ret, test_turn, test_exp, _ = simulate_equal_weight_positions(test_close, test_pos)
        test_metrics = summarize_returns(test_ret, test_turn, test_exp, BARS_PER_YEAR_2M)
        allowed = set(best_params["allowed_states"])
        builder = lambda panel: apply_regime_decomposition(
            build_feature_universe(panel, regime_params), allowed_states=allowed
        )
        ok, msg = assert_causal_positions_builder(test_close, builder=builder)
        best_params = dict(best_params)
        best_params["allowed_states"] = ",".join(
            str(x) for x in sorted(set(best_params["allowed_states"]))
        )
        return best_params, train_metrics, test_metrics, test_ret, test_turn, test_exp, ok, msg

    if model_name == "quality_filter":
        best_params, train_metrics = tune_quality_filter(train_close, train_features)
        test_pos = apply_quality_filter(
            test_features,
            spread_z_threshold=float(best_params["spread_z_threshold"]),
            require_ret30_positive=bool(best_params["require_ret30_positive"]),
            trend_tstat_threshold=float(best_params["trend_tstat_threshold"]),
            vol_ratio_max=float(best_params["vol_ratio_max"]),
            require_ret5_positive=bool(best_params["require_ret5_positive"]),
        )
        test_ret, test_turn, test_exp, _ = simulate_equal_weight_positions(test_close, test_pos)
        test_metrics = summarize_returns(test_ret, test_turn, test_exp, BARS_PER_YEAR_2M)
        spread_thr = float(best_params["spread_z_threshold"])
        mom_flag = bool(best_params["require_ret30_positive"])
        builder = lambda panel: apply_quality_filter(
            build_feature_universe(panel, regime_params),
            spread_z_threshold=spread_thr,
            require_ret30_positive=mom_flag,
            trend_tstat_threshold=float(best_params["trend_tstat_threshold"]),
            vol_ratio_max=float(best_params["vol_ratio_max"]),
            require_ret5_positive=bool(best_params["require_ret5_positive"]),
        )
        ok, msg = assert_causal_positions_builder(test_close, builder=builder)
        return best_params, train_metrics, test_metrics, test_ret, test_turn, test_exp, ok, msg

    if model_name == "meta_model":
        best_params, train_metrics, models = tune_meta_model(train_close, train_features)
        test_pos = apply_meta_model_gate(
            test_features,
            models=models,
            threshold=float(best_params["prob_threshold"]),
        )
        test_ret, test_turn, test_exp, _ = simulate_equal_weight_positions(test_close, test_pos)
        test_metrics = summarize_returns(test_ret, test_turn, test_exp, BARS_PER_YEAR_2M)
        threshold = float(best_params["prob_threshold"])
        builder = lambda panel: apply_meta_model_gate(
            build_feature_universe(panel, regime_params),
            models=models,
            threshold=threshold,
        )
        ok, msg = assert_causal_positions_builder(test_close, builder=builder)
        return best_params, train_metrics, test_metrics, test_ret, test_turn, test_exp, ok, msg

    if model_name == "dynamic_sizing":
        best_params, train_metrics = tune_dynamic_sizing(train_close, train_features)
        test_pos = apply_dynamic_sizing(
            test_features,
            scale=float(best_params["scale"]),
            offset=float(best_params["offset"]),
            vol_exponent=float(best_params["vol_exponent"]),
            smooth_span=int(best_params["smooth_span"]),
            min_size=float(best_params["min_size"]),
        )
        test_ret, test_turn, test_exp, _ = simulate_equal_weight_positions(test_close, test_pos)
        test_metrics = summarize_returns(test_ret, test_turn, test_exp, BARS_PER_YEAR_2M)
        scale = float(best_params["scale"])
        offset = float(best_params["offset"])
        builder = lambda panel: apply_dynamic_sizing(
            build_feature_universe(panel, regime_params),
            scale=scale,
            offset=offset,
            vol_exponent=float(best_params["vol_exponent"]),
            smooth_span=int(best_params["smooth_span"]),
            min_size=float(best_params["min_size"]),
        )
        ok, msg = assert_causal_positions_builder(test_close, builder=builder)
        return best_params, train_metrics, test_metrics, test_ret, test_turn, test_exp, ok, msg

    if model_name == "exit_analysis":
        best_params, train_metrics = tune_exit_strategy(train_close, train_features)
        if best_params["mode"] == "baseline":
            test_pos = {symbol: df["base_long"] for symbol, df in test_features.items()}
        else:
            test_pos = apply_exit_strategy(
                test_features,
                mode=str(best_params["mode"]),
                time_bars=best_params["time_bars"],
                trail_stop=best_params["trail_stop"],
                profit_take=best_params["profit_take"],
                min_hold_bars=int(best_params["min_hold_bars"]),
            )
        test_ret, test_turn, test_exp, _ = simulate_equal_weight_positions(test_close, test_pos)
        test_metrics = summarize_returns(test_ret, test_turn, test_exp, BARS_PER_YEAR_2M)
        if best_params["mode"] == "baseline":
            builder = lambda panel: {
                sym: df["base_long"] for sym, df in build_feature_universe(panel, regime_params).items()
            }
        else:
            mode = str(best_params["mode"])
            time_bars = best_params["time_bars"]
            trail_stop = best_params["trail_stop"]
            builder = lambda panel: apply_exit_strategy(
                build_feature_universe(panel, regime_params),
                mode=mode,
                time_bars=time_bars,
                trail_stop=trail_stop,
                profit_take=best_params["profit_take"],
                min_hold_bars=int(best_params["min_hold_bars"]),
            )
        ok, msg = assert_causal_positions_builder(test_close, builder=builder)
        return best_params, train_metrics, test_metrics, test_ret, test_turn, test_exp, ok, msg

    if model_name == "cross_asset_overlay":
        best_params, train_metrics = tune_overlay_mode(train_close, train_features)
        weights_test = build_overlay_weights(test_features, mode=str(best_params["overlay_mode"]))
        test_ret, test_turn, test_exp = simulate_weighted_portfolio(test_close, weights_test)
        test_metrics = summarize_returns(test_ret, test_turn, test_exp, BARS_PER_YEAR_2M)
        overlay_mode = str(best_params["overlay_mode"])
        builder = lambda panel: build_overlay_weights(
            build_feature_universe(panel, regime_params), mode=overlay_mode
        )
        ok, msg = assert_causal_weights_builder(test_close, builder=builder)
        return best_params, train_metrics, test_metrics, test_ret, test_turn, test_exp, ok, msg

    raise ValueError(f"Unknown model name: {model_name}")


def to_param_text(params: Dict[str, object], regime_params: RegimeParams) -> str:
    if not params:
        return f"regime[{regime_params.to_string()}]"
    extras = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"regime[{regime_params.to_string()}] | {extras}"


def pooled_summary_from_fold_returns(
    fold_df: pd.DataFrame,
    oos_returns_by_model: Dict[str, List[pd.Series]],
    oos_turnover_by_model: Dict[str, List[pd.Series]],
    oos_exposure_by_model: Dict[str, List[pd.Series]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model, chunks in oos_returns_by_model.items():
        model_rows = fold_df[fold_df["model"] == model]
        if model_rows.empty:
            continue
        if chunks:
            pooled = pd.concat(chunks).sort_index()
            pooled = pooled[~pooled.index.duplicated(keep="last")]
        else:
            pooled = pd.Series(dtype=float)
        turn_chunks = oos_turnover_by_model.get(model, [])
        exp_chunks = oos_exposure_by_model.get(model, [])
        if turn_chunks:
            pooled_turn = pd.concat(turn_chunks).sort_index()
            pooled_turn = pooled_turn[~pooled_turn.index.duplicated(keep="last")]
        else:
            pooled_turn = pd.Series(dtype=float)
        if exp_chunks:
            pooled_exp = pd.concat(exp_chunks).sort_index()
            pooled_exp = pooled_exp[~pooled_exp.index.duplicated(keep="last")]
        else:
            pooled_exp = pd.Series(dtype=float)

        if len(pooled) == 0:
            pooled_metrics = {
                "total_return": 0.0,
                "net_sharpe": 0.0,
                "net_ann_return": 0.0,
                "net_ann_vol": 0.0,
                "net_max_drawdown": 0.0,
                "hit_rate": 0.0,
                "avg_turnover": 0.0,
                "total_turnover": 0.0,
                "avg_exposure": 0.0,
                "max_exposure": 0.0,
                "calmar": 0.0,
                "bars": 0.0,
            }
        else:
            if len(pooled_turn) == 0:
                pooled_turn = pd.Series(0.0, index=pooled.index)
            if len(pooled_exp) == 0:
                pooled_exp = pd.Series(0.0, index=pooled.index)
            pooled_metrics = summarize_returns(
                pooled,
                turnover=pooled_turn.reindex(pooled.index).fillna(0.0),
                exposure=pooled_exp.reindex(pooled.index).fillna(0.0),
                bars_per_year=BARS_PER_YEAR_2M,
            )

        rows.append(
            {
                "model": model,
                "folds": int(model_rows.shape[0]),
                "mean_fold_net_sharpe": float(model_rows["test_net_sharpe"].mean()),
                "median_fold_net_sharpe": float(model_rows["test_net_sharpe"].median()),
                "mean_fold_total_return": float(model_rows["test_total_return"].mean()),
                "mean_fold_net_ann_return": float(model_rows["test_net_ann_return"].mean()),
                "mean_fold_net_max_drawdown": float(model_rows["test_net_max_drawdown"].mean()),
                "mean_fold_avg_turnover": float(model_rows["test_avg_turnover"].mean()),
                "mean_fold_avg_exposure": float(model_rows["test_avg_exposure"].mean()),
                "positive_sharpe_fold_pct": float((model_rows["test_net_sharpe"] > 0).mean()),
                "pooled_total_return": float(pooled_metrics["total_return"]),
                "pooled_net_sharpe": float(pooled_metrics["net_sharpe"]),
                "pooled_net_ann_return": float(pooled_metrics["net_ann_return"]),
                "pooled_net_max_drawdown": float(pooled_metrics["net_max_drawdown"]),
                "pooled_hit_rate": float(pooled_metrics["hit_rate"]),
                "pooled_bars": float(pooled_metrics["bars"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["pooled_total_return", "pooled_net_sharpe"], ascending=False)


def recommendation_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=["model", "recommendation", "reason"])
    baseline = summary_df[summary_df["model"] == "base_regime_tuned"]
    baseline_total_return = float(baseline["pooled_total_return"].iloc[0]) if not baseline.empty else -np.inf
    baseline_dd = float(abs(baseline["pooled_net_max_drawdown"].iloc[0])) if not baseline.empty else np.inf

    rows: List[Dict[str, str]] = []
    for _, row in summary_df.iterrows():
        model = str(row["model"])
        pooled_total_return = float(row["pooled_total_return"])
        positive_folds = float(row["positive_sharpe_fold_pct"])
        dd = float(abs(row["pooled_net_max_drawdown"]))
        mean_exposure = float(row["mean_fold_avg_exposure"])

        if model == "base_regime_tuned":
            reco = "control"
            reason = "Reference control model."
        elif (
            pooled_total_return > (baseline_total_return + 0.02)
            and positive_folds >= 0.75
            and dd <= baseline_dd
            and mean_exposure >= 0.05
        ):
            reco = "add_now"
            reason = "Strong pooled-return lift with good stability and no drawdown degradation."
        elif (
            pooled_total_return > baseline_total_return
            and positive_folds >= 0.5
            and dd <= baseline_dd * 1.5
        ):
            reco = "paper_test"
            reason = "Moderate improvement; validate with shadow paper runs before promotion."
        else:
            reco = "do_not_add"
            reason = "No robust OOS edge relative to control."
        rows.append({"model": model, "recommendation": reco, "reason": reason})
    return pd.DataFrame(rows)


def render_report(
    args: argparse.Namespace,
    folds: List[FoldWindow],
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    reco_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("# Crypto Strategy Experiment Report")
    lines.append("")
    lines.append("## Objective")
    lines.append(
        "Run walk-forward experiments (`cost_bps=0`) for six strategy ideas and decide which are strong enough to add."
    )
    lines.append("")
    lines.append("## Data and Setup")
    lines.append(f"- Data glob: `{args.crypto_glob}`")
    lines.append(f"- Train/Test/Step days: {args.train_days}/{args.test_days}/{args.step_days}")
    lines.append(
        f"- Bars/year used for annualization (2-minute crypto): {BARS_PER_YEAR_2M:,}"
    )
    lines.append(
        f"- Fold count attempted: {len(folds)} | Fold count evaluated: {fold_df['fold_id'].nunique() if not fold_df.empty else 0}"
    )
    lines.append("- Cost model: 0 bps linear transaction cost (paper competition assumption).")
    if "causal_audit_passed" in fold_df.columns and len(fold_df):
        audit_ok = bool(fold_df["causal_audit_passed"].all())
        passed = int(fold_df["causal_audit_passed"].sum())
        total = int(len(fold_df))
        lines.append(
            f"- Lookahead audit (truncation invariance): {'PASS' if audit_ok else 'FAIL'} ({passed}/{total})."
        )
    lines.append("")
    lines.append("## Experiment Families")
    lines.append("- `base_regime_tuned`: baseline EMA regime model with fold-wise train tuning.")
    lines.append("- `regime_decomposition`: train-time regime-state expectancy map, then test-time state gating.")
    lines.append("- `quality_filter`: spread-z and medium-horizon momentum quality gates.")
    lines.append("- `meta_model`: sklearn classifier probability gate on top of baseline entries.")
    lines.append("- `dynamic_sizing`: fractional sizing from trend strength and volatility normalization.")
    lines.append("- `exit_analysis`: baseline vs time-stop/trailing-stop/combined exits.")
    lines.append("- `cross_asset_overlay`: active-equal/strength/inv-vol/top1 portfolio overlays.")
    lines.append(
        "- Optimization protocol for meta/sizing/quality/exit: per-fold inner train/validation split (time-ordered) for parameter selection."
    )
    lines.append("")
    lines.append("## OOS Summary")
    lines.append("")
    lines.append(
        "| Model | Folds | Mean Fold Return | Mean Fold Sharpe | Median Fold Sharpe | Positive Fold % | Pooled Return | Pooled Sharpe | Pooled Ann Return | Pooled Max DD | Mean Exposure |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['model']} | {int(row['folds'])} | {row['mean_fold_total_return']:.3f} | "
            f"{row['mean_fold_net_sharpe']:.3f} | {row['median_fold_net_sharpe']:.3f} | {row['positive_sharpe_fold_pct']:.2%} | "
            f"{row['pooled_total_return']:.3f} | {row['pooled_net_sharpe']:.3f} | {row['pooled_net_ann_return']:.3f} | "
            f"{row['pooled_net_max_drawdown']:.3f} | {row['mean_fold_avg_exposure']:.3f} |"
        )
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append("| Model | Decision | Reason |")
    lines.append("| --- | --- | --- |")
    for _, row in reco_df.iterrows():
        lines.append(f"| {row['model']} | {row['recommendation']} | {row['reason']} |")
    lines.append("")
    lines.append("## Fold-Level Notes")
    lines.append("")
    lines.append(
        "| Fold | Model | Train Return | Train Sharpe | Test Return | Test Sharpe | Test Ann Return | Test Max DD | Params |"
    )
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for _, row in fold_df.sort_values(["fold_id", "model"]).iterrows():
        lines.append(
            f"| {int(row['fold_id'])} | {row['model']} | {row['train_total_return']:.3f} | {row['train_net_sharpe']:.3f} | "
            f"{row['test_total_return']:.3f} | {row['test_net_sharpe']:.3f} | {row['test_net_ann_return']:.3f} | "
            f"{row['test_net_max_drawdown']:.3f} | {row['params']} |"
        )
    lines.append("")
    lines.append("## Caveats")
    lines.append("- Sample window is limited to available local files; conclusions are provisional.")
    lines.append(
        "- `cost_bps=0` matches competition optimization, but high-turnover variants still carry operational risk."
    )
    lines.append(
        "- Meta model uses sklearn `RandomForestClassifier` with per-symbol training and probability gating."
    )
    lines.append(
        "- Lookahead audit checks whether decisions at timestamp t are unchanged when recomputed with data truncated at t."
    )
    lines.append(
        "- Overlay models can increase gross exposure vs baseline equal-average construction; compare with that in mind."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward experiment suite for crypto EMA strategy enhancements."
    )
    parser.add_argument(
        "--crypto-glob",
        default="data/legacy/raw_data_crypto/*_2m_data.csv",
        help="Glob pattern for crypto CSV files.",
    )
    parser.add_argument("--train-days", type=int, default=10, help="Train window length in days.")
    parser.add_argument("--test-days", type=int, default=5, help="Test window length in days.")
    parser.add_argument("--step-days", type=int, default=5, help="Fold step in days.")
    parser.add_argument(
        "--min-train-bars",
        type=int,
        default=5000,
        help="Minimum bars required in each symbol's train slice.",
    )
    parser.add_argument(
        "--min-test-bars",
        type=int,
        default=2000,
        help="Minimum bars required in each symbol's test slice.",
    )
    parser.add_argument("--short-grid", default="20,30", help="Short EMA grid.")
    parser.add_argument("--long-grid", default="120,180", help="Long EMA grid.")
    parser.add_argument("--slope-grid", default="120,240,720", help="Slope lookback grid.")
    parser.add_argument("--vol-window-grid", default="720,1440", help="Vol window grid.")
    parser.add_argument("--vol-quantile-grid", default="0.6,0.7,0.8", help="Vol quantile grid.")
    parser.add_argument(
        "--output-dir",
        default="notebooks/results/crypto_experiment_suite",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    short_grid = parse_int_grid(args.short_grid)
    long_grid = parse_int_grid(args.long_grid)
    slope_grid = parse_int_grid(args.slope_grid)
    vol_window_grid = parse_int_grid(args.vol_window_grid)
    vol_quantile_grid = parse_float_grid(args.vol_quantile_grid)

    close_by_symbol = load_universe_from_glob(args.crypto_glob)
    if not close_by_symbol:
        raise RuntimeError("No symbols loaded.")

    common_start = max(series.index.min() for series in close_by_symbol.values())
    common_end = min(series.index.max() for series in close_by_symbol.values())
    folds = make_folds(
        start=common_start,
        end=common_end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    if not folds:
        raise RuntimeError("No folds produced. Reduce train/test windows.")

    models = [
        "base_regime_tuned",
        "regime_decomposition",
        "quality_filter",
        "meta_model",
        "dynamic_sizing",
        "exit_analysis",
        "cross_asset_overlay",
    ]

    fold_rows: List[Dict[str, object]] = []
    oos_returns_by_model: Dict[str, List[pd.Series]] = {model: [] for model in models}
    oos_turnover_by_model: Dict[str, List[pd.Series]] = {model: [] for model in models}
    oos_exposure_by_model: Dict[str, List[pd.Series]] = {model: [] for model in models}

    for fold in folds:
        train_close = slice_universe(
            close_by_symbol=close_by_symbol,
            start=fold.train_start,
            end=fold.train_end,
            min_bars=args.min_train_bars,
        )
        test_close = slice_universe(
            close_by_symbol=close_by_symbol,
            start=fold.test_start,
            end=fold.test_end,
            min_bars=args.min_test_bars,
        )
        common_symbols = sorted(set(train_close.keys()) & set(test_close.keys()))
        train_close = {s: train_close[s] for s in common_symbols}
        test_close = {s: test_close[s] for s in common_symbols}
        if len(common_symbols) < 2:
            continue

        regime_params, regime_train_metrics = tune_regime_params(
            train_close=train_close,
            short_grid=short_grid,
            long_grid=long_grid,
            slope_grid=slope_grid,
            vol_window_grid=vol_window_grid,
            vol_quantile_grid=vol_quantile_grid,
        )
        train_features = build_feature_universe(train_close, regime_params)
        test_features = build_feature_universe(test_close, regime_params)

        for model_name in models:
            (
                params,
                train_metrics,
                test_metrics,
                test_ret,
                test_turn,
                test_exp,
                causal_ok,
                causal_msg,
            ) = evaluate_model(
                model_name=model_name,
                regime_params=regime_params,
                train_close=train_close,
                test_close=test_close,
                train_features=train_features,
                test_features=test_features,
            )
            if not causal_ok:
                raise RuntimeError(
                    f"Lookahead audit failed: fold={fold.fold_id} model={model_name} details={causal_msg}"
                )
            row: Dict[str, object] = {
                "fold_id": fold.fold_id,
                "model": model_name,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "n_symbols": len(common_symbols),
                "regime_train_sharpe": regime_train_metrics["net_sharpe"],
                "regime_params": regime_params.to_string(),
                "params": to_param_text(params, regime_params),
                "train_total_return": train_metrics["total_return"],
                "train_net_sharpe": train_metrics["net_sharpe"],
                "train_net_ann_return": train_metrics["net_ann_return"],
                "train_net_max_drawdown": train_metrics["net_max_drawdown"],
                "train_avg_turnover": train_metrics["avg_turnover"],
                "train_avg_exposure": train_metrics["avg_exposure"],
                "test_total_return": test_metrics["total_return"],
                "test_net_sharpe": test_metrics["net_sharpe"],
                "test_net_ann_return": test_metrics["net_ann_return"],
                "test_net_max_drawdown": test_metrics["net_max_drawdown"],
                "test_avg_turnover": test_metrics["avg_turnover"],
                "test_avg_exposure": test_metrics["avg_exposure"],
                "test_hit_rate": test_metrics["hit_rate"],
                "causal_audit_passed": bool(causal_ok),
                "causal_audit_msg": str(causal_msg),
            }
            fold_rows.append(row)
            oos_returns_by_model[model_name].append(
                test_ret.rename(f"fold_{fold.fold_id}_{model_name}")
            )
            oos_turnover_by_model[model_name].append(
                test_turn.rename(f"fold_{fold.fold_id}_{model_name}")
            )
            oos_exposure_by_model[model_name].append(
                test_exp.rename(f"fold_{fold.fold_id}_{model_name}")
            )

    if not fold_rows:
        raise RuntimeError("No folds evaluated after data-quality filtering.")

    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "fold_id"])
    summary_df = pooled_summary_from_fold_returns(
        fold_df, oos_returns_by_model, oos_turnover_by_model, oos_exposure_by_model
    )
    reco_df = recommendation_table(summary_df)

    fold_path = out_dir / "fold_results.csv"
    summary_path = out_dir / "summary.csv"
    reco_path = out_dir / "recommendations.csv"
    report_path = out_dir / "report.md"

    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    reco_df.to_csv(reco_path, index=False)

    report_text = render_report(
        args=args,
        folds=folds,
        fold_df=fold_df,
        summary_df=summary_df,
        reco_df=reco_df,
    )
    report_path.write_text(report_text)

    print("Crypto experiment suite complete")
    print(f"Fold results: {fold_path}")
    print(f"Summary: {summary_path}")
    print(f"Recommendations: {reco_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
