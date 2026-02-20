"""
Strategy base classes and built-in strategies.

To create your own strategy:
1. Create a new class that inherits from Strategy
2. Implement add_indicators() to calculate your technical indicators
3. Implement generate_signals() to generate buy/sell signals

Required output columns from generate_signals():
    - signal: 1 for buy, -1 for sell, 0 for hold
    - target_qty: position size (shares for stocks, USD for crypto)
    - position: current position state (1=long, -1=short, 0=flat)

Optional output columns:
    - limit_price: if set, places a limit order instead of market

Example:
    class MyStrategy(Strategy):
        def __init__(self, lookback=20, position_size=10.0):
            self.lookback = lookback
            self.position_size = position_size

        def add_indicators(self, df):
            df['sma'] = df['Close'].rolling(self.lookback).mean()
            return df

        def generate_signals(self, df):
            df['signal'] = 0
            df.loc[df['Close'] > df['sma'], 'signal'] = 1
            df.loc[df['Close'] < df['sma'], 'signal'] = -1
            df['position'] = df['signal']
            df['target_qty'] = self.position_size
            return df
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - optional dependency guard
    RandomForestClassifier = None  # type: ignore[assignment]
    ExtraTreesClassifier = None  # type: ignore[assignment]
    HistGradientBoostingClassifier = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency guard
    XGBClassifier = None  # type: ignore[assignment]


class Strategy:
    """
    Base Strategy interface for adding indicators and generating trading signals.

    All strategies must implement:
        - add_indicators(df): Add technical indicators to the DataFrame
        - generate_signals(df): Generate trading signals

    The DataFrame must contain these columns:
        - Datetime, Open, High, Low, Close, Volume (input)
        - signal, target_qty, position (output from generate_signals)
    """

    def add_indicators(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:  # pragma: no cover - interface
        """Add technical indicators to the DataFrame. Override this method."""
        raise NotImplementedError

    def generate_signals(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:  # pragma: no cover - interface
        """Generate trading signals. Override this method."""
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full strategy pipeline. Do not override."""
        df = df.copy()
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df


class CrossSectionalStrategy(Strategy):
    """Base class for portfolio strategies that operate on symbol panels."""

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def run_panel(
        self,
        panel_df: pd.DataFrame,
        current_positions: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError


class MovingAverageStrategy(Strategy):
    """
    Moving average crossover strategy with explicitly defined entry/exit rules.
    """

    def __init__(
        self, short_window: int = 20, long_window: int = 60, position_size: float = 10.0
    ):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=1).mean()
        df["MA_long"] = df["Close"].rolling(self.long_window, min_periods=1).mean()
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.long_window).std().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = (df["MA_short"].shift(1) <= df["MA_long"].shift(1)) & (
            df["MA_short"] > df["MA_long"]
        )
        sell = (df["MA_short"].shift(1) >= df["MA_long"].shift(1)) & (
            df["MA_short"] < df["MA_long"]
        )

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = 0
        df.loc[df["MA_short"] > df["MA_long"], "position"] = 1
        df.loc[df["MA_short"] < df["MA_long"], "position"] = -1
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class TemplateStrategy(Strategy):
    """
    Starter strategy template for students. Modify the indicator and signal
    logic to build your own ideas.
    """

    def __init__(
        self,
        lookback: int = 14,
        position_size: float = 10.0,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
    ):
        if lookback < 1:
            raise ValueError("lookback must be at least 1.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.lookback = lookback
        self.position_size = position_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["momentum"] = df["Close"].pct_change(self.lookback).fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = df["momentum"] > self.buy_threshold
        sell = df["momentum"] < self.sell_threshold

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class CryptoTrendStrategy(Strategy):
    """
    Crypto trend-following strategy using fast/slow EMAs (long-only).
    """

    def __init__(
        self, short_window: int = 7, long_window: int = 21, position_size: float = 100.0
    ):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        long_regime = df["EMA_fast"] > df["EMA_slow"]
        flips = long_regime.astype(int).diff().fillna(0)
        df.loc[flips > 0, "signal"] = 1
        df.loc[flips < 0, "signal"] = -1
        df["position"] = long_regime.astype(int)
        df["target_qty"] = self.position_size
        return df


class CryptoRegimeTrendStrategy(Strategy):
    """
    Regime-aware crypto trend strategy.

    Improves on basic EMA cross by requiring:
    - fast EMA above slow EMA,
    - slow EMA slope positive over a lookback window,
    - realized volatility below a rolling quantile gate (causal threshold).
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 120,
        slope_lookback: int = 240,
        volatility_window: int = 1440,
        volatility_quantile: float = 0.70,
        position_size: float = 100.0,
    ):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if slope_lookback < 1:
            raise ValueError("slope_lookback must be at least 1.")
        if volatility_window < 20:
            raise ValueError("volatility_window must be at least 20.")
        if not 0.0 < volatility_quantile <= 1.0:
            raise ValueError("volatility_quantile must be in (0, 1].")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")

        self.short_window = short_window
        self.long_window = long_window
        self.slope_lookback = slope_lookback
        self.volatility_window = volatility_window
        self.volatility_quantile = volatility_quantile
        self.position_size = position_size

    @property
    def required_lookback(self) -> int:
        # Vol gate needs realized_vol history plus quantile-window history, then shift(1).
        vol_min = max(20, self.volatility_window // 3)
        quantile_min = max(50, self.volatility_window // 2)
        vol_gate_ready = vol_min + quantile_min + 1
        return max(self.long_window + 5, self.slope_lookback + 5, vol_gate_ready + 5)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        df["EMA_slow_slope"] = (
            df["EMA_slow"].pct_change(self.slope_lookback).fillna(0.0)
        )

        returns = df["Close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        min_vol_samples = max(20, self.volatility_window // 3)
        min_quantile_samples = max(50, self.volatility_window // 2)
        df["realized_vol"] = returns.rolling(
            self.volatility_window, min_periods=min_vol_samples
        ).std()
        vol_threshold = df["realized_vol"].rolling(
            self.volatility_window, min_periods=min_quantile_samples
        ).quantile(self.volatility_quantile)
        # Shift threshold by one bar so the gate is strictly causal.
        df["vol_gate_threshold"] = vol_threshold.shift(1)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        long_regime = (df["EMA_fast"] > df["EMA_slow"]) & (df["EMA_slow_slope"] > 0.0)
        vol_ok = df["realized_vol"] <= df["vol_gate_threshold"]
        long_regime = long_regime & vol_ok.fillna(False)

        flips = long_regime.astype(int).diff().fillna(0.0)
        df.loc[flips > 0, "signal"] = 1
        df.loc[flips < 0, "signal"] = -1
        df["position"] = long_regime.astype(int)
        df["target_qty"] = self.position_size
        return df


class CryptoCompetitionStrategy(Strategy):
    """
    Competition-focused crypto strategy that combines:
    - regime trend filter,
    - quality gates,
    - causal (walk-forward) meta-model probability gate,
    - dynamic entry sizing,
    - exit-analysis rules.

    Parameters are initialized to the strongest robust settings found in
    notebooks/results/crypto_experiment_suite_20260220_optimized_v2.
    """

    META_FEATURE_COLS = (
        "spread_z",
        "EMA_slow_slope",
        "vol_ratio",
        "ret1",
        "ret5",
        "ret30",
        "trend_tstat",
    )

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 120,
        slope_lookback: int = 120,
        volatility_window: int = 1440,
        volatility_quantile: float = 0.70,
        spread_z_threshold: float = 0.0,
        trend_tstat_threshold: float = 0.0,
        vol_ratio_max: float = 1.2,
        require_ret5_positive: bool = True,
        require_ret30_positive: bool = False,
        meta_n_estimators: int = 350,
        meta_max_depth: int = 4,
        meta_min_samples_leaf: int = 15,
        meta_max_features: str = "sqrt",
        meta_prob_threshold: float = 0.50,
        meta_no_trade_band: float = 0.03,
        meta_min_train_samples: int = 300,
        meta_refit_interval: int = 120,
        meta_train_window: int | None = 3000,
        meta_calibration_frac: float = 0.20,
        meta_use_xgboost: bool = False,
        sizing_scale: float = 1.5,
        sizing_offset: float = 0.3,
        sizing_vol_exponent: float = 0.75,
        sizing_smooth_span: int = 1,
        sizing_min_size: float = 0.05,
        exit_mode: str = "combo",
        exit_time_bars: int | None = 180,
        exit_trail_stop: float | None = 0.008,
        exit_profit_take: float | None = None,
        exit_min_hold_bars: int = 0,
        position_size: float = 100.0,
    ):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if slope_lookback < 1:
            raise ValueError("slope_lookback must be at least 1.")
        if volatility_window < 20:
            raise ValueError("volatility_window must be at least 20.")
        if not 0.0 < volatility_quantile <= 1.0:
            raise ValueError("volatility_quantile must be in (0, 1].")
        if meta_prob_threshold <= 0.0 or meta_prob_threshold >= 1.0:
            raise ValueError("meta_prob_threshold must be in (0, 1).")
        if meta_no_trade_band < 0.0 or meta_no_trade_band >= 0.5:
            raise ValueError("meta_no_trade_band must be in [0, 0.5).")
        if meta_n_estimators < 10:
            raise ValueError("meta_n_estimators must be at least 10.")
        if meta_max_depth < 2:
            raise ValueError("meta_max_depth must be at least 2.")
        if meta_min_samples_leaf < 1:
            raise ValueError("meta_min_samples_leaf must be at least 1.")
        if meta_min_train_samples < 100:
            raise ValueError("meta_min_train_samples must be at least 100.")
        if meta_refit_interval < 1:
            raise ValueError("meta_refit_interval must be at least 1.")
        if meta_train_window is not None and meta_train_window < meta_min_train_samples:
            raise ValueError(
                "meta_train_window must be >= meta_min_train_samples when provided."
            )
        if meta_calibration_frac <= 0.0 or meta_calibration_frac >= 0.5:
            raise ValueError("meta_calibration_frac must be in (0, 0.5).")
        if sizing_scale <= 0.0:
            raise ValueError("sizing_scale must be positive.")
        if sizing_vol_exponent <= 0.0:
            raise ValueError("sizing_vol_exponent must be positive.")
        if sizing_smooth_span < 1:
            raise ValueError("sizing_smooth_span must be at least 1.")
        if sizing_min_size < 0.0:
            raise ValueError("sizing_min_size must be non-negative.")
        if exit_mode not in {"baseline", "time", "trail", "combo"}:
            raise ValueError("exit_mode must be one of baseline/time/trail/combo.")
        if exit_min_hold_bars < 0:
            raise ValueError("exit_min_hold_bars must be non-negative.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        if (
            RandomForestClassifier is None
            or ExtraTreesClassifier is None
            or HistGradientBoostingClassifier is None
            or LogisticRegression is None
        ):
            raise ImportError(
                "CryptoCompetitionStrategy requires scikit-learn. "
                "Install dependencies from requirements.txt."
            )

        self.short_window = short_window
        self.long_window = long_window
        self.slope_lookback = slope_lookback
        self.volatility_window = volatility_window
        self.volatility_quantile = volatility_quantile

        self.spread_z_threshold = spread_z_threshold
        self.trend_tstat_threshold = trend_tstat_threshold
        self.vol_ratio_max = vol_ratio_max
        self.require_ret5_positive = require_ret5_positive
        self.require_ret30_positive = require_ret30_positive

        self.meta_n_estimators = meta_n_estimators
        self.meta_max_depth = meta_max_depth
        self.meta_min_samples_leaf = meta_min_samples_leaf
        self.meta_max_features = meta_max_features
        self.meta_prob_threshold = meta_prob_threshold
        self.meta_no_trade_band = meta_no_trade_band
        self.meta_min_train_samples = meta_min_train_samples
        self.meta_refit_interval = meta_refit_interval
        self.meta_train_window = meta_train_window
        self.meta_calibration_frac = meta_calibration_frac
        self.meta_use_xgboost = bool(meta_use_xgboost)

        self.sizing_scale = sizing_scale
        self.sizing_offset = sizing_offset
        self.sizing_vol_exponent = sizing_vol_exponent
        self.sizing_smooth_span = sizing_smooth_span
        self.sizing_min_size = sizing_min_size

        self.exit_mode = exit_mode
        self.exit_time_bars = exit_time_bars
        self.exit_trail_stop = exit_trail_stop
        self.exit_profit_take = exit_profit_take
        self.exit_min_hold_bars = exit_min_hold_bars
        self.position_size = position_size

        self._reset_meta_cache()

    def _reset_meta_cache(self) -> None:
        self._meta_cache_index: pd.Index | None = None
        self._meta_cache_probs: np.ndarray = np.array([], dtype=float)
        self._meta_cache_bundle: dict[str, object] | None = None
        self._meta_cache_last_refit_bar: int = -10**9

    @property
    def required_lookback(self) -> int:
        vol_min = max(20, self.volatility_window // 3)
        quantile_min = max(50, self.volatility_window // 2)
        vol_gate_ready = vol_min + quantile_min + 1
        exit_ready = (
            int(self.exit_time_bars) + 5
            if self.exit_time_bars is not None and self.exit_time_bars > 0
            else 5
        )
        return max(
            self.long_window + 5,
            self.slope_lookback + 5,
            vol_gate_ready + 5,
            730,  # spread_z rolling stats warmup
            self.meta_min_train_samples + self.meta_refit_interval + 5,
            exit_ready,
        )

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ret1"] = (
            df["Close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        df["ret5"] = (
            df["Close"].pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        df["ret30"] = (
            df["Close"].pct_change(30).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        # Keep NaN at the end so unknown future labels are excluded from training.
        df["fwd_ret"] = df["Close"].pct_change().shift(-1).replace(
            [np.inf, -np.inf], np.nan
        )

        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        df["EMA_slow_slope"] = (
            df["EMA_slow"].pct_change(self.slope_lookback).fillna(0.0)
        )

        min_vol_samples = max(20, self.volatility_window // 3)
        min_q_samples = max(50, self.volatility_window // 2)
        df["realized_vol"] = df["ret1"].rolling(
            self.volatility_window, min_periods=min_vol_samples
        ).std()
        vol_threshold = (
            df["realized_vol"]
            .rolling(self.volatility_window, min_periods=min_q_samples)
            .quantile(self.volatility_quantile)
        )
        df["vol_gate_threshold"] = vol_threshold.shift(1)
        df["vol_med_threshold"] = (
            df["realized_vol"]
            .rolling(self.volatility_window, min_periods=min_q_samples)
            .quantile(0.5)
            .shift(1)
        )
        df["vol_ok"] = (df["realized_vol"] <= df["vol_gate_threshold"]).fillna(False)

        spread = ((df["EMA_fast"] - df["EMA_slow"]) / df["EMA_slow"]).replace(
            [np.inf, -np.inf], np.nan
        )
        spread_mean = spread.rolling(720, min_periods=100).mean()
        spread_std = spread.rolling(720, min_periods=100).std()
        df["spread_z"] = ((spread - spread_mean) / (spread_std + 1e-12)).replace(
            [np.inf, -np.inf], np.nan
        )
        df["trend_tstat"] = (
            (df["EMA_fast"] - df["EMA_slow"]).abs()
            / (df["Close"] * df["realized_vol"] + 1e-12)
        ).replace([np.inf, -np.inf], np.nan)
        df["vol_ratio"] = (df["realized_vol"] / (df["vol_gate_threshold"] + 1e-12)).replace(
            [np.inf, -np.inf], np.nan
        )

        df["trend_pre"] = (df["EMA_fast"] > df["EMA_slow"]).fillna(False)
        df["base_long"] = (
            df["trend_pre"] & (df["EMA_slow_slope"] > 0.0) & df["vol_ok"]
        ).astype(float)
        return df

    def _meta_model_names(self) -> list[str]:
        names = ["rf", "et", "hgb"]
        if self.meta_use_xgboost and XGBClassifier is not None:
            names.append("xgb")
        return names

    def _fit_single_meta_model(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        seed: int,
    ) -> object | None:
        if model_name == "rf":
            model = RandomForestClassifier(
                n_estimators=int(self.meta_n_estimators),
                max_depth=int(self.meta_max_depth),
                min_samples_leaf=int(self.meta_min_samples_leaf),
                max_features=self.meta_max_features,
                random_state=int(seed),
            )
        elif model_name == "et":
            model = ExtraTreesClassifier(
                n_estimators=int(self.meta_n_estimators),
                max_depth=int(self.meta_max_depth),
                min_samples_leaf=int(self.meta_min_samples_leaf),
                max_features=self.meta_max_features,
                random_state=int(seed),
            )
        elif model_name == "hgb":
            model = HistGradientBoostingClassifier(
                max_iter=max(80, int(self.meta_n_estimators)),
                max_depth=int(self.meta_max_depth),
                min_samples_leaf=int(self.meta_min_samples_leaf),
                learning_rate=0.05,
                random_state=int(seed),
            )
        elif model_name == "xgb":
            if XGBClassifier is None:
                return None
            model = XGBClassifier(
                n_estimators=int(self.meta_n_estimators),
                max_depth=int(self.meta_max_depth),
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                min_child_weight=1,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=int(seed),
                n_jobs=1,
                verbosity=0,
            )
        else:
            return None
        try:
            model.fit(X, y.astype(int))
        except Exception:
            return None
        return model

    def _predict_meta_model(self, model: object, X: np.ndarray) -> np.ndarray:
        try:
            p = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        except Exception:
            return np.full(len(X), 0.5, dtype=float)
        return np.clip(p, 1e-6, 1.0 - 1e-6)

    def _fit_meta_model(self, train_df: pd.DataFrame) -> dict[str, object]:
        if train_df.empty:
            return {
                "models": [],
                "weights": np.array([], dtype=float),
                "stacker": None,
                "const_prob": 0.5,
            }

        y = (train_df["fwd_ret"].values > 0.0).astype(int)
        positive_rate = float(y.mean()) if len(y) else 0.5
        if len(train_df) < self.meta_min_train_samples:
            return {
                "models": [],
                "weights": np.array([], dtype=float),
                "stacker": None,
                "const_prob": positive_rate,
            }
        if positive_rate <= 0.01 or positive_rate >= 0.99:
            return {
                "models": [],
                "weights": np.array([], dtype=float),
                "stacker": None,
                "const_prob": positive_rate,
            }

        X = (
            train_df[list(self.META_FEATURE_COLS)]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .values.astype(float)
        )
        X = np.clip(X, -50.0, 50.0)

        n = len(train_df)
        min_cal = max(60, int(self.meta_min_train_samples // 4))
        split = int(n * (1.0 - float(self.meta_calibration_frac)))
        split = max(int(self.meta_min_train_samples), min(split, n - min_cal))
        if split <= 0 or split >= n:
            split = int(n * 0.8)
        if split <= 0 or split >= n:
            split = n

        if split < n:
            X_fit, y_fit = X[:split], y[:split]
            X_cal, y_cal = X[split:], y[split:]
        else:
            X_fit, y_fit = X, y
            X_cal, y_cal = X, y

        models: list[tuple[str, object]] = []
        for k, name in enumerate(self._meta_model_names()):
            model = self._fit_single_meta_model(name, X_fit, y_fit, seed=42 + 11 * k)
            if model is not None:
                models.append((name, model))

        if not models:
            return {
                "models": [],
                "weights": np.array([], dtype=float),
                "stacker": None,
                "const_prob": positive_rate,
            }

        pred_cols: list[np.ndarray] = []
        inv_losses: list[float] = []
        for _, model in models:
            p = self._predict_meta_model(model, X_cal)
            pred_cols.append(p)
            brier = float(np.mean((p - y_cal) ** 2))
            inv_losses.append(1.0 / max(brier, 1e-6))

        weights = np.asarray(inv_losses, dtype=float)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = np.ones(len(models), dtype=float)
        weights = weights / float(weights.sum())

        stacker: LogisticRegression | None = None
        if len(models) >= 2 and len(y_cal) >= 80 and len(np.unique(y_cal)) >= 2:
            Z = np.vstack(pred_cols).T
            try:
                stacker = LogisticRegression(max_iter=200, solver="lbfgs")
                stacker.fit(Z, y_cal.astype(int))
            except Exception:
                stacker = None

        return {
            "models": models,
            "weights": weights,
            "stacker": stacker,
            "const_prob": positive_rate,
        }

    def _predict_meta_bundle(self, X_row: np.ndarray, bundle: dict[str, object]) -> float:
        models = bundle.get("models", [])
        if not isinstance(models, list) or not models:
            return float(bundle.get("const_prob", 0.5))

        weights = np.asarray(bundle.get("weights", np.array([], dtype=float)), dtype=float)
        if len(weights) != len(models) or float(weights.sum()) <= 0:
            weights = np.ones(len(models), dtype=float) / float(len(models))

        preds: list[float] = []
        for _, model in models:
            p = self._predict_meta_model(model, X_row)
            preds.append(float(p[0]) if len(p) else 0.5)
        pred_vec = np.asarray(preds, dtype=float)
        weighted = float(np.dot(weights, pred_vec))

        stacker = bundle.get("stacker")
        if stacker is not None and len(pred_vec) >= 2:
            try:
                stacked = float(stacker.predict_proba(pred_vec.reshape(1, -1))[0, 1])
                weighted = 0.6 * weighted + 0.4 * stacked
            except Exception:
                pass
        return float(np.clip(weighted, 1e-6, 1.0 - 1e-6))

    def _compute_causal_meta_prob(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        probs = pd.Series(0.5, index=df.index, dtype=float)
        if n == 0:
            self._reset_meta_cache()
            return probs

        # Fast path: identical index means we can return cached probabilities.
        if self._meta_cache_index is not None and df.index.equals(self._meta_cache_index):
            if len(self._meta_cache_probs) == n:
                return pd.Series(self._meta_cache_probs, index=df.index, dtype=float)

        feature_frame = df[list(self.META_FEATURE_COLS)].replace(
            [np.inf, -np.inf], np.nan
        )

        incremental = (
            self._meta_cache_index is not None
            and len(self._meta_cache_probs) + 1 == n
            and df.index[:-1].equals(self._meta_cache_index)
        )
        sliding_incremental = (
            self._meta_cache_index is not None
            and len(self._meta_cache_probs) == n
            and n > 1
            and df.index[:-1].equals(self._meta_cache_index[1:])
        )

        if incremental:
            probs_np = np.concatenate([self._meta_cache_probs, np.array([0.5])])
            start_i = n - 1
            bundle = self._meta_cache_bundle
            last_refit = int(self._meta_cache_last_refit_bar)
        elif sliding_incremental:
            probs_np = np.concatenate([self._meta_cache_probs[1:], np.array([0.5])])
            start_i = n - 1
            bundle = self._meta_cache_bundle
            # Window shifted by one bar, so rebase refit pointer by -1.
            last_refit = int(self._meta_cache_last_refit_bar) - 1
        else:
            probs_np = np.full(n, 0.5, dtype=float)
            start_i = 0
            bundle = None
            last_refit = -10**9

        for i in range(start_i, n):
            if bundle is None or (i - last_refit) >= self.meta_refit_interval:
                lo = 0
                if self.meta_train_window is not None:
                    lo = max(0, i - int(self.meta_train_window))
                train_block = df.iloc[lo:i]
                if len(train_block):
                    valid_train = (
                        train_block["trend_pre"].astype(bool)
                        & train_block[list(self.META_FEATURE_COLS)].replace(
                            [np.inf, -np.inf], np.nan
                        ).notna().all(axis=1)
                        & train_block["fwd_ret"].notna()
                    )
                    train_df = train_block.loc[valid_train]
                else:
                    train_df = train_block
                bundle = self._fit_meta_model(train_df)
                last_refit = i

            row = (
                feature_frame.iloc[i]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .values.astype(float)
            )
            row = np.clip(row, -50.0, 50.0).reshape(1, -1)
            if bundle is None:
                p = 0.5
            else:
                p = self._predict_meta_bundle(row, bundle)
            probs_np[i] = float(np.clip(p, 1e-6, 1.0 - 1e-6))

        self._meta_cache_index = df.index.copy()
        self._meta_cache_probs = probs_np.copy()
        self._meta_cache_bundle = bundle
        self._meta_cache_last_refit_bar = int(last_refit)
        return pd.Series(probs_np, index=df.index, dtype=float)

    def _apply_quality_filter(self, df: pd.DataFrame) -> pd.Series:
        q = df["base_long"] > 0.0
        q = q & (df["spread_z"] > self.spread_z_threshold).fillna(False)
        q = q & (df["trend_tstat"] > self.trend_tstat_threshold).fillna(False)
        q = q & (df["vol_ratio"] <= self.vol_ratio_max).fillna(False)
        if self.require_ret30_positive:
            q = q & (df["ret30"] > 0.0).fillna(False)
        if self.require_ret5_positive:
            q = q & (df["ret5"] > 0.0).fillna(False)
        return q.astype(float)

    def _dynamic_size(self, df: pd.DataFrame) -> pd.Series:
        strength = (df["spread_z"] - self.sizing_offset).clip(lower=0.0).fillna(0.0)
        vol_ratio = df["vol_ratio"].clip(lower=0.5, upper=5.0).fillna(1.0)
        vol_adj = 1.0 / (vol_ratio ** self.sizing_vol_exponent)
        size = (self.sizing_scale * strength * vol_adj).clip(lower=0.0, upper=1.0)
        if self.sizing_smooth_span > 1:
            size = size.ewm(span=self.sizing_smooth_span, adjust=False).mean()
        size = size.where(size >= self.sizing_min_size, 0.0)
        return size.astype(float)

    def _position_with_exit_and_size(
        self,
        entry_gate: pd.Series,
        size_series: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        sig = entry_gate.fillna(0.0).astype(float).values
        size = size_series.fillna(0.0).clip(lower=0.0, upper=1.0).astype(float).values
        px = close.astype(float).values

        n = len(entry_gate)
        out = np.zeros(n, dtype=float)
        in_pos = False
        held_size = 0.0
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
                    held_size = float(size[i])
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
                    if hold >= int(max(0, self.exit_min_hold_bars)):
                        if (
                            self.exit_mode in {"time", "combo"}
                            and self.exit_time_bars is not None
                            and hold >= int(self.exit_time_bars)
                        ):
                            exit_now = True
                            forced_exit = True
                        if (
                            self.exit_mode in {"trail", "combo"}
                            and self.exit_trail_stop is not None
                            and peak > 0
                        ):
                            drawdown = price / peak - 1.0
                            if drawdown <= -float(self.exit_trail_stop):
                                exit_now = True
                                forced_exit = True
                        if self.exit_profit_take is not None and entry_price > 0:
                            pnl = price / entry_price - 1.0
                            if pnl >= float(self.exit_profit_take):
                                exit_now = True
                                forced_exit = True
                if exit_now:
                    in_pos = False
                    held_size = 0.0
                    hold = 0
                    peak = 0.0
                    entry_price = 0.0
                    if forced_exit:
                        lock_until_reset = True
            out[i] = held_size if in_pos else 0.0

        return pd.Series(out, index=entry_gate.index, dtype=float)

    def _hysteresis_meta_gate(self, prob: pd.Series) -> pd.Series:
        lower = max(1e-6, float(self.meta_prob_threshold) - float(self.meta_no_trade_band))
        upper = min(1.0 - 1e-6, float(self.meta_prob_threshold) + float(self.meta_no_trade_band))
        out = np.zeros(len(prob), dtype=float)
        state = False
        vals = prob.fillna(0.5).astype(float).values
        for i, p in enumerate(vals):
            if p >= upper:
                state = True
            elif p <= lower:
                state = False
            out[i] = 1.0 if state else 0.0
        return pd.Series(out, index=prob.index, dtype=float)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["quality_long"] = self._apply_quality_filter(df)
        df["meta_prob"] = self._compute_causal_meta_prob(df)
        df["meta_long"] = self._hysteresis_meta_gate(df["meta_prob"])

        entry_gate = (
            (df["quality_long"] > 0.0) & (df["meta_long"] > 0.0)
        ).astype(float)
        size = self._dynamic_size(df)
        position_frac = self._position_with_exit_and_size(
            entry_gate=entry_gate,
            size_series=size,
            close=df["Close"],
        )

        desired_notional = (position_frac * self.position_size).fillna(0.0)
        delta = desired_notional.diff().fillna(desired_notional).fillna(0.0)
        eps = 1e-9

        df["signal"] = 0
        df.loc[delta > eps, "signal"] = 1
        df.loc[delta < -eps, "signal"] = -1
        df["target_qty"] = delta.abs().astype(float)
        # Keep position neutral in output so live runner relies on explicit signals only.
        df["position"] = 0
        df["desired_position_frac"] = position_frac.astype(float)
        return df


class CryptoCompetitionPortfolioStrategy(CrossSectionalStrategy):
    """
    Multi-asset portfolio wrapper around CryptoCompetitionStrategy.

    Each symbol gets its own causal model state. The strategy emits delta orders
    toward target notional per symbol, where target notional is:

        desired_position_frac * (portfolio_notional / active_symbols)

    This matches the vectorized equal-weight portfolio spirit better than
    per-symbol standalone sizing.
    """

    def __init__(
        self,
        portfolio_notional: float = 100_000.0,
        allow_fractional_qty: bool = True,
        min_order_notional: float = 5.0,
        **competition_kwargs,
    ):
        if portfolio_notional <= 0:
            raise ValueError("portfolio_notional must be positive.")
        if min_order_notional < 0:
            raise ValueError("min_order_notional must be non-negative.")

        self.portfolio_notional = float(portfolio_notional)
        self.allow_fractional_qty = bool(allow_fractional_qty)
        self.min_order_notional = float(min_order_notional)

        # Use per-symbol position_size=1 so desired_position_frac remains the
        # sizing driver, while portfolio wrapper controls gross notional.
        self._competition_kwargs = dict(competition_kwargs)
        self._competition_kwargs["position_size"] = 1.0
        self._strategy_by_symbol: dict[str, CryptoCompetitionStrategy] = {}
        self._required_lookback = int(
            CryptoCompetitionStrategy(**self._competition_kwargs).required_lookback
        )

    @property
    def required_lookback(self) -> int:
        return self._required_lookback

    def _strategy_for_symbol(self, symbol: str) -> CryptoCompetitionStrategy:
        sym = str(symbol).upper()
        strategy = self._strategy_by_symbol.get(sym)
        if strategy is None:
            strategy = CryptoCompetitionStrategy(**self._competition_kwargs)
            self._strategy_by_symbol[sym] = strategy
        return strategy

    @staticmethod
    def _normalize_panel(panel_df: pd.DataFrame) -> pd.DataFrame:
        panel = panel_df.copy()
        rename: dict[str, str] = {}
        for col in panel.columns:
            key = str(col).strip().lower()
            if key in {"datetime", "timestamp", "date", "time"}:
                rename[col] = "Datetime"
            elif key in {"symbol", "ticker"}:
                rename[col] = "symbol"
            elif key == "open":
                rename[col] = "Open"
            elif key == "high":
                rename[col] = "High"
            elif key == "low":
                rename[col] = "Low"
            elif key == "close":
                rename[col] = "Close"
            elif key == "volume":
                rename[col] = "Volume"
        if rename:
            panel = panel.rename(columns=rename)

        required = ["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in panel.columns]
        if missing:
            raise ValueError(f"Panel data missing required columns: {missing}")

        out = panel[required].copy()
        out["Datetime"] = pd.to_datetime(out["Datetime"], utc=True, errors="coerce")
        out["symbol"] = out["symbol"].astype(str).str.upper()
        out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
        out = out.dropna(subset=["Datetime", "symbol", "Close"])
        return out.sort_values(["Datetime", "symbol"]).drop_duplicates(
            ["Datetime", "symbol"], keep="last"
        )

    def _qty_from_notional_delta(self, delta_notional: float, close: float) -> float:
        if close <= 0:
            return 0.0
        qty = abs(float(delta_notional)) / float(close)
        if self.allow_fractional_qty:
            qty = np.floor(qty * 1_000_000.0) / 1_000_000.0
        else:
            qty = float(int(qty))
        return float(max(0.0, qty))

    def run_panel(
        self,
        panel_df: pd.DataFrame,
        current_positions: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["symbol", "signal", "target_qty", "limit_price"])
        if panel_df is None or panel_df.empty:
            return empty

        panel = self._normalize_panel(panel_df)
        if panel.empty:
            return empty

        desired_frac_by_symbol: dict[str, float] = {}
        close_by_symbol: dict[str, float] = {}

        for symbol, sym_df in panel.groupby("symbol"):
            sym = str(symbol).upper()
            local = sym_df.sort_values("Datetime").copy()
            if len(local) < self.required_lookback:
                continue

            strategy = self._strategy_for_symbol(sym)
            out = strategy.run(local[["Datetime", "Open", "High", "Low", "Close", "Volume"]])
            if out.empty:
                continue

            latest = out.iloc[-1]
            close = float(latest.get("Close", np.nan))
            if not np.isfinite(close) or close <= 0:
                continue

            frac_val = latest.get("desired_position_frac", np.nan)
            if pd.notna(frac_val):
                frac = float(np.clip(float(frac_val), 0.0, 1.0))
            else:
                signal = int(latest.get("signal", 0) or 0)
                frac = 1.0 if signal > 0 else 0.0

            desired_frac_by_symbol[sym] = frac
            close_by_symbol[sym] = close

        if not desired_frac_by_symbol:
            return empty

        active_symbols = sorted(desired_frac_by_symbol.keys())
        per_symbol_budget = float(self.portfolio_notional) / float(len(active_symbols))

        pos_map = {
            str(k).upper(): float(v)
            for k, v in (current_positions or {}).items()
            if pd.notna(v)
        }

        rows: list[dict[str, float | int | str]] = []
        for sym in active_symbols:
            close = float(close_by_symbol[sym])
            target_notional = float(desired_frac_by_symbol[sym]) * per_symbol_budget
            current_qty = float(pos_map.get(sym, 0.0))
            current_notional = current_qty * close
            delta_notional = target_notional - current_notional

            if abs(delta_notional) < float(self.min_order_notional):
                continue

            signal = 1 if delta_notional > 0 else -1
            qty = self._qty_from_notional_delta(delta_notional, close)
            if qty <= 0:
                continue

            if signal < 0:
                # Long-only behavior for this crypto competition portfolio.
                max_sell = max(0.0, current_qty)
                qty = min(qty, max_sell)
                if qty <= 0:
                    continue

            rows.append(
                {
                    "symbol": sym,
                    "signal": int(signal),
                    "target_qty": float(qty),
                    "limit_price": float(close),
                }
            )

        if not rows:
            return empty
        out = pd.DataFrame(rows)
        out = out.dropna(subset=["symbol", "signal", "target_qty"])
        out["signal"] = out["signal"].astype(int)
        out["target_qty"] = out["target_qty"].astype(float)
        out = out[out["target_qty"] > 0]
        return out[["symbol", "signal", "target_qty", "limit_price"]]


class CrossSectionalPaperReversalStrategy(CrossSectionalStrategy):
    """
    Cross-sectional intraday reversal portfolio strategy.

    Emits order intents for multiple symbols using columns:
    - symbol
    - signal (1 buy, -1 sell)
    - target_qty (shares)
    - limit_price (optional)
    """

    ANNUALIZATION_MINUTES = np.sqrt(252 * 6.5 * 60)

    def __init__(
        self,
        lookback_minutes: int = 15,
        hold_minutes: int = 30,
        tail_quantile: float = 0.016,
        top_n: int = 600,
        liquidity_lookback: int = 30,
        min_universe_size: int = 40,
        base_notional_per_name: float = 1_000.0,
        target_annual_vol: float = 0.60,
        volatility_window: int = 60,
        max_leverage: float = 10.0,
        min_annual_vol: float = 0.01,
        allow_flips: bool = True,
        refresh_hold_on_signal: bool = True,
    ):
        if lookback_minutes < 1:
            raise ValueError("lookback_minutes must be at least 1.")
        if hold_minutes < 1:
            raise ValueError("hold_minutes must be at least 1.")
        if not 0 < tail_quantile < 0.5:
            raise ValueError("tail_quantile must be between 0 and 0.5.")
        if top_n < 2:
            raise ValueError("top_n must be at least 2.")
        if liquidity_lookback < 5:
            raise ValueError("liquidity_lookback must be at least 5.")
        if min_universe_size < 2:
            raise ValueError("min_universe_size must be at least 2.")
        if base_notional_per_name <= 0:
            raise ValueError("base_notional_per_name must be positive.")
        if target_annual_vol < 0:
            raise ValueError("target_annual_vol must be non-negative.")
        if volatility_window < 5:
            raise ValueError("volatility_window must be at least 5.")
        if max_leverage <= 0:
            raise ValueError("max_leverage must be positive.")
        if min_annual_vol <= 0:
            raise ValueError("min_annual_vol must be positive.")

        self.lookback_minutes = lookback_minutes
        self.hold_minutes = hold_minutes
        self.tail_quantile = tail_quantile
        self.top_n = top_n
        self.liquidity_lookback = liquidity_lookback
        self.min_universe_size = min_universe_size
        self.base_notional_per_name = float(base_notional_per_name)
        self.target_annual_vol = target_annual_vol
        self.volatility_window = volatility_window
        self.max_leverage = max_leverage
        self.min_annual_vol = min_annual_vol
        self.allow_flips = allow_flips
        self.refresh_hold_on_signal = refresh_hold_on_signal

        self.active_positions: dict[str, dict[str, float]] = {}

    @property
    def required_lookback(self) -> int:
        return max(
            self.lookback_minutes + 5,
            self.liquidity_lookback + 5,
            self.volatility_window + 5,
        )

    def _prepare_panel(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        df = panel_df.copy()
        rename = {}
        for col in df.columns:
            key = str(col).strip().lower()
            if key in {"datetime", "timestamp", "date", "time"}:
                rename[col] = "Datetime"
            elif key in {"symbol", "ticker"}:
                rename[col] = "symbol"
            elif key == "close":
                rename[col] = "Close"
            elif key == "volume":
                rename[col] = "Volume"
        if rename:
            df = df.rename(columns=rename)

        required = ["Datetime", "symbol", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Cross-sectional strategy missing columns: {missing}")

        out = df[required].copy()
        out["Datetime"] = pd.to_datetime(out["Datetime"], utc=True, errors="coerce")
        out["symbol"] = out["symbol"].astype(str).str.upper()
        out = out.dropna(subset=["Datetime", "symbol", "Close", "Volume"])
        out = out.sort_values(["symbol", "Datetime"]).drop_duplicates(
            ["symbol", "Datetime"], keep="last"
        )
        return out

    def _sync_with_positions(self, current_positions: dict[str, float] | None) -> None:
        if current_positions is None:
            return
        normalized = {
            str(symbol).upper(): float(qty)
            for symbol, qty in current_positions.items()
            if float(qty) != 0.0
        }
        for symbol in list(self.active_positions.keys()):
            if symbol not in normalized:
                self.active_positions.pop(symbol, None)
        for symbol, qty in normalized.items():
            side = 1.0 if qty > 0 else -1.0
            abs_qty = abs(qty)
            state = self.active_positions.get(symbol)
            if state is None:
                self.active_positions[symbol] = {
                    "side": side,
                    "qty": abs_qty,
                    "bars_left": float(self.hold_minutes),
                }
                continue

            previous_side = float(np.sign(state.get("side", side)))
            state["side"] = side
            state["qty"] = abs_qty
            if previous_side != side:
                state["bars_left"] = float(self.hold_minutes)

    def run_panel(
        self,
        panel_df: pd.DataFrame,
        current_positions: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        panel = self._prepare_panel(panel_df)
        if panel.empty:
            return pd.DataFrame(
                columns=["symbol", "signal", "target_qty", "limit_price"]
            )

        self._sync_with_positions(current_positions)

        panel["ret_lb"] = panel.groupby("symbol")["Close"].pct_change(
            self.lookback_minutes
        )
        panel["ret_1m"] = panel.groupby("symbol")["Close"].pct_change()
        panel["dollar_volume"] = panel["Close"] * panel["Volume"]
        panel["adv"] = panel.groupby("symbol")["dollar_volume"].transform(
            lambda s: s.rolling(
                self.liquidity_lookback,
                min_periods=max(5, self.liquidity_lookback // 3),
            ).mean()
        )
        panel["ann_vol"] = (
            panel.groupby("symbol")["ret_1m"].transform(
                lambda s: s.rolling(
                    self.volatility_window,
                    min_periods=max(5, self.volatility_window // 3),
                ).std()
            )
            * self.ANNUALIZATION_MINUTES
        )

        latest_rows = (
            panel.sort_values("Datetime").groupby("symbol", as_index=False).tail(1)
        )
        latest_prices = dict(zip(latest_rows["symbol"], latest_rows["Close"]))

        orders: list[dict[str, float | int | str]] = []

        for symbol in list(self.active_positions.keys()):
            state = self.active_positions[symbol]
            state["bars_left"] -= 1

        liquid = latest_rows.dropna(subset=["ret_lb", "adv"]).copy()
        if not liquid.empty:
            liquid["liquidity_rank"] = liquid["adv"].rank(
                ascending=False, method="first"
            )
            liquid = liquid[liquid["liquidity_rank"] <= float(self.top_n)].copy()

        desired_side: dict[str, int] = {}
        qty_for_symbol: dict[str, int] = {}

        if len(liquid) >= self.min_universe_size:
            tail_count = max(1, int(np.floor(len(liquid) * self.tail_quantile)))
            long_candidates = liquid.sort_values("ret_lb", ascending=True).head(
                tail_count
            )
            short_candidates = liquid.sort_values("ret_lb", ascending=False).head(
                tail_count
            )

            side_count = int(min(len(long_candidates), len(short_candidates)))
            if side_count > 0:
                long_candidates = long_candidates.head(side_count)
                short_candidates = short_candidates.head(side_count)

                vol_proxy = (
                    float(liquid["ann_vol"].dropna().median())
                    if liquid["ann_vol"].notna().any()
                    else self.min_annual_vol
                )
                vol_proxy = max(vol_proxy, self.min_annual_vol)
                leverage = 1.0
                if self.target_annual_vol > 0:
                    leverage = min(
                        self.max_leverage, self.target_annual_vol / vol_proxy
                    )

                notional_per_name = self.base_notional_per_name * leverage
                for _, row in long_candidates.iterrows():
                    symbol = str(row["symbol"])
                    close = float(row["Close"])
                    qty = max(1, int(notional_per_name / close)) if close > 0 else 0
                    if qty <= 0:
                        continue
                    desired_side[symbol] = 1
                    qty_for_symbol[symbol] = qty

                for _, row in short_candidates.iterrows():
                    symbol = str(row["symbol"])
                    close = float(row["Close"])
                    qty = max(1, int(notional_per_name / close)) if close > 0 else 0
                    if qty <= 0:
                        continue
                    desired_side[symbol] = -1
                    qty_for_symbol[symbol] = qty

        for symbol in list(self.active_positions.keys()):
            state = self.active_positions[symbol]
            side = int(np.sign(state["side"]))
            qty = int(max(1, round(float(state["qty"]))))
            desired = int(desired_side.get(symbol, 0))

            if desired == side:
                if self.refresh_hold_on_signal:
                    state["bars_left"] = float(self.hold_minutes)
                    state["qty"] = float(qty_for_symbol.get(symbol, qty))
                continue

            if desired != 0 and self.allow_flips:
                new_qty = int(max(1, qty_for_symbol.get(symbol, qty)))
                flip_qty = qty + new_qty
                orders.append(
                    {
                        "symbol": symbol,
                        "signal": desired,
                        "target_qty": flip_qty,
                        "limit_price": float(latest_prices.get(symbol, np.nan)),
                    }
                )
                self.active_positions[symbol] = {
                    "side": float(desired),
                    "qty": float(new_qty),
                    "bars_left": float(self.hold_minutes),
                }
                continue

            if state["bars_left"] <= 0:
                orders.append(
                    {
                        "symbol": symbol,
                        "signal": -side,
                        "target_qty": qty,
                        "limit_price": float(latest_prices.get(symbol, np.nan)),
                    }
                )
                self.active_positions.pop(symbol, None)

        for symbol, desired in desired_side.items():
            if symbol in self.active_positions:
                continue
            qty = int(max(1, qty_for_symbol.get(symbol, 1)))
            orders.append(
                {
                    "symbol": symbol,
                    "signal": int(desired),
                    "target_qty": qty,
                    "limit_price": float(latest_prices.get(symbol, np.nan)),
                }
            )
            self.active_positions[symbol] = {
                "side": float(desired),
                "qty": float(qty),
                "bars_left": float(self.hold_minutes),
            }

        if not orders:
            return pd.DataFrame(
                columns=["symbol", "signal", "target_qty", "limit_price"]
            )

        out = pd.DataFrame(orders)
        out = out.dropna(subset=["symbol", "signal", "target_qty"])
        out["symbol"] = out["symbol"].astype(str).str.upper()
        out["signal"] = out["signal"].astype(int)
        out["target_qty"] = out["target_qty"].astype(float)
        out = out[out["target_qty"] > 0]
        return out[["symbol", "signal", "target_qty", "limit_price"]]


class DemoStrategy(Strategy):
    """
    Simple demo strategy - buys 1 share when price up, sells 1 share when price down.
    Uses tiny position size to avoid margin/locate issues.

    Usage:
        python run_live.py --symbol AAPL --strategy demo --timeframe 1Min --sleep 5 --live
    """

    def __init__(self, position_size: float = 1.0):
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["change"] = df["Close"].diff().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df.loc[df["change"] > 0, "signal"] = 1  # Price went up -> buy
        df.loc[df["change"] < 0, "signal"] = -1  # Price went down -> sell
        df["position"] = df["signal"]
        df["target_qty"] = self.position_size
        return df


## =============================================================================
## CREATE YOUR OWN STRATEGIES BELOW
## =============================================================================
##
## Example: RSI Strategy
##
## class RSIStrategy(Strategy):
##     """Buy when RSI is oversold, sell when overbought."""
##
##     def __init__(self, period=14, oversold=30, overbought=70, position_size=10.0):
##         self.period = period
##         self.oversold = oversold
##         self.overbought = overbought
##         self.position_size = position_size
##
##     def add_indicators(self, df):
##         delta = df['Close'].diff()
##         gain = delta.where(delta > 0, 0).rolling(self.period).mean()
##         loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
##         rs = gain / loss
##         df['RSI'] = 100 - (100 / (1 + rs))
##         return df
##
##     def generate_signals(self, df):
##         df['signal'] = 0
##         df.loc[df['RSI'] < self.oversold, 'signal'] = 1   # Buy when oversold
##         df.loc[df['RSI'] > self.overbought, 'signal'] = -1  # Sell when overbought
##         df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
##         df['target_qty'] = self.position_size
##         return df
##
## To use your strategy:
##   python run_live.py --symbol AAPL --strategy mystrategy --live
##
