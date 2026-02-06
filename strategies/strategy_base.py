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

import numpy as np
import pandas as pd


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
        lookback_minutes: int = 5,
        hold_minutes: int = 10,
        tail_quantile: float = 0.02,
        top_n: int = 2000,
        liquidity_lookback: int = 390,
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
            if symbol in self.active_positions:
                continue
            self.active_positions[symbol] = {
                "side": 1.0 if qty > 0 else -1.0,
                "qty": abs(qty),
                "bars_left": float(self.hold_minutes),
            }

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
