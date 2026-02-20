from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core.asset_eligibility import (
    evaluate_asset_eligibility,
    normalize_asset_flags_by_symbol,
)
from core.paper_parity import (
    PaperParityConfig,
    normalize_paper_parity_config,
    short_open_unit_value,
)
from strategies import Strategy


@dataclass
class MultiAssetTradeRecord:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    qty: float
    price: float


class MultiAssetBacktester:
    """
    Multi-symbol backtester that reuses single-symbol strategies per symbol.

    The input must be long-form OHLCV data with at least:
    Datetime, symbol, Open, High, Low, Close, Volume.
    """

    def __init__(
        self,
        panel_df: pd.DataFrame,
        strategy_factory: Callable[[], Strategy],
        initial_capital: float = 50_000.0,
        max_notional_per_order: Optional[float] = None,
        max_short_notional: Optional[float] = None,
        paper_parity: Optional[PaperParityConfig] = None,
        asset_flags_by_symbol: Optional[Dict[str, object]] = None,
    ):
        self.panel_df = self._prepare_panel(panel_df)
        self.strategy_factory = strategy_factory
        self.initial_capital = float(initial_capital)
        self.max_notional_per_order = max_notional_per_order
        self.max_short_notional = max_short_notional
        self.paper_parity = normalize_paper_parity_config(paper_parity)
        self.asset_flags_by_symbol = normalize_asset_flags_by_symbol(asset_flags_by_symbol)

        strategy_probe = strategy_factory()
        self.portfolio_strategy: Optional[Strategy] = None
        self.required_lookback = int(getattr(strategy_probe, "required_lookback", 1))
        if hasattr(strategy_probe, "run_panel"):
            self.portfolio_strategy = strategy_probe

        self.cash = self.initial_capital
        self.position_by_symbol: Dict[str, float] = {}
        self.latest_price_by_symbol: Dict[str, float] = {}
        self.max_history_rows = max(self.required_lookback + 20, 50)
        self.history_by_symbol: Dict[str, pd.DataFrame] = {
            symbol: pd.DataFrame(
                columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
            )
            for symbol in sorted(self.panel_df["symbol"].unique())
        }

        self.equity_curve: List[float] = []
        self.cash_curve: List[float] = []
        self.timestamp_curve: List[pd.Timestamp] = []
        self.trades: List[MultiAssetTradeRecord] = []
        self.rejections: List[dict] = []
        self.reservation_events: List[dict] = []
        self._batch_reserved_buying_power = 0.0

    def _projected_total_short_notional(
        self,
        symbol: str,
        new_qty: float,
        trade_price: float,
    ) -> float:
        total = 0.0
        for sym, qty in self.position_by_symbol.items():
            projected_qty = float(new_qty) if sym == symbol else float(qty)
            if projected_qty >= 0:
                continue
            px = trade_price if sym == symbol else self.latest_price_by_symbol.get(sym)
            if px is None or px <= 0:
                continue
            total += abs(projected_qty) * float(px)
        if symbol not in self.position_by_symbol and new_qty < 0:
            total += abs(new_qty) * trade_price
        return total

    def _current_short_notional(self) -> float:
        total = 0.0
        for symbol, qty in self.position_by_symbol.items():
            if qty >= 0:
                continue
            px = self.latest_price_by_symbol.get(symbol)
            if px is None or px <= 0:
                continue
            total += abs(qty) * float(px)
        return total

    def _opening_long_qty(self, current_position: float, qty: float) -> float:
        short_to_cover = max(0.0, -float(current_position))
        return max(0.0, float(qty) - short_to_cover)

    def _opening_short_qty(self, current_position: float, qty: float) -> float:
        long_to_close = max(0.0, float(current_position))
        return max(0.0, float(qty) - long_to_close)

    def _reservation_cost(
        self,
        *,
        side: str,
        current_position: float,
        qty: float,
        limit_price: float,
        reference_price: float,
    ) -> float:
        if not self.paper_parity.enabled or not self.paper_parity.reserve_open_orders:
            return 0.0

        opening_long_qty = (
            self._opening_long_qty(current_position=current_position, qty=qty)
            if side == "buy"
            else 0
        )
        opening_short_qty = (
            self._opening_short_qty(current_position=current_position, qty=qty)
            if side == "sell"
            else 0
        )
        if self.paper_parity.reserve_open_orders_for_shorts_only:
            opening_long_qty = 0

        long_notional = limit_price * float(opening_long_qty)
        short_notional = 0.0
        if opening_short_qty > 0:
            short_notional = short_open_unit_value(
                limit_price=limit_price,
                reference_price=reference_price,
                buffer=float(self.paper_parity.short_open_valuation_buffer),
            ) * float(opening_short_qty)
        return max(0.0, long_notional + short_notional)

    def _prepare_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        df = panel.copy()
        rename = {}
        for col in df.columns:
            low = str(col).strip().lower()
            if low in {"datetime", "timestamp", "date", "time"}:
                rename[col] = "Datetime"
            elif low in {"symbol", "ticker"}:
                rename[col] = "symbol"
            elif low == "open":
                rename[col] = "Open"
            elif low == "high":
                rename[col] = "High"
            elif low == "low":
                rename[col] = "Low"
            elif low == "close":
                rename[col] = "Close"
            elif low == "volume":
                rename[col] = "Volume"
        if rename:
            df = df.rename(columns=rename)

        required = ["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Panel data is missing required columns: {missing}")

        out = df[required].copy()
        out["Datetime"] = pd.to_datetime(out["Datetime"], utc=True, errors="coerce")
        out["symbol"] = out["symbol"].astype(str).str.upper()
        out = out.dropna(subset=["Datetime", "symbol", "Close", "Volume"])
        out = out.sort_values(["Datetime", "symbol"]).drop_duplicates(
            ["Datetime", "symbol"], keep="last"
        )
        return out.reset_index(drop=True)

    def _build_signal_panel(self) -> pd.DataFrame:
        per_symbol: List[pd.DataFrame] = []
        for symbol, sym_df in self.panel_df.groupby("symbol"):
            strategy = self.strategy_factory()
            in_df = sym_df.drop(columns=["symbol"]).reset_index(drop=True)
            out_df = strategy.run(in_df)
            keep = ["Datetime", "Close"]
            for col in ["signal", "position", "target_qty", "limit_price"]:
                if col in out_df.columns:
                    keep.append(col)
            local = out_df[keep].copy()
            local["symbol"] = symbol
            if "signal" not in local.columns:
                local["signal"] = 0
            if "target_qty" not in local.columns:
                local["target_qty"] = 0.0
            if "position" not in local.columns:
                local["position"] = 0.0
            per_symbol.append(local)

        signals = pd.concat(per_symbol, ignore_index=True)
        signals = signals.sort_values(["Datetime", "symbol"]).reset_index(drop=True)
        return signals

    def _append_history(self, chunk: pd.DataFrame) -> None:
        for _, row in chunk.iterrows():
            symbol = str(row["symbol"])
            history = self.history_by_symbol[symbol]
            local = pd.DataFrame(
                {
                    "Datetime": [row["Datetime"]],
                    "Open": [row["Open"]],
                    "High": [row["High"]],
                    "Low": [row["Low"]],
                    "Close": [row["Close"]],
                    "Volume": [row["Volume"]],
                }
            )
            merged = pd.concat([history, local], ignore_index=True)
            merged = merged.sort_values("Datetime").drop_duplicates(
                ["Datetime"], keep="last"
            )
            if len(merged) > self.max_history_rows:
                merged = merged.iloc[-self.max_history_rows :]
            self.history_by_symbol[symbol] = merged.reset_index(drop=True)

    def _history_panel(self) -> pd.DataFrame:
        rows = []
        for symbol, history in self.history_by_symbol.items():
            if history.empty:
                continue
            local = history.copy()
            local["symbol"] = symbol
            rows.append(local)
        if not rows:
            return pd.DataFrame(
                columns=["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
            )
        out = pd.concat(rows, ignore_index=True)
        return out.sort_values(["Datetime", "symbol"]).drop_duplicates(
            ["Datetime", "symbol"], keep="last"
        )

    def _apply_trade(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        signal: int,
        qty: float,
        price: float,
    ) -> None:
        if qty <= 0 or price <= 0:
            return
        current = float(self.position_by_symbol.get(symbol, 0.0))
        side = "buy" if signal > 0 else "sell"

        eligible, reason = evaluate_asset_eligibility(
            symbol=symbol,
            side=side,
            qty=qty,
            current_position=current,
            parity=self.paper_parity,
            asset_flags_by_symbol=self.asset_flags_by_symbol,
        )
        if not eligible:
            self.rejections.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "reason": reason,
                }
            )
            return

        if self.portfolio_strategy is None:
            if side == "buy" and current > 0:
                return
            if side == "sell" and current < 0:
                return

        if self.max_notional_per_order is not None:
            max_qty = float(self.max_notional_per_order) / float(price)
            qty = min(qty, max_qty)
        if qty <= 0:
            return

        if side == "buy":
            affordable_cash = self.cash
            if self.paper_parity.enabled and self.paper_parity.reserve_open_orders:
                affordable_cash -= self._batch_reserved_buying_power
            affordable = float(max(0.0, affordable_cash) / float(price))
            qty = min(qty, affordable)
            if qty <= 0:
                if self.paper_parity.enabled and self.paper_parity.reserve_open_orders:
                    self.rejections.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "side": side,
                            "qty": qty,
                            "reason": "reject_reserved_buying_power",
                        }
                    )
                return
            reservation = self._reservation_cost(
                side=side,
                current_position=current,
                qty=qty,
                limit_price=price,
                reference_price=float(self.latest_price_by_symbol.get(symbol, price)),
            )
            if reservation > 0:
                self._batch_reserved_buying_power += reservation
                self.reservation_events.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "action": "reserve",
                        "amount": reservation,
                    }
                )
            self.cash -= qty * price
            self.position_by_symbol[symbol] = current + qty
        else:
            projected_position = current - qty
            if projected_position < 0 and self.max_short_notional is not None:
                projected_short_notional = self._projected_total_short_notional(
                    symbol=symbol,
                    new_qty=projected_position,
                    trade_price=price,
                )
                if projected_short_notional > float(self.max_short_notional):
                    return

            opening_short_qty = max(0.0, float(qty) - max(0.0, float(current)))
            if opening_short_qty > 0 and self.paper_parity.enabled:
                reference_price = self.latest_price_by_symbol.get(symbol, price)
                if reference_price is None or reference_price <= 0:
                    self.rejections.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "side": side,
                            "qty": qty,
                            "reason": "reject_invalid_short_reference",
                        }
                    )
                    return
                try:
                    short_value = short_open_unit_value(
                        limit_price=price,
                        reference_price=float(reference_price),
                        buffer=float(self.paper_parity.short_open_valuation_buffer),
                    )
                except ValueError:
                    self.rejections.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "side": side,
                            "qty": qty,
                            "reason": "reject_invalid_short_valuation",
                        }
                    )
                    return

                anchor_equity = (
                    float(self.paper_parity.account_equity)
                    if self.paper_parity.account_equity is not None
                    else float(self.initial_capital)
                )
                buying_power_limit = anchor_equity * float(
                    self.paper_parity.buying_power_multiplier()
                )
                available_short_bp = max(
                    0.0,
                    buying_power_limit
                    - self._current_short_notional()
                    - (
                        self._batch_reserved_buying_power
                        if self.paper_parity.reserve_open_orders
                        else 0.0
                    ),
                )
                required_short_bp = short_value * float(opening_short_qty)
                if required_short_bp > available_short_bp:
                    self.rejections.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "side": side,
                            "qty": qty,
                            "reason": "reject_short_open_buying_power",
                        }
                    )
                    return

            reservation = self._reservation_cost(
                side=side,
                current_position=current,
                qty=qty,
                limit_price=price,
                reference_price=float(self.latest_price_by_symbol.get(symbol, price)),
            )
            if reservation > 0:
                self._batch_reserved_buying_power += reservation
                self.reservation_events.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "action": "reserve",
                        "amount": reservation,
                    }
                )
            self.cash += qty * price
            self.position_by_symbol[symbol] = projected_position

        self.trades.append(
            MultiAssetTradeRecord(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
            )
        )

    def run(self) -> pd.DataFrame:
        if self.portfolio_strategy is not None:
            for timestamp, chunk in self.panel_df.groupby("Datetime"):
                self._batch_reserved_buying_power = 0.0
                for _, row in chunk.iterrows():
                    symbol = str(row["symbol"])
                    self.latest_price_by_symbol[symbol] = float(row["Close"])

                self._append_history(chunk)
                history_panel = self._history_panel()
                if not history_panel.empty:
                    current_positions = {
                        symbol: float(qty)
                        for symbol, qty in self.position_by_symbol.items()
                        if qty != 0
                    }
                    decisions = self.portfolio_strategy.run_panel(
                        history_panel,
                        current_positions=current_positions,
                    )
                    if decisions is not None and not decisions.empty:
                        for _, row in decisions.iterrows():
                            symbol = str(row.get("symbol", "")).upper()
                            if not symbol:
                                continue
                            if symbol not in self.latest_price_by_symbol:
                                continue
                            signal_val = row.get("signal", 0)
                            signal = int(signal_val) if pd.notna(signal_val) else 0
                            if signal == 0:
                                continue

                            qty_val = row.get("target_qty", 0)
                            qty = float(qty_val) if pd.notna(qty_val) else 0.0
                            if qty <= 0:
                                continue

                            close = self.latest_price_by_symbol[symbol]
                            limit = row.get("limit_price", np.nan)
                            price = float(limit) if pd.notna(limit) else close
                            self._apply_trade(timestamp, symbol, signal, qty, price)

                if self._batch_reserved_buying_power > 0:
                    self.reservation_events.append(
                        {
                            "timestamp": timestamp,
                            "symbol": "*",
                            "action": "release",
                            "amount": self._batch_reserved_buying_power,
                        }
                    )
                    self._batch_reserved_buying_power = 0.0

                equity = self.cash
                for symbol, qty in self.position_by_symbol.items():
                    if qty == 0:
                        continue
                    price = self.latest_price_by_symbol.get(symbol)
                    if price is None:
                        continue
                    equity += qty * price

                self.timestamp_curve.append(timestamp)
                self.cash_curve.append(self.cash)
                self.equity_curve.append(equity)
        else:
            signals = self._build_signal_panel()

            for timestamp, chunk in signals.groupby("Datetime"):
                self._batch_reserved_buying_power = 0.0
                for _, row in chunk.iterrows():
                    symbol = str(row["symbol"])
                    close = float(row["Close"])
                    self.latest_price_by_symbol[symbol] = close

                    signal_val = row.get("signal", 0)
                    signal = int(signal_val) if pd.notna(signal_val) else 0
                    if signal == 0:
                        continue

                    qty_val = row.get("target_qty", 0)
                    qty = float(qty_val) if pd.notna(qty_val) else 0.0
                    if qty <= 0:
                        continue

                    limit = row.get("limit_price", np.nan)
                    price = float(limit) if pd.notna(limit) else close
                    self._apply_trade(timestamp, symbol, signal, qty, price)

                if self._batch_reserved_buying_power > 0:
                    self.reservation_events.append(
                        {
                            "timestamp": timestamp,
                            "symbol": "*",
                            "action": "release",
                            "amount": self._batch_reserved_buying_power,
                        }
                    )
                    self._batch_reserved_buying_power = 0.0

                equity = self.cash
                for symbol, qty in self.position_by_symbol.items():
                    if qty == 0:
                        continue
                    price = self.latest_price_by_symbol.get(symbol)
                    if price is None:
                        continue
                    equity += qty * price

                self.timestamp_curve.append(timestamp)
                self.cash_curve.append(self.cash)
                self.equity_curve.append(equity)

        return self.equity_frame()

    def equity_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Datetime": self.timestamp_curve,
                "equity": self.equity_curve,
                "cash": self.cash_curve,
            }
        )
