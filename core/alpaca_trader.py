from __future__ import annotations

import time
from dataclasses import dataclass
import math
import os
from typing import Optional, Tuple

import alpaca_trade_api as tradeapi
import pandas as pd
from alpaca_trade_api.rest import APIError

from core.logger import get_logger, get_trade_logger
from pipeline.alpaca import fetch_crypto_bars, fetch_stock_bars, get_rest
from strategies import Strategy

logger = get_logger("alpaca_trader")


@dataclass
class TradeDecision:
    side: str
    qty: float
    price: float
    order_type: str
    limit_price: Optional[float] = None


CRYPTO_QUOTE_SUFFIXES = ("USDT", "USDC", "USD")


def normalize_crypto_symbols(symbol: str) -> tuple[str, str]:
    sym = symbol.strip().upper()
    if not sym:
        return sym, sym
    sym = sym.replace("-", "/").replace("_", "/")
    if "/" in sym:
        base, quote = sym.split("/", 1)
        if not base or not quote:
            return sym, sym
        trade_symbol = f"{base}{quote}"
        data_symbol = f"{base}/{quote}"
        return trade_symbol, data_symbol
    for quote in CRYPTO_QUOTE_SUFFIXES:
        if sym.endswith(quote) and len(sym) > len(quote):
            base = sym[: -len(quote)]
            return f"{base}{quote}", f"{base}/{quote}"
    return sym, sym


class AlpacaTrader:
    """
    Simple paper-trading loop that uses Alpaca for data and orders.
    """

    def __init__(
        self,
        symbol: str,
        asset_class: str,
        timeframe: str,
        lookback: int,
        strategy: Strategy,
        feed: Optional[str] = None,
        dry_run: bool = False,
        max_order_notional: Optional[float] = None,
        api: Optional[tradeapi.REST] = None,
    ):
        asset_class = asset_class.lower()
        if asset_class not in {"stock", "crypto"}:
            raise ValueError("asset_class must be 'stock' or 'crypto'.")

        self.asset_class = asset_class
        if asset_class == "crypto":
            trade_symbol, data_symbol = normalize_crypto_symbols(symbol)
            if "/" not in data_symbol:
                raise ValueError(
                    "Crypto symbols must include a quote currency, e.g. BTC/USD or BTCUSD."
                )
            self.symbol = trade_symbol
            self.data_symbol = data_symbol
            self.display_symbol = data_symbol
        else:
            self.symbol = symbol.upper()
            self.data_symbol = self.symbol
            self.display_symbol = self.symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.strategy = strategy
        self.strategy_name = strategy.__class__.__name__
        self.feed = feed
        self.dry_run = dry_run
        self.max_order_notional = None
        if max_order_notional is not None:
            self.max_order_notional = float(max_order_notional)
        else:
            env_value = os.environ.get("ALPACA_MAX_ORDER_NOTIONAL") or os.environ.get(
                "ALPACA_MAX_NOTIONAL"
            )
            if env_value:
                try:
                    self.max_order_notional = float(env_value)
                except ValueError:
                    self.max_order_notional = None
        self.api = api or get_rest()
        self.trade_logger = get_trade_logger()
        self.starting_equity = self._get_equity()

        logger.info(
            f"Initialized trader: {self.display_symbol} | strategy={self.strategy_name} | equity=${self.starting_equity:,.2f}"
        )

    def _get_equity(self) -> float:
        account = self.api.get_account()
        return float(account.equity)

    def _get_net_position(self) -> float:
        try:
            position = self.api.get_position(self.symbol)
        except APIError as exc:
            if getattr(exc, "status_code", None) == 404:
                return 0.0
            raise

        qty = float(position.qty)
        if getattr(position, "side", "long") == "short":
            qty = -qty
        return qty

    def _has_open_order(self) -> bool:
        orders = self.api.list_orders(status="open", symbols=[self.symbol])
        return len(orders) > 0

    def fetch_latest_bars(self) -> pd.DataFrame:
        if self.asset_class == "crypto":
            return fetch_crypto_bars(
                self.data_symbol,
                timeframe=self.timeframe,
                limit=self.lookback,
                api=self.api,
            )
        return fetch_stock_bars(
            self.symbol,
            timeframe=self.timeframe,
            limit=self.lookback,
            feed=self.feed,
            api=self.api,
        )

    def _format_qty(self, qty: float) -> str:
        if self.asset_class == "crypto":
            return f"{qty:.6f}".rstrip("0").rstrip(".")
        return str(int(qty))

    @staticmethod
    def _normalize_stock_limit_price(price: float) -> float:
        decimals = 4 if price < 1.0 else 2
        return float(f"{price:.{decimals}f}")

    def _build_decision(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[TradeDecision], Optional[str]]:
        if df is None or df.empty:
            return None, "no market data returned"

        signals_df = self.strategy.run(df)
        if signals_df.empty:
            return None, "strategy returned no rows"
        latest = signals_df.iloc[-1]
        signal_value = latest.get("signal", 0)
        signal = int(signal_value) if pd.notna(signal_value) else 0
        position_value = latest.get("position", None)
        position = float(position_value) if pd.notna(position_value) else None

        raw_price = latest.get("Close", 0.0)
        price = float(raw_price) if pd.notna(raw_price) else 0.0
        if price <= 0:
            return None, "missing price data"

        limit_value = latest.get("limit_price", None)
        limit_price = float(limit_value) if pd.notna(limit_value) else None
        if self.asset_class == "stock" and limit_price is not None:
            limit_price = self._normalize_stock_limit_price(limit_price)
            if limit_price <= 0:
                return None, "invalid limit price"
        order_type = "limit" if limit_price is not None else "market"
        price_for_qty = limit_price if limit_price is not None else price
        if price_for_qty <= 0:
            return None, "invalid price for sizing"

        qty_value = latest.get("target_qty", 0)
        qty = float(qty_value) if pd.notna(qty_value) else 0.0
        if qty <= 0:
            return None, "target_qty is zero"
        if self.asset_class == "crypto":
            # Interpret target_qty as USD notional for crypto and convert to units.
            qty = qty / price_for_qty
            qty = math.floor(qty * 1_000_000) / 1_000_000
            if qty <= 0:
                return None, "target_qty too small for crypto price"

        if signal != 0:
            side = "buy" if signal > 0 else "sell"
        elif position is not None and position != 0:
            side = "buy" if position > 0 else "sell"
        else:
            return None, "no signal from strategy"
        return (
            TradeDecision(
                side=side,
                qty=qty,
                price=price,
                order_type=order_type,
                limit_price=limit_price,
            ),
            None,
        )

    def _adjust_qty_for_position(
        self, decision: TradeDecision, net_position: float
    ) -> Tuple[float, Optional[str]]:
        qty = decision.qty

        # Simple logic: just trade the requested quantity
        # Skip if we'd be doubling down in same direction
        if decision.side == "buy" and net_position > 0:
            return 0.0, "already long"
        if decision.side == "sell" and net_position < 0:
            return 0.0, "already short"

        # For crypto, don't allow shorting
        if (
            self.asset_class == "crypto"
            and decision.side == "sell"
            and net_position <= 0
        ):
            return 0.0, "crypto shorting disabled"

        if self.asset_class == "stock":
            qty = float(int(qty))
            if qty <= 0:
                return 0.0, "share quantity too small"
        return qty, None

    def _cap_qty_for_notional(self, decision: TradeDecision, qty: float) -> float:
        if qty <= 0:
            return qty
        if self.max_order_notional is None:
            return qty
        price = (
            decision.limit_price if decision.order_type == "limit" else decision.price
        )
        if price <= 0:
            return qty
        max_qty = self.max_order_notional / price
        if self.asset_class == "stock":
            max_qty = float(int(max_qty))
        else:
            max_qty = math.floor(max_qty * 1_000_000) / 1_000_000
        if max_qty <= 0:
            return 0.0
        return min(qty, max_qty)

    def _submit_order(self, decision: TradeDecision, qty: float) -> Optional[str]:
        tif = "gtc" if self.asset_class == "crypto" else "day"
        order_kwargs = {"type": decision.order_type, "time_in_force": tif}
        if decision.order_type == "limit":
            limit_price = (
                float(decision.limit_price)
                if decision.limit_price is not None
                else None
            )
            if self.asset_class == "stock" and limit_price is not None:
                limit_price = self._normalize_stock_limit_price(limit_price)
            order_kwargs["limit_price"] = limit_price

        if self.dry_run:
            return "dry_run"

        qty_to_send = int(qty) if self.asset_class == "stock" else qty
        order = self.api.submit_order(
            symbol=self.symbol,
            qty=qty_to_send,
            side=decision.side,
            **order_kwargs,
        )
        return order.id

    def _print_trade(self, decision: TradeDecision, qty: float, order_id: str) -> None:
        equity = self._get_equity()
        net_pnl = equity - self.starting_equity
        qty_display = self._format_qty(qty)
        print_price = (
            decision.limit_price if decision.order_type == "limit" else decision.price
        )
        price_display = f"{print_price:.2f}" if print_price else "market"

        # Log to file
        self.trade_logger.log_trade(
            symbol=self.display_symbol,
            side=decision.side,
            qty=qty,
            price=print_price or decision.price,
            order_type=decision.order_type,
            order_id=order_id,
            status="submitted" if order_id != "dry_run" else "dry_run",
            equity=equity,
            net_pnl=net_pnl,
            strategy=self.strategy_name,
        )

        # Log to console and system log
        msg = (
            f"{decision.side.upper()} {qty_display} {self.display_symbol} @ {price_display} "
            f"| order_id={order_id} | equity=${equity:,.2f} | net_pnl={net_pnl:+.2f}"
        )
        logger.info(msg)

    def _skip_trade(self, reason: str) -> None:
        logger.debug(f"No trade ({self.display_symbol}): {reason}")

    def run_once(self) -> Optional[pd.DataFrame]:
        try:
            df = self.fetch_latest_bars()
        except ValueError as exc:
            self._skip_trade(str(exc))
            return None
        decision, reason = self._build_decision(df)
        if decision is None:
            if reason:
                self._skip_trade(reason)
            return df

        if self._has_open_order():
            self._skip_trade("open order already exists")
            return df

        net_position = self._get_net_position()
        qty, reason = self._adjust_qty_for_position(decision, net_position)
        if qty <= 0:
            if reason:
                self._skip_trade(reason)
            return df
        qty = self._cap_qty_for_notional(decision, qty)
        if qty <= 0:
            self._skip_trade("quantity too small after notional cap")
            return df

        try:
            order_id = self._submit_order(decision, qty)
        except APIError as exc:
            self._skip_trade(f"order rejected: {exc}")
            return df
        self._print_trade(decision, qty, order_id or "unknown")
        return df

    def run(self, iterations: int = 1, sleep_seconds: int = 60) -> None:
        for i in range(iterations):
            self.run_once()
            if i < iterations - 1:
                time.sleep(sleep_seconds)
