from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from alpaca_trade_api.rest import APIError

from core.logger import get_logger, get_trade_logger
from core.rate_limiter import RequestRateLimiter
from pipeline.alpaca import _parse_timeframe, get_rest
from strategies import Strategy

logger = get_logger("multi_asset_trader")


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    tf = timeframe.strip().lower()
    if tf.endswith("min"):
        return pd.Timedelta(minutes=int(tf[:-3]))
    if tf.endswith("hour"):
        return pd.Timedelta(hours=int(tf[:-4]))
    if tf.endswith("h") and tf[:-1].isdigit():
        return pd.Timedelta(hours=int(tf[:-1]))
    if tf.endswith("day"):
        return pd.Timedelta(days=int(tf[:-3]))
    if tf.endswith("d") and tf[:-1].isdigit():
        return pd.Timedelta(days=int(tf[:-1]))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _chunked(values: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    return [values[i : i + size] for i in range(0, len(values), size)]


@dataclass
class MultiTradeDecision:
    symbol: str
    side: str
    qty: float
    price: float
    order_type: str
    limit_price: Optional[float] = None


class MultiAssetAlpacaTrader:
    """
    Multi-symbol paper trader with batched market-data calls and API rate limiting.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframe: str,
        lookback: int,
        strategy_factory: Callable[[], Strategy],
        feed: Optional[str] = None,
        dry_run: bool = False,
        max_order_notional: Optional[float] = None,
        max_api_requests_per_minute: int = 190,
        batch_size: int = 100,
        max_orders_per_cycle: int = 25,
        api: Optional[tradeapi.REST] = None,
    ):
        if not symbols:
            raise ValueError("At least one symbol is required for multi-asset mode.")
        if lookback <= 0:
            raise ValueError("lookback must be positive")

        self.symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.timeframe = timeframe
        self.lookback = int(lookback)
        self.feed = feed or "iex"
        self.dry_run = dry_run
        self.max_order_notional = (
            float(max_order_notional) if max_order_notional is not None else None
        )
        self.batch_size = int(batch_size)
        self.max_orders_per_cycle = int(max_orders_per_cycle)

        self.api = api or get_rest()
        self.rate_limiter = RequestRateLimiter(
            max_requests_per_minute=max_api_requests_per_minute
        )
        self.trade_logger = get_trade_logger()

        strategy_probe = strategy_factory()
        self.portfolio_strategy: Optional[Strategy] = None
        self.strategy_by_symbol: Dict[str, Strategy] = {}
        if hasattr(strategy_probe, "run_panel"):
            self.portfolio_strategy = strategy_probe
            self.required_lookback = max(
                self.lookback,
                int(getattr(strategy_probe, "required_lookback", self.lookback)),
            )
        else:
            self.strategy_by_symbol = {
                symbol: strategy_factory() for symbol in self.symbols
            }
            self.required_lookback = self.lookback
            for strategy in self.strategy_by_symbol.values():
                strategy_lb = getattr(strategy, "required_lookback", self.lookback)
                self.required_lookback = max(self.required_lookback, int(strategy_lb))

        self.timeframe_delta = _timeframe_to_timedelta(timeframe)
        self.max_history_rows = self.required_lookback + 20
        self.history_by_symbol: Dict[str, pd.DataFrame] = {
            symbol: pd.DataFrame(
                columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
            )
            for symbol in self.symbols
        }

        self.last_fetch_end: Optional[pd.Timestamp] = None
        self.starting_equity = self._get_equity()
        self.latest_equity = self.starting_equity

        logger.info(
            "Initialized multi-asset trader: symbols=%s timeframe=%s lookback=%s dry_run=%s portfolio_mode=%s",
            len(self.symbols),
            self.timeframe,
            self.required_lookback,
            self.dry_run,
            bool(self.portfolio_strategy),
        )

    # ------------------------------------------------------------------ api

    def _api_call(self, func, *args, **kwargs):
        self.rate_limiter.wait_for_slot()
        return func(*args, **kwargs)

    def _get_equity(self) -> float:
        account = self._api_call(self.api.get_account)
        return float(account.equity)

    # ------------------------------------------------------------------ data

    def _fetch_batch_bars(
        self, symbols: List[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        tf = str(_parse_timeframe(self.timeframe))
        page_token: Optional[str] = None
        chunks: List[pd.DataFrame] = []

        while True:
            data: dict[str, object] = {
                "symbols": ",".join(symbols),
                "timeframe": tf,
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "limit": 10_000,
            }
            if page_token:
                data["page_token"] = page_token

            resp = self._api_call(
                self.api.data_get,
                "/stocks/bars",
                data=data,
                feed=self.feed,
                api_version="v2",
            )
            bars_by_symbol = resp.get("bars", {}) if isinstance(resp, dict) else {}

            rows: List[dict] = []
            for symbol in symbols:
                for item in bars_by_symbol.get(symbol, []):
                    rows.append(
                        {
                            "Datetime": item.get("t"),
                            "symbol": symbol,
                            "Open": item.get("o"),
                            "High": item.get("h"),
                            "Low": item.get("l"),
                            "Close": item.get("c"),
                            "Volume": item.get("v"),
                        }
                    )

            if rows:
                chunk = pd.DataFrame.from_records(rows)
                chunk["Datetime"] = pd.to_datetime(
                    chunk["Datetime"], utc=True, errors="coerce"
                )
                chunk = chunk.dropna(
                    subset=["Datetime", "Close", "Volume"]
                )  # pragma: no branch
                chunks.append(chunk)

            page_token = resp.get("next_page_token") if isinstance(resp, dict) else None
            if not page_token:
                break

        if not chunks:
            return pd.DataFrame(
                columns=["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
            )

        out = pd.concat(chunks, ignore_index=True)
        out["symbol"] = out["symbol"].astype(str).str.upper()
        out = out.sort_values(["Datetime", "symbol"]).drop_duplicates(
            ["Datetime", "symbol"], keep="last"
        )
        return out

    def _fetch_market_panel(self) -> pd.DataFrame:
        end = pd.Timestamp.now(tz="UTC").floor("min")
        if self.last_fetch_end is None:
            start = end - self.timeframe_delta * (self.required_lookback + 2)
        else:
            start = self.last_fetch_end - self.timeframe_delta

        chunks = _chunked(self.symbols, self.batch_size)
        panel_parts: List[pd.DataFrame] = []
        for batch in chunks:
            batch_df = self._fetch_batch_bars(batch, start=start, end=end)
            if not batch_df.empty:
                panel_parts.append(batch_df)

        self.last_fetch_end = end
        if not panel_parts:
            return pd.DataFrame(
                columns=["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
            )

        panel = pd.concat(panel_parts, ignore_index=True)
        panel = panel.sort_values(["Datetime", "symbol"]).drop_duplicates(
            ["Datetime", "symbol"], keep="last"
        )
        return panel

    def _merge_history(self, panel: pd.DataFrame) -> None:
        if panel.empty:
            return
        for symbol, chunk in panel.groupby("symbol"):
            history = self.history_by_symbol.get(symbol)
            if history is None:
                continue
            merged = pd.concat(
                [history, chunk.drop(columns=["symbol"])], ignore_index=True
            )
            merged = merged.sort_values("Datetime").drop_duplicates(
                ["Datetime"], keep="last"
            )
            if len(merged) > self.max_history_rows:
                merged = merged.iloc[-self.max_history_rows :]
            self.history_by_symbol[symbol] = merged.reset_index(drop=True)

    def _history_panel(self) -> pd.DataFrame:
        rows: List[pd.DataFrame] = []
        for symbol in self.symbols:
            history = self.history_by_symbol.get(symbol)
            if history is None or history.empty:
                continue
            local = history.copy()
            local["symbol"] = symbol
            rows.append(local)
        if not rows:
            return pd.DataFrame(
                columns=["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
            )
        out = pd.concat(rows, ignore_index=True)
        out = out.sort_values(["Datetime", "symbol"]).drop_duplicates(
            ["Datetime", "symbol"], keep="last"
        )
        return out

    # ---------------------------------------------------------------- decision

    def _list_position_map(self) -> Dict[str, float]:
        positions = self._api_call(self.api.list_positions)
        out: Dict[str, float] = {}
        for pos in positions:
            symbol = str(getattr(pos, "symbol", "")).upper()
            if not symbol:
                continue
            qty = float(getattr(pos, "qty", 0.0))
            if getattr(pos, "side", "long") == "short":
                qty = -qty
            out[symbol] = qty
        return out

    def _list_open_order_symbols(self) -> set[str]:
        orders = self._api_call(self.api.list_orders, status="open")
        return {str(getattr(order, "symbol", "")).upper() for order in orders}

    @staticmethod
    def _normalize_stock_limit_price(price: float) -> float:
        decimals = 4 if price < 1.0 else 2
        return float(f"{price:.{decimals}f}")

    def _build_decision(
        self, symbol: str, latest: pd.Series, net_position: float
    ) -> tuple[Optional[MultiTradeDecision], Optional[str]]:
        signal_value = latest.get("signal", 0)
        signal = int(signal_value) if pd.notna(signal_value) else 0
        position_value = latest.get("position", None)
        position = float(position_value) if pd.notna(position_value) else None

        close_value = latest.get("Close", 0)
        price = float(close_value) if pd.notna(close_value) else 0.0
        if price <= 0:
            return None, "missing price"

        limit_value = latest.get("limit_price", None)
        limit_price = float(limit_value) if pd.notna(limit_value) else None
        if limit_price is not None:
            limit_price = self._normalize_stock_limit_price(limit_price)
            if limit_price <= 0:
                return None, "invalid limit price"
        order_type = "limit" if limit_price is not None else "market"
        price_for_qty = limit_price if limit_price is not None else price

        qty_value = latest.get("target_qty", 0)
        qty = float(qty_value) if pd.notna(qty_value) else 0.0
        qty = float(int(max(0.0, qty)))
        if qty <= 0:
            return None, "target_qty is zero"

        if signal != 0:
            side = "buy" if signal > 0 else "sell"
        elif position is not None and position != 0:
            side = "buy" if position > 0 else "sell"
        else:
            return None, "no signal"

        if side == "buy" and net_position > 0:
            return None, "already long"
        if side == "sell" and net_position < 0:
            return None, "already short"

        if self.max_order_notional is not None and price_for_qty > 0:
            max_qty = float(int(self.max_order_notional / price_for_qty))
            qty = min(qty, max_qty)
        if qty <= 0:
            return None, "quantity too small after notional cap"

        return (
            MultiTradeDecision(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                order_type=order_type,
                limit_price=limit_price,
            ),
            None,
        )

    def _build_decision_from_order_row(
        self,
        order_row: pd.Series,
        position_map: Dict[str, float],
    ) -> tuple[Optional[MultiTradeDecision], Optional[str]]:
        symbol = str(order_row.get("symbol", "")).upper().strip()
        if not symbol or symbol not in self.history_by_symbol:
            return None, "unknown symbol"
        history = self.history_by_symbol[symbol]
        if history.empty:
            return None, "missing symbol history"
        close_price = float(history.iloc[-1]["Close"])
        synthetic = pd.Series(
            {
                "signal": order_row.get("signal", 0),
                "target_qty": order_row.get("target_qty", 0),
                "limit_price": order_row.get("limit_price", np.nan),
                "Close": close_price,
            }
        )
        return self._build_decision(symbol, synthetic, position_map.get(symbol, 0.0))

    def _submit_order(self, decision: MultiTradeDecision) -> str:
        if self.dry_run:
            return "dry_run"

        kwargs = {"type": decision.order_type, "time_in_force": "day"}
        if decision.order_type == "limit" and decision.limit_price is not None:
            kwargs["limit_price"] = self._normalize_stock_limit_price(
                float(decision.limit_price)
            )

        order = self._api_call(
            self.api.submit_order,
            symbol=decision.symbol,
            qty=int(decision.qty),
            side=decision.side,
            **kwargs,
        )
        return str(order.id)

    # ------------------------------------------------------------------ public

    def run_once(self) -> Optional[pd.DataFrame]:
        panel = self._fetch_market_panel()
        if panel.empty:
            logger.debug("No bars returned in this cycle.")
            return panel

        self._merge_history(panel)
        open_order_symbols = self._list_open_order_symbols()
        position_map = self._list_position_map()

        decisions: List[MultiTradeDecision] = []
        if self.portfolio_strategy is not None:
            history_panel = self._history_panel()
            if not history_panel.empty:
                order_df = self.portfolio_strategy.run_panel(
                    history_panel,
                    current_positions=position_map,
                )
                if order_df is not None and not order_df.empty:
                    for _, row in order_df.iterrows():
                        symbol = str(row.get("symbol", "")).upper()
                        if not symbol or symbol in open_order_symbols:
                            continue
                        decision, reason = self._build_decision_from_order_row(
                            row,
                            position_map,
                        )
                        if decision is None:
                            if reason:
                                logger.debug("No trade %s (%s)", symbol, reason)
                            continue
                        decisions.append(decision)
        else:
            for symbol in self.symbols:
                if symbol in open_order_symbols:
                    continue

                history = self.history_by_symbol[symbol]
                if len(history) < self.required_lookback:
                    continue

                strategy = self.strategy_by_symbol[symbol]
                signals_df = strategy.run(history)
                if signals_df.empty:
                    continue
                latest = signals_df.iloc[-1]

                decision, reason = self._build_decision(
                    symbol, latest, position_map.get(symbol, 0.0)
                )
                if decision is None:
                    if reason:
                        logger.debug("No trade %s (%s)", symbol, reason)
                    continue
                decisions.append(decision)

        if not decisions:
            return panel

        decisions.sort(key=lambda d: d.qty * (d.limit_price or d.price), reverse=True)
        if self.max_orders_per_cycle > 0:
            decisions = decisions[: self.max_orders_per_cycle]

        try:
            self.latest_equity = self._get_equity()
        except Exception:
            pass

        for decision in decisions:
            try:
                order_id = self._submit_order(decision)
            except APIError as exc:
                logger.warning("Order rejected for %s: %s", decision.symbol, exc)
                continue

            print_price = (
                decision.limit_price
                if decision.limit_price is not None
                else decision.price
            )
            net_pnl = self.latest_equity - self.starting_equity
            self.trade_logger.log_trade(
                symbol=decision.symbol,
                side=decision.side,
                qty=decision.qty,
                price=print_price,
                order_type=decision.order_type,
                order_id=order_id,
                status="submitted" if order_id != "dry_run" else "dry_run",
                equity=self.latest_equity,
                net_pnl=net_pnl,
                strategy=(
                    self.portfolio_strategy.__class__.__name__
                    if self.portfolio_strategy is not None
                    else self.strategy_by_symbol[decision.symbol].__class__.__name__
                ),
            )
            logger.info(
                "%s %s %s @ %.2f | order_id=%s | equity=$%.2f | net_pnl=%+.2f",
                decision.side.upper(),
                int(decision.qty),
                decision.symbol,
                print_price,
                order_id,
                self.latest_equity,
                net_pnl,
            )

        return panel
