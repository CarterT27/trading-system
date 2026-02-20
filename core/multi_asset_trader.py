from __future__ import annotations

from dataclasses import dataclass, replace
import re
import time
from typing import Callable, Dict, List, Optional

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from alpaca_trade_api.rest import APIError

from core.logger import get_logger, get_trade_logger
from core.rate_limiter import RequestRateLimiter
from pipeline.alpaca import _parse_timeframe, fetch_crypto_bars, get_rest
from core.alpaca_trader import normalize_crypto_symbols
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
        asset_class: str,
        timeframe: str,
        lookback: int,
        strategy_factory: Callable[[], Strategy],
        feed: Optional[str] = None,
        dry_run: bool = False,
        max_order_notional: Optional[float] = None,
        max_api_requests_per_minute: int = 190,
        batch_size: int = 100,
        max_orders_per_cycle: int = 25,
        data_fetch_retries: int = 3,
        data_fetch_backoff_seconds: float = 0.75,
        buying_power_buffer: float = 0.05,
        buying_power_cooldown_cycles: int = 3,
        api: Optional[tradeapi.REST] = None,
    ):
        if not symbols:
            raise ValueError("At least one symbol is required for multi-asset mode.")
        if lookback <= 0:
            raise ValueError("lookback must be positive")

        asset_class = str(asset_class).strip().lower()
        if asset_class not in {"stock", "crypto"}:
            raise ValueError("asset_class must be 'stock' or 'crypto'.")
        self.asset_class = asset_class

        raw_symbols = [s.strip().upper() for s in symbols if s.strip()]
        self.symbols: List[str] = []
        self.trade_symbol_by_symbol: Dict[str, str] = {}
        self.symbol_by_trade_symbol: Dict[str, str] = {}
        for sym in raw_symbols:
            if self.asset_class == "crypto":
                trade_symbol, data_symbol = normalize_crypto_symbols(sym)
                data_symbol = data_symbol.upper()
                trade_symbol = trade_symbol.upper()
                if "/" not in data_symbol:
                    raise ValueError(
                        f"Crypto symbol '{sym}' must include quote currency (e.g., BTC/USD or BTCUSD)."
                    )
            else:
                data_symbol = sym
                trade_symbol = sym

            if data_symbol not in self.trade_symbol_by_symbol:
                self.symbols.append(data_symbol)
            self.trade_symbol_by_symbol[data_symbol] = trade_symbol
            self.symbol_by_trade_symbol[trade_symbol] = data_symbol

        self.timeframe = timeframe
        self.lookback = int(lookback)
        self.feed = feed or ("iex" if self.asset_class == "stock" else None)
        self.dry_run = dry_run
        self.max_order_notional = (
            float(max_order_notional) if max_order_notional is not None else None
        )
        self.batch_size = int(batch_size)
        self.max_orders_per_cycle = int(max_orders_per_cycle)
        self.data_fetch_retries = max(1, int(data_fetch_retries))
        self.data_fetch_backoff_seconds = max(0.0, float(data_fetch_backoff_seconds))
        self.buying_power_buffer = min(max(0.0, float(buying_power_buffer)), 0.50)
        self.buying_power_cooldown_cycles = max(1, int(buying_power_cooldown_cycles))

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
        self.iteration_count = 0
        self.buying_power_cooldown_until: Dict[str, int] = {}
        self.starting_equity = self._get_equity()
        self.latest_equity = self.starting_equity

        logger.info(
            "Initialized multi-asset trader: asset_class=%s symbols=%s timeframe=%s lookback=%s dry_run=%s portfolio_mode=%s",
            self.asset_class,
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

    def _get_account_snapshot(self) -> tuple[float, float]:
        account = self._api_call(self.api.get_account)
        equity = float(getattr(account, "equity", 0.0) or 0.0)
        buying_power_raw = getattr(account, "buying_power", 0.0)
        try:
            buying_power = float(buying_power_raw)
        except (TypeError, ValueError):
            buying_power = 0.0
        return equity, max(0.0, buying_power)

    def _data_api_call_with_retries(self, endpoint: str, data: dict) -> dict:
        attempts = self.data_fetch_retries
        for attempt in range(1, attempts + 1):
            try:
                return self._api_call(
                    self.api.data_get,
                    endpoint,
                    data=data,
                    feed=self.feed,
                    api_version="v2",
                )
            except Exception as exc:
                if attempt >= attempts:
                    raise
                sleep_for = self.data_fetch_backoff_seconds * attempt
                logger.warning(
                    "Data fetch failed (attempt %s/%s): %s; retrying in %.2fs",
                    attempt,
                    attempts,
                    exc,
                    sleep_for,
                )
                if sleep_for > 0:
                    time.sleep(sleep_for)

    # ------------------------------------------------------------------ data

    def _fetch_batch_bars(
        self, symbols: List[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        if self.asset_class != "stock":
            raise RuntimeError("_fetch_batch_bars is only valid for stock mode.")
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

            resp = self._data_api_call_with_retries("/stocks/bars", data)
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
        if self.asset_class == "crypto":
            panel_parts: List[pd.DataFrame] = []
            for symbol in self.symbols:
                try:
                    bars = fetch_crypto_bars(
                        symbol,
                        timeframe=self.timeframe,
                        limit=self.required_lookback,
                        api=self.api,
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping %s after fetch errors: %s",
                        symbol,
                        exc,
                    )
                    continue
                if bars.empty:
                    continue
                local = bars.copy()
                local["symbol"] = symbol
                panel_parts.append(
                    local[
                        ["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
                    ].copy()
                )

            if not panel_parts:
                return pd.DataFrame(
                    columns=["Datetime", "symbol", "Open", "High", "Low", "Close", "Volume"]
                )
            panel = pd.concat(panel_parts, ignore_index=True)
            panel = panel.sort_values(["Datetime", "symbol"]).drop_duplicates(
                ["Datetime", "symbol"], keep="last"
            )
            return panel

        end = pd.Timestamp.now(tz="UTC").floor("min")
        if self.last_fetch_end is None:
            start = end - self.timeframe_delta * (self.required_lookback + 2)
        else:
            start = self.last_fetch_end - self.timeframe_delta

        chunks = _chunked(self.symbols, self.batch_size)
        panel_parts: List[pd.DataFrame] = []
        for batch in chunks:
            try:
                batch_df = self._fetch_batch_bars(batch, start=start, end=end)
            except Exception as exc:
                logger.warning(
                    "Skipping batch %s..%s (%s symbols) after repeated fetch errors: %s",
                    batch[0],
                    batch[-1],
                    len(batch),
                    exc,
                )
                continue
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
            trade_symbol = str(getattr(pos, "symbol", "")).upper()
            if not trade_symbol:
                continue
            symbol = self.symbol_by_trade_symbol.get(trade_symbol, trade_symbol)
            qty = float(getattr(pos, "qty", 0.0))
            if getattr(pos, "side", "long") == "short":
                qty = -qty
            out[symbol] = qty
        return out

    def _list_open_order_symbols(self) -> set[str]:
        orders = self._api_call(self.api.list_orders, status="open")
        out: set[str] = set()
        for order in orders:
            trade_symbol = str(getattr(order, "symbol", "")).upper()
            if not trade_symbol:
                continue
            out.add(self.symbol_by_trade_symbol.get(trade_symbol, trade_symbol))
        return out

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
        if self.asset_class == "stock" and limit_price is not None:
            limit_price = self._normalize_stock_limit_price(limit_price)
            if limit_price <= 0:
                return None, "invalid limit price"
        order_type = "limit" if limit_price is not None else "market"
        price_for_qty = limit_price if limit_price is not None else price

        qty_value = latest.get("target_qty", 0)
        qty = float(qty_value) if pd.notna(qty_value) else 0.0
        qty = max(0.0, qty)
        if self.asset_class == "stock":
            qty = float(int(qty))
        else:
            qty = np.floor(qty * 1_000_000.0) / 1_000_000.0
        if qty <= 0:
            return None, "target_qty is zero"

        if signal != 0:
            side = "buy" if signal > 0 else "sell"
        elif position is not None and position != 0:
            side = "buy" if position > 0 else "sell"
        else:
            return None, "no signal"

        if self.portfolio_strategy is None:
            if side == "buy" and net_position > 0:
                return None, "already long"
            if side == "sell" and net_position < 0:
                return None, "already short"

        if self.asset_class == "crypto" and side == "sell" and net_position <= 0:
            return None, "crypto shorting disabled"

        if self.max_order_notional is not None and price_for_qty > 0:
            max_qty = float(self.max_order_notional) / float(price_for_qty)
            if self.asset_class == "stock":
                max_qty = float(int(max_qty))
            else:
                max_qty = np.floor(max_qty * 1_000_000.0) / 1_000_000.0
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

        qty = float(max(0.0, decision.qty))
        if self.asset_class == "stock":
            qty = float(int(qty))
        else:
            qty = np.floor(qty * 1_000_000.0) / 1_000_000.0
        if qty <= 0:
            raise ValueError("Order quantity must be positive.")

        kwargs = {
            "type": decision.order_type,
            "time_in_force": "gtc" if self.asset_class == "crypto" else "day",
        }
        if decision.order_type == "limit" and decision.limit_price is not None:
            limit_price = float(decision.limit_price)
            if self.asset_class == "stock":
                limit_price = self._normalize_stock_limit_price(limit_price)
            kwargs["limit_price"] = limit_price

        trade_symbol = self.trade_symbol_by_symbol.get(decision.symbol, decision.symbol)
        order = self._api_call(
            self.api.submit_order,
            symbol=trade_symbol,
            qty=int(qty) if self.asset_class == "stock" else float(qty),
            side=decision.side,
            **kwargs,
        )
        return str(order.id)

    @staticmethod
    def _extract_available_qty_from_error(exc: Exception) -> Optional[float]:
        text = str(exc)
        match = re.search(
            r"insufficient qty available for order \(requested:\s*([0-9]+(?:\.[0-9]+)?),\s*available:\s*([0-9]+(?:\.[0-9]+)?)\)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return float(match.group(2))
        if "insufficient qty available" not in text.lower():
            return None
        fallback = re.search(
            r"available:\s*([0-9]+(?:\.[0-9]+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if fallback:
            return float(fallback.group(1))
        return None

    @staticmethod
    def _is_insufficient_buying_power_error(exc: Exception) -> bool:
        return "insufficient buying power" in str(exc).lower()

    @staticmethod
    def _effective_price(decision: MultiTradeDecision) -> float:
        if decision.limit_price is not None:
            return float(decision.limit_price)
        return float(decision.price)

    def _cap_qty_for_buying_power(
        self,
        decision: MultiTradeDecision,
        position_qty: float,
        remaining_buying_power: float,
    ) -> tuple[float, float]:
        qty = float(max(0.0, decision.qty))
        if self.asset_class == "stock":
            qty = float(int(qty))
        else:
            qty = np.floor(qty * 1_000_000.0) / 1_000_000.0
        price = self._effective_price(decision)
        if qty <= 0 or price <= 0:
            return 0.0, 0.0

        affordable_incremental = max(0.0, float(remaining_buying_power) / float(price))
        if self.asset_class == "stock":
            affordable_incremental = float(int(affordable_incremental))
        else:
            affordable_incremental = np.floor(affordable_incremental * 1_000_000.0) / 1_000_000.0
        if decision.side == "buy":
            flatten_qty = min(qty, max(0.0, -position_qty))
        else:
            flatten_qty = min(qty, max(0.0, position_qty))

        capped_qty = min(qty, flatten_qty + affordable_incremental)
        capped_qty = max(0.0, float(capped_qty))
        if self.asset_class == "stock":
            capped_qty = float(int(capped_qty))
        else:
            capped_qty = np.floor(capped_qty * 1_000_000.0) / 1_000_000.0
        if capped_qty <= 0:
            return 0.0, 0.0

        incremental_qty = max(0.0, capped_qty - flatten_qty)
        reserved_bp = incremental_qty * price
        return capped_qty, reserved_bp

    # ------------------------------------------------------------------ public

    def preload_history(self) -> pd.DataFrame:
        started = time.monotonic()
        logger.info(
            "Preloading history window: symbols=%s lookback=%s",
            len(self.symbols),
            self.required_lookback,
        )
        panel = self._fetch_market_panel()
        if panel.empty:
            logger.warning("Startup preload returned no bars.")
            return panel
        self._merge_history(panel)
        logger.info(
            "Startup preload complete: rows=%s elapsed=%.1fs",
            len(panel),
            time.monotonic() - started,
        )
        return panel

    def run_once(self) -> Optional[pd.DataFrame]:
        self.iteration_count += 1
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

        filtered_decisions: List[MultiTradeDecision] = []
        for decision in decisions:
            cooldown_until = self.buying_power_cooldown_until.get(decision.symbol, 0)
            if cooldown_until >= self.iteration_count:
                logger.debug(
                    "No trade %s (buying-power cooldown for %s more cycles)",
                    decision.symbol,
                    cooldown_until - self.iteration_count + 1,
                )
                continue
            filtered_decisions.append(decision)
        decisions = filtered_decisions
        if not decisions:
            return panel

        def decision_sort_key(decision: MultiTradeDecision) -> tuple[int, float]:
            position_qty = float(position_map.get(decision.symbol, 0.0))
            is_position_reducing = (decision.side == "sell" and position_qty > 0) or (
                decision.side == "buy" and position_qty < 0
            )
            notional = decision.qty * self._effective_price(decision)
            return (0 if is_position_reducing else 1, -notional)

        decisions.sort(key=decision_sort_key)
        if self.max_orders_per_cycle > 0:
            decisions = decisions[: self.max_orders_per_cycle]

        remaining_buying_power = 0.0
        try:
            self.latest_equity, buying_power = self._get_account_snapshot()
            remaining_buying_power = max(
                0.0, buying_power * (1.0 - self.buying_power_buffer)
            )
        except Exception:
            pass

        working_position_map = dict(position_map)
        for decision in decisions:
            position_qty = float(working_position_map.get(decision.symbol, 0.0))
            adjusted_qty = decision.qty
            if decision.side == "sell" and position_qty > 0:
                if self.asset_class == "stock":
                    max_sell_qty = float(int(position_qty))
                else:
                    max_sell_qty = (
                        np.floor(float(position_qty) * 1_000_000.0) / 1_000_000.0
                    )
                adjusted_qty = min(adjusted_qty, max_sell_qty)

            adjusted_qty, reserved_buying_power = self._cap_qty_for_buying_power(
                decision,
                position_qty,
                remaining_buying_power,
            )
            if adjusted_qty <= 0:
                logger.debug(
                    "No trade %s (insufficient available quantity)", decision.symbol
                )
                continue
            if adjusted_qty != decision.qty:
                logger.debug(
                    "Adjusted %s sell qty from %s to %s based on live position.",
                    decision.symbol,
                    (f"{decision.qty:.6f}".rstrip("0").rstrip(".") if self.asset_class == "crypto" else int(decision.qty)),
                    (f"{adjusted_qty:.6f}".rstrip("0").rstrip(".") if self.asset_class == "crypto" else int(adjusted_qty)),
                )
            order_decision = replace(decision, qty=adjusted_qty)

            try:
                order_id = self._submit_order(order_decision)
            except APIError as exc:
                available_qty = self._extract_available_qty_from_error(exc)
                if (
                    order_decision.side == "sell"
                    and available_qty is not None
                    and available_qty > 0
                    and float(available_qty) < float(order_decision.qty)
                ):
                    if self.asset_class == "stock":
                        retry_qty = float(int(available_qty))
                    else:
                        retry_qty = np.floor(float(available_qty) * 1_000_000.0) / 1_000_000.0
                    retry_decision = replace(order_decision, qty=retry_qty)
                    logger.warning(
                        "Order rejected for %s with qty=%s; retrying with available qty=%s",
                        retry_decision.symbol,
                        (
                            f"{order_decision.qty:.6f}".rstrip("0").rstrip(".")
                            if self.asset_class == "crypto"
                            else int(order_decision.qty)
                        ),
                        (
                            f"{retry_qty:.6f}".rstrip("0").rstrip(".")
                            if self.asset_class == "crypto"
                            else int(retry_qty)
                        ),
                    )
                    try:
                        order_id = self._submit_order(retry_decision)
                        order_decision = retry_decision
                    except APIError as retry_exc:
                        logger.warning(
                            "Order rejected for %s after qty retry: %s",
                            retry_decision.symbol,
                            retry_exc,
                        )
                        continue
                else:
                    if self._is_insufficient_buying_power_error(exc):
                        self.buying_power_cooldown_until[order_decision.symbol] = (
                            self.iteration_count + self.buying_power_cooldown_cycles
                        )
                        logger.warning(
                            "Order rejected for %s: %s (cooldown %s cycles)",
                            order_decision.symbol,
                            exc,
                            self.buying_power_cooldown_cycles,
                        )
                        remaining_buying_power = max(
                            0.0,
                            remaining_buying_power - reserved_buying_power,
                        )
                        continue
                    logger.warning(
                        "Order rejected for %s: %s", order_decision.symbol, exc
                    )
                    continue

            print_price = (
                order_decision.limit_price
                if order_decision.limit_price is not None
                else order_decision.price
            )
            net_pnl = self.latest_equity - self.starting_equity
            self.trade_logger.log_trade(
                symbol=order_decision.symbol,
                side=order_decision.side,
                qty=order_decision.qty,
                price=print_price,
                order_type=order_decision.order_type,
                order_id=order_id,
                status="submitted" if order_id != "dry_run" else "dry_run",
                equity=self.latest_equity,
                net_pnl=net_pnl,
                strategy=(
                    self.portfolio_strategy.__class__.__name__
                    if self.portfolio_strategy is not None
                    else self.strategy_by_symbol[
                        order_decision.symbol
                    ].__class__.__name__
                ),
            )
            logger.info(
                "%s %s %s @ %.2f | order_id=%s | equity=$%.2f | net_pnl=%+.2f",
                order_decision.side.upper(),
                (
                    f"{order_decision.qty:.6f}".rstrip("0").rstrip(".")
                    if self.asset_class == "crypto"
                    else int(order_decision.qty)
                ),
                order_decision.symbol,
                print_price,
                order_id,
                self.latest_equity,
                net_pnl,
            )

            signed_qty = (
                order_decision.qty
                if order_decision.side == "buy"
                else -order_decision.qty
            )
            working_position_map[order_decision.symbol] = (
                working_position_map.get(order_decision.symbol, 0.0) + signed_qty
            )

            remaining_buying_power = max(
                0.0,
                remaining_buying_power - reserved_buying_power,
            )
            if order_decision.side == "sell" and position_qty > 0:
                released_qty = min(order_decision.qty, max(0.0, position_qty))
                remaining_buying_power += released_qty * self._effective_price(
                    order_decision
                )

        return panel
