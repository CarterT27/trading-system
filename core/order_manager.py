import json
import time
from pathlib import Path
from typing import Optional

from core.paper_parity import PaperParityConfig, short_open_unit_value


class OrderManager:
    """
    Validates new orders and tracks capital/position state for risk checks.
    """

    def __init__(
        self,
        capital: float = 100_000.0,
        max_long_position: int = 500,
        max_short_position: int = 500,
        max_orders_per_min: int = 30,
    ):
        self.initial_capital = float(capital)
        self.cash = self.initial_capital
        self.max_long_position = max_long_position
        self.max_short_position = max_short_position
        self.max_orders_per_min = max_orders_per_min

        self.order_timestamps: list[float] = []
        self.long_position = 0
        self.short_position = 0
        self.reserved_buying_power = 0.0
        self._reserved_by_order_id: dict[str, float] = {}

    # ------------------------------------------------------------------ utils

    @property
    def net_position(self) -> int:
        return self.long_position - self.short_position

    def portfolio_value(self, price: float) -> float:
        return self.cash + self.long_position * price - self.short_position * price

    # ----------------------------------------------------------------- checks

    def _check_capital(self, order, *, reserved_buffer: float = 0.0) -> bool:
        if order.side == "buy":
            available_cash = max(0.0, float(self.cash) - max(0.0, float(reserved_buffer)))
            return float(order.price) * int(order.qty) <= available_cash
        return True

    def _project_positions(self, order):
        long_after = self.long_position
        short_after = self.short_position
        qty_remaining = order.qty

        if order.side == "buy":
            if short_after > 0:
                cover = min(qty_remaining, short_after)
                short_after -= cover
                qty_remaining -= cover
            long_after += qty_remaining
        else:
            if long_after > 0:
                cover = min(qty_remaining, long_after)
                long_after -= cover
                qty_remaining -= cover
            short_after += qty_remaining

        return long_after, short_after

    def _check_position_limit(self, order) -> bool:
        long_after, short_after = self._project_positions(order)
        return (long_after <= self.max_long_position) and (
            short_after <= self.max_short_position
        )

    def _check_order_rate(self) -> bool:
        now = time.time()
        self.order_timestamps = [t for t in self.order_timestamps if now - t < 60]
        return len(self.order_timestamps) < self.max_orders_per_min

    def _opening_short_qty(self, order) -> int:
        if order.side != "sell":
            return 0
        return max(0, int(order.qty) - max(0, int(self.long_position)))

    def _check_short_open_buying_power(
        self,
        order,
        *,
        reference_price: Optional[float],
        paper_parity: Optional[PaperParityConfig],
    ) -> tuple[bool, str]:
        if paper_parity is None or not paper_parity.enabled:
            return True, "Order approved"

        opening_short_qty = self._opening_short_qty(order)
        if opening_short_qty <= 0:
            return True, "Order approved"

        if reference_price is None or float(reference_price) <= 0:
            return False, "Invalid short reference price"

        try:
            short_unit_value = short_open_unit_value(
                limit_price=float(order.price),
                reference_price=float(reference_price),
                buffer=float(paper_parity.short_open_valuation_buffer),
            )
        except ValueError:
            return False, "Invalid short valuation inputs"

        required_notional = short_unit_value * float(opening_short_qty)
        anchor_equity = (
            float(paper_parity.account_equity)
            if paper_parity.account_equity is not None
            else float(self.initial_capital)
        )
        buying_power_limit = anchor_equity * float(paper_parity.buying_power_multiplier())
        existing_short_notional = float(self.short_position) * float(reference_price)
        reserved = self.reserved_buying_power if paper_parity.reserve_open_orders else 0.0
        available = max(0.0, buying_power_limit - existing_short_notional - reserved)
        if required_notional > available:
            return False, "Insufficient short buying power"

        return True, "Order approved"

    def _opening_long_qty(self, order) -> int:
        if order.side != "buy":
            return 0
        return max(0, int(order.qty) - max(0, int(self.short_position)))

    def _reservation_cost(
        self,
        order,
        *,
        reference_price: Optional[float],
        paper_parity: Optional[PaperParityConfig],
    ) -> float:
        if paper_parity is None or not paper_parity.enabled:
            return 0.0
        if not paper_parity.reserve_open_orders:
            return 0.0

        opening_long_qty = self._opening_long_qty(order)
        opening_short_qty = self._opening_short_qty(order)
        if paper_parity.reserve_open_orders_for_shorts_only:
            opening_long_qty = 0

        long_notional = float(order.price) * float(opening_long_qty)
        short_notional = 0.0
        if opening_short_qty > 0 and reference_price is not None and float(reference_price) > 0:
            short_notional = short_open_unit_value(
                limit_price=float(order.price),
                reference_price=float(reference_price),
                buffer=float(paper_parity.short_open_valuation_buffer),
            ) * float(opening_short_qty)
        return max(0.0, long_notional + short_notional)

    def _set_order_reservation(self, order_id: str, amount: float) -> None:
        prior = float(self._reserved_by_order_id.get(order_id, 0.0))
        if prior > 0:
            self.reserved_buying_power = max(0.0, self.reserved_buying_power - prior)

        amt = max(0.0, float(amount))
        if amt <= 0:
            self._reserved_by_order_id.pop(order_id, None)
            return

        self._reserved_by_order_id[order_id] = amt
        self.reserved_buying_power += amt

    def reserved_for_order(self, order_id: str) -> float:
        return float(self._reserved_by_order_id.get(order_id, 0.0))

    def reconcile_reservation(
        self,
        order,
        *,
        filled_qty: int,
        status: str,
        reference_price: Optional[float],
        paper_parity: Optional[PaperParityConfig],
    ) -> float:
        if paper_parity is None or not paper_parity.enabled or not paper_parity.reserve_open_orders:
            return 0.0
        reserved = float(self._reserved_by_order_id.pop(order.order_id, 0.0))
        if reserved <= 0:
            return 0.0
        self.reserved_buying_power = max(0.0, self.reserved_buying_power - reserved)
        return reserved

    # ----------------------------------------------------------------- public

    def validate(
        self,
        order,
        *,
        reference_price: Optional[float] = None,
        paper_parity: Optional[PaperParityConfig] = None,
    ):
        reserved_buffer = (
            self.reserved_buying_power
            if paper_parity is not None and paper_parity.enabled and paper_parity.reserve_open_orders
            else 0.0
        )
        if not self._check_capital(order, reserved_buffer=reserved_buffer):
            if reserved_buffer > 0 and order.side == "buy":
                return False, "Not enough capital after reserved buying power"
            return False, "Not enough capital"
        if not self._check_position_limit(order):
            return False, "Position limit exceeded"
        short_ok, short_reason = self._check_short_open_buying_power(
            order,
            reference_price=reference_price,
            paper_parity=paper_parity,
        )
        if not short_ok:
            return False, short_reason
        if not self._check_order_rate():
            return False, "Order rate limit exceeded"

        reservation = self._reservation_cost(
            order,
            reference_price=reference_price,
            paper_parity=paper_parity,
        )
        if reservation > 0:
            self._set_order_reservation(order.order_id, reservation)

        self.order_timestamps.append(time.time())
        return True, "Order approved"

    def record_execution(self, order, filled_qty: int, price: float) -> None:
        """
        Update capital and open positions after an execution report.
        """
        if filled_qty <= 0:
            return

        qty_remaining = filled_qty
        if order.side == "buy":
            if self.short_position > 0:
                cover = min(qty_remaining, self.short_position)
                self.short_position -= cover
                qty_remaining -= cover
                self.cash -= price * cover
            if qty_remaining > 0:
                self.long_position += qty_remaining
                self.cash -= price * qty_remaining
        else:
            if self.long_position > 0:
                close = min(qty_remaining, self.long_position)
                self.long_position -= close
                qty_remaining -= close
                self.cash += price * close
            if qty_remaining > 0:
                self.short_position += qty_remaining
                self.cash += price * qty_remaining


class OrderLoggingGateway:
    """
    Logs all order events: new, modified, canceled, filled.
    """

    def __init__(self, file_path="data/order_log.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type, data):
        event = {"event": event_type, "timestamp": time.time(), "data": data}
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
