import math
import random

from core.order_book import Order


class MatchingEngine:
    """
    Simulates exchange execution outcomes such as fills, partial fills, and
    cancellations for the provided trade intent.
    """

    def __init__(self, seed: int | None = 42):
        # Deterministic by default for reproducible backtests.
        self.seed = seed
        self._rng = random.Random(seed)

    def simulate_execution(self, order: Order, intended_qty: float, trade_price: float):
        r = self._rng.random()
        qty = float(intended_qty)
        if qty <= 0:
            return {
                "order_id": order.order_id,
                "status": "cancelled",
                "filled_qty": 0.0,
                "avg_price": trade_price,
            }

        if r < 0.70:
            filled_qty = qty
            status = "filled"
        elif r < 0.90:
            part = qty * self._rng.uniform(0.1, 0.9)
            if qty >= 1.0:
                filled_qty = max(1.0, float(int(part)))
            else:
                filled_qty = max(1e-6, math.floor(part * 1_000_000.0) / 1_000_000.0)
            status = "partial"
        else:
            filled_qty = 0.0
            status = "cancelled"

        return {
            "order_id": order.order_id,
            "status": status,
            "filled_qty": filled_qty,
            "avg_price": trade_price,
        }
