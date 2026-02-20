from __future__ import annotations

import unittest

import pandas as pd

from core.backtester import Backtester
from core.matching_engine import MatchingEngine
from core.order_book import OrderBook
from core.order_manager import OrderManager


class _RowsGateway:
    def __init__(self, rows: list[dict], symbol: str = "TEST"):
        self._rows = rows
        self.length = len(rows)
        self.symbol = symbol

    def stream(self):
        for row in self._rows:
            yield row


def _make_rows(n: int) -> list[dict]:
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    rows: list[dict] = []
    for i in range(n):
        price = 100.0 + float(i)
        rows.append(
            {
                "Datetime": start + pd.Timedelta(minutes=i),
                "Open": price,
                "High": price + 1.0,
                "Low": price - 1.0,
                "Close": price,
                "Volume": 1_000.0,
            }
        )
    return rows


class _IncrementalOnlyStrategy:
    required_lookback = 5

    def __init__(self):
        self.reset_calls = 0
        self.incremental_calls = 0

    def reset_incremental_state(self) -> None:
        self.reset_calls += 1

    def run_incremental(self, row: dict, bar_index: int | None = None):
        self.incremental_calls += 1
        return {
            "Close": float(row["Close"]),
            "signal": 0,
            "position": 0,
            "target_qty": 0.0,
        }

    def run(self, df):
        raise AssertionError("run(df) should not be called when incremental output is returned.")


class _WindowProbeStrategy:
    required_lookback = 5

    def __init__(self):
        self.lengths: list[int] = []
        self.index_ranges: list[tuple[int, int]] = []

    def run(self, df):
        self.lengths.append(int(len(df)))
        if len(df):
            self.index_ranges.append((int(df.index[0]), int(df.index[-1])))
        out = df.copy()
        out["signal"] = 0
        out["position"] = 0
        out["target_qty"] = 0.0
        return out


class BacktesterIncrementalExecutionTests(unittest.TestCase):
    def _build_backtester(self, rows: list[dict], strategy) -> Backtester:
        return Backtester(
            data_gateway=_RowsGateway(rows),
            strategy=strategy,
            order_manager=OrderManager(capital=10_000),
            order_book=OrderBook(),
            matching_engine=MatchingEngine(seed=42),
            logger=None,
            verbose=False,
            show_progress=False,
            asset_class="stock",
        )

    def test_uses_incremental_strategy_output_when_available(self) -> None:
        rows = _make_rows(12)
        strategy = _IncrementalOnlyStrategy()
        bt = self._build_backtester(rows=rows, strategy=strategy)

        equity_df = bt.run()

        self.assertEqual(len(equity_df), len(rows))
        self.assertEqual(strategy.reset_calls, 1)
        self.assertEqual(strategy.incremental_calls, len(rows))

    def test_fallback_strategy_receives_bounded_sliding_window(self) -> None:
        rows = _make_rows(12)
        strategy = _WindowProbeStrategy()
        bt = self._build_backtester(rows=rows, strategy=strategy)

        _ = bt.run()

        self.assertEqual(len(strategy.lengths), len(rows))
        self.assertEqual(max(strategy.lengths), strategy.required_lookback)
        self.assertEqual(strategy.lengths[-1], strategy.required_lookback)
        self.assertEqual(strategy.index_ranges[-1], (7, 11))


if __name__ == "__main__":
    unittest.main()
