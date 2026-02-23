from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from core.logger import TradeLogger


class TradeLoggerSummaryTests(unittest.TestCase):
    def _write_trade_rows(self, trade_logger: TradeLogger, rows: list[dict]) -> None:
        with open(trade_logger.trade_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TradeLogger.HEADERS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_session_summary_filters_to_current_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trade_logger = TradeLogger(log_dir=Path(tmp_dir))
            self._write_trade_rows(
                trade_logger,
                [
                    {
                        "timestamp": "2026-02-20T10:00:00Z",
                        "symbol": "AAPL",
                        "side": "buy",
                        "qty": 1,
                        "price": 100,
                        "order_type": "market",
                        "order_id": "old-order",
                        "status": "submitted",
                        "equity": 1010,
                        "net_pnl": 10,
                        "strategy": "test",
                        "notes": "",
                    },
                    {
                        "timestamp": "2026-02-22T10:00:00Z",
                        "symbol": "AAPL",
                        "side": "sell",
                        "qty": 1,
                        "price": 101,
                        "order_type": "market",
                        "order_id": "new-order",
                        "status": "submitted",
                        "equity": 1025,
                        "net_pnl": 25,
                        "strategy": "test",
                        "notes": "",
                    },
                ],
            )

            summary = trade_logger.get_session_summary(
                start_equity=1000.0,
                session_started_at=datetime(2026, 2, 21, tzinfo=timezone.utc),
            )

            self.assertEqual(summary["total_trades"], 1)
            self.assertEqual(summary["buys"], 0)
            self.assertEqual(summary["sells"], 1)
            self.assertEqual(summary["end_equity"], 1025.0)
            self.assertEqual(summary["net_pnl"], 25.0)

    def test_session_summary_empty_when_no_rows_in_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trade_logger = TradeLogger(log_dir=Path(tmp_dir))
            self._write_trade_rows(
                trade_logger,
                [
                    {
                        "timestamp": "2026-02-20T10:00:00Z",
                        "symbol": "AAPL",
                        "side": "buy",
                        "qty": 1,
                        "price": 100,
                        "order_type": "market",
                        "order_id": "old-order",
                        "status": "submitted",
                        "equity": 1010,
                        "net_pnl": 10,
                        "strategy": "test",
                        "notes": "",
                    }
                ],
            )

            summary = trade_logger.get_session_summary(
                start_equity=1000.0,
                session_started_at=datetime(2026, 2, 21, tzinfo=timezone.utc),
            )

            self.assertEqual(summary["total_trades"], 0)
            self.assertEqual(summary["net_pnl"], 0.0)
            self.assertEqual(summary["start_equity"], 1000.0)
            self.assertEqual(summary["end_equity"], 1000.0)

    def test_session_summary_can_start_from_trade_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trade_logger = TradeLogger(log_dir=Path(tmp_dir))
            self._write_trade_rows(
                trade_logger,
                [
                    {
                        "timestamp": "2026-02-20T10:00:00Z",
                        "symbol": "AAPL",
                        "side": "buy",
                        "qty": 1,
                        "price": 100,
                        "order_type": "market",
                        "order_id": "old-order",
                        "status": "submitted",
                        "equity": 1010,
                        "net_pnl": 10,
                        "strategy": "test",
                        "notes": "",
                    },
                    {
                        "timestamp": "2026-02-20T10:01:00Z",
                        "symbol": "AAPL",
                        "side": "sell",
                        "qty": 1,
                        "price": 101,
                        "order_type": "market",
                        "order_id": "new-order",
                        "status": "submitted",
                        "equity": 1020,
                        "net_pnl": 20,
                        "strategy": "test",
                        "notes": "",
                    },
                ],
            )

            summary = trade_logger.get_session_summary(
                start_equity=1000.0,
                start_trade_count=1,
            )

            self.assertEqual(summary["total_trades"], 1)
            self.assertEqual(summary["buys"], 0)
            self.assertEqual(summary["sells"], 1)
            self.assertEqual(summary["end_equity"], 1020.0)
            self.assertEqual(summary["net_pnl"], 20.0)


if __name__ == "__main__":
    unittest.main()
