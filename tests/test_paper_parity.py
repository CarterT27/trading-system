from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import pandas as pd

from core.backtester import Backtester
from core.multi_asset_backtester import MultiAssetBacktester
from core.order_book import Order, OrderBook
from core.order_manager import OrderManager
from core.paper_parity import (
    BuyingPowerConfig,
    PaperParityConfig,
    normalize_paper_parity_config,
)
from run_backtest import (
    build_paper_parity_config,
    load_asset_flags_by_symbol,
    parse_args,
)


class _DummyGateway:
    symbol = "DUMMY"

    def stream(self):
        return iter(())


class _DummyStrategy:
    required_lookback = 1

    def run(self, df):
        return df


class _RowsGateway:
    def __init__(self, symbol: str, rows: list[dict]):
        self.symbol = symbol
        self._rows = rows

    def stream(self):
        for row in self._rows:
            yield row


class _StaticSignalStrategy:
    def __init__(self, signal: int, qty: int, limit_price: float | None = None):
        self._signal = signal
        self._qty = qty
        self._limit_price = limit_price

    def run(self, df):
        out = df.copy()
        out["signal"] = self._signal
        out["target_qty"] = self._qty
        if self._limit_price is not None:
            out["limit_price"] = self._limit_price
        return out


class _AlwaysFillEngine:
    def simulate_execution(self, order, intended_qty: int, trade_price: float):
        return {
            "order_id": order.order_id,
            "status": "filled",
            "filled_qty": intended_qty,
            "avg_price": trade_price,
        }


class _CaptureLogger:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def log(self, event_type, data):
        self.events.append((event_type, data))


class PaperParityConfigTests(unittest.TestCase):
    def test_disabled_defaults(self) -> None:
        cfg = PaperParityConfig()
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.require_tradable)
        self.assertFalse(cfg.require_shortable)
        self.assertFalse(cfg.require_easy_to_borrow)
        self.assertTrue(cfg.allow_crypto_shorts)
        self.assertAlmostEqual(cfg.short_open_valuation_buffer, 1.03)
        self.assertFalse(cfg.reserve_open_orders)
        self.assertAlmostEqual(cfg.buying_power_multiplier(), 1.0)

    def test_invalid_short_open_buffer(self) -> None:
        with self.assertRaises(ValueError):
            PaperParityConfig(short_open_valuation_buffer=0)

    def test_invalid_easy_to_borrow_dependency(self) -> None:
        with self.assertRaises(ValueError):
            PaperParityConfig(
                enabled=True,
                require_shortable=False,
                require_easy_to_borrow=True,
            )

    def test_invalid_buying_power_mode(self) -> None:
        with self.assertRaises(ValueError):
            BuyingPowerConfig(mode="bogus")

    def test_invalid_multiplier_mode_without_value(self) -> None:
        with self.assertRaises(ValueError):
            BuyingPowerConfig(mode="multiplier", explicit_multiplier=None)

    def test_invalid_explicit_multiplier_value(self) -> None:
        with self.assertRaises(ValueError):
            BuyingPowerConfig(mode="multiplier", explicit_multiplier=0)

    def test_invalid_tier_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            BuyingPowerConfig(
                mode="tiered",
                tier_no_margin_equity=5_000,
                tier_day_trader_equity=5_000,
            )

    def test_tiered_mode_requires_equity_for_resolution(self) -> None:
        cfg = BuyingPowerConfig(mode="tiered")
        with self.assertRaises(ValueError):
            cfg.resolve_multiplier(None)

    def test_tiered_mode_resolves_expected_multipliers(self) -> None:
        cfg = BuyingPowerConfig(mode="tiered")
        self.assertAlmostEqual(cfg.resolve_multiplier(1_000), 1.0)
        self.assertAlmostEqual(cfg.resolve_multiplier(10_000), 2.0)
        self.assertAlmostEqual(cfg.resolve_multiplier(30_000), 4.0)

    def test_multiplier_mode_resolves_explicit_value(self) -> None:
        cfg = BuyingPowerConfig(mode="multiplier", explicit_multiplier=2.5)
        self.assertAlmostEqual(cfg.resolve_multiplier(None), 2.5)
        self.assertAlmostEqual(cfg.resolve_multiplier(123.0), 2.5)

    def test_enabled_tiered_requires_account_equity(self) -> None:
        with self.assertRaises(ValueError):
            PaperParityConfig(enabled=True, buying_power=BuyingPowerConfig(mode="tiered"))

    def test_enabled_defaults_constructor(self) -> None:
        cfg = PaperParityConfig.enabled_defaults()
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.require_tradable)
        self.assertTrue(cfg.require_shortable)
        self.assertFalse(cfg.allow_crypto_shorts)

    def test_normalize_none_returns_disabled(self) -> None:
        cfg = normalize_paper_parity_config(None)
        self.assertIsInstance(cfg, PaperParityConfig)
        self.assertFalse(cfg.enabled)

    def test_normalize_rejects_invalid_type(self) -> None:
        with self.assertRaises(TypeError):
            normalize_paper_parity_config("bad-type")  # type: ignore[arg-type]


class PaperParityWiringTests(unittest.TestCase):
    def test_both_backtesters_accept_same_config(self) -> None:
        cfg = PaperParityConfig(
            enabled=True,
            buying_power=BuyingPowerConfig(mode="multiplier", explicit_multiplier=2.0),
        )

        backtester = Backtester(
            data_gateway=_DummyGateway(),
            strategy=_DummyStrategy(),
            order_manager=OrderManager(capital=10_000),
            order_book=object(),
            matching_engine=object(),
            logger=None,
            verbose=False,
            paper_parity=cfg,
        )

        panel = pd.DataFrame(
            {
                "Datetime": [pd.Timestamp("2024-01-01T09:30:00Z")],
                "symbol": ["AAPL"],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1_000],
            }
        )
        multi = MultiAssetBacktester(
            panel_df=panel,
            strategy_factory=lambda: _DummyStrategy(),
            paper_parity=cfg,
        )

        self.assertIs(backtester.paper_parity, cfg)
        self.assertIs(multi.paper_parity, cfg)

    def test_backtester_uses_disabled_default_when_none_passed(self) -> None:
        backtester = Backtester(
            data_gateway=_DummyGateway(),
            strategy=_DummyStrategy(),
            order_manager=OrderManager(capital=10_000),
            order_book=object(),
            matching_engine=object(),
            logger=None,
            verbose=False,
            paper_parity=None,
        )
        self.assertFalse(backtester.paper_parity.enabled)

    def test_multi_asset_uses_disabled_default_when_none_passed(self) -> None:
        panel = pd.DataFrame(
            {
                "Datetime": [pd.Timestamp("2024-01-01T09:30:00Z")],
                "symbol": ["AAPL"],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1_000],
            }
        )
        multi = MultiAssetBacktester(
            panel_df=panel,
            strategy_factory=lambda: _DummyStrategy(),
            paper_parity=None,
        )
        self.assertFalse(multi.paper_parity.enabled)

    def test_multi_asset_max_short_notional_cap_blocks_excess_short(self) -> None:
        panel = pd.DataFrame(
            {
                "Datetime": [
                    pd.Timestamp("2024-01-01T09:30:00Z"),
                    pd.Timestamp("2024-01-01T09:31:00Z"),
                ],
                "symbol": ["AAPL", "MSFT"],
                "Open": [100.0, 100.0],
                "High": [101.0, 101.0],
                "Low": [99.0, 99.0],
                "Close": [100.0, 100.0],
                "Volume": [1_000, 1_000],
            }
        )

        class _SignalStrategy:
            required_lookback = 1

            def run(self, df):
                out = df.copy()
                out["signal"] = -1
                out["target_qty"] = 2
                return out

        multi = MultiAssetBacktester(
            panel_df=panel,
            strategy_factory=lambda: _SignalStrategy(),
            max_short_notional=250.0,
        )
        multi.run()
        self.assertEqual(len(multi.trades), 1)
        self.assertEqual(multi.trades[0].symbol, "AAPL")


class RunBacktestParityBuilderTests(unittest.TestCase):
    def test_builder_defaults_disabled_without_flag(self) -> None:
        args = argparse.Namespace(
            paper_parity=False,
            buying_power_mode="tiered",
            buying_power_multiplier=None,
            account_equity=None,
            reserve_open_orders=False,
        )
        cfg = build_paper_parity_config(args)
        self.assertFalse(cfg.enabled)

    def test_builder_enables_tiered_defaults_with_flag(self) -> None:
        args = argparse.Namespace(
            paper_parity=True,
            buying_power_mode="tiered",
            buying_power_multiplier=None,
            account_equity=None,
            reserve_open_orders=True,
        )
        cfg = build_paper_parity_config(args)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.buying_power.mode, "tiered")
        self.assertAlmostEqual(cfg.account_equity, 25_000.0)
        self.assertTrue(cfg.reserve_open_orders)


class PaperParityEligibilityTests(unittest.TestCase):
    def test_opening_short_rejected_when_not_shortable_single_and_multi(self) -> None:
        cfg = PaperParityConfig(
            enabled=True,
            require_tradable=True,
            require_shortable=True,
            account_equity=25_000.0,
        )
        flags = {
            "AAPL": {
                "tradable": True,
                "shortable": False,
                "easy_to_borrow": False,
            }
        }

        single = Backtester(
            data_gateway=_RowsGateway(
                symbol="AAPL",
                rows=[
                    {
                        "Datetime": pd.Timestamp("2024-01-01T09:30:00Z"),
                        "Open": 100.0,
                        "High": 101.0,
                        "Low": 99.0,
                        "Close": 100.0,
                        "Volume": 1_000,
                    }
                ],
            ),
            strategy=_StaticSignalStrategy(signal=-1, qty=5),
            order_manager=OrderManager(capital=10_000),
            order_book=OrderBook(),
            matching_engine=_AlwaysFillEngine(),
            logger=None,
            verbose=False,
            paper_parity=cfg,
            asset_flags_by_symbol=flags,
        )
        single.run()
        self.assertEqual(len(single.trades), 0)

        panel = pd.DataFrame(
            {
                "Datetime": [pd.Timestamp("2024-01-01T09:30:00Z")],
                "symbol": ["AAPL"],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.0],
                "Volume": [1_000],
            }
        )
        multi = MultiAssetBacktester(
            panel_df=panel,
            strategy_factory=lambda: _StaticSignalStrategy(signal=-1, qty=5),
            paper_parity=cfg,
            asset_flags_by_symbol=flags,
        )
        multi.run()
        self.assertEqual(len(multi.trades), 0)

    def test_closing_long_allowed_when_short_flags_fail(self) -> None:
        cfg = PaperParityConfig(
            enabled=True,
            require_tradable=True,
            require_shortable=True,
            require_easy_to_borrow=True,
            account_equity=25_000.0,
        )
        flags = {
            "AAPL": {
                "tradable": True,
                "shortable": False,
                "easy_to_borrow": False,
            }
        }
        single = Backtester(
            data_gateway=_RowsGateway(
                symbol="AAPL",
                rows=[
                    {
                        "Datetime": pd.Timestamp("2024-01-01T09:30:00Z"),
                        "Open": 100.0,
                        "High": 101.0,
                        "Low": 99.0,
                        "Close": 100.0,
                        "Volume": 1_000,
                    }
                ],
            ),
            strategy=_StaticSignalStrategy(signal=-1, qty=3),
            order_manager=OrderManager(capital=10_000),
            order_book=OrderBook(),
            matching_engine=_AlwaysFillEngine(),
            logger=None,
            verbose=False,
            paper_parity=cfg,
            asset_flags_by_symbol=flags,
        )
        single.order_manager.long_position = 5
        single.run()
        self.assertEqual(len(single.trades), 1)
        self.assertEqual(single.order_manager.long_position, 2)


class PaperParityShortValuationTests(unittest.TestCase):
    def test_short_open_rejected_when_buffered_value_exceeds_bp(self) -> None:
        cfg = PaperParityConfig(
            enabled=True,
            buying_power=BuyingPowerConfig(mode="multiplier", explicit_multiplier=1.0),
            account_equity=1_000.0,
        )
        manager = OrderManager(capital=10_000, max_short_position=10_000)
        order = Order(
            order_id="o1",
            side="sell",
            price=90.0,
            qty=10,
            timestamp=pd.Timestamp("2024-01-01T09:30:00Z").timestamp(),
        )

        valid, reason = manager.validate(
            order,
            reference_price=100.0,
            paper_parity=cfg,
        )
        self.assertFalse(valid)
        self.assertIn("short", reason.lower())

    def test_short_open_uses_max_of_limit_and_buffered_reference(self) -> None:
        cfg = PaperParityConfig(
            enabled=True,
            buying_power=BuyingPowerConfig(mode="multiplier", explicit_multiplier=1.0),
            account_equity=1_200.0,
        )
        manager = OrderManager(capital=10_000, max_short_position=10_000)
        order = Order(
            order_id="o2",
            side="sell",
            price=110.0,
            qty=10,
            timestamp=pd.Timestamp("2024-01-01T09:30:00Z").timestamp(),
        )
        valid, _ = manager.validate(
            order,
            reference_price=100.0,
            paper_parity=cfg,
        )
        self.assertTrue(valid)

    def test_multi_asset_short_open_rejects_using_buffered_valuation(self) -> None:
        cfg = PaperParityConfig(
            enabled=True,
            buying_power=BuyingPowerConfig(mode="multiplier", explicit_multiplier=1.0),
            account_equity=1_000.0,
        )
        panel = pd.DataFrame(
            {
                "Datetime": [pd.Timestamp("2024-01-01T09:30:00Z")],
                "symbol": ["AAPL"],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.0],
                "Volume": [1_000],
            }
        )
        multi = MultiAssetBacktester(
            panel_df=panel,
            strategy_factory=lambda: _StaticSignalStrategy(signal=-1, qty=10, limit_price=90.0),
            paper_parity=cfg,
        )
        multi.run()
        self.assertEqual(len(multi.trades), 0)
        self.assertEqual(multi.rejections[0]["reason"], "reject_short_open_buying_power")


class PaperParityReservationTests(unittest.TestCase):
    def test_reserve_open_orders_blocks_second_order(self) -> None:
        cfg = PaperParityConfig(
            enabled=True,
            reserve_open_orders=True,
            buying_power=BuyingPowerConfig(mode="multiplier", explicit_multiplier=1.0),
            account_equity=1_000.0,
        )
        manager = OrderManager(
            capital=1_000.0,
            max_long_position=10_000,
            max_short_position=10_000,
        )
        o1 = Order("r1", "buy", 100.0, 5, timestamp=time.time())
        ok1, _ = manager.validate(o1, reference_price=100.0, paper_parity=cfg)
        self.assertTrue(ok1)
        self.assertAlmostEqual(manager.reserved_buying_power, 500.0)

        o2 = Order("r2", "buy", 100.0, 6, timestamp=time.time())
        ok2, reason2 = manager.validate(o2, reference_price=100.0, paper_parity=cfg)
        self.assertFalse(ok2)
        self.assertIn("reserved", reason2.lower())

    def test_cancel_releases_reserved_buying_power(self) -> None:
        cfg = PaperParityConfig(
            enabled=True,
            reserve_open_orders=True,
            buying_power=BuyingPowerConfig(mode="multiplier", explicit_multiplier=1.0),
            account_equity=1_000.0,
        )
        manager = OrderManager(
            capital=1_000.0,
            max_long_position=10_000,
            max_short_position=10_000,
        )
        order = Order("r3", "buy", 100.0, 5, timestamp=time.time())
        ok, _ = manager.validate(order, reference_price=100.0, paper_parity=cfg)
        self.assertTrue(ok)
        self.assertGreater(manager.reserved_buying_power, 0.0)

        manager.reconcile_reservation(
            order,
            filled_qty=0,
            status="cancelled",
            reference_price=100.0,
            paper_parity=cfg,
        )
        self.assertAlmostEqual(manager.reserved_buying_power, 0.0)

    def test_multi_asset_timestamp_batch_uses_reservations(self) -> None:
        cfg = PaperParityConfig(
            enabled=True,
            reserve_open_orders=True,
            buying_power=BuyingPowerConfig(mode="multiplier", explicit_multiplier=1.0),
            account_equity=1_000.0,
        )
        panel = pd.DataFrame(
            {
                "Datetime": [
                    pd.Timestamp("2024-01-01T09:30:00Z"),
                    pd.Timestamp("2024-01-01T09:30:00Z"),
                ],
                "symbol": ["AAPL", "MSFT"],
                "Open": [100.0, 100.0],
                "High": [101.0, 101.0],
                "Low": [99.0, 99.0],
                "Close": [100.0, 100.0],
                "Volume": [1_000, 1_000],
            }
        )

        multi = MultiAssetBacktester(
            panel_df=panel,
            strategy_factory=lambda: _StaticSignalStrategy(signal=1, qty=6),
            initial_capital=1_000.0,
            paper_parity=cfg,
        )
        multi.run()
        self.assertEqual(len(multi.trades), 1)
        self.assertGreaterEqual(len(multi.reservation_events), 2)

    def test_single_backtester_logs_reservation_lifecycle(self) -> None:
        cfg = PaperParityConfig(
            enabled=True,
            reserve_open_orders=True,
            buying_power=BuyingPowerConfig(mode="multiplier", explicit_multiplier=1.0),
            account_equity=1_000.0,
        )
        logger = _CaptureLogger()
        backtester = Backtester(
            data_gateway=_RowsGateway(
                symbol="AAPL",
                rows=[
                    {
                        "Datetime": pd.Timestamp("2024-01-01T09:30:00Z"),
                        "Open": 100.0,
                        "High": 101.0,
                        "Low": 99.0,
                        "Close": 100.0,
                        "Volume": 1_000,
                    }
                ],
            ),
            strategy=_StaticSignalStrategy(signal=1, qty=5),
            order_manager=OrderManager(capital=1_000.0),
            order_book=OrderBook(),
            matching_engine=_AlwaysFillEngine(),
            logger=logger,
            verbose=False,
            paper_parity=cfg,
        )
        backtester.run()
        reservation_actions = [
            event[1].get("action")
            for event in logger.events
            if event[0] == "reservation"
        ]
        self.assertIn("reserve", reservation_actions)
        self.assertIn("release", reservation_actions)


class RunBacktestCliWiringTests(unittest.TestCase):
    def test_parse_args_reads_multi_asset_caps(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run_backtest.py",
                "--max-notional-per-order",
                "2500",
                "--max-short-notional",
                "9000",
            ],
        ):
            args = parse_args()
        self.assertAlmostEqual(args.max_notional_per_order, 2500.0)
        self.assertAlmostEqual(args.max_short_notional, 9000.0)

    def test_load_asset_flags_by_symbol(self) -> None:
        csv_path = "tests/.tmp_asset_flags.csv"
        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "tradable": [True, True],
                "shortable": [False, True],
                "easy_to_borrow": [False, True],
            }
        )
        df.to_csv(csv_path, index=False)
        try:
            flags = load_asset_flags_by_symbol(Path(csv_path))
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

        self.assertTrue(flags["AAPL"].tradable)
        self.assertFalse(flags["AAPL"].shortable)
        self.assertFalse(flags["AAPL"].easy_to_borrow)

    def test_buy_to_cover_not_blocked_by_short_open_rule(self) -> None:
        cfg = PaperParityConfig(
            enabled=True,
            buying_power=BuyingPowerConfig(mode="multiplier", explicit_multiplier=1.0),
            account_equity=100.0,
        )
        manager = OrderManager(capital=10_000, max_short_position=10_000)
        manager.short_position = 5
        order = Order(
            order_id="o3",
            side="buy",
            price=100.0,
            qty=3,
            timestamp=pd.Timestamp("2024-01-01T09:30:00Z").timestamp(),
        )
        valid, _ = manager.validate(
            order,
            reference_price=100.0,
            paper_parity=cfg,
        )
        self.assertTrue(valid)


if __name__ == "__main__":
    unittest.main()
