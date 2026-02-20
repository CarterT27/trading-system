from __future__ import annotations

import unittest

import pandas as pd

from core.backtester import Backtester
from core.multi_asset_backtester import MultiAssetBacktester
from core.order_manager import OrderManager
from core.paper_parity import (
    BuyingPowerConfig,
    PaperParityConfig,
    normalize_paper_parity_config,
)


class _DummyGateway:
    symbol = "DUMMY"

    def stream(self):
        return iter(())


class _DummyStrategy:
    required_lookback = 1

    def run(self, df):
        return df


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


if __name__ == "__main__":
    unittest.main()
