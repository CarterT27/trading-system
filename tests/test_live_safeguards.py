from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from core.multi_asset_trader import MultiAssetAlpacaTrader
from strategies.strategy_base import CryptoCompetitionPortfolioStrategy


class _FakeCompetitionSignal:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = 1
        out["desired_position_frac"] = 1.0
        return out


class CryptoPortfolioSafeguardTests(unittest.TestCase):
    def test_max_symbol_weight_caps_target_qty(self) -> None:
        with (
            patch("strategies.strategy_base.RandomForestClassifier", object),
            patch("strategies.strategy_base.ExtraTreesClassifier", object),
            patch("strategies.strategy_base.HistGradientBoostingClassifier", object),
            patch("strategies.strategy_base.LogisticRegression", object),
        ):
            strategy = CryptoCompetitionPortfolioStrategy(
                portfolio_notional=100_000.0,
                min_active_symbols=1,
                max_symbol_weight=0.20,
            )
        strategy._required_lookback = 1
        fake = _FakeCompetitionSignal()
        strategy._strategy_for_symbol = lambda _symbol: fake  # type: ignore[assignment]

        panel = pd.DataFrame(
            {
                "Datetime": [
                    pd.Timestamp("2026-02-22T10:00:00Z"),
                    pd.Timestamp("2026-02-22T10:00:00Z"),
                ],
                "symbol": ["BTC/USD", "ETH/USD"],
                "Open": [100.0, 100.0],
                "High": [100.0, 100.0],
                "Low": [100.0, 100.0],
                "Close": [100.0, 100.0],
                "Volume": [1_000, 1_000],
            }
        )

        orders = strategy.run_panel(panel, current_positions={})
        self.assertEqual(len(orders), 2)
        self.assertTrue((orders["signal"] == 1).all())
        self.assertTrue((orders["target_qty"] == 200.0).all())

    def test_under_min_active_symbols_flattens_existing_longs(self) -> None:
        with (
            patch("strategies.strategy_base.RandomForestClassifier", object),
            patch("strategies.strategy_base.ExtraTreesClassifier", object),
            patch("strategies.strategy_base.HistGradientBoostingClassifier", object),
            patch("strategies.strategy_base.LogisticRegression", object),
        ):
            strategy = CryptoCompetitionPortfolioStrategy(
                portfolio_notional=100_000.0,
                min_active_symbols=3,
                max_symbol_weight=0.35,
            )
        strategy._required_lookback = 1
        fake = _FakeCompetitionSignal()
        strategy._strategy_for_symbol = lambda _symbol: fake  # type: ignore[assignment]

        panel = pd.DataFrame(
            {
                "Datetime": [pd.Timestamp("2026-02-22T10:00:00Z")],
                "symbol": ["BTC/USD"],
                "Open": [50.0],
                "High": [50.0],
                "Low": [50.0],
                "Close": [50.0],
                "Volume": [1_000],
            }
        )

        orders = strategy.run_panel(panel, current_positions={"BTC/USD": 2.0})
        self.assertEqual(len(orders), 1)
        self.assertEqual(int(orders.iloc[0]["signal"]), -1)
        self.assertAlmostEqual(float(orders.iloc[0]["target_qty"]), 2.0)


class BuyingPowerErrorParsingTests(unittest.TestCase):
    def test_buying_power_error_detects_insufficient_balance(self) -> None:
        exc = RuntimeError(
            "insufficient balance for USD (requested: 97.68, available: 61.07)"
        )
        self.assertTrue(MultiAssetAlpacaTrader._is_insufficient_buying_power_error(exc))

    def test_extract_requested_available_balance(self) -> None:
        exc = RuntimeError(
            "insufficient balance for USD (requested: 97.68, available: 61.07)"
        )
        requested, available = (
            MultiAssetAlpacaTrader._extract_requested_available_balance(exc)
        )
        self.assertAlmostEqual(float(requested or 0.0), 97.68)
        self.assertAlmostEqual(float(available or 0.0), 61.07)


if __name__ == "__main__":
    unittest.main()
