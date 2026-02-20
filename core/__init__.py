"""Core backtesting components."""

from .logger import get_logger, get_trade_logger, TradeLogger

__all__ = [
    "get_logger",
    "get_trade_logger",
    "TradeLogger",
]

try:
    from .alpaca_trader import AlpacaTrader

    __all__.append("AlpacaTrader")
except ModuleNotFoundError:
    AlpacaTrader = None  # type: ignore[assignment]

try:
    from .multi_asset_trader import MultiAssetAlpacaTrader

    __all__.append("MultiAssetAlpacaTrader")
except ModuleNotFoundError:
    MultiAssetAlpacaTrader = None  # type: ignore[assignment]
