"""Strategy implementations and lookup helpers."""

from __future__ import annotations

import inspect
from typing import Dict, Type

from . import strategy_base
from .strategy_base import (
    CrossSectionalPaperReversalStrategy,
    CrossSectionalStrategy,
    CryptoCompetitionStrategy,
    CryptoRegimeTrendStrategy,
    CryptoTrendStrategy,
    DemoStrategy,
    MovingAverageStrategy,
    Strategy,
    TemplateStrategy,
)


_BUILTIN_STRATEGIES: Dict[str, Type[Strategy]] = {
    "ma": MovingAverageStrategy,
    "moving_average": MovingAverageStrategy,
    "template": TemplateStrategy,
    "student": TemplateStrategy,
    "crypto_trend": CryptoTrendStrategy,
    "crypto_competition": CryptoCompetitionStrategy,
    "crypto_meta": CryptoCompetitionStrategy,
    "crypto_comp": CryptoCompetitionStrategy,
    "crypto_regime_trend": CryptoRegimeTrendStrategy,
    "crypto_regime": CryptoRegimeTrendStrategy,
    "crypto": CryptoTrendStrategy,
    "demo": DemoStrategy,
    "cross_sectional_reversal": CrossSectionalPaperReversalStrategy,
}

_BUILTIN_CLASSES = {
    MovingAverageStrategy,
    TemplateStrategy,
    CryptoTrendStrategy,
    CryptoCompetitionStrategy,
    CryptoRegimeTrendStrategy,
    DemoStrategy,
    CrossSectionalPaperReversalStrategy,
}


def _build_registry() -> Dict[str, Type[Strategy]]:
    registry: Dict[str, Type[Strategy]] = dict(_BUILTIN_STRATEGIES)
    for name, obj in inspect.getmembers(strategy_base, inspect.isclass):
        if obj in {Strategy, CrossSectionalStrategy}:
            continue
        if issubclass(obj, Strategy) and obj.__module__ == strategy_base.__name__:
            if obj not in _BUILTIN_CLASSES:
                registry[name.lower()] = obj
    return registry


_REGISTRY = _build_registry()


def get_strategy_class(name: str) -> Type[Strategy]:
    key = name.strip().lower()
    if not key:
        raise ValueError("Strategy name cannot be empty.")
    if key not in _REGISTRY:
        options = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: {options}")
    return _REGISTRY[key]


def list_strategies() -> list[str]:
    custom = sorted(k for k in _REGISTRY.keys() if k not in _BUILTIN_STRATEGIES)
    return sorted(_BUILTIN_STRATEGIES.keys()) + custom


__all__ = [
    "Strategy",
    "TemplateStrategy",
    "CrossSectionalStrategy",
    "CrossSectionalPaperReversalStrategy",
    "MovingAverageStrategy",
    "CryptoTrendStrategy",
    "CryptoCompetitionStrategy",
    "CryptoRegimeTrendStrategy",
    "DemoStrategy",
    "get_strategy_class",
    "list_strategies",
]
