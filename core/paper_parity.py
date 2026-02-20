from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


_VALID_BUYING_POWER_MODES = {"disabled", "multiplier", "tiered"}


@dataclass(frozen=True)
class BuyingPowerConfig:
    """
    Shared buying-power policy used by parity-aware backtest paths.

    mode:
    - "disabled": keep legacy backtest behavior (no parity multiplier logic).
    - "multiplier": use explicit_multiplier for all checks.
    - "tiered": derive multiplier from account_equity and equity tiers.
    """

    mode: str = "disabled"
    explicit_multiplier: Optional[float] = None
    tier_no_margin_equity: float = 2_000.0
    tier_day_trader_equity: float = 25_000.0
    tier_multiplier_below_min_equity: float = 1.0
    tier_multiplier_reg_t: float = 2.0
    tier_multiplier_day_trader: float = 4.0

    def __post_init__(self) -> None:
        if self.mode not in _VALID_BUYING_POWER_MODES:
            raise ValueError(
                f"Invalid buying_power.mode={self.mode!r}. "
                f"Expected one of {sorted(_VALID_BUYING_POWER_MODES)}."
            )

        if self.explicit_multiplier is not None and self.explicit_multiplier <= 0:
            raise ValueError("explicit_multiplier must be > 0 when provided.")

        if self.mode == "multiplier" and self.explicit_multiplier is None:
            raise ValueError("explicit_multiplier is required when mode='multiplier'.")

        if self.tier_no_margin_equity < 0:
            raise ValueError("tier_no_margin_equity must be >= 0.")
        if self.tier_day_trader_equity <= self.tier_no_margin_equity:
            raise ValueError(
                "tier_day_trader_equity must be greater than tier_no_margin_equity."
            )

        multipliers = (
            self.tier_multiplier_below_min_equity,
            self.tier_multiplier_reg_t,
            self.tier_multiplier_day_trader,
        )
        if any(m <= 0 for m in multipliers):
            raise ValueError("All tier multipliers must be > 0.")

    def resolve_multiplier(self, account_equity: Optional[float]) -> float:
        if self.mode == "disabled":
            return 1.0

        if self.mode == "multiplier":
            # __post_init__ enforces explicit_multiplier for this mode.
            return float(self.explicit_multiplier)  # type: ignore[arg-type]

        if account_equity is None:
            raise ValueError("account_equity is required when mode='tiered'.")
        if account_equity < 0:
            raise ValueError("account_equity must be >= 0.")

        if account_equity < self.tier_no_margin_equity:
            return self.tier_multiplier_below_min_equity
        if account_equity < self.tier_day_trader_equity:
            return self.tier_multiplier_reg_t
        return self.tier_multiplier_day_trader


@dataclass(frozen=True)
class PaperParityConfig:
    """
    Shared parity feature switches consumed by single-symbol and multi-asset backtests.
    """

    enabled: bool = False
    require_tradable: bool = False
    require_shortable: bool = False
    require_easy_to_borrow: bool = False
    allow_crypto_shorts: bool = True
    buying_power: BuyingPowerConfig = field(default_factory=BuyingPowerConfig)
    account_equity: Optional[float] = None
    short_open_valuation_buffer: float = 1.03
    reserve_open_orders: bool = False
    reserve_open_orders_for_shorts_only: bool = False

    def __post_init__(self) -> None:
        if self.short_open_valuation_buffer <= 0:
            raise ValueError("short_open_valuation_buffer must be > 0.")
        if self.account_equity is not None and self.account_equity < 0:
            raise ValueError("account_equity must be >= 0 when provided.")
        if self.require_easy_to_borrow and not self.require_shortable:
            raise ValueError(
                "require_easy_to_borrow=True requires require_shortable=True."
            )

        if (
            self.enabled
            and self.buying_power.mode == "tiered"
            and self.account_equity is None
        ):
            raise ValueError(
                "account_equity is required for parity mode when buying_power.mode='tiered'."
            )

    @classmethod
    def disabled(cls) -> "PaperParityConfig":
        return cls(enabled=False)

    @classmethod
    def enabled_defaults(cls) -> "PaperParityConfig":
        buying_power = BuyingPowerConfig(mode="tiered")
        return cls(
            enabled=True,
            require_tradable=True,
            require_shortable=True,
            require_easy_to_borrow=False,
            allow_crypto_shorts=False,
            buying_power=buying_power,
            account_equity=buying_power.tier_day_trader_equity,
        )

    def buying_power_multiplier(self) -> float:
        if not self.enabled:
            return 1.0
        return self.buying_power.resolve_multiplier(self.account_equity)


def normalize_paper_parity_config(
    config: Optional[PaperParityConfig],
) -> PaperParityConfig:
    if config is None:
        return PaperParityConfig.disabled()
    if not isinstance(config, PaperParityConfig):
        raise TypeError(
            "paper_parity must be a PaperParityConfig or None; "
            f"received {type(config).__name__}."
        )
    return config
