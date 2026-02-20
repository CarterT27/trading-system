from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional

from core.paper_parity import PaperParityConfig


ELIGIBILITY_OK = "eligible"
REJECT_NOT_TRADABLE = "reject_not_tradable"
REJECT_NOT_SHORTABLE = "reject_not_shortable"
REJECT_NOT_EASY_TO_BORROW = "reject_not_easy_to_borrow"


@dataclass(frozen=True)
class AssetEligibilityFlags:
    tradable: bool = True
    shortable: bool = True
    easy_to_borrow: bool = True


def parse_asset_flag_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return default
        if numeric == 1.0:
            return True
        if numeric == 0.0:
            return False
        return default

    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n"}:
        return False
    return default


def normalize_asset_flags_by_symbol(
    raw: Optional[Mapping[str, Any]],
) -> dict[str, AssetEligibilityFlags]:
    if not raw:
        return {}

    out: dict[str, AssetEligibilityFlags] = {}
    for symbol, value in raw.items():
        sym = str(symbol).strip().upper()
        if not sym:
            continue
        if isinstance(value, AssetEligibilityFlags):
            out[sym] = value
            continue
        if isinstance(value, Mapping):
            out[sym] = AssetEligibilityFlags(
                tradable=parse_asset_flag_bool(value.get("tradable", True), default=True),
                shortable=parse_asset_flag_bool(value.get("shortable", True), default=True),
                easy_to_borrow=parse_asset_flag_bool(
                    value.get("easy_to_borrow", True),
                    default=True,
                ),
            )
            continue
        raise TypeError(
            f"Unsupported asset flag payload for symbol {sym}: {type(value).__name__}"
        )
    return out


def opening_short_qty(*, side: str, qty: float, current_position: float) -> float:
    if side != "sell" or qty <= 0:
        return 0.0
    long_to_close = max(0.0, float(current_position))
    return max(0.0, float(qty) - long_to_close)


def evaluate_asset_eligibility(
    *,
    symbol: str,
    side: str,
    qty: float,
    current_position: float,
    parity: PaperParityConfig,
    asset_flags_by_symbol: Optional[Mapping[str, AssetEligibilityFlags]] = None,
) -> tuple[bool, str]:
    if not parity.enabled:
        return True, ELIGIBILITY_OK

    flags_map = asset_flags_by_symbol or {}
    flags = flags_map.get(str(symbol).strip().upper(), AssetEligibilityFlags())

    if parity.require_tradable and not bool(flags.tradable):
        return False, REJECT_NOT_TRADABLE

    short_open_qty = opening_short_qty(
        side=side,
        qty=float(qty),
        current_position=float(current_position),
    )
    if short_open_qty <= 0:
        return True, ELIGIBILITY_OK

    if parity.require_shortable and not bool(flags.shortable):
        return False, REJECT_NOT_SHORTABLE
    if parity.require_easy_to_borrow and not bool(flags.easy_to_borrow):
        return False, REJECT_NOT_EASY_TO_BORROW

    return True, ELIGIBILITY_OK
