from __future__ import annotations

from dataclasses import dataclass
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
                tradable=bool(value.get("tradable", True)),
                shortable=bool(value.get("shortable", True)),
                easy_to_borrow=bool(value.get("easy_to_borrow", True)),
            )
            continue
        raise TypeError(
            f"Unsupported asset flag payload for symbol {sym}: {type(value).__name__}"
        )
    return out


def opening_short_qty(*, side: str, qty: int, current_position: int) -> int:
    if side != "sell" or qty <= 0:
        return 0
    long_to_close = max(0, int(current_position))
    return max(0, int(qty) - long_to_close)


def evaluate_asset_eligibility(
    *,
    symbol: str,
    side: str,
    qty: int,
    current_position: int,
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
        qty=int(qty),
        current_position=int(current_position),
    )
    if short_open_qty <= 0:
        return True, ELIGIBILITY_OK

    if parity.require_shortable and not bool(flags.shortable):
        return False, REJECT_NOT_SHORTABLE
    if parity.require_easy_to_borrow and not bool(flags.easy_to_borrow):
        return False, REJECT_NOT_EASY_TO_BORROW

    return True, ELIGIBILITY_OK
