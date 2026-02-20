"""
Market data gateway used by the backtester to stream cleaned historical data
row-by-row, simulating a live feed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional
import time

import pandas as pd


class MarketDataGateway:
    """
    Streams historical market data to consumers. Supports iterator interface and
    an explicit generator via the `stream` method.
    """

    def __init__(
        self,
        csv_path: str | Path,
        symbol: Optional[str] = None,
        tail_bars: Optional[int] = None,
    ):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.symbol = symbol or self._infer_symbol()
        self.data = pd.read_csv(self.csv_path)
        if "Datetime" not in self.data.columns:
            raise ValueError("CSV must contain a Datetime column.")

        self.data["Datetime"] = pd.to_datetime(self.data["Datetime"], errors="coerce")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        # Drop malformed rows so NaNs cannot poison equity calculations.
        self.data = self.data.dropna(subset=["Datetime", "Close"]).copy()
        if self.data.empty:
            raise ValueError(
                f"{self.csv_path} has no valid rows after cleaning Datetime/Close."
            )

        self.data.sort_values("Datetime", inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        if tail_bars is not None:
            tail_n = int(tail_bars)
            if tail_n <= 0:
                raise ValueError("tail_bars must be positive when provided.")
            self.data = self.data.tail(tail_n).reset_index(drop=True)

        self.length = len(self.data)
        self.pointer = 0

    def _infer_symbol(self) -> str:
        stem = self.csv_path.stem
        token = stem.split("_")[0] if stem else "ASSET"
        return token.upper()

    # Iterator protocol -----------------------------------------------------

    def __iter__(self) -> Iterator[Dict]:
        self.reset()
        return self

    def __next__(self) -> Dict:
        if self.pointer >= self.length:
            raise StopIteration

        row = self.data.iloc[self.pointer].to_dict()
        row["Datetime"] = pd.Timestamp(row["Datetime"])
        self.pointer += 1
        return row

    # Helpers ----------------------------------------------------------------

    def reset(self) -> None:
        self.pointer = 0

    def has_next(self) -> bool:
        return self.pointer < self.length

    def get_next(self) -> Optional[Dict]:
        try:
            return next(self)
        except StopIteration:
            return None

    def peek(self) -> Optional[Dict]:
        if not self.has_next():
            return None
        row = self.data.iloc[self.pointer].to_dict()
        row["Datetime"] = pd.Timestamp(row["Datetime"])
        return row

    # Generator --------------------------------------------------------------

    def stream(self, delay: Optional[float] = None, reset: bool = False):
        """
        Yields rows sequentially. Optional delay (seconds) mimics websocket feed.
        """
        if reset:
            self.reset()

        while self.has_next():
            row = next(self)
            yield row

            if delay:
                time.sleep(delay)


# Backwards compatible alias for historical imports.
Gateway = MarketDataGateway
