"""
Trade logging system for the trading platform.

Logs all trades, signals, and system events to both console and files.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path("logs")
TRADE_LOG_FILE = "trades.csv"
SIGNAL_LOG_FILE = "signals.csv"
SYSTEM_LOG_FILE = "system.log"
_console_log_level = logging.INFO


def _ensure_log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def _resolve_log_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    normalized = str(level).strip().upper()
    resolved = logging.getLevelName(normalized)
    if isinstance(resolved, int):
        return resolved
    raise ValueError(f"Unknown log level: {level}")


def _set_logger_console_level(logger: logging.Logger, level: int) -> None:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(level)


def set_console_log_level(level: int | str) -> None:
    """Set console log level for all configured loggers."""
    global _console_log_level
    resolved_level = _resolve_log_level(level)
    _console_log_level = resolved_level

    logger_manager = logging.Logger.manager
    for logger_obj in logger_manager.loggerDict.values():
        if isinstance(logger_obj, logging.Logger):
            _set_logger_console_level(logger_obj, resolved_level)


def get_logger(name: str = "trading") -> logging.Logger:
    """Get a configured logger that writes to both console and file."""
    logger = logging.getLogger(name)

    if logger.handlers:
        _set_logger_console_level(logger, _console_log_level)
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(_console_log_level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    _ensure_log_dir()
    file_handler = logging.FileHandler(LOG_DIR / SYSTEM_LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


class TradeLogger:
    """
    Logs trades to a CSV file for analysis and record-keeping.

    Each trade includes: timestamp, symbol, side, quantity, price,
    order_id, order_type, status, equity, net_pnl, strategy, and notes.
    """

    HEADERS = [
        "timestamp",
        "symbol",
        "side",
        "qty",
        "price",
        "order_type",
        "order_id",
        "status",
        "equity",
        "net_pnl",
        "strategy",
        "notes",
    ]

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trade_file = self.log_dir / TRADE_LOG_FILE
        self.signal_file = self.log_dir / SIGNAL_LOG_FILE
        self._init_csv(self.trade_file, self.HEADERS)
        self._init_signal_csv()

    def _init_csv(self, filepath: Path, headers: list) -> None:
        if not filepath.exists():
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def _init_signal_csv(self) -> None:
        headers = ["timestamp", "symbol", "signal", "price", "strategy", "indicators"]
        self._init_csv(self.signal_file, headers)

    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        order_type: str = "market",
        order_id: Optional[str] = None,
        status: str = "submitted",
        equity: Optional[float] = None,
        net_pnl: Optional[float] = None,
        strategy: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log a trade to CSV and return the trade record."""
        timestamp = datetime.utcnow().isoformat() + "Z"

        record = {
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "order_type": order_type,
            "order_id": order_id or "",
            "status": status,
            "equity": equity or 0.0,
            "net_pnl": net_pnl or 0.0,
            "strategy": strategy or "",
            "notes": notes or "",
        }

        with open(self.trade_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writerow(record)

        return record

    def log_signal(
        self,
        symbol: str,
        signal: int,
        price: float,
        strategy: str,
        indicators: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log a trading signal to CSV."""
        timestamp = datetime.utcnow().isoformat() + "Z"

        with open(self.signal_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    symbol,
                    signal,
                    price,
                    strategy,
                    json.dumps(indicators or {}),
                ]
            )

    def log_skip(
        self,
        symbol: str,
        reason: str,
        strategy: Optional[str] = None,
    ) -> None:
        """Log a skipped trade (no action taken)."""
        self.log_trade(
            symbol=symbol,
            side="none",
            qty=0,
            price=0,
            status="skipped",
            strategy=strategy,
            notes=reason,
        )

    def get_trades(self, limit: Optional[int] = None) -> list:
        """Read recent trades from the log file."""
        if not self.trade_file.exists():
            return []

        trades = []
        with open(self.trade_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)

        if limit:
            trades = trades[-limit:]
        return trades

    def get_session_summary(
        self,
        start_equity: float,
        session_started_at: Optional[datetime] = None,
        start_trade_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive summary of the current trading session."""
        import numpy as np

        trades = self.get_trades()
        if start_trade_count is not None:
            offset = max(0, int(start_trade_count))
            if offset > 0:
                trades = trades[offset:]
        if session_started_at is not None:
            cutoff = session_started_at
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
            filtered_trades = []
            for t in trades:
                ts_raw = str(t.get("timestamp", "")).strip()
                if not ts_raw:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if ts >= cutoff:
                    filtered_trades.append(t)
            trades = filtered_trades

        empty_result = {
            "total_trades": 0,
            "buys": 0,
            "sells": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "start_equity": start_equity,
            "end_equity": start_equity,
            "sharpe_ratio": 0.0,
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "avg_trade_pnl": 0.0,
        }

        if not trades:
            return empty_result

        executed_trades = [
            t for t in trades if t["status"] not in ("skipped", "rejected", "none")
        ]

        if not executed_trades:
            return empty_result

        # Basic counts
        buys = sum(1 for t in executed_trades if t["side"] == "buy")
        sells = sum(1 for t in executed_trades if t["side"] == "sell")

        # Extract equity values for performance calculation
        equities = []
        for t in executed_trades:
            try:
                eq = float(t.get("equity", 0) or 0)
                if eq > 0:
                    equities.append(eq)
            except (ValueError, TypeError):
                pass

        if not equities:
            equities = [start_equity]

        end_equity = equities[-1] if equities else start_equity
        net_pnl = end_equity - start_equity

        # Calculate returns from equity curve
        if len(equities) >= 2:
            equity_arr = np.array(equities)
            returns = np.diff(equity_arr) / equity_arr[:-1]
            returns = returns[np.isfinite(returns)]  # Remove inf/nan

            if len(returns) > 0:
                # Volatility (annualized, assuming ~252 trading days, ~6.5 hours/day, ~60 mins/hour)
                # For intraday, we annualize based on minutes
                volatility = float(np.std(returns)) if len(returns) > 1 else 0.0

                # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
                mean_return = float(np.mean(returns))
                sharpe_ratio = (mean_return / volatility) if volatility > 0 else 0.0
                # Annualize: multiply by sqrt(trading periods per year)
                # ~252 days * 6.5 hours * 60 minutes = ~98,280 minutes
                sharpe_ratio = sharpe_ratio * np.sqrt(252)

                # Max drawdown
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / running_max
                max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0

        # Count wins/losses based on P&L changes
        wins = 0
        losses = 0
        trade_pnls = []

        for i, t in enumerate(executed_trades):
            try:
                pnl = float(t.get("net_pnl", 0) or 0)
                # For the first trade, compare to 0; for subsequent, compare to previous
                if i > 0:
                    prev_pnl = float(executed_trades[i - 1].get("net_pnl", 0) or 0)
                    trade_pnl = pnl - prev_pnl
                else:
                    trade_pnl = pnl

                trade_pnls.append(trade_pnl)
                if trade_pnl > 0:
                    wins += 1
                elif trade_pnl < 0:
                    losses += 1
            except (ValueError, TypeError):
                pass

        total_with_outcome = wins + losses
        win_rate = (wins / total_with_outcome * 100) if total_with_outcome > 0 else 0.0
        avg_trade_pnl = float(np.mean(trade_pnls)) if trade_pnls else 0.0

        return {
            "total_trades": len(executed_trades),
            "buys": buys,
            "sells": sells,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "net_pnl": net_pnl,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility * 100,  # As percentage
            "max_drawdown": max_drawdown * 100,  # As percentage
            "avg_trade_pnl": avg_trade_pnl,
        }


# Global logger instance
_trade_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get the global trade logger instance."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger
