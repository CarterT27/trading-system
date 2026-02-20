from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from core.asset_eligibility import (
    evaluate_asset_eligibility,
    normalize_asset_flags_by_symbol,
)
from core.gateway import MarketDataGateway
from core.matching_engine import MatchingEngine
from core.order_book import Order, OrderBook
from core.order_manager import OrderLoggingGateway, OrderManager
from core.paper_parity import PaperParityConfig, normalize_paper_parity_config
from strategies import MovingAverageStrategy, Strategy

DATA_DIR = Path("data")


@dataclass
class TradeRecord:
    timestamp: pd.Timestamp
    side: str
    price: float
    qty: float
    status: str
    pnl: float


class Backtester:
    """
    Integrates market data, strategy, order management, order book, and matching
    engine components to simulate trading activity.
    """

    def __init__(
        self,
        data_gateway: MarketDataGateway,
        strategy: Strategy,
        order_manager: OrderManager,
        order_book: OrderBook,
        matching_engine: MatchingEngine,
        logger: Optional[OrderLoggingGateway] = None,
        default_position_size: int = 10,
        verbose: bool = True,
        show_progress: bool = False,
        asset_class: str = "stock",
        paper_parity: Optional[PaperParityConfig] = None,
        asset_flags_by_symbol: Optional[Dict[str, object]] = None,
    ):
        self.data_gateway = data_gateway
        self.strategy = strategy
        self.order_manager = order_manager
        self.order_book = order_book
        self.matching_engine = matching_engine
        self.logger = logger
        self.default_position_size = default_position_size
        self.verbose = verbose
        self.show_progress = bool(show_progress)
        asset_class = str(asset_class).strip().lower()
        if asset_class not in {"stock", "crypto"}:
            raise ValueError("asset_class must be 'stock' or 'crypto'.")
        self.asset_class = asset_class
        self.paper_parity = normalize_paper_parity_config(paper_parity)
        self.asset_flags_by_symbol = normalize_asset_flags_by_symbol(asset_flags_by_symbol)

        self.market_history: List[Dict] = []
        self.equity_curve: List[float] = []
        self.cash_history: List[float] = []
        self.position_history: List[float] = []
        self.trades: List[TradeRecord] = []

        self._order_counter = 0
        self._long_inventory = 0.0
        self._short_inventory = 0.0
        self._long_avg_price = 0.0
        self._short_avg_price = 0.0

    # ----------------------------------------------------------------- helpers

    def _log(self, event_type: str, data: Dict) -> None:
        if self.logger:
            self.logger.log(event_type, data)

    def _next_order_id(self) -> str:
        order_id = f"order_{self._order_counter}"
        self._order_counter += 1
        return order_id

    def _create_order(
        self, signal: int, price: float, timestamp: pd.Timestamp, qty: float
    ) -> Order:
        return Order(
            order_id=self._next_order_id(),
            side="buy" if signal > 0 else "sell",
            price=price,
            qty=qty,
            timestamp=timestamp.timestamp(),
        )

    def _format_qty(self, qty: float) -> str:
        if self.asset_class == "crypto":
            return f"{float(qty):.6f}".rstrip("0").rstrip(".")
        return str(int(round(float(qty))))

    def _update_equity(self, price: float) -> None:
        equity = self.order_manager.portfolio_value(price)
        self.equity_curve.append(equity)
        self.cash_history.append(self.order_manager.cash)
        self.position_history.append(self.order_manager.net_position)

    def _apply_fill(self, order: Order, filled_qty: float, price: float) -> float:
        """
        Update inventory tracking for realized PnL statistics.
        """
        realized = 0.0
        qty_remaining = float(filled_qty)

        if order.side == "buy":
            if self._short_inventory > 0:
                cover = min(qty_remaining, self._short_inventory)
                pnl = (self._short_avg_price - price) * cover
                realized += pnl
                self._short_inventory -= cover
                qty_remaining -= cover
                if self._short_inventory == 0:
                    self._short_avg_price = 0.0
            if qty_remaining > 0:
                total_cost = self._long_avg_price * self._long_inventory + price * qty_remaining
                self._long_inventory += qty_remaining
                self._long_avg_price = total_cost / self._long_inventory

        else:
            if self._long_inventory > 0:
                close = min(qty_remaining, self._long_inventory)
                pnl = (price - self._long_avg_price) * close
                realized += pnl
                self._long_inventory -= close
                qty_remaining -= close
                if self._long_inventory == 0:
                    self._long_avg_price = 0.0
            if qty_remaining > 0:
                total_credit = self._short_avg_price * self._short_inventory + price * qty_remaining
                self._short_inventory += qty_remaining
                self._short_avg_price = total_credit / self._short_inventory

        return realized

    def _print_trade(
        self,
        order: Order,
        filled_qty: float,
        price: float,
        timestamp: pd.Timestamp,
        status: str,
    ) -> None:
        if not self.verbose:
            return
        symbol = getattr(self.data_gateway, "symbol", "ASSET")
        net_pnl = self.order_manager.portfolio_value(price) - self.order_manager.initial_capital
        side = order.side.upper()
        qty_text = self._format_qty(filled_qty)
        print(
            f"{timestamp:%Y-%m-%d %H:%M:%S} | {side} {qty_text} {symbol} @ {price:.2f} "
            f"| status={status} | net_pnl={net_pnl:+.2f}"
        )

    def _submit_order(self, order: Order, timestamp: pd.Timestamp, quantity: float) -> None:
        reserved = self.order_manager.reserved_for_order(order.order_id)
        if reserved > 0:
            self._log(
                "reservation",
                {
                    "action": "reserve",
                    "order_id": order.order_id,
                    "amount": reserved,
                },
            )
        self.order_book.add_order(order)
        self._log("submitted", order.__dict__)

        # Add synthetic liquidity so the order book can match.
        liquidity_order = Order(
            order_id=f"liq_{order.order_id}",
            side="sell" if order.side == "buy" else "buy",
            price=order.price,
            qty=quantity,
            timestamp=timestamp.timestamp(),
        )
        self.order_book.add_order(liquidity_order)

        trades = self.order_book.match()
        for trade in trades:
            if order.order_id not in (trade["bid_id"], trade["ask_id"]):
                continue

            exec_report = self.matching_engine.simulate_execution(order, trade["qty"], trade["price"])
            self._log("execution", exec_report)
            status = exec_report["status"]
            if status == "cancelled":
                self._log("cancelled", {"order_id": order.order_id})

            filled_qty = exec_report["filled_qty"]
            realized = 0.0
            if filled_qty > 0:
                realized = self._apply_fill(order, filled_qty, trade["price"])
                self.order_manager.record_execution(order, filled_qty, trade["price"])
                self._print_trade(order, filled_qty, trade["price"], timestamp, status)

            released = self.order_manager.reconcile_reservation(
                order,
                filled_qty=filled_qty,
                status=status,
                reference_price=float(trade["price"]),
                paper_parity=self.paper_parity,
            )
            if released > 0:
                self._log(
                    "reservation",
                    {
                        "action": "release",
                        "order_id": order.order_id,
                        "amount": released,
                        "status": status,
                    },
                )

            self.trades.append(
                TradeRecord(
                    timestamp=timestamp,
                    side=order.side,
                    price=trade["price"],
                    qty=filled_qty,
                    status=status,
                    pnl=realized,
                )
            )

    def _asset_eligibility(self, order: Order) -> tuple[bool, str]:
        symbol = str(getattr(self.data_gateway, "symbol", "ASSET")).upper()
        return evaluate_asset_eligibility(
            symbol=symbol,
            side=order.side,
            qty=order.qty,
            current_position=self.order_manager.net_position,
            parity=self.paper_parity,
            asset_flags_by_symbol=self.asset_flags_by_symbol,
        )

    # ------------------------------------------------------------------- main

    def run(self) -> pd.DataFrame:
        stream_iter = self.data_gateway.stream()
        progress = None
        if self.show_progress:
            total = int(getattr(self.data_gateway, "length", 0) or 0)
            progress = tqdm(
                stream_iter,
                total=total if total > 0 else None,
                desc=f"Backtest {getattr(self.data_gateway, 'symbol', 'ASSET')}",
                unit="bar",
                dynamic_ncols=True,
            )
            row_iter = progress
        else:
            row_iter = stream_iter

        required_lookback = int(getattr(self.strategy, "required_lookback", 0) or 0)
        if required_lookback > 0:
            strategy_window_rows: deque[Dict] = deque(maxlen=required_lookback)
            strategy_window_index: deque[int] = deque(maxlen=required_lookback)
        else:
            strategy_window_rows = deque()
            strategy_window_index = deque()
        incremental_bar_index = 0

        reset_incremental = getattr(self.strategy, "reset_incremental_state", None)
        if callable(reset_incremental):
            reset_incremental()

        try:
            for row in row_iter:
                self.market_history.append(row)
                if hasattr(self.strategy, "update_context"):
                    try:
                        self.strategy.update_context(position=self.order_manager.net_position)
                    except TypeError:
                        # Backwards compatibility if a strategy ignores context.
                        pass

                latest_map: Dict[str, object] | None = None
                latest_columns: set[str] = set()
                run_incremental = getattr(self.strategy, "run_incremental", None)
                if callable(run_incremental):
                    try:
                        incremental_out = run_incremental(
                            row=row, bar_index=incremental_bar_index
                        )
                    except TypeError:
                        try:
                            incremental_out = run_incremental(row)
                        except TypeError:
                            incremental_out = run_incremental()
                    if isinstance(incremental_out, pd.Series):
                        latest_map = incremental_out.to_dict()
                        latest_columns = set(incremental_out.index)
                    elif isinstance(incremental_out, dict):
                        latest_map = dict(incremental_out)
                        latest_columns = set(latest_map.keys())
                    elif incremental_out is not None:
                        raise TypeError(
                            "run_incremental must return dict/pd.Series/None."
                        )

                if latest_map is None:
                    strategy_window_rows.append(dict(row))
                    strategy_window_index.append(int(incremental_bar_index))
                    strategy_df = pd.DataFrame(
                        list(strategy_window_rows),
                        index=list(strategy_window_index),
                    )
                    signals_df = self.strategy.run(strategy_df)
                    latest = signals_df.iloc[-1]
                    latest_map = latest.to_dict()
                    latest_columns = set(signals_df.columns)

                timestamp = pd.Timestamp(row["Datetime"])
                close_value = latest_map.get("Close", row.get("Close"))
                if close_value is None or pd.isna(close_value):
                    incremental_bar_index += 1
                    continue
                latest_close = float(close_value)
                price = latest_close
                self._update_equity(price)

                # ------------------------------------------------------------------
                # Strategy can either emit per-side quotes (bid/ask) or a single
                # directional signal (legacy). Prefer the richer quote interface.
                # ------------------------------------------------------------------
                submitted_any = False

                if {"bid_price", "ask_price"} <= latest_columns:
                    orders_to_submit = []

                    bid_active = bool(latest_map.get("bid_active", True))
                    ask_active = bool(latest_map.get("ask_active", True))
                    bid_price = latest_map.get("bid_price")
                    ask_price = latest_map.get("ask_price")

                    if bid_active and pd.notna(bid_price):
                        bid_qty_val = latest_map.get("bid_qty", self.default_position_size)
                        bid_qty = (
                            float(bid_qty_val)
                            if pd.notna(bid_qty_val) and bid_qty_val > 0
                            else float(self.default_position_size)
                        )
                        orders_to_submit.append((1, float(bid_price), bid_qty))

                    if ask_active and pd.notna(ask_price):
                        ask_qty_val = latest_map.get("ask_qty", self.default_position_size)
                        ask_qty = (
                            float(ask_qty_val)
                            if pd.notna(ask_qty_val) and ask_qty_val > 0
                            else float(self.default_position_size)
                        )
                        orders_to_submit.append((-1, float(ask_price), ask_qty))

                    for sig, px, qty in orders_to_submit:
                        if self.asset_class == "crypto" and sig < 0:
                            net_pos = float(self.order_manager.net_position)
                            if net_pos <= 0:
                                self._log(
                                    "rejected",
                                    {
                                        "order_id": "pending",
                                        "reason": "crypto shorting disabled",
                                    },
                                )
                                continue
                            qty = min(float(qty), net_pos)
                            qty = np.floor(qty * 1_000_000.0) / 1_000_000.0
                            if qty <= 0:
                                continue
                        order = self._create_order(sig, px, timestamp, qty)
                        eligible, reason = self._asset_eligibility(order)
                        if not eligible:
                            self._log("rejected", {"order_id": order.order_id, "reason": reason})
                            continue
                        valid, reason = self.order_manager.validate(
                            order,
                            reference_price=latest_close,
                            paper_parity=self.paper_parity,
                        )
                        if not valid:
                            self._log("rejected", {"order_id": order.order_id, "reason": reason})
                            continue
                        self._submit_order(order, timestamp, qty)
                        submitted_any = True

                if submitted_any:
                    incremental_bar_index += 1
                    continue

                # Fallback: classic single signal / limit_price pattern.
                signal_value = latest_map.get("signal", 0)
                signal = int(signal_value) if pd.notna(signal_value) else 0
                if signal == 0:
                    incremental_bar_index += 1
                    continue

                limit_price = latest_map.get("limit_price", latest_close)
                price = float(limit_price) if pd.notna(limit_price) else latest_close
                qty_value = latest_map.get("target_qty", self.default_position_size)
                qty_raw = (
                    float(qty_value)
                    if pd.notna(qty_value) and float(qty_value) > 0
                    else float(self.default_position_size)
                )
                if qty_raw <= 0:
                    incremental_bar_index += 1
                    continue

                if self.asset_class == "crypto":
                    qty = np.floor((qty_raw / max(price, 1e-12)) * 1_000_000.0) / 1_000_000.0
                    if qty <= 0:
                        incremental_bar_index += 1
                        continue
                    if signal < 0:
                        net_pos = float(self.order_manager.net_position)
                        if net_pos <= 0:
                            self._log("rejected", {"order_id": "pending", "reason": "crypto shorting disabled"})
                            incremental_bar_index += 1
                            continue
                        qty = min(qty, net_pos)
                        qty = np.floor(qty * 1_000_000.0) / 1_000_000.0
                        if qty <= 0:
                            incremental_bar_index += 1
                            continue
                else:
                    qty = float(int(qty_raw))
                    if qty <= 0:
                        incremental_bar_index += 1
                        continue

                order = self._create_order(signal, price, timestamp, qty)
                eligible, reason = self._asset_eligibility(order)
                if not eligible:
                    self._log("rejected", {"order_id": order.order_id, "reason": reason})
                    incremental_bar_index += 1
                    continue
                valid, reason = self.order_manager.validate(
                    order,
                    reference_price=latest_close,
                    paper_parity=self.paper_parity,
                )
                if not valid:
                    self._log("rejected", {"order_id": order.order_id, "reason": reason})
                    incremental_bar_index += 1
                    continue

                self._submit_order(order, timestamp, qty)
                incremental_bar_index += 1
        finally:
            if progress is not None:
                progress.close()

        return pd.DataFrame(
            {
                "equity": self.equity_curve,
                "cash": self.cash_history,
                "position": self.position_history,
            }
        )


class PerformanceAnalyzer:
    def __init__(self, equity_curve: List[float], trades: List[TradeRecord]):
        self.equity_curve = np.array(equity_curve, dtype=float)
        self.trades = trades

    def pnl(self) -> float:
        if self.equity_curve.size == 0:
            return 0.0
        return float(self.equity_curve[-1] - self.equity_curve[0])

    def returns(self) -> np.ndarray:
        if self.equity_curve.size < 2:
            return np.array([])
        return np.diff(self.equity_curve) / self.equity_curve[:-1]

    def sharpe(self, rf: float = 0.0) -> float:
        r = self.returns()
        if r.size == 0 or r.std() == 0:
            return 0.0
        return float((r.mean() - rf) / r.std() * np.sqrt(252 * 6.5 * 60))

    def max_drawdown(self) -> float:
        if self.equity_curve.size == 0:
            return 0.0
        cummax = np.maximum.accumulate(self.equity_curve)
        drawdowns = (self.equity_curve - cummax) / cummax
        return float(drawdowns.min())

    def win_rate(self) -> float:
        realized = [t.pnl for t in self.trades if t.pnl != 0]
        if not realized:
            return 0.0
        wins = sum(1 for pnl in realized if pnl > 0)
        return wins / len(realized)


def plot_equity(equity_df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(equity_df["equity"], label="Equity")
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def run_sample_backtest(
    csv_path: str,
    strategy: Optional[Strategy] = None,
    title: Optional[str] = None,
) -> PerformanceAnalyzer:
    gateway = MarketDataGateway(csv_path)
    strategy = strategy or MovingAverageStrategy(short_window=5, long_window=15, position_size=10)
    order_book = OrderBook()
    order_manager = OrderManager(capital=50_000, max_long_position=1_000, max_short_position=1_000)
    matching_engine = MatchingEngine()
    logger = OrderLoggingGateway()

    bt = Backtester(
        data_gateway=gateway,
        strategy=strategy,
        order_manager=order_manager,
        order_book=order_book,
        matching_engine=matching_engine,
        logger=logger,
    )

    equity_df = bt.run()
    analyzer = PerformanceAnalyzer(equity_df["equity"].tolist(), bt.trades)

    if title:
        print(f"\n=== {title} ===")
    print("PnL:", analyzer.pnl())
    print("Sharpe:", analyzer.sharpe())
    print("Max Drawdown:", analyzer.max_drawdown())
    print("Win Rate:", analyzer.win_rate())
    print(f"Trades executed: {len([t for t in bt.trades if t.qty > 0])}")
    return analyzer


if __name__ == "__main__":
    sample_csv = DATA_DIR / "sample_system_test_data.csv"
    if not sample_csv.exists():
        # Create a lightweight dataset for demonstration.
        dates = pd.date_range(start="2024-01-01 09:30", periods=200, freq="T")
        df = pd.DataFrame(
            {
                "Datetime": dates,
                "Open": np.random.uniform(100, 105, len(dates)),
                "High": np.random.uniform(105, 110, len(dates)),
                "Low": np.random.uniform(95, 100, len(dates)),
                "Close": np.random.uniform(100, 110, len(dates)),
                "Volume": np.random.randint(1_000, 5_000, len(dates)),
            }
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(sample_csv, index=False)

    ma_strategy = MovingAverageStrategy(short_window=5, long_window=15, position_size=10)
    run_sample_backtest(str(sample_csv), strategy=ma_strategy, title="Moving Average Baseline")
