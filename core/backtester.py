from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    qty: int
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
        paper_parity: Optional[PaperParityConfig] = None,
    ):
        self.data_gateway = data_gateway
        self.strategy = strategy
        self.order_manager = order_manager
        self.order_book = order_book
        self.matching_engine = matching_engine
        self.logger = logger
        self.default_position_size = default_position_size
        self.verbose = verbose
        self.paper_parity = normalize_paper_parity_config(paper_parity)

        self.market_history: List[Dict] = []
        self.equity_curve: List[float] = []
        self.cash_history: List[float] = []
        self.position_history: List[int] = []
        self.trades: List[TradeRecord] = []

        self._order_counter = 0
        self._long_inventory = 0
        self._short_inventory = 0
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

    def _create_order(self, signal: int, price: float, timestamp: pd.Timestamp, qty: int) -> Order:
        return Order(
            order_id=self._next_order_id(),
            side="buy" if signal > 0 else "sell",
            price=price,
            qty=qty,
            timestamp=timestamp.timestamp(),
        )

    def _update_equity(self, price: float) -> None:
        equity = self.order_manager.portfolio_value(price)
        self.equity_curve.append(equity)
        self.cash_history.append(self.order_manager.cash)
        self.position_history.append(self.order_manager.net_position)

    def _apply_fill(self, order: Order, filled_qty: int, price: float) -> float:
        """
        Update inventory tracking for realized PnL statistics.
        """
        realized = 0.0
        qty_remaining = filled_qty

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
        filled_qty: int,
        price: float,
        timestamp: pd.Timestamp,
        status: str,
    ) -> None:
        if not self.verbose:
            return
        symbol = getattr(self.data_gateway, "symbol", "ASSET")
        net_pnl = self.order_manager.portfolio_value(price) - self.order_manager.initial_capital
        side = order.side.upper()
        print(
            f"{timestamp:%Y-%m-%d %H:%M:%S} | {side} {filled_qty} {symbol} @ {price:.2f} "
            f"| status={status} | net_pnl={net_pnl:+.2f}"
        )

    def _submit_order(self, order: Order, timestamp: pd.Timestamp, quantity: int) -> None:
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

    # ------------------------------------------------------------------- main

    def run(self) -> pd.DataFrame:
        for row in self.data_gateway.stream():
            self.market_history.append(row)
            market_df = pd.DataFrame(self.market_history)
            if hasattr(self.strategy, "update_context"):
                try:
                    self.strategy.update_context(position=self.order_manager.net_position)
                except TypeError:
                    # Backwards compatibility if a strategy ignores context.
                    pass

            signals_df = self.strategy.run(market_df)
            latest = signals_df.iloc[-1]
            timestamp = pd.Timestamp(row["Datetime"])

            price = float(latest["Close"])
            self._update_equity(price)

            # ------------------------------------------------------------------
            # Strategy can either emit per-side quotes (bid/ask) or a single
            # directional signal (legacy). Prefer the richer quote interface.
            # ------------------------------------------------------------------
            submitted_any = False

            if {"bid_price", "ask_price"} <= set(signals_df.columns):
                orders_to_submit = []

                bid_active = bool(latest.get("bid_active", True))
                ask_active = bool(latest.get("ask_active", True))
                bid_price = latest.get("bid_price")
                ask_price = latest.get("ask_price")

                if bid_active and pd.notna(bid_price):
                    bid_qty_val = latest.get("bid_qty", self.default_position_size)
                    bid_qty = int(bid_qty_val) if pd.notna(bid_qty_val) and bid_qty_val > 0 else self.default_position_size
                    orders_to_submit.append((1, float(bid_price), bid_qty))

                if ask_active and pd.notna(ask_price):
                    ask_qty_val = latest.get("ask_qty", self.default_position_size)
                    ask_qty = int(ask_qty_val) if pd.notna(ask_qty_val) and ask_qty_val > 0 else self.default_position_size
                    orders_to_submit.append((-1, float(ask_price), ask_qty))

                for sig, px, qty in orders_to_submit:
                    order = self._create_order(sig, px, timestamp, qty)
                    valid, reason = self.order_manager.validate(order)
                    if not valid:
                        self._log("rejected", {"order_id": order.order_id, "reason": reason})
                        continue
                    self._submit_order(order, timestamp, qty)
                    submitted_any = True

            if submitted_any:
                continue

            # Fallback: classic single signal / limit_price pattern.
            signal_value = latest.get("signal", 0)
            signal = int(signal_value) if pd.notna(signal_value) else 0
            if signal == 0:
                continue

            limit_price = latest.get("limit_price", latest["Close"])
            price = float(limit_price) if pd.notna(limit_price) else float(latest["Close"])
            qty_value = latest.get("target_qty", self.default_position_size)
            qty = int(qty_value) if pd.notna(qty_value) and qty_value > 0 else self.default_position_size

            order = self._create_order(signal, price, timestamp, qty)
            valid, reason = self.order_manager.validate(order)
            if not valid:
                self._log("rejected", {"order_id": order.order_id, "reason": reason})
                continue

            self._submit_order(order, timestamp, qty)

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
