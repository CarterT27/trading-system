import heapq
import math
import time

class Order:
    """
    Basic order object.
    """
    def __init__(self, order_id, side, price, qty, timestamp=None):
        self.order_id = order_id
        self.side = side      # 'buy' or 'sell'
        self.price = price
        self.qty = qty
        self.timestamp = timestamp if timestamp else time.time()

    def __lt__(self, other):
        """
        Ensures heap compares based on price priority then time.
        """
        if self.side == "buy":
            return (-self.price, self.timestamp) < (-other.price, other.timestamp)
        else:
            return (self.price, self.timestamp) < (other.price, other.timestamp)


class OrderBook:
    """
    Manages bid/ask orders and performs price-time priority matching.
    """
    QTY_EPS = 1e-12

    def __init__(self):
        self.bids = []  # max heap
        self.asks = []  # min heap
        self.order_map = {}  # track active orders by ID

    def add_order(self, order: Order):
        self.order_map[order.order_id] = order
        if order.side == "buy":
            heapq.heappush(self.bids, order)
        else:
            heapq.heappush(self.asks, order)

    def cancel_order(self, order_id):
        if order_id in self.order_map:
            self.order_map[order_id].qty = 0  # mark as canceled
            del self.order_map[order_id]

    def modify_order(self, order_id, new_price, new_qty):
        """
        Simplest implementation: cancel and re-add.
        """
        if order_id in self.order_map:
            old = self.order_map[order_id]
            self.cancel_order(order_id)

            new_order = Order(
                order_id,
                side=old.side,
                price=new_price,
                qty=new_qty,
                timestamp=old.timestamp
            )
            self.add_order(new_order)

    def match(self):
        """
        Matches orders based on best bid >= best ask.
        Returns list of trades executed.
        """
        trades = []

        while self.bids and self.asks:
            best_bid = self.bids[0]
            best_ask = self.asks[0]

            if best_bid.price < best_ask.price:
                break

            qty = min(best_bid.qty, best_ask.qty)
            trades.append({
                "price": best_ask.price,
                "qty": qty,
                "bid_id": best_bid.order_id,
                "ask_id": best_ask.order_id
            })

            best_bid.qty -= qty
            best_ask.qty -= qty

            if math.isclose(best_bid.qty, 0.0, abs_tol=self.QTY_EPS) or best_bid.qty < self.QTY_EPS:
                heapq.heappop(self.bids)
                self.order_map.pop(best_bid.order_id, None)

            if math.isclose(best_ask.qty, 0.0, abs_tol=self.QTY_EPS) or best_ask.qty < self.QTY_EPS:
                heapq.heappop(self.asks)
                self.order_map.pop(best_ask.order_id, None)
        return trades
