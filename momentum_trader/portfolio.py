"""Portfolio and order management for the trading engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(slots=True)
class Position:
    ticker: str
    quantity: float


@dataclass(slots=True)
class Order:
    ticker: str
    quantity: float
    price: float


class Portfolio:
    """Keep track of holdings and compute orders towards target weights."""

    def __init__(self, cash: float) -> None:
        if cash <= 0:
            raise ValueError("Initial cash must be positive")
        self.cash = cash
        self.positions: Dict[str, Position] = {}
        self.market_prices: Dict[str, float] = {}

    def update_market_prices(self, prices: Dict[str, float]) -> None:
        self.market_prices.update({ticker.upper(): price for ticker, price in prices.items()})

    @property
    def equity(self) -> float:
        return self.cash + sum(
            position.quantity * self.market_prices.get(position.ticker, 0.0)
            for position in self.positions.values()
        )

    def desired_orders(
        self,
        target_weights: Dict[str, float],
        transaction_cost: float = 0.0,
    ) -> List[Order]:
        orders: List[Order] = []
        equity = self.equity
        for ticker, weight in target_weights.items():
            price = self.market_prices.get(ticker)
            if price is None or price <= 0:
                continue
            target_value = weight * equity
            current_qty = self.positions.get(ticker, Position(ticker, 0.0)).quantity
            current_value = current_qty * price
            delta_value = target_value - current_value
            if abs(delta_value) < 1e-8:
                continue
            target_qty = delta_value / price
            effective_qty = self._adjust_for_costs(target_qty, transaction_cost)
            if abs(effective_qty) < 1e-6:
                continue
            orders.append(Order(ticker=ticker, quantity=effective_qty, price=price))
        # Close remaining positions that no longer have a target weight
        for ticker, position in self.positions.items():
            if ticker in target_weights:
                continue
            price = self.market_prices.get(ticker)
            if price is None:
                continue
            if abs(position.quantity) > 1e-6:
                orders.append(Order(ticker=ticker, quantity=-position.quantity, price=price))
        return orders

    def apply_execution(self, orders: Iterable[Order], transaction_cost: float = 0.0) -> None:
        for order in orders:
            value = order.quantity * order.price
            cost = abs(value) * transaction_cost
            self.cash -= value + cost
            position = self.positions.get(order.ticker)
            if position is None:
                position = Position(order.ticker, 0.0)
                self.positions[order.ticker] = position
            position.quantity += order.quantity
            if abs(position.quantity) < 1e-6:
                del self.positions[order.ticker]

    @staticmethod
    def _adjust_for_costs(quantity: float, transaction_cost: float) -> float:
        if transaction_cost <= 0:
            return quantity
        # Adjust order size so that the post-cost value matches the target delta.
        # For small percentages the difference is minimal but this keeps the
        # accounting consistent.
        return quantity / (1 + transaction_cost)

    def snapshot(self) -> Dict[str, Tuple[float, float]]:
        return {
            ticker: (position.quantity, self.market_prices.get(ticker, 0.0))
            for ticker, position in self.positions.items()
        }
