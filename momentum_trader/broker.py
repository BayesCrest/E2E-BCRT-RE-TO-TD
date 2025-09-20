"""Execution layer for the momentum trading engine."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from .portfolio import Order, Portfolio

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutionReport:
    orders: list[Order]
    cash: float
    equity: float


class PaperBroker:
    """Simplified broker that executes orders immediately at the provided price."""

    def __init__(self, portfolio: Portfolio, transaction_cost: float) -> None:
        self.portfolio = portfolio
        self.transaction_cost = transaction_cost

    def execute(self, orders: Iterable[Order]) -> ExecutionReport:
        orders = list(orders)
        if not orders:
            LOGGER.debug("No orders to execute.")
            return ExecutionReport(orders=[], cash=self.portfolio.cash, equity=self.portfolio.equity)

        LOGGER.info("Executing %d orders", len(orders))
        for order in orders:
            LOGGER.info("%s %+.2f @ %.2f", order.ticker, order.quantity, order.price)

        self.portfolio.apply_execution(orders, self.transaction_cost)
        return ExecutionReport(orders=orders, cash=self.portfolio.cash, equity=self.portfolio.equity)
