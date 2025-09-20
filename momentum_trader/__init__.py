"""Momentum trading automation package."""
from .broker import ExecutionReport, PaperBroker
from .config import DateRange, MomentumConfig
from .data import CSVPriceProvider, PriceDataProvider, YahooPriceProvider
from .engine import EngineResult, MomentumEngine
from .portfolio import Order, Portfolio, Position
from .strategy import MomentumSignal, MomentumStrategy

__all__ = [
    "ExecutionReport",
    "PaperBroker",
    "DateRange",
    "MomentumConfig",
    "CSVPriceProvider",
    "PriceDataProvider",
    "YahooPriceProvider",
    "EngineResult",
    "MomentumEngine",
    "Order",
    "Portfolio",
    "Position",
    "MomentumSignal",
    "MomentumStrategy",
]
