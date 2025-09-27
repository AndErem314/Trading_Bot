"""
Backtesting module for trading strategies.

This module provides comprehensive backtesting capabilities including:
- Event-driven simulation engine
- Realistic trade execution with slippage and commissions
- Multi-timeframe support
- Performance metrics and analysis
"""

from .ichimoku_backtester import (
    IchimokuBacktester,
    Order,
    OrderType,
    OrderStatus,
    PositionSide,
    Position,
    TradeRecord,
    Fill,
    BacktestEvent,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent
)

__all__ = [
    'IchimokuBacktester',
    'Order',
    'OrderType', 
    'OrderStatus',
    'PositionSide',
    'Position',
    'TradeRecord',
    'Fill',
    'BacktestEvent',
    'MarketEvent',
    'SignalEvent',
    'OrderEvent',
    'FillEvent'
]