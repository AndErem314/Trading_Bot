"""
Executable Trading Strategies Package

This package contains executable implementations of trading strategies
that conform to the TradingStrategy interface.
"""

from .bollinger_bands_strategy import BollingerBandsMeanReversion
from .rsi_momentum_strategy import RSIMomentumDivergence
from .macd_momentum_strategy import MACDMomentumCrossover
from .sma_golden_cross_strategy import SMAGoldenCross
from .ichimoku_cloud_strategy import IchimokuCloudBreakout
from .parabolic_sar_strategy import ParabolicSARTrendFollowing
from .fibonacci_retracement_strategy import FibonacciRetracementSupportResistance
from .gaussian_channel_strategy import GaussianChannelBreakoutMeanReversion

# Dynamic strategy builder
from .strategy_builder import StrategyBuilder, Position, Trade, ExitType

__all__ = [
    # Static strategy implementations
    'BollingerBandsMeanReversion',
    'RSIMomentumDivergence',
    'MACDMomentumCrossover',
    'SMAGoldenCross',
    'IchimokuCloudBreakout',
    'ParabolicSARTrendFollowing',
    'FibonacciRetracementSupportResistance',
    'GaussianChannelBreakoutMeanReversion',
    # Dynamic strategy builder
    'StrategyBuilder',
    'Position',
    'Trade',
    'ExitType'
]
