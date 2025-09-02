"""
Trading Strategies Module

This module contains various trading strategies for the Trading Bot.
Each strategy is implemented as a separate class with standardized methods
for entry/exit conditions, risk management, and SQL queries.
"""

from .RSI_Momentum_Divergence_Swing_Strategy import RSIMomentumDivergenceSwingStrategy
from .MACD_Momentum_Crossover_Strategy import MACDMomentumCrossoverStrategy
from .Bollinger_Bands_Mean_Reversion_Strategy import BollingerBandsMeanReversionStrategy
from .Ichimoku_Cloud_Breakout_Strategy import IchimokuCloudBreakoutStrategy
from .Parabolic_SAR_Trend_Following_Strategy import ParabolicSARTrendFollowingStrategy
from .Fibonacci_Retracement_Support_Resistance_Strategy import FibonacciRetracementSupportResistanceStrategy
from .Gaussian_Channel_Breakout_Mean_Reversion_Strategy import GaussianChannelBreakoutMeanReversionStrategy
__all__ = [
    'RSIMomentumDivergenceSwingStrategy',
    'MACDMomentumCrossoverStrategy',
    'BollingerBandsMeanReversionStrategy',
    'IchimokuCloudBreakoutStrategy',
    'ParabolicSARTrendFollowingStrategy',
    'FibonacciRetracementSupportResistanceStrategy',
    'GaussianChannelBreakoutMeanReversionStrategy'
]
