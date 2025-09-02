"""
Trading Strategy Interface

This module defines the formal contract that all trading strategies must implement
to be compatible with the MetaStrategyOrchestrator.

"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TradingStrategy(ABC):
    """
    Abstract base class defining the mandatory interface for all trading strategies.
    
    This class ensures consistency and predictability between the orchestrator
    and any concrete strategy implementation.
    """
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize the strategy with market data and optional configuration.
        
        Args:
            data: DataFrame containing OHLCV data with columns:
                  ['open', 'high', 'low', 'close', 'volume']
            config: Optional configuration dictionary for strategy parameters
        """
        self.data = data
        self.config = config or {}
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        
        # Validate data
        if data is None or data.empty:
            logger.warning(f"{self.name}: Initialized with empty data")
        else:
            self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate that the data contains required columns."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(self.data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    @abstractmethod
    def calculate_signal(self) -> Dict[str, float]:
        """
        Calculate the trading signal based on the strategy's logic.
        
        This is the core method that must be implemented by all strategies.
        
        Returns:
            Dictionary containing at least:
            - 'signal': float between -1 (strong sell) and +1 (strong buy)
            - Additional keys for metadata (e.g., 'confidence', indicator values)
            
            Returns {'signal': 0} if there is insufficient data or an error occurs.
        """
        pass
    
    @abstractmethod
    def get_required_data_points(self) -> int:
        """
        Get the minimum number of data points (bars) required for the strategy.
        
        This ensures the strategy has enough historical data to calculate
        its indicators without NaN values.
        
        Returns:
            Minimum number of bars needed
        """
        pass
    
    @abstractmethod
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        Determine if this strategy should be active given the current market regime.
        
        Args:
            market_bias: Current market regime (e.g., 'Bullish', 'Neutral', 'Bearish')
            
        Returns:
            True if the strategy is suitable for the current market regime
        """
        pass
    
    def has_sufficient_data(self) -> bool:
        """
        Check if there is sufficient data to calculate the strategy.
        
        Returns:
            True if data length >= required data points
        """
        if self.data is None or self.data.empty:
            return False
        return len(self.data) >= self.get_required_data_points()
    
    def get_latest_signal(self) -> Dict[str, float]:
        """
        Convenience method to get the most recent signal.
        
        Returns:
            Signal dictionary or {'signal': 0} if unavailable
        """
        if not self.has_sufficient_data():
            logger.warning(f"{self.name}: Insufficient data for signal calculation")
            return {'signal': 0}
        
        try:
            return self.calculate_signal()
        except Exception as e:
            logger.error(f"{self.name}: Error calculating signal: {e}")
            return {'signal': 0}


class ExampleSMACrossover(TradingStrategy):
    """
    Example implementation of a Simple Moving Average Crossover strategy.
    
    Demonstrates how to implement the TradingStrategy interface.
    """
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """Initialize with default SMA periods."""
        super().__init__(data, config)
        self.name = "SMA Crossover Strategy"
        self.version = "1.0.0"
        
        # Get parameters from config or use defaults
        self.fast_period = self.config.get('fast_period', 10)
        self.slow_period = self.config.get('slow_period', 30)
        
    def calculate_signal(self) -> Dict[str, float]:
        """
        Calculate signal based on SMA crossover.
        
        Returns:
            Signal dictionary with crossover information
        """
        if not self.has_sufficient_data():
            return {'signal': 0, 'fast_sma': None, 'slow_sma': None}
        
        try:
            # Calculate SMAs
            fast_sma = self.data['close'].rolling(window=self.fast_period).mean()
            slow_sma = self.data['close'].rolling(window=self.slow_period).mean()
            
            # Get latest values
            current_fast = fast_sma.iloc[-1]
            current_slow = slow_sma.iloc[-1]
            prev_fast = fast_sma.iloc[-2]
            prev_slow = slow_sma.iloc[-2]
            
            # Detect crossovers
            signal = 0.0
            
            # Bullish crossover
            if prev_fast <= prev_slow and current_fast > current_slow:
                signal = 1.0
            # Bearish crossover
            elif prev_fast >= prev_slow and current_fast < current_slow:
                signal = -1.0
            # Trending
            elif current_fast > current_slow:
                # Bullish trend - moderate buy signal
                signal = 0.5
            elif current_fast < current_slow:
                # Bearish trend - moderate sell signal
                signal = -0.5
            
            # Calculate confidence based on separation
            separation = abs(current_fast - current_slow) / current_slow
            confidence = min(separation * 100, 1.0)  # Cap at 1.0
            
            return {
                'signal': signal,
                'confidence': confidence,
                'fast_sma': current_fast,
                'slow_sma': current_slow,
                'separation_pct': separation * 100
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error in calculation: {e}")
            return {'signal': 0}
    
    def get_required_data_points(self) -> int:
        """Return the slow SMA period as minimum required."""
        return self.slow_period + 1  # +1 for crossover detection
    
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        SMA crossover works best in trending markets.
        
        Args:
            market_bias: Current market regime
            
        Returns:
            True if market is trending
        """
        trending_regimes = ['Strong Bullish', 'Bullish', 'Bearish', 'Strong Bearish']
        return market_bias in trending_regimes
