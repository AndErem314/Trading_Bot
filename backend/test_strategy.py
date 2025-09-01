"""
Test Strategy for MetaStrategyOrchestrator
A simple strategy that demonstrates the expected interface
"""

import pandas as pd
import numpy as np


class SimpleTestStrategy:
    """
    A simple test strategy that generates random signals
    for demonstration purposes.
    """
    
    def __init__(self, data=None, symbol=None, timeframe=None):
        """Initialize with data."""
        self.data = data
        self.symbol = symbol
        self.timeframe = timeframe
        self.name = "SimpleTestStrategy"
        
    def generate_signals(self):
        """Generate trading signals based on simple MA crossover."""
        if self.data is None or self.data.empty:
            return pd.Series()
        
        # Simple moving average crossover
        sma_short = self.data['close'].rolling(window=10).mean()
        sma_long = self.data['close'].rolling(window=30).mean()
        
        # Generate signals
        signals = pd.Series(index=self.data.index, data=0)
        signals[sma_short > sma_long] = 1  # Buy signal
        signals[sma_short < sma_long] = -1  # Sell signal
        
        return signals
    
    def get_signal(self):
        """Get the current signal (last value)."""
        signals = self.generate_signals()
        if signals.empty:
            return 0.0
        return float(signals.iloc[-1])
