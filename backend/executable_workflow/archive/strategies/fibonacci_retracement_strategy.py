"""
Fibonacci Retracement Support Resistance Strategy - Executable Implementation

This module provides an executable implementation of the Fibonacci Retracement
Support Resistance strategy that conforms to the TradingStrategy interface.

"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
import pandas_ta as ta
import sys
import os

# Add parent directory to path to import the interface
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.executable_workflow.interfaces.trading_strategy_interface import TradingStrategy

logger = logging.getLogger(__name__)


class FibonacciRetracementSupportResistance(TradingStrategy):
    """
    Executable Fibonacci Retracement Support Resistance Strategy.
    
    This strategy identifies key Fibonacci retracement levels from recent
    price swings and generates signals when price approaches these levels
    with confirmation from momentum and volume indicators.
    """
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize the Fibonacci Retracement strategy.
        
        Args:
            data: OHLCV DataFrame
            config: Optional configuration with keys:
                - lookback_period: Period to find swing highs/lows (default: 50)
                - fib_levels: List of Fibonacci levels (default: [0.236, 0.382, 0.5, 0.618, 0.786])
                - rsi_period: RSI period for momentum (default: 14)
                - volume_period: Volume analysis period (default: 20)
                - proximity_threshold: % proximity to level (default: 0.5%)
        """
        super().__init__(data, config)
        self.name = "Fibonacci Retracement Support Resistance"
        self.version = "1.0.0"
        
        # Strategy parameters
        self.lookback_period = self.config.get('lookback_period', 50)
        self.fib_levels = self.config.get('fib_levels', [0.236, 0.382, 0.5, 0.618, 0.786])
        self.rsi_period = self.config.get('rsi_period', 14)
        self.volume_period = self.config.get('volume_period', 20)
        self.proximity_threshold = self.config.get('proximity_threshold', 0.005)  # 0.5%
        
        # Initialize indicators
        self.indicators = {}
        self.fib_data = {}
        if self.has_sufficient_data():
            self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """Calculate Fibonacci levels and supporting indicators."""
        try:
            # Find swing highs and lows
            self.indicators['swing_highs'] = self._find_swing_highs()
            self.indicators['swing_lows'] = self._find_swing_lows()
            
            # Calculate current Fibonacci levels
            self.fib_data = self._calculate_fibonacci_levels()
            
            # Calculate RSI for momentum
            self.indicators['rsi'] = ta.rsi(self.data['close'], length=self.rsi_period)
            
            # Calculate MACD for trend confirmation
            macd_result = ta.macd(self.data['close'], fast=12, slow=26, signal=9)
            self.indicators['macd'] = macd_result.iloc[:, 0]
            self.indicators['macd_signal'] = macd_result.iloc[:, 1]
            self.indicators['macd_histogram'] = macd_result.iloc[:, 2]
            
            # Volume analysis
            self.indicators['volume_sma'] = ta.sma(self.data['volume'], length=self.volume_period)
            self.indicators['volume_ratio'] = self.data['volume'] / self.indicators['volume_sma']
            
            # ATR for volatility
            self.indicators['atr'] = ta.atr(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                length=14
            )
            
            # Price position relative to Fibonacci levels
            self.indicators['nearest_fib'] = self._find_nearest_fib_level()
            self.indicators['fib_proximity'] = self._calculate_fib_proximity()
            
            # Support/Resistance strength
            self.indicators['level_strength'] = self._calculate_level_strength()
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating indicators: {e}")
            self.indicators = {}
    
    def _find_swing_highs(self) -> pd.Series:
        """Find swing high points."""
        highs = pd.Series(np.nan, index=self.data.index)
        
        for i in range(2, len(self.data) - 2):
            if (self.data['high'].iloc[i] > self.data['high'].iloc[i-1] and
                self.data['high'].iloc[i] > self.data['high'].iloc[i-2] and
                self.data['high'].iloc[i] > self.data['high'].iloc[i+1] and
                self.data['high'].iloc[i] > self.data['high'].iloc[i+2]):
                highs.iloc[i] = self.data['high'].iloc[i]
        
        return highs
    
    def _find_swing_lows(self) -> pd.Series:
        """Find swing low points."""
        lows = pd.Series(np.nan, index=self.data.index)
        
        for i in range(2, len(self.data) - 2):
            if (self.data['low'].iloc[i] < self.data['low'].iloc[i-1] and
                self.data['low'].iloc[i] < self.data['low'].iloc[i-2] and
                self.data['low'].iloc[i] < self.data['low'].iloc[i+1] and
                self.data['low'].iloc[i] < self.data['low'].iloc[i+2]):
                lows.iloc[i] = self.data['low'].iloc[i]
        
        return lows
    
    def _calculate_fibonacci_levels(self) -> Dict:
        """Calculate current Fibonacci retracement levels."""
        # Get recent data
        recent_data = self.data.tail(self.lookback_period)
        
        # Find highest high and lowest low
        high = recent_data['high'].max()
        low = recent_data['low'].min()
        high_idx = recent_data['high'].idxmax()
        low_idx = recent_data['low'].idxmin()
        
        # Determine if we're in uptrend or downtrend
        is_uptrend = high_idx > low_idx
        
        # Calculate range
        range_size = high - low
        
        # Calculate Fibonacci levels
        fib_levels = {}
        
        if is_uptrend:
            # Retracement from high
            for level in self.fib_levels:
                fib_levels[f'fib_{level}'] = high - (range_size * level)
        else:
            # Retracement from low
            for level in self.fib_levels:
                fib_levels[f'fib_{level}'] = low + (range_size * level)
        
        return {
            'high': high,
            'low': low,
            'is_uptrend': is_uptrend,
            'levels': fib_levels,
            'range': range_size
        }
    
    def _find_nearest_fib_level(self) -> pd.Series:
        """Find the nearest Fibonacci level for each price point."""
        nearest = pd.Series(index=self.data.index, dtype=float)
        
        if not self.fib_data or 'levels' not in self.fib_data:
            return nearest
        
        fib_values = list(self.fib_data['levels'].values())
        
        for i in range(len(self.data)):
            current_price = self.data['close'].iloc[i]
            distances = [abs(current_price - level) for level in fib_values]
            min_idx = distances.index(min(distances))
            nearest.iloc[i] = fib_values[min_idx]
        
        return nearest
    
    def _calculate_fib_proximity(self) -> pd.Series:
        """Calculate proximity to nearest Fibonacci level as percentage."""
        proximity = pd.Series(0.0, index=self.data.index)
        
        nearest_fib = self.indicators.get('nearest_fib', pd.Series())
        
        for i in range(len(self.data)):
            if pd.notna(nearest_fib.iloc[i]):
                distance = abs(self.data['close'].iloc[i] - nearest_fib.iloc[i])
                proximity.iloc[i] = distance / self.data['close'].iloc[i]
        
        return proximity
    
    def _calculate_level_strength(self) -> pd.Series:
        """Calculate strength of support/resistance at current level."""
        strength = pd.Series(0.0, index=self.data.index)
        
        # Count how many times price has bounced off each level
        if self.fib_data and 'levels' in self.fib_data:
            for i in range(20, len(self.data)):
                for level_name, level_value in self.fib_data['levels'].items():
                    # Check if price touched this level in recent past
                    recent_window = self.data.iloc[i-20:i]
                    touches = 0
                    
                    for j in range(len(recent_window)):
                        if abs(recent_window['low'].iloc[j] - level_value) / level_value < 0.01:
                            touches += 1
                        if abs(recent_window['high'].iloc[j] - level_value) / level_value < 0.01:
                            touches += 1
                    
                    strength.iloc[i] = max(strength.iloc[i], min(touches / 5, 1.0))
        
        return strength
    
    def calculate_signal(self) -> Dict[str, float]:
        """
        Calculate trading signal based on Fibonacci Retracement strategy.
        
        Returns:
            Dictionary with signal and metadata
        """
        if not self.has_sufficient_data():
            return {'signal': 0, 'reason': 'Insufficient data'}
        
        try:
            # Ensure indicators are calculated
            if not self.indicators:
                self._calculate_indicators()
            
            # Get current values
            idx = -1
            current_close = self.data['close'].iloc[idx]
            nearest_fib = self.indicators['nearest_fib'].iloc[idx]
            fib_proximity = self.indicators['fib_proximity'].iloc[idx]
            rsi = self.indicators['rsi'].iloc[idx]
            macd = self.indicators['macd'].iloc[idx]
            macd_signal = self.indicators['macd_signal'].iloc[idx]
            volume_ratio = self.indicators['volume_ratio'].iloc[idx]
            level_strength = self.indicators['level_strength'].iloc[idx]
            
            # Check for NaN values
            if pd.isna(nearest_fib) or pd.isna(rsi):
                return {'signal': 0, 'reason': 'Indicators not ready'}
            
            # Initialize signal
            signal = 0.0
            confidence = 0.0
            reason = "No signal"
            
            # Determine which Fibonacci level we're near
            fib_level_name = None
            for name, value in self.fib_data['levels'].items():
                if abs(value - nearest_fib) < 0.01:
                    fib_level_name = name
                    break
            
            # Check if we're close enough to a level
            if fib_proximity < self.proximity_threshold:
                is_uptrend = self.fib_data['is_uptrend']
                
                # SUPPORT BOUNCE (BUY) SIGNALS
                if is_uptrend and current_close > nearest_fib:
                    # Price bouncing off Fibonacci support
                    if rsi < 50 and macd < macd_signal:
                        signal = 0.8
                        confidence = 0.8
                        reason = f"Bounce off Fibonacci support at {fib_level_name}"
                        
                        # Strong support level
                        if level_strength > 0.5:
                            confidence = min(confidence + 0.1, 0.95)
                            reason += " (strong level)"
                        
                        # Volume confirmation
                        if volume_ratio > 1.3:
                            confidence = min(confidence + 0.05, 0.95)
                            reason += " + volume"
                    
                    # Oversold at support
                    elif rsi < 30:
                        signal = 0.9
                        confidence = 0.85
                        reason = f"Oversold at Fibonacci support {fib_level_name}"
                
                # RESISTANCE REJECTION (SELL) SIGNALS
                elif not is_uptrend and current_close < nearest_fib:
                    # Price rejecting from Fibonacci resistance
                    if rsi > 50 and macd > macd_signal:
                        signal = -0.8
                        confidence = 0.8
                        reason = f"Rejection from Fibonacci resistance at {fib_level_name}"
                        
                        # Strong resistance level
                        if level_strength > 0.5:
                            confidence = min(confidence + 0.1, 0.95)
                            reason += " (strong level)"
                        
                        # Volume confirmation
                        if volume_ratio > 1.3:
                            confidence = min(confidence + 0.05, 0.95)
                            reason += " + volume"
                    
                    # Overbought at resistance
                    elif rsi > 70:
                        signal = -0.9
                        confidence = 0.85
                        reason = f"Overbought at Fibonacci resistance {fib_level_name}"
                
                # BREAKOUT SIGNALS
                # Bullish breakout
                if (self.data['close'].iloc[idx-1] < nearest_fib and 
                    current_close > nearest_fib and
                    volume_ratio > 1.5):
                    
                    signal = 0.7
                    confidence = 0.7
                    reason = f"Bullish breakout above {fib_level_name}"
                
                # Bearish breakdown
                elif (self.data['close'].iloc[idx-1] > nearest_fib and 
                      current_close < nearest_fib and
                      volume_ratio > 1.5):
                    
                    signal = -0.7
                    confidence = 0.7
                    reason = f"Bearish breakdown below {fib_level_name}"
            
            # GOLDEN RATIO SPECIAL HANDLING
            if fib_level_name and '0.618' in fib_level_name:
                # Golden ratio is the strongest level
                if abs(signal) > 0:
                    confidence = min(confidence * 1.1, 0.95)
                    reason += " (Golden Ratio)"
            
            return {
                'signal': round(signal, 3),
                'confidence': round(confidence, 3),
                'reason': reason,
                'nearest_fib_level': round(nearest_fib, 2) if pd.notna(nearest_fib) else None,
                'fib_level': fib_level_name,
                'proximity': round(fib_proximity * 100, 3),  # As percentage
                'level_strength': round(level_strength, 2),
                'rsi': round(rsi, 2),
                'trend': 'up' if self.fib_data.get('is_uptrend', False) else 'down',
                'swing_high': round(self.fib_data.get('high', 0), 2),
                'swing_low': round(self.fib_data.get('low', 0), 2),
                'price': round(current_close, 2)
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating signal: {e}")
            return {'signal': 0, 'error': str(e)}
    
    def get_required_data_points(self) -> int:
        """
        Get minimum required data points.
        
        Returns:
            Lookback period plus buffer
        """
        return self.lookback_period + 20
    
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        Determine if strategy is suitable for current market regime.
        
        Fibonacci works in all market conditions but best in ranging/retracing markets.
        
        Args:
            market_bias: Current market regime
            
        Returns:
            True if market is suitable
        """
        # Strategy works in all conditions but best in ranging markets
        ideal_regimes = ['Neutral', 'Ranging', 'Consolidating']
        good_regimes = ['Bullish', 'Bearish']
        
        if market_bias in ideal_regimes:
            return True
        elif market_bias in good_regimes:
            return True
        else:
            # Less effective in strong trends but still usable
            return True
