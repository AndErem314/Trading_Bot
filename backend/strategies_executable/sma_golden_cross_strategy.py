"""
SMA Golden Cross Strategy - Executable Implementation

This module provides an executable implementation of the SMA Golden Cross
strategy that conforms to the TradingStrategy interface.

Author: Trading Bot Team
Date: 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
import pandas_ta as ta
import sys
import os

# Add parent directory to path to import the interface
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_strategy_interface import TradingStrategy

logger = logging.getLogger(__name__)


class SMAGoldenCross(TradingStrategy):
    """
    Executable SMA Golden Cross Strategy.
    
    This classic trend-following strategy uses the 50-day and 200-day
    simple moving average crossover to identify major trend changes.
    """
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize the SMA Golden Cross strategy.
        
        Args:
            data: OHLCV DataFrame
            config: Optional configuration with keys:
                - fast_sma: Fast SMA period (default: 50)
                - slow_sma: Slow SMA period (default: 200)
                - volume_sma: Volume SMA period (default: 20)
                - atr_period: ATR period (default: 14)
                - distance_threshold: Min distance between SMAs (default: 0.01)
        """
        super().__init__(data, config)
        self.name = "SMA Golden Cross"
        self.version = "1.0.0"
        
        # Strategy parameters
        self.fast_sma = self.config.get('fast_sma', 50)
        self.slow_sma = self.config.get('slow_sma', 200)
        self.volume_sma = self.config.get('volume_sma', 20)
        self.atr_period = self.config.get('atr_period', 14)
        self.distance_threshold = self.config.get('distance_threshold', 0.01)
        
        # Initialize indicators
        self.indicators = {}
        if self.has_sufficient_data():
            self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """Calculate all required indicators."""
        try:
            # Calculate SMAs
            self.indicators['sma_fast'] = ta.sma(self.data['close'], length=self.fast_sma)
            self.indicators['sma_slow'] = ta.sma(self.data['close'], length=self.slow_sma)
            
            # Calculate SMA slope (momentum)
            self.indicators['sma_fast_slope'] = self.indicators['sma_fast'].diff(5) / 5
            self.indicators['sma_slow_slope'] = self.indicators['sma_slow'].diff(10) / 10
            
            # Calculate distance between SMAs
            self.indicators['sma_distance'] = (
                (self.indicators['sma_fast'] - self.indicators['sma_slow']) / 
                self.indicators['sma_slow'] * 100
            )
            
            # Price position relative to SMAs
            self.indicators['price_to_fast'] = (
                (self.data['close'] - self.indicators['sma_fast']) / 
                self.indicators['sma_fast'] * 100
            )
            self.indicators['price_to_slow'] = (
                (self.data['close'] - self.indicators['sma_slow']) / 
                self.indicators['sma_slow'] * 100
            )
            
            # Volume analysis
            self.indicators['volume_sma'] = ta.sma(self.data['volume'], length=self.volume_sma)
            self.indicators['volume_ratio'] = self.data['volume'] / self.indicators['volume_sma']
            
            # ATR for volatility
            self.indicators['atr'] = ta.atr(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                length=self.atr_period
            )
            
            # Detect crossovers
            self.indicators['golden_cross'] = self._detect_crossovers(
                self.indicators['sma_fast'],
                self.indicators['sma_slow']
            )
            
            # Calculate trend strength
            self.indicators['trend_strength'] = self._calculate_trend_strength()
            
            # Support/resistance levels
            self.indicators['resistance'] = self.data['high'].rolling(window=20).max()
            self.indicators['support'] = self.data['low'].rolling(window=20).min()
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating indicators: {e}")
            self.indicators = {}
    
    def _detect_crossovers(self, fast: pd.Series, slow: pd.Series) -> pd.Series:
        """
        Detect golden/death crosses between SMAs.
        
        Returns:
            Series with 1 for golden cross, -1 for death cross, 0 otherwise
        """
        crossovers = pd.Series(0, index=fast.index)
        
        # Golden cross: fast crosses above slow
        golden = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        crossovers[golden] = 1
        
        # Death cross: fast crosses below slow
        death = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        crossovers[death] = -1
        
        return crossovers
    
    def _calculate_trend_strength(self) -> pd.Series:
        """
        Calculate trend strength based on SMA alignment and slopes.
        
        Returns:
            Series with trend strength values
        """
        trend_strength = pd.Series(0.0, index=self.data.index)
        
        try:
            for i in range(len(self.data)):
                if pd.isna(self.indicators['sma_fast'].iloc[i]) or pd.isna(self.indicators['sma_slow'].iloc[i]):
                    continue
                
                fast_slope = self.indicators['sma_fast_slope'].iloc[i]
                slow_slope = self.indicators['sma_slow_slope'].iloc[i]
                distance = self.indicators['sma_distance'].iloc[i]
                
                # Strong uptrend
                if distance > 5 and fast_slope > 0 and slow_slope > 0:
                    trend_strength.iloc[i] = 1.0
                # Moderate uptrend
                elif distance > 2 and fast_slope > 0:
                    trend_strength.iloc[i] = 0.7
                # Weak uptrend
                elif distance > 0:
                    trend_strength.iloc[i] = 0.3
                # Strong downtrend
                elif distance < -5 and fast_slope < 0 and slow_slope < 0:
                    trend_strength.iloc[i] = -1.0
                # Moderate downtrend
                elif distance < -2 and fast_slope < 0:
                    trend_strength.iloc[i] = -0.7
                # Weak downtrend
                elif distance < 0:
                    trend_strength.iloc[i] = -0.3
                    
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            
        return trend_strength
    
    def calculate_signal(self) -> Dict[str, float]:
        """
        Calculate trading signal based on SMA Golden Cross strategy.
        
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
            sma_fast = self.indicators['sma_fast'].iloc[idx]
            sma_slow = self.indicators['sma_slow'].iloc[idx]
            golden_cross = self.indicators['golden_cross'].iloc[idx]
            sma_distance = self.indicators['sma_distance'].iloc[idx]
            price_to_fast = self.indicators['price_to_fast'].iloc[idx]
            price_to_slow = self.indicators['price_to_slow'].iloc[idx]
            fast_slope = self.indicators['sma_fast_slope'].iloc[idx]
            slow_slope = self.indicators['sma_slow_slope'].iloc[idx]
            volume_ratio = self.indicators['volume_ratio'].iloc[idx]
            trend_strength = self.indicators['trend_strength'].iloc[idx]
            
            # Historical context
            recent_crosses = self.indicators['golden_cross'].iloc[-10:].sum()
            
            # Check for NaN values
            if pd.isna(sma_fast) or pd.isna(sma_slow):
                return {'signal': 0, 'reason': 'Indicators not ready'}
            
            # Initialize signal
            signal = 0.0
            confidence = 0.0
            reason = "No signal"
            
            # GOLDEN CROSS - STRONG BUY
            if golden_cross == 1:
                signal = 1.0
                confidence = 0.9
                reason = "Golden Cross detected"
                
                # Boost confidence for clean crosses
                if abs(sma_distance) > self.distance_threshold * 100:
                    confidence = min(confidence + 0.1, 1.0)
                    reason += " with clear separation"
                
                # Volume confirmation
                if volume_ratio > 1.5:
                    confidence = min(confidence + 0.05, 1.0)
                    reason += " + volume surge"
            
            # DEATH CROSS - STRONG SELL
            elif golden_cross == -1:
                signal = -1.0
                confidence = 0.9
                reason = "Death Cross detected"
                
                # Boost confidence for clean crosses
                if abs(sma_distance) > self.distance_threshold * 100:
                    confidence = min(confidence + 0.1, 1.0)
                    reason += " with clear separation"
                
                # Volume confirmation
                if volume_ratio > 1.5:
                    confidence = min(confidence + 0.05, 1.0)
                    reason += " + volume surge"
            
            # UPTREND CONTINUATION
            elif (sma_fast > sma_slow and 
                  fast_slope > 0 and 
                  slow_slope > 0 and
                  price_to_fast > -2):  # Price not too far below fast SMA
                
                # Scale signal based on trend strength
                signal = 0.3 + (0.4 * abs(trend_strength))
                signal = min(signal, 0.8)  # Cap at 0.8
                confidence = 0.7
                reason = "Uptrend continuation"
                
                # Pullback to support
                if price_to_fast < 0 and price_to_fast > -3:
                    signal = min(signal + 0.2, 1.0)
                    confidence = 0.8
                    reason = "Uptrend pullback to support"
            
            # DOWNTREND CONTINUATION
            elif (sma_fast < sma_slow and 
                  fast_slope < 0 and 
                  slow_slope < 0 and
                  price_to_fast < 2):  # Price not too far above fast SMA
                
                # Scale signal based on trend strength
                signal = -0.3 - (0.4 * abs(trend_strength))
                signal = max(signal, -0.8)  # Cap at -0.8
                confidence = 0.7
                reason = "Downtrend continuation"
                
                # Rally to resistance
                if price_to_fast > 0 and price_to_fast < 3:
                    signal = max(signal - 0.2, -1.0)
                    confidence = 0.8
                    reason = "Downtrend rally to resistance"
            
            # TREND REVERSAL SIGNALS
            elif (sma_fast < sma_slow and  # Currently in downtrend
                  fast_slope > 0 and  # But fast SMA turning up
                  price_to_fast > 0 and  # Price above fast SMA
                  abs(sma_distance) < 3):  # SMAs converging
                
                signal = 0.4
                confidence = 0.6
                reason = "Potential trend reversal (bullish)"
            
            elif (sma_fast > sma_slow and  # Currently in uptrend
                  fast_slope < 0 and  # But fast SMA turning down
                  price_to_fast < 0 and  # Price below fast SMA
                  abs(sma_distance) < 3):  # SMAs converging
                
                signal = -0.4
                confidence = 0.6
                reason = "Potential trend reversal (bearish)"
            
            # Reduce confidence for choppy markets
            if recent_crosses > 2:
                confidence *= 0.7
                reason += " (caution: choppy market)"
            
            return {
                'signal': round(signal, 3),
                'confidence': round(confidence, 3),
                'reason': reason,
                'sma_fast': round(sma_fast, 2),
                'sma_slow': round(sma_slow, 2),
                'sma_distance': round(sma_distance, 2),
                'price_to_fast': round(price_to_fast, 2),
                'fast_slope': round(fast_slope, 4),
                'slow_slope': round(slow_slope, 4),
                'trend_strength': round(trend_strength, 2),
                'volume_ratio': round(volume_ratio, 2)
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating signal: {e}")
            return {'signal': 0, 'error': str(e)}
    
    def get_required_data_points(self) -> int:
        """
        Get minimum required data points.
        
        Returns:
            Slow SMA period plus buffer
        """
        return self.slow_sma + 20  # Extra buffer for slope calculation
    
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        Determine if strategy is suitable for current market regime.
        
        Golden Cross works best in trending markets with clear direction.
        
        Args:
            market_bias: Current market regime
            
        Returns:
            True if market is suitable for trend following
        """
        # Strategy works best in clear trends
        suitable_regimes = ['Strong Bullish', 'Bullish', 'Bearish', 'Strong Bearish']
        # Less effective in neutral/ranging markets
        unsuitable_regimes = ['Neutral', 'Ranging', 'Crash']
        
        return market_bias not in unsuitable_regimes
