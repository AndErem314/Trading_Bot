"""
MACD Momentum Crossover Strategy - Executable Implementation

This module provides an executable implementation of the MACD Momentum
Crossover strategy that conforms to the TradingStrategy interface.

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
try:
    from interfaces.trading_strategy_interface import TradingStrategy
except ImportError:
    # Try alternative import path
    from executable_workflow.interfaces.trading_strategy_interface import TradingStrategy

logger = logging.getLogger(__name__)


class MACDMomentumCrossover(TradingStrategy):
    """
    Executable MACD Momentum Crossover Strategy.
    
    This strategy combines MACD signals with momentum confirmation
    for high-probability trend-following trades.
    """
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize the MACD Momentum strategy.
        
        Args:
            data: OHLCV DataFrame
            config: Optional configuration with keys:
                - macd_fast: Fast EMA period (default: 12)
                - macd_slow: Slow EMA period (default: 26)
                - macd_signal: Signal line EMA period (default: 9)
                - momentum_period: Momentum lookback (default: 14)
                - atr_period: ATR period for volatility (default: 14)
                - volume_threshold: Volume spike threshold (default: 1.5)
        """
        super().__init__(data, config)
        self.name = "MACD Momentum Crossover"
        self.version = "1.0.0"
        
        # Strategy parameters
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.momentum_period = self.config.get('momentum_period', 14)
        self.atr_period = self.config.get('atr_period', 14)
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        
        # Initialize indicators
        self.indicators = {}
        if self.has_sufficient_data():
            self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """Calculate all required indicators."""
        try:
            # Calculate MACD
            macd_result = ta.macd(
                self.data['close'],
                fast=self.macd_fast,
                slow=self.macd_slow,
                signal=self.macd_signal
            )
            
            self.indicators['macd'] = macd_result[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            self.indicators['macd_signal'] = macd_result[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            self.indicators['macd_histogram'] = macd_result[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            
            # Calculate momentum
            self.indicators['momentum'] = ta.mom(self.data['close'], length=self.momentum_period)
            
            # Calculate rate of change
            self.indicators['roc'] = ta.roc(self.data['close'], length=self.momentum_period)
            
            # Calculate ATR for volatility
            self.indicators['atr'] = ta.atr(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                length=self.atr_period
            )
            
            # Calculate volume indicators
            self.indicators['volume_sma'] = self.data['volume'].rolling(window=20).mean()
            self.indicators['volume_ratio'] = self.data['volume'] / self.indicators['volume_sma']
            
            # Calculate trend strength using ADX
            adx_result = ta.adx(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                length=14
            )
            self.indicators['adx'] = adx_result[f'ADX_14']
            
            # Detect MACD crossovers
            self.indicators['macd_cross'] = self._detect_crossovers(
                self.indicators['macd'],
                self.indicators['macd_signal']
            )
            
            # Detect histogram divergence
            self.indicators['histogram_divergence'] = self._detect_histogram_divergence()
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating indicators: {e}")
            self.indicators = {}
    
    def _detect_crossovers(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Detect crossovers between two series.
        
        Returns:
            Series with 1 for bullish crossover, -1 for bearish crossover, 0 otherwise
        """
        crossovers = pd.Series(0, index=series1.index)
        
        # Bullish crossover: series1 crosses above series2
        bullish = (series1 > series2) & (series1.shift(1) <= series2.shift(1))
        crossovers[bullish] = 1
        
        # Bearish crossover: series1 crosses below series2
        bearish = (series1 < series2) & (series1.shift(1) >= series2.shift(1))
        crossovers[bearish] = -1
        
        return crossovers
    
    def _detect_histogram_divergence(self) -> pd.Series:
        """
        Detect divergence between price and MACD histogram.
        
        Returns:
            Series with divergence signals
        """
        divergence = pd.Series('none', index=self.data.index)
        
        try:
            close = self.data['close']
            histogram = self.indicators['macd_histogram']
            lookback = 20
            
            for i in range(lookback, len(self.data)):
                # Get recent values
                recent_close = close.iloc[i-lookback:i+1]
                recent_hist = histogram.iloc[i-lookback:i+1]
                
                # Find peaks and troughs
                price_peaks = self._find_peaks(recent_close)
                hist_peaks = self._find_peaks(recent_hist)
                
                # Check for divergence
                if len(price_peaks) >= 2 and len(hist_peaks) >= 2:
                    # Bearish divergence: higher price high, lower histogram high
                    if (recent_close.iloc[price_peaks[-1]] > recent_close.iloc[price_peaks[-2]] and
                        recent_hist.iloc[hist_peaks[-1]] < recent_hist.iloc[hist_peaks[-2]]):
                        divergence.iloc[i] = 'bearish'
                    # Bullish divergence: lower price low, higher histogram low
                    elif (recent_close.iloc[price_peaks[-1]] < recent_close.iloc[price_peaks[-2]] and
                          recent_hist.iloc[hist_peaks[-1]] > recent_hist.iloc[hist_peaks[-2]]):
                        divergence.iloc[i] = 'bullish'
                        
        except Exception as e:
            logger.error(f"Error detecting histogram divergence: {e}")
            
        return divergence
    
    def _find_peaks(self, series: pd.Series) -> list:
        """Find local peaks in a series."""
        peaks = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] > series.iloc[i-1] and series.iloc[i] > series.iloc[i+1]:
                peaks.append(i)
        return peaks
    
    def calculate_signal(self) -> Dict[str, float]:
        """
        Calculate trading signal based on MACD momentum strategy.
        
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
            macd = self.indicators['macd'].iloc[idx]
            macd_signal = self.indicators['macd_signal'].iloc[idx]
            macd_histogram = self.indicators['macd_histogram'].iloc[idx]
            macd_cross = self.indicators['macd_cross'].iloc[idx]
            momentum = self.indicators['momentum'].iloc[idx]
            roc = self.indicators['roc'].iloc[idx]
            volume_ratio = self.indicators['volume_ratio'].iloc[idx]
            adx = self.indicators['adx'].iloc[idx]
            histogram_divergence = self.indicators['histogram_divergence'].iloc[idx]
            
            # Previous values for trend detection
            prev_histogram = self.indicators['macd_histogram'].iloc[idx-1]
            
            # Check for NaN values
            if pd.isna(macd) or pd.isna(momentum) or pd.isna(adx):
                return {'signal': 0, 'reason': 'Indicators not ready'}
            
            # Initialize signal
            signal = 0.0
            confidence = 0.0
            reason = "No signal"
            
            # STRONG BUY SIGNAL
            if (macd_cross == 1 and  # MACD bullish crossover
                macd < 0 and  # Crossover below zero line (more reliable)
                momentum > 0 and  # Positive momentum
                roc > 0 and  # Positive rate of change
                volume_ratio > self.volume_threshold and  # Volume confirmation
                adx > 25):  # Trending market
                
                signal = 1.0
                confidence = 0.9
                reason = "MACD bullish crossover with momentum and volume confirmation"
                
                # Boost for histogram expansion
                if macd_histogram > prev_histogram and prev_histogram < 0:
                    confidence = min(confidence + 0.1, 1.0)
                    reason += " + histogram expansion"
            
            # STRONG SELL SIGNAL
            elif (macd_cross == -1 and  # MACD bearish crossover
                  macd > 0 and  # Crossover above zero line (more reliable)
                  momentum < 0 and  # Negative momentum
                  roc < 0 and  # Negative rate of change
                  volume_ratio > self.volume_threshold and  # Volume confirmation
                  adx > 25):  # Trending market
                
                signal = -1.0
                confidence = 0.9
                reason = "MACD bearish crossover with momentum and volume confirmation"
                
                # Boost for histogram contraction
                if macd_histogram < prev_histogram and prev_histogram > 0:
                    confidence = min(confidence + 0.1, 1.0)
                    reason += " + histogram contraction"
            
            # MODERATE BUY SIGNAL
            elif (macd > macd_signal and  # MACD above signal
                  macd_histogram > 0 and  # Positive histogram
                  macd_histogram > prev_histogram and  # Growing histogram
                  momentum > 0):  # Positive momentum
                
                # Scale signal based on histogram strength
                hist_strength = min(abs(macd_histogram) / 0.5, 1.0)  # Normalize to 0-1
                signal = 0.3 + (0.4 * hist_strength)  # 0.3 to 0.7
                confidence = 0.6
                reason = "MACD uptrend with growing histogram"
            
            # MODERATE SELL SIGNAL
            elif (macd < macd_signal and  # MACD below signal
                  macd_histogram < 0 and  # Negative histogram
                  macd_histogram < prev_histogram and  # Declining histogram
                  momentum < 0):  # Negative momentum
                
                # Scale signal based on histogram strength
                hist_strength = min(abs(macd_histogram) / 0.5, 1.0)  # Normalize to 0-1
                signal = -0.3 - (0.4 * hist_strength)  # -0.3 to -0.7
                confidence = 0.6
                reason = "MACD downtrend with declining histogram"
            
            # DIVERGENCE SIGNALS
            elif histogram_divergence == 'bullish' and macd < 0:
                signal = 0.5
                confidence = 0.7
                reason = "Bullish divergence detected"
                
            elif histogram_divergence == 'bearish' and macd > 0:
                signal = -0.5
                confidence = 0.7
                reason = "Bearish divergence detected"
            
            # Reduce confidence in choppy markets
            if adx < 20:
                confidence *= 0.7
                reason += " (low ADX - weak trend)"
            
            return {
                'signal': round(signal, 3),
                'confidence': round(confidence, 3),
                'reason': reason,
                'macd': round(macd, 4),
                'macd_signal': round(macd_signal, 4),
                'macd_histogram': round(macd_histogram, 4),
                'momentum': round(momentum, 2),
                'roc': round(roc, 2),
                'volume_ratio': round(volume_ratio, 2),
                'adx': round(adx, 2),
                'divergence': histogram_divergence
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating signal: {e}")
            return {'signal': 0, 'error': str(e)}
    
    def get_required_data_points(self) -> int:
        """
        Get minimum required data points.
        
        Returns:
            Maximum of MACD slow period and other indicators
        """
        return max(self.macd_slow + self.macd_signal, 50)  # Need extra for signal line
    
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        Determine if strategy is suitable for current market regime.
        
        MACD works best in trending markets.
        
        Args:
            market_bias: Current market regime
            
        Returns:
            True if market is suitable for trend following
        """
        # Strategy works best in trending markets
        suitable_regimes = ['Strong Bullish', 'Bullish', 'Bearish', 'Strong Bearish']
        # Can work in neutral if there's micro-trends
        moderately_suitable = ['Neutral']
        
        if market_bias in suitable_regimes:
            return True
        elif market_bias in moderately_suitable:
            return True  # Orchestrator should reduce weight
        else:
            # Not suitable for crash conditions
            return False
