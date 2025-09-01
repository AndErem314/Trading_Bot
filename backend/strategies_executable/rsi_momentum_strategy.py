"""
RSI Momentum Divergence Strategy - Executable Implementation

This module provides an executable implementation of the RSI Momentum Divergence
Swing Trading strategy that conforms to the TradingStrategy interface.

Author: Trading Bot Team
Date: 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
import pandas_ta as ta
import sys
import os

# Add parent directory to path to import the interface
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_strategy_interface import TradingStrategy

logger = logging.getLogger(__name__)


class RSIMomentumDivergence(TradingStrategy):
    """
    Executable RSI Momentum Divergence Swing Trading Strategy.
    
    This strategy identifies swing trading opportunities by combining:
    1. RSI extreme levels (oversold/overbought)
    2. RSI moving average crossovers for trend confirmation
    3. Momentum shifts for entry timing
    4. Divergence signals for additional confirmation
    5. Trend strength filters to avoid false signals
    """
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize the RSI Momentum Divergence strategy.
        
        Args:
            data: OHLCV DataFrame
            config: Optional configuration with keys:
                - rsi_length: RSI period (default: 14)
                - rsi_sma_fast: Fast RSI SMA period (default: 5)
                - rsi_sma_slow: Slow RSI SMA period (default: 10)
                - rsi_oversold: Oversold threshold (default: 30)
                - rsi_overbought: Overbought threshold (default: 70)
                - momentum_lookback: Periods for momentum calculation (default: 5)
        """
        super().__init__(data, config)
        self.name = "RSI Momentum Divergence Swing"
        self.version = "1.1.0"
        
        # Strategy parameters
        self.rsi_length = self.config.get('rsi_length', 14)
        self.rsi_sma_fast = self.config.get('rsi_sma_fast', 5)
        self.rsi_sma_slow = self.config.get('rsi_sma_slow', 10)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.momentum_lookback = self.config.get('momentum_lookback', 5)
        
        # Divergence detection parameters
        self.divergence_lookback = self.config.get('divergence_lookback', 20)
        self.divergence_threshold = self.config.get('divergence_threshold', 0.02)
        
        # Initialize indicators
        self.indicators = {}
        if self.has_sufficient_data():
            self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """Calculate all required indicators."""
        try:
            # Calculate RSI
            self.indicators['rsi'] = ta.rsi(self.data['close'], length=self.rsi_length)
            
            # Calculate RSI moving averages
            self.indicators['rsi_sma_fast'] = self.indicators['rsi'].rolling(window=self.rsi_sma_fast).mean()
            self.indicators['rsi_sma_slow'] = self.indicators['rsi'].rolling(window=self.rsi_sma_slow).mean()
            
            # Calculate momentum
            self.indicators['momentum'] = self.data['close'].pct_change(self.momentum_lookback)
            
            # Calculate trend strength using ADX
            adx_result = ta.adx(self.data['high'], self.data['low'], self.data['close'], length=14)
            self.indicators['adx'] = adx_result[f'ADX_{14}']
            self.indicators['di_plus'] = adx_result[f'DMP_{14}']
            self.indicators['di_minus'] = adx_result[f'DMN_{14}']
            
            # Detect divergences
            self.indicators['divergence'] = self._detect_divergence()
            
            # Detect momentum shifts
            self.indicators['momentum_shift'] = self._detect_momentum_shift()
            
            # Classify trend strength
            self.indicators['trend_strength'] = self._classify_trend_strength()
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating indicators: {e}")
            self.indicators = {}
    
    def _detect_divergence(self) -> pd.Series:
        """
        Detect bullish and bearish divergences between price and RSI.
        
        Returns:
            Series with values: 'bullish', 'bearish', or 'none'
        """
        divergence = pd.Series('none', index=self.data.index)
        
        try:
            rsi = self.indicators['rsi']
            close = self.data['close']
            
            for i in range(self.divergence_lookback, len(self.data)):
                # Get recent price and RSI values
                recent_close = close.iloc[i-self.divergence_lookback:i+1]
                recent_rsi = rsi.iloc[i-self.divergence_lookback:i+1]
                
                # Find local extremes
                price_peaks = self._find_peaks(recent_close)
                price_troughs = self._find_troughs(recent_close)
                rsi_peaks = self._find_peaks(recent_rsi)
                rsi_troughs = self._find_troughs(recent_rsi)
                
                # Check for bearish divergence (price higher high, RSI lower high)
                if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                    if (recent_close.iloc[price_peaks[-1]] > recent_close.iloc[price_peaks[-2]] and
                        recent_rsi.iloc[rsi_peaks[-1]] < recent_rsi.iloc[rsi_peaks[-2]]):
                        divergence.iloc[i] = 'bearish'
                
                # Check for bullish divergence (price lower low, RSI higher low)
                elif len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
                    if (recent_close.iloc[price_troughs[-1]] < recent_close.iloc[price_troughs[-2]] and
                        recent_rsi.iloc[rsi_troughs[-1]] > recent_rsi.iloc[rsi_troughs[-2]]):
                        divergence.iloc[i] = 'bullish'
                        
        except Exception as e:
            logger.error(f"Error detecting divergence: {e}")
            
        return divergence
    
    def _find_peaks(self, series: pd.Series) -> list:
        """Find local peaks in a series."""
        peaks = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] > series.iloc[i-1] and series.iloc[i] > series.iloc[i+1]:
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, series: pd.Series) -> list:
        """Find local troughs in a series."""
        troughs = []
        for i in range(1, len(series) - 1):
            if series.iloc[i] < series.iloc[i-1] and series.iloc[i] < series.iloc[i+1]:
                troughs.append(i)
        return troughs
    
    def _detect_momentum_shift(self) -> pd.Series:
        """
        Detect momentum shifts based on rate of change.
        
        Returns:
            Boolean series indicating momentum shifts
        """
        momentum_shift = pd.Series(False, index=self.data.index)
        
        try:
            mom = self.indicators['momentum']
            mom_change = mom.diff()
            
            # Detect significant momentum changes
            mom_std = mom_change.rolling(window=20).std()
            momentum_shift = abs(mom_change) > (2 * mom_std)
            
        except Exception as e:
            logger.error(f"Error detecting momentum shift: {e}")
            
        return momentum_shift
    
    def _classify_trend_strength(self) -> pd.Series:
        """
        Classify trend strength based on ADX and DI values.
        
        Returns:
            Series with trend strength classifications
        """
        trend_strength = pd.Series('neutral', index=self.data.index)
        
        try:
            adx = self.indicators['adx']
            di_plus = self.indicators['di_plus']
            di_minus = self.indicators['di_minus']
            
            for i in range(len(self.data)):
                if pd.isna(adx.iloc[i]):
                    continue
                    
                adx_val = adx.iloc[i]
                
                # Strong trend
                if adx_val > 40:
                    if di_plus.iloc[i] > di_minus.iloc[i]:
                        trend_strength.iloc[i] = 'strong_bullish'
                    else:
                        trend_strength.iloc[i] = 'strong_bearish'
                # Moderate trend
                elif adx_val > 25:
                    if di_plus.iloc[i] > di_minus.iloc[i]:
                        trend_strength.iloc[i] = 'bullish'
                    else:
                        trend_strength.iloc[i] = 'bearish'
                # Weak/no trend
                else:
                    trend_strength.iloc[i] = 'neutral'
                    
        except Exception as e:
            logger.error(f"Error classifying trend strength: {e}")
            
        return trend_strength
    
    def calculate_signal(self) -> Dict[str, float]:
        """
        Calculate trading signal based on RSI momentum divergence strategy.
        
        Returns:
            Dictionary with signal and metadata
        """
        if not self.has_sufficient_data():
            return {'signal': 0, 'reason': 'Insufficient data'}
        
        try:
            # Ensure indicators are calculated
            if not self.indicators:
                self._calculate_indicators()
            
            # Get current and previous values
            idx = -1
            prev_idx = -2
            
            rsi = self.indicators['rsi'].iloc[idx]
            prev_rsi = self.indicators['rsi'].iloc[prev_idx]
            rsi_sma_fast = self.indicators['rsi_sma_fast'].iloc[idx]
            rsi_sma_slow = self.indicators['rsi_sma_slow'].iloc[idx]
            prev_rsi_sma_fast = self.indicators['rsi_sma_fast'].iloc[prev_idx]
            prev_rsi_sma_slow = self.indicators['rsi_sma_slow'].iloc[prev_idx]
            
            momentum_shift = self.indicators['momentum_shift'].iloc[idx]
            divergence = self.indicators['divergence'].iloc[idx]
            trend_strength = self.indicators['trend_strength'].iloc[idx]
            
            # Check for NaN values
            if pd.isna(rsi) or pd.isna(rsi_sma_fast) or pd.isna(rsi_sma_slow):
                return {'signal': 0, 'reason': 'Indicators not ready'}
            
            # Initialize signal
            signal = 0.0
            confidence = 0.0
            reason = "No signal"
            
            # LONG ENTRY CONDITIONS
            if (prev_rsi <= self.rsi_oversold and rsi > self.rsi_oversold and  # RSI crosses above oversold
                prev_rsi_sma_fast <= prev_rsi_sma_slow and rsi_sma_fast > rsi_sma_slow and  # Bullish MA crossover
                momentum_shift and  # Momentum shift detected
                divergence != 'bearish' and  # No bearish divergence
                trend_strength != 'strong_bearish'):  # Not in strong downtrend
                
                signal = 1.0
                confidence = 0.8
                reason = "RSI oversold recovery with momentum confirmation"
                
                # Boost confidence for bullish divergence
                if divergence == 'bullish':
                    confidence = min(confidence + 0.2, 1.0)
                    reason += " + bullish divergence"
            
            # SHORT ENTRY CONDITIONS
            elif (prev_rsi >= self.rsi_overbought and rsi < self.rsi_overbought and  # RSI crosses below overbought
                  prev_rsi_sma_fast >= prev_rsi_sma_slow and rsi_sma_fast < rsi_sma_slow and  # Bearish MA crossover
                  momentum_shift and  # Momentum shift detected
                  divergence != 'bullish' and  # No bullish divergence
                  trend_strength != 'strong_bullish'):  # Not in strong uptrend
                
                signal = -1.0
                confidence = 0.8
                reason = "RSI overbought reversal with momentum confirmation"
                
                # Boost confidence for bearish divergence
                if divergence == 'bearish':
                    confidence = min(confidence + 0.2, 1.0)
                    reason += " + bearish divergence"
            
            # MODERATE SIGNALS (without full confirmation)
            elif rsi < 35 and trend_strength in ['neutral', 'bullish']:
                signal = 0.3
                confidence = 0.5
                reason = "RSI approaching oversold"
                
            elif rsi > 65 and trend_strength in ['neutral', 'bearish']:
                signal = -0.3
                confidence = 0.5
                reason = "RSI approaching overbought"
            
            return {
                'signal': round(signal, 3),
                'confidence': round(confidence, 3),
                'reason': reason,
                'rsi': round(rsi, 2),
                'rsi_sma_fast': round(rsi_sma_fast, 2),
                'rsi_sma_slow': round(rsi_sma_slow, 2),
                'momentum_shift': momentum_shift,
                'divergence': divergence,
                'trend_strength': trend_strength,
                'adx': round(self.indicators['adx'].iloc[idx], 2) if not pd.isna(self.indicators['adx'].iloc[idx]) else None
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating signal: {e}")
            return {'signal': 0, 'error': str(e)}
    
    def get_required_data_points(self) -> int:
        """
        Get minimum required data points.
        
        Returns:
            Minimum periods needed for all indicators
        """
        # Need enough for ADX, RSI, SMAs, and divergence detection
        return max(
            self.rsi_length + self.rsi_sma_slow,
            self.divergence_lookback + 10,
            50  # Minimum for ADX calculation
        )
    
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        Determine if strategy is suitable for current market regime.
        
        RSI momentum strategy works in most market conditions except crashes.
        
        Args:
            market_bias: Current market regime
            
        Returns:
            True if market is suitable
        """
        # Strategy works in all regimes except extreme crash conditions
        unsuitable_regimes = ['Crash', 'Flash Crash']
        return market_bias not in unsuitable_regimes
