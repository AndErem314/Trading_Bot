"""
Gaussian Channel Breakout Mean Reversion Strategy - Executable Implementation

This module provides an executable implementation of the Gaussian Channel
Breakout Mean Reversion strategy that conforms to the TradingStrategy interface.

"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
import pandas_ta as ta
from scipy import stats
import sys
import os

# Add parent directory to path to import the interface
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_strategy_interface import TradingStrategy

logger = logging.getLogger(__name__)


class GaussianChannelBreakoutMeanReversion(TradingStrategy):
    """
    Executable Gaussian Channel Breakout Mean Reversion Strategy.
    
    This strategy uses Gaussian (Normal) distribution properties to create
    adaptive channels that adjust to market volatility. It identifies both
    breakout and mean reversion opportunities.
    """
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize the Gaussian Channel strategy.
        
        Args:
            data: OHLCV DataFrame
            config: Optional configuration with keys:
                - period: Channel calculation period (default: 20)
                - std_dev: Standard deviation multiplier (default: 2.0)
                - atr_period: ATR period for volatility (default: 14)
                - rsi_period: RSI period (default: 14)
                - volume_period: Volume analysis period (default: 20)
                - adaptive: Use adaptive std dev (default: True)
        """
        super().__init__(data, config)
        self.name = "Gaussian Channel Breakout Mean Reversion"
        self.version = "1.0.0"
        
        # Strategy parameters
        self.period = self.config.get('period', 20)
        self.std_dev = self.config.get('std_dev', 2.0)
        self.atr_period = self.config.get('atr_period', 14)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.volume_period = self.config.get('volume_period', 20)
        self.adaptive = self.config.get('adaptive', True)
        
        # Initialize indicators
        self.indicators = {}
        if self.has_sufficient_data():
            self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """Calculate Gaussian Channel and supporting indicators."""
        try:
            # Calculate Gaussian Channel
            self.indicators['channel_mean'] = self._calculate_gaussian_mean()
            self.indicators['channel_std'] = self._calculate_gaussian_std()
            
            # Calculate channel bands
            if self.adaptive:
                # Adaptive multiplier based on volatility
                self.indicators['adaptive_mult'] = self._calculate_adaptive_multiplier()
                upper_mult = self.std_dev * self.indicators['adaptive_mult']
                lower_mult = self.std_dev * self.indicators['adaptive_mult']
            else:
                upper_mult = self.std_dev
                lower_mult = self.std_dev
            
            self.indicators['upper_band'] = (
                self.indicators['channel_mean'] + 
                self.indicators['channel_std'] * upper_mult
            )
            self.indicators['lower_band'] = (
                self.indicators['channel_mean'] - 
                self.indicators['channel_std'] * lower_mult
            )
            
            # Calculate channel width
            self.indicators['channel_width'] = (
                self.indicators['upper_band'] - self.indicators['lower_band']
            )
            self.indicators['channel_width_pct'] = (
                self.indicators['channel_width'] / self.indicators['channel_mean'] * 100
            )
            
            # Price position in channel (0-1 scale)
            self.indicators['channel_position'] = self._calculate_channel_position()
            
            # Calculate RSI
            self.indicators['rsi'] = ta.rsi(self.data['close'], length=self.rsi_period)
            
            # Calculate ATR
            self.indicators['atr'] = ta.atr(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                length=self.atr_period
            )
            
            # Volume analysis
            self.indicators['volume_sma'] = ta.sma(self.data['volume'], length=self.volume_period)
            self.indicators['volume_ratio'] = self.data['volume'] / self.indicators['volume_sma']
            
            # Detect breakouts
            self.indicators['breakout'] = self._detect_breakouts()
            
            # Calculate momentum
            self.indicators['momentum'] = ta.mom(self.data['close'], length=10)
            
            # Z-score for mean reversion
            self.indicators['z_score'] = self._calculate_z_score()
            
            # Channel slope for trend detection
            self.indicators['channel_slope'] = self._calculate_channel_slope()
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating indicators: {e}")
            self.indicators = {}
    
    def _calculate_gaussian_mean(self) -> pd.Series:
        """Calculate Gaussian-weighted moving average."""
        # Create Gaussian weights
        weights = stats.norm.pdf(range(self.period), self.period/2, self.period/4)
        weights = weights / weights.sum()
        
        # Apply weights to calculate mean
        gaussian_mean = self.data['close'].rolling(window=self.period).apply(
            lambda x: np.sum(x * weights) if len(x) == self.period else np.nan
        )
        
        return gaussian_mean
    
    def _calculate_gaussian_std(self) -> pd.Series:
        """Calculate Gaussian-weighted standard deviation."""
        mean = self.indicators.get('channel_mean', self._calculate_gaussian_mean())
        
        # Calculate weighted standard deviation
        def weighted_std(values):
            if len(values) < self.period:
                return np.nan
            weights = stats.norm.pdf(range(self.period), self.period/2, self.period/4)
            weights = weights / weights.sum()
            weighted_mean = np.sum(values * weights)
            variance = np.sum(weights * (values - weighted_mean)**2)
            return np.sqrt(variance)
        
        gaussian_std = self.data['close'].rolling(window=self.period).apply(weighted_std)
        
        return gaussian_std
    
    def _calculate_adaptive_multiplier(self) -> pd.Series:
        """Calculate adaptive multiplier based on market conditions."""
        # Use ATR ratio for volatility adjustment
        atr = self.indicators.get('atr', ta.atr(self.data['high'], self.data['low'], 
                                                 self.data['close'], length=self.atr_period))
        atr_sma = atr.rolling(window=50).mean()
        
        # Adaptive multiplier: increase in high volatility, decrease in low volatility
        adaptive_mult = np.where(atr > atr_sma * 1.5, 1.2,  # High volatility
                                 np.where(atr < atr_sma * 0.5, 0.8,  # Low volatility
                                         1.0))  # Normal
        
        return pd.Series(adaptive_mult, index=self.data.index)
    
    def _calculate_channel_position(self) -> pd.Series:
        """Calculate price position within channel (0 = lower band, 1 = upper band)."""
        position = (
            (self.data['close'] - self.indicators['lower_band']) / 
            (self.indicators['upper_band'] - self.indicators['lower_band'])
        )
        return position.clip(0, 1)
    
    def _detect_breakouts(self) -> pd.Series:
        """Detect channel breakouts."""
        breakout = pd.Series(0, index=self.data.index)
        
        # Upper band breakout
        upper_break = (
            (self.data['close'] > self.indicators['upper_band']) & 
            (self.data['close'].shift(1) <= self.indicators['upper_band'].shift(1))
        )
        breakout[upper_break] = 1
        
        # Lower band breakout
        lower_break = (
            (self.data['close'] < self.indicators['lower_band']) & 
            (self.data['close'].shift(1) >= self.indicators['lower_band'].shift(1))
        )
        breakout[lower_break] = -1
        
        return breakout
    
    def _calculate_z_score(self) -> pd.Series:
        """Calculate Z-score for mean reversion."""
        z_score = (
            (self.data['close'] - self.indicators['channel_mean']) / 
            self.indicators['channel_std']
        )
        return z_score
    
    def _calculate_channel_slope(self) -> pd.Series:
        """Calculate channel slope for trend detection."""
        # Use linear regression on channel mean
        def calculate_slope(values):
            if len(values) < 5:
                return 0
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return slope
        
        slope = self.indicators['channel_mean'].rolling(window=5).apply(calculate_slope)
        
        # Normalize slope
        return slope / self.data['close'] * 100
    
    def calculate_signal(self) -> Dict[str, float]:
        """
        Calculate trading signal based on Gaussian Channel strategy.
        
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
            upper_band = self.indicators['upper_band'].iloc[idx]
            lower_band = self.indicators['lower_band'].iloc[idx]
            channel_mean = self.indicators['channel_mean'].iloc[idx]
            channel_position = self.indicators['channel_position'].iloc[idx]
            breakout = self.indicators['breakout'].iloc[idx]
            rsi = self.indicators['rsi'].iloc[idx]
            z_score = self.indicators['z_score'].iloc[idx]
            channel_slope = self.indicators['channel_slope'].iloc[idx]
            volume_ratio = self.indicators['volume_ratio'].iloc[idx]
            channel_width_pct = self.indicators['channel_width_pct'].iloc[idx]
            momentum = self.indicators['momentum'].iloc[idx]
            
            # Check for NaN values
            if pd.isna(upper_band) or pd.isna(rsi):
                return {'signal': 0, 'reason': 'Indicators not ready'}
            
            # Initialize signal
            signal = 0.0
            confidence = 0.0
            reason = "No signal"
            
            # BREAKOUT SIGNALS
            if breakout == 1 and channel_slope > 0.1:
                # Bullish breakout
                signal = 0.8
                confidence = 0.8
                reason = "Bullish channel breakout"
                
                # Strong volume confirmation
                if volume_ratio > 1.5:
                    confidence = min(confidence + 0.1, 0.95)
                    reason += " + volume surge"
                
                # Momentum confirmation
                if momentum > 0 and rsi > 60:
                    confidence = min(confidence + 0.05, 0.95)
                    reason += " + momentum"
            
            elif breakout == -1 and channel_slope < -0.1:
                # Bearish breakout
                signal = -0.8
                confidence = 0.8
                reason = "Bearish channel breakout"
                
                # Strong volume confirmation
                if volume_ratio > 1.5:
                    confidence = min(confidence + 0.1, 0.95)
                    reason += " + volume surge"
                
                # Momentum confirmation
                if momentum < 0 and rsi < 40:
                    confidence = min(confidence + 0.05, 0.95)
                    reason += " + momentum"
            
            # MEAN REVERSION SIGNALS
            elif abs(z_score) > 2:
                # Extreme deviation from mean
                if z_score > 2 and rsi > 70:
                    # Overbought - sell signal
                    signal = -0.9
                    confidence = 0.85
                    reason = "Extreme overbought (Z > 2)"
                    
                    # In narrow channel (more reliable)
                    if channel_width_pct < 2:
                        confidence = min(confidence + 0.1, 0.95)
                        reason += " in tight channel"
                
                elif z_score < -2 and rsi < 30:
                    # Oversold - buy signal
                    signal = 0.9
                    confidence = 0.85
                    reason = "Extreme oversold (Z < -2)"
                    
                    # In narrow channel (more reliable)
                    if channel_width_pct < 2:
                        confidence = min(confidence + 0.1, 0.95)
                        reason += " in tight channel"
            
            # BAND TOUCH SIGNALS
            elif channel_position >= 0.95 and abs(channel_slope) < 0.5:
                # At upper band in ranging market
                signal = -0.6
                confidence = 0.7
                reason = "Upper band resistance"
                
                # RSI confirmation
                if rsi > 65:
                    confidence = min(confidence + 0.1, 0.85)
                    reason += " + RSI overbought"
            
            elif channel_position <= 0.05 and abs(channel_slope) < 0.5:
                # At lower band in ranging market
                signal = 0.6
                confidence = 0.7
                reason = "Lower band support"
                
                # RSI confirmation
                if rsi < 35:
                    confidence = min(confidence + 0.1, 0.85)
                    reason += " + RSI oversold"
            
            # TREND FOLLOWING IN WIDE CHANNELS
            elif channel_width_pct > 4:
                # Wide channel suggests trending market
                if channel_position > 0.7 and channel_slope > 0.2:
                    signal = 0.5
                    confidence = 0.6
                    reason = "Uptrend in expanding channel"
                
                elif channel_position < 0.3 and channel_slope < -0.2:
                    signal = -0.5
                    confidence = 0.6
                    reason = "Downtrend in expanding channel"
            
            # Reduce confidence in very wide channels (high volatility)
            if channel_width_pct > 6:
                confidence *= 0.8
                reason += " (high volatility)"
            
            return {
                'signal': round(signal, 3),
                'confidence': round(confidence, 3),
                'reason': reason,
                'upper_band': round(upper_band, 2),
                'lower_band': round(lower_band, 2),
                'channel_mean': round(channel_mean, 2),
                'channel_position': round(channel_position, 3),
                'channel_width_pct': round(channel_width_pct, 2),
                'z_score': round(z_score, 2),
                'channel_slope': round(channel_slope, 3),
                'rsi': round(rsi, 2),
                'volume_ratio': round(volume_ratio, 2),
                'price': round(current_close, 2)
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating signal: {e}")
            return {'signal': 0, 'error': str(e)}
    
    def get_required_data_points(self) -> int:
        """
        Get minimum required data points.
        
        Returns:
            Maximum of period requirements
        """
        return max(self.period + 20, 70)  # Need extra for slope calculation
    
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        Determine if strategy is suitable for current market regime.
        
        Gaussian Channel works in all markets but excels in different ways.
        
        Args:
            market_bias: Current market regime
            
        Returns:
            True if market is suitable
        """
        # Strategy adapts to all market conditions
        # Mean reversion in ranging markets
        # Breakout in trending markets
        return True  # Works in all regimes with different approaches
