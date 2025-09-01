"""
Parabolic SAR Trend Following Strategy - Executable Implementation

This module provides an executable implementation of the Parabolic SAR
Trend Following strategy that conforms to the TradingStrategy interface.

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


class ParabolicSARTrendFollowing(TradingStrategy):
    """
    Executable Parabolic SAR Trend Following Strategy.
    
    This strategy uses the Parabolic SAR (Stop and Reverse) indicator to
    identify trend direction and generate signals when the SAR flips from
    one side of price to the other.
    """
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize the Parabolic SAR strategy.
        
        Args:
            data: OHLCV DataFrame
            config: Optional configuration with keys:
                - start: Initial acceleration factor (default: 0.02)
                - increment: Acceleration increment (default: 0.02)
                - maximum: Maximum acceleration factor (default: 0.2)
                - atr_period: ATR period for volatility (default: 14)
                - adx_period: ADX period for trend strength (default: 14)
        """
        super().__init__(data, config)
        self.name = "Parabolic SAR Trend Following"
        self.version = "1.0.0"
        
        # Strategy parameters
        self.start = self.config.get('start', 0.02)
        self.increment = self.config.get('increment', 0.02)
        self.maximum = self.config.get('maximum', 0.2)
        self.atr_period = self.config.get('atr_period', 14)
        self.adx_period = self.config.get('adx_period', 14)
        
        # Initialize indicators
        self.indicators = {}
        if self.has_sufficient_data():
            self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """Calculate Parabolic SAR and supporting indicators."""
        try:
            # Calculate Parabolic SAR
            psar_result = ta.psar(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                af0=self.start,
                af=self.increment,
                max_af=self.maximum
            )
            
            # Extract PSAR components
            if isinstance(psar_result, pd.DataFrame):
                self.indicators['psar_long'] = psar_result.iloc[:, 0]  # Long/Support SAR
                self.indicators['psar_short'] = psar_result.iloc[:, 1]  # Short/Resistance SAR
                self.indicators['psar_af'] = psar_result.iloc[:, 2] if psar_result.shape[1] > 2 else None
                self.indicators['psar_reversal'] = psar_result.iloc[:, 3] if psar_result.shape[1] > 3 else None
            else:
                # Single series output
                self.indicators['psar'] = psar_result
            
            # Calculate trend direction
            self.indicators['trend'] = self._calculate_trend()
            
            # Calculate SAR flips
            self.indicators['sar_flip'] = self._detect_sar_flips()
            
            # Calculate ATR for volatility
            self.indicators['atr'] = ta.atr(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                length=self.atr_period
            )
            
            # Calculate ADX for trend strength
            adx_result = ta.adx(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                length=self.adx_period
            )
            self.indicators['adx'] = adx_result[f'ADX_{self.adx_period}']
            self.indicators['di_plus'] = adx_result[f'DMP_{self.adx_period}']
            self.indicators['di_minus'] = adx_result[f'DMN_{self.adx_period}']
            
            # Calculate distance from SAR
            self.indicators['distance_from_sar'] = self._calculate_distance_from_sar()
            
            # Moving averages for additional confirmation
            self.indicators['sma_20'] = ta.sma(self.data['close'], length=20)
            self.indicators['sma_50'] = ta.sma(self.data['close'], length=50)
            
            # Volume analysis
            self.indicators['volume_sma'] = ta.sma(self.data['volume'], length=20)
            self.indicators['volume_ratio'] = self.data['volume'] / self.indicators['volume_sma']
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating indicators: {e}")
            self.indicators = {}
    
    def _calculate_trend(self) -> pd.Series:
        """Calculate current trend based on PSAR position."""
        trend = pd.Series('neutral', index=self.data.index)
        
        if 'psar_long' in self.indicators and 'psar_short' in self.indicators:
            # When PSAR is below price (long SAR active), trend is up
            trend[self.indicators['psar_long'].notna()] = 'up'
            # When PSAR is above price (short SAR active), trend is down
            trend[self.indicators['psar_short'].notna()] = 'down'
        elif 'psar' in self.indicators:
            # Single PSAR series
            trend[self.data['close'] > self.indicators['psar']] = 'up'
            trend[self.data['close'] < self.indicators['psar']] = 'down'
        
        return trend
    
    def _detect_sar_flips(self) -> pd.Series:
        """Detect when SAR flips from one side to another."""
        flips = pd.Series(0, index=self.data.index)
        
        trend = self.indicators['trend']
        
        # Bullish flip: trend changes from down to up
        bullish_flip = (trend == 'up') & (trend.shift(1) == 'down')
        flips[bullish_flip] = 1
        
        # Bearish flip: trend changes from up to down
        bearish_flip = (trend == 'down') & (trend.shift(1) == 'up')
        flips[bearish_flip] = -1
        
        return flips
    
    def _calculate_distance_from_sar(self) -> pd.Series:
        """Calculate percentage distance from current SAR."""
        distance = pd.Series(0.0, index=self.data.index)
        
        if 'psar_long' in self.indicators and 'psar_short' in self.indicators:
            # Use active SAR
            long_mask = self.indicators['psar_long'].notna()
            short_mask = self.indicators['psar_short'].notna()
            
            distance[long_mask] = (
                (self.data['close'][long_mask] - self.indicators['psar_long'][long_mask]) / 
                self.data['close'][long_mask] * 100
            )
            distance[short_mask] = (
                (self.indicators['psar_short'][short_mask] - self.data['close'][short_mask]) / 
                self.data['close'][short_mask] * 100
            )
        elif 'psar' in self.indicators:
            distance = (
                (self.data['close'] - self.indicators['psar']) / 
                self.data['close'] * 100
            )
        
        return distance
    
    def calculate_signal(self) -> Dict[str, float]:
        """
        Calculate trading signal based on Parabolic SAR strategy.
        
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
            trend = self.indicators['trend'].iloc[idx]
            sar_flip = self.indicators['sar_flip'].iloc[idx]
            distance_from_sar = self.indicators['distance_from_sar'].iloc[idx]
            
            # Get ADX values
            adx = self.indicators['adx'].iloc[idx]
            di_plus = self.indicators['di_plus'].iloc[idx]
            di_minus = self.indicators['di_minus'].iloc[idx]
            
            # Get moving averages
            sma_20 = self.indicators['sma_20'].iloc[idx]
            sma_50 = self.indicators['sma_50'].iloc[idx]
            
            # Volume ratio
            volume_ratio = self.indicators['volume_ratio'].iloc[idx]
            
            # Get current SAR value
            if 'psar_long' in self.indicators and 'psar_short' in self.indicators:
                if pd.notna(self.indicators['psar_long'].iloc[idx]):
                    current_sar = self.indicators['psar_long'].iloc[idx]
                else:
                    current_sar = self.indicators['psar_short'].iloc[idx]
            else:
                current_sar = self.indicators['psar'].iloc[idx]
            
            # Check for NaN values
            if pd.isna(current_sar) or pd.isna(adx):
                return {'signal': 0, 'reason': 'Indicators not ready'}
            
            # Initialize signal
            signal = 0.0
            confidence = 0.0
            reason = "No signal"
            
            # STRONG BUY SIGNAL - SAR flip to bullish
            if sar_flip == 1 and adx > 25:
                signal = 1.0
                confidence = 0.9
                reason = "Bullish SAR flip with strong trend"
                
                # Boost confidence with volume
                if volume_ratio > 1.5:
                    confidence = min(confidence + 0.1, 1.0)
                    reason += " + volume confirmation"
                
                # Check MA alignment
                if current_close > sma_20 > sma_50:
                    confidence = min(confidence + 0.05, 1.0)
                    reason += " + MA alignment"
            
            # MODERATE BUY SIGNAL - Uptrend continuation
            elif (trend == 'up' and 
                  distance_from_sar > 1 and  # Some distance from SAR
                  di_plus > di_minus and
                  adx > 20):
                
                signal = 0.6
                confidence = 0.7
                reason = "Uptrend continuation"
                
                # Stronger signal if pulling back to SAR
                if distance_from_sar < 3:
                    signal = 0.8
                    confidence = 0.8
                    reason = "Uptrend pullback to SAR support"
            
            # STRONG SELL SIGNAL - SAR flip to bearish
            elif sar_flip == -1 and adx > 25:
                signal = -1.0
                confidence = 0.9
                reason = "Bearish SAR flip with strong trend"
                
                # Boost confidence with volume
                if volume_ratio > 1.5:
                    confidence = min(confidence + 0.1, 1.0)
                    reason += " + volume confirmation"
                
                # Check MA alignment
                if current_close < sma_20 < sma_50:
                    confidence = min(confidence + 0.05, 1.0)
                    reason += " + MA alignment"
            
            # MODERATE SELL SIGNAL - Downtrend continuation
            elif (trend == 'down' and 
                  distance_from_sar > 1 and
                  di_minus > di_plus and
                  adx > 20):
                
                signal = -0.6
                confidence = 0.7
                reason = "Downtrend continuation"
                
                # Stronger signal if rallying to SAR
                if distance_from_sar < 3:
                    signal = -0.8
                    confidence = 0.8
                    reason = "Downtrend rally to SAR resistance"
            
            # WEAK TREND WARNING
            elif adx < 20:
                if trend == 'up':
                    signal = 0.2
                elif trend == 'down':
                    signal = -0.2
                confidence = 0.4
                reason = f"Weak {trend}trend (low ADX)"
            
            # Reduce confidence if SAR is too far (overextended)
            if abs(distance_from_sar) > 10:
                confidence *= 0.8
                reason += " (overextended from SAR)"
            
            return {
                'signal': round(signal, 3),
                'confidence': round(confidence, 3),
                'reason': reason,
                'current_sar': round(current_sar, 2),
                'trend': trend,
                'distance_from_sar': round(distance_from_sar, 2),
                'adx': round(adx, 2),
                'di_plus': round(di_plus, 2),
                'di_minus': round(di_minus, 2),
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
            Maximum of indicator periods
        """
        return max(50, self.adx_period + 10)  # Need sufficient data for ADX
    
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        Determine if strategy is suitable for current market regime.
        
        Parabolic SAR works best in trending markets.
        
        Args:
            market_bias: Current market regime
            
        Returns:
            True if market is suitable for trend following
        """
        # Strategy works best in strong trends
        suitable_regimes = ['Strong Bullish', 'Bullish', 'Bearish', 'Strong Bearish']
        # Less effective in neutral/ranging markets
        unsuitable_regimes = ['Neutral', 'Ranging', 'Consolidating']
        
        return market_bias not in unsuitable_regimes
