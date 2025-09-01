"""
Bollinger Bands Mean Reversion Strategy - Executable Implementation

This module provides an executable implementation of the Bollinger Bands
Mean Reversion strategy that conforms to the TradingStrategy interface.

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


class BollingerBandsMeanReversion(TradingStrategy):
    """
    Executable Bollinger Bands Mean Reversion Strategy.
    
    This strategy generates buy signals when price touches the lower band
    with oversold RSI, and sell signals when price touches the upper band
    with overbought RSI.
    """
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize the Bollinger Bands strategy.
        
        Args:
            data: OHLCV DataFrame
            config: Optional configuration with keys:
                - bb_length: Bollinger Bands period (default: 20)
                - bb_std: Standard deviation multiplier (default: 2.0)
                - rsi_length: RSI period (default: 14)
                - rsi_oversold: RSI oversold threshold (default: 35)
                - rsi_overbought: RSI overbought threshold (default: 65)
        """
        super().__init__(data, config)
        self.name = "Bollinger Bands Mean Reversion"
        self.version = "1.0.0"
        
        # Strategy parameters from config
        self.bb_length = self.config.get('bb_length', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        self.rsi_length = self.config.get('rsi_length', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 35)
        self.rsi_overbought = self.config.get('rsi_overbought', 65)
        
        # Calculate indicators on initialization if data is available
        self.indicators = {}
        if self.has_sufficient_data():
            self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """Calculate Bollinger Bands and RSI indicators."""
        try:
            # Calculate Bollinger Bands
            bb = ta.bbands(
                self.data['close'], 
                length=self.bb_length, 
                std=self.bb_std
            )
            
            # Extract BB components
            self.indicators['bb_lower'] = bb[f'BBL_{self.bb_length}_{self.bb_std}']
            self.indicators['bb_middle'] = bb[f'BBM_{self.bb_length}_{self.bb_std}']
            self.indicators['bb_upper'] = bb[f'BBU_{self.bb_length}_{self.bb_std}']
            self.indicators['bb_bandwidth'] = bb[f'BBB_{self.bb_length}_{self.bb_std}']
            self.indicators['bb_percent'] = bb[f'BBP_{self.bb_length}_{self.bb_std}']
            
            # Calculate RSI
            self.indicators['rsi'] = ta.rsi(self.data['close'], length=self.rsi_length)
            
            # Calculate price position relative to bands
            self.indicators['price_to_lower'] = (
                (self.data['close'] - self.indicators['bb_lower']) / 
                self.data['close'] * 100
            )
            self.indicators['price_to_upper'] = (
                (self.indicators['bb_upper'] - self.data['close']) / 
                self.data['close'] * 100
            )
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating indicators: {e}")
            self.indicators = {}
    
    def calculate_signal(self) -> Dict[str, float]:
        """
        Calculate trading signal based on Bollinger Bands and RSI.
        
        Returns:
            Dictionary with signal and metadata
        """
        if not self.has_sufficient_data():
            return {'signal': 0, 'reason': 'Insufficient data'}
        
        try:
            # Ensure indicators are calculated
            if not self.indicators:
                self._calculate_indicators()
            
            # Get latest values
            current_close = self.data['close'].iloc[-1]
            current_low = self.data['low'].iloc[-1]
            current_high = self.data['high'].iloc[-1]
            
            bb_lower = self.indicators['bb_lower'].iloc[-1]
            bb_upper = self.indicators['bb_upper'].iloc[-1]
            bb_middle = self.indicators['bb_middle'].iloc[-1]
            bb_bandwidth = self.indicators['bb_bandwidth'].iloc[-1]
            bb_percent = self.indicators['bb_percent'].iloc[-1]
            rsi = self.indicators['rsi'].iloc[-1]
            
            # Check for NaN values
            if pd.isna(rsi) or pd.isna(bb_lower) or pd.isna(bb_upper):
                return {'signal': 0, 'reason': 'Indicators not ready'}
            
            # Initialize signal and metadata
            signal = 0.0
            confidence = 0.0
            reason = "No signal"
            
            # Buy signal conditions
            if (current_low <= bb_lower or current_close <= bb_lower * 1.01) and rsi < self.rsi_oversold:
                # Strong buy signal
                signal = 1.0
                confidence = min((self.rsi_oversold - rsi) / self.rsi_oversold, 1.0)
                reason = "Price at lower band with oversold RSI"
                
            elif current_close < bb_middle and rsi < 50:
                # Moderate buy signal
                distance_pct = (bb_middle - current_close) / bb_middle * 100
                signal = min(distance_pct / 5, 0.5)  # Max 0.5 for moderate signal
                confidence = 0.5
                reason = "Price below middle band"
                
            # Sell signal conditions
            elif (current_high >= bb_upper or current_close >= bb_upper * 0.99) and rsi > self.rsi_overbought:
                # Strong sell signal
                signal = -1.0
                confidence = min((rsi - self.rsi_overbought) / (100 - self.rsi_overbought), 1.0)
                reason = "Price at upper band with overbought RSI"
                
            elif current_close > bb_middle and rsi > 50:
                # Moderate sell signal
                distance_pct = (current_close - bb_middle) / bb_middle * 100
                signal = -min(distance_pct / 5, 0.5)  # Max -0.5 for moderate signal
                confidence = 0.5
                reason = "Price above middle band"
            
            # Additional confidence adjustments based on bandwidth
            # Narrow bands suggest impending volatility
            if bb_bandwidth < 0.1:  # Very narrow bands
                confidence *= 1.2  # Increase confidence
            elif bb_bandwidth > 0.5:  # Very wide bands
                confidence *= 0.8  # Decrease confidence
                
            confidence = min(confidence, 1.0)  # Cap at 1.0
            
            return {
                'signal': round(signal, 3),
                'confidence': round(confidence, 3),
                'reason': reason,
                'rsi': round(rsi, 2),
                'bb_lower': round(bb_lower, 2),
                'bb_middle': round(bb_middle, 2),
                'bb_upper': round(bb_upper, 2),
                'bb_bandwidth': round(bb_bandwidth, 4),
                'bb_percent': round(bb_percent, 3),
                'price': round(current_close, 2)
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating signal: {e}")
            return {'signal': 0, 'error': str(e)}
    
    def get_required_data_points(self) -> int:
        """
        Get minimum required data points.
        
        Returns:
            Maximum of BB length and RSI length plus buffer
        """
        return max(self.bb_length, self.rsi_length) + 10  # Extra buffer for stability
    
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        Determine if strategy is suitable for current market regime.
        
        Mean reversion works best in ranging/neutral markets.
        
        Args:
            market_bias: Current market regime
            
        Returns:
            True if market is suitable for mean reversion
        """
        suitable_regimes = ['Neutral', 'Ranging', 'Consolidating']
        # Also allow in weak trending markets
        weakly_suitable = ['Bullish', 'Bearish']
        
        if market_bias in suitable_regimes:
            return True
        elif market_bias in weakly_suitable:
            # Allow but the orchestrator should reduce weight
            return True
        else:
            # Strong trends are not suitable for mean reversion
            return False
    
    def get_strategy_metrics(self) -> Dict[str, float]:
        """
        Get additional strategy metrics for analysis.
        
        Returns:
            Dictionary of strategy-specific metrics
        """
        if not self.indicators:
            return {}
        
        try:
            current_idx = -1
            return {
                'bb_squeeze': self.indicators['bb_bandwidth'].iloc[current_idx] < 0.05,
                'price_percentile': self.indicators['bb_percent'].iloc[current_idx],
                'rsi_momentum': self.indicators['rsi'].iloc[current_idx] - self.indicators['rsi'].iloc[current_idx - 5],
                'volatility_rank': self._calculate_volatility_rank()
            }
        except:
            return {}
    
    def _calculate_volatility_rank(self) -> float:
        """
        Calculate volatility rank (0-1) based on current bandwidth vs historical.
        
        Returns:
            Volatility rank between 0 (low vol) and 1 (high vol)
        """
        try:
            current_bandwidth = self.indicators['bb_bandwidth'].iloc[-1]
            historical_bandwidth = self.indicators['bb_bandwidth'].iloc[-252:]  # 1 year
            rank = (historical_bandwidth < current_bandwidth).sum() / len(historical_bandwidth)
            return round(rank, 3)
        except:
            return 0.5  # Default to middle if calculation fails
