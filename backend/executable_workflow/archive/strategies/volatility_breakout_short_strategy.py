"""
Volatility Breakout Short Strategy - Executable Implementation

This module provides a specialized strategy for trading during crash/panic
market conditions with extreme volatility.
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
from backend.executable_workflow.interfaces.trading_strategy_interface import TradingStrategy

logger = logging.getLogger(__name__)


class VolatilityBreakoutShort(TradingStrategy):
    """
    Specialized strategy for crash/panic market conditions.
    
    This strategy is designed to capitalize on extreme volatility and
    cascading selloffs during market crashes. It uses wider stops and
    reduced position sizing to manage the extreme risk.
    """
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize the Volatility Breakout Short strategy.
        
        Args:
            data: OHLCV DataFrame
            config: Optional configuration with keys:
                - atr_period: ATR calculation period (default: 14)
                - lookback_period: Period for support detection (default: 20)
                - volume_multiplier: Volume surge threshold (default: 2.0)
                - rsi_period: RSI period (default: 14)
                - rsi_extreme: Extreme oversold level (default: 20)
                - atr_stop_multiplier: Stop loss ATR multiplier (default: 2.0)
                - atr_trail_multiplier: Trailing stop ATR multiplier (default: 1.5)
        """
        super().__init__(data, config)
        self.name = "Volatility Breakout Short"
        self.version = "1.0.0"
        
        # Strategy parameters
        self.atr_period = self.config.get('atr_period', 14)
        self.lookback_period = self.config.get('lookback_period', 20)
        self.volume_multiplier = self.config.get('volume_multiplier', 2.0)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_extreme = self.config.get('rsi_extreme', 20)
        self.atr_stop_multiplier = self.config.get('atr_stop_multiplier', 2.0)
        self.atr_trail_multiplier = self.config.get('atr_trail_multiplier', 1.5)
        
        # Initialize indicators
        self.indicators = {}
        if self.has_sufficient_data():
            self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """Calculate all required indicators for crash trading."""
        try:
            # ATR for volatility and stops
            self.indicators['atr'] = ta.atr(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                length=self.atr_period
            )
            
            # ATR percentage
            self.indicators['atr_percentage'] = (
                self.indicators['atr'] / self.data['close']
            ) * 100
            
            # RSI for extreme oversold conditions
            self.indicators['rsi'] = ta.rsi(
                self.data['close'],
                length=self.rsi_period
            )
            
            # Volume analysis
            self.indicators['volume_sma'] = self.data['volume'].rolling(
                window=20
            ).mean()
            self.indicators['volume_ratio'] = (
                self.data['volume'] / self.indicators['volume_sma']
            )
            
            # Support levels (20-period low)
            self.indicators['support'] = self.data['low'].rolling(
                window=self.lookback_period
            ).min()
            
            # Rate of change for momentum
            self.indicators['roc'] = ta.roc(
                self.data['close'],
                length=7
            )
            
            # Bollinger Bands for volatility expansion
            bb_result = ta.bbands(
                self.data['close'],
                length=20,
                std=2
            )
            self.indicators['bb_width'] = bb_result[f'BBB_20_2']
            
            # ADX for trend strength confirmation
            adx_result = ta.adx(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                length=14
            )
            self.indicators['adx'] = adx_result[f'ADX_14']
            
            # Detect support breakdowns
            self.indicators['support_break'] = (
                self.data['close'] < self.indicators['support']
            )
            
            # Calculate divergence for exit signals
            self.indicators['rsi_divergence'] = self._detect_rsi_divergence()
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating indicators: {e}")
            self.indicators = {}
    
    def _detect_rsi_divergence(self) -> pd.Series:
        """
        Detect bullish divergence in RSI for exit signals.
        
        Returns:
            Boolean series indicating bullish divergence
        """
        divergence = pd.Series(False, index=self.data.index)
        
        try:
            rsi = self.indicators['rsi']
            close = self.data['close']
            lookback = 10
            
            for i in range(lookback, len(self.data)):
                # Find recent lows
                recent_close = close.iloc[i-lookback:i+1]
                recent_rsi = rsi.iloc[i-lookback:i+1]
                
                # Simple divergence: price makes lower low, RSI makes higher low
                if len(recent_close) > 5:
                    price_min_idx = recent_close.idxmin()
                    rsi_at_price_min = rsi.loc[price_min_idx]
                    
                    # Check if current RSI is higher than RSI at price minimum
                    if (close.iloc[i] <= recent_close.min() * 1.01 and
                        rsi.iloc[i] > rsi_at_price_min + 5):
                        divergence.iloc[i] = True
                        
        except Exception as e:
            logger.error(f"Error detecting RSI divergence: {e}")
            
        return divergence
    
    def calculate_signal(self) -> Dict[str, float]:
        """
        Calculate trading signal for crash/panic conditions.
        
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
            current_low = self.data['low'].iloc[idx]
            atr = self.indicators['atr'].iloc[idx]
            atr_percentage = self.indicators['atr_percentage'].iloc[idx]
            rsi = self.indicators['rsi'].iloc[idx]
            volume_ratio = self.indicators['volume_ratio'].iloc[idx]
            support = self.indicators['support'].iloc[idx]
            support_break = self.indicators['support_break'].iloc[idx]
            roc = self.indicators['roc'].iloc[idx]
            bb_width = self.indicators['bb_width'].iloc[idx]
            adx = self.indicators['adx'].iloc[idx]
            rsi_divergence = self.indicators['rsi_divergence'].iloc[idx]
            
            # Previous values for trend
            prev_close = self.data['close'].iloc[idx-1]
            prev_support_break = self.indicators['support_break'].iloc[idx-1]
            
            # Check for NaN values
            if pd.isna(atr) or pd.isna(rsi) or pd.isna(adx):
                return {'signal': 0, 'reason': 'Indicators not ready'}
            
            # Initialize signal
            signal = 0.0
            confidence = 0.0
            reason = "No signal"
            
            # ENTRY CONDITIONS - SHORT ONLY IN CRASH
            if (atr_percentage > 2.5 and  # Extreme volatility confirmed
                support_break and  # Breaking support
                not prev_support_break and  # Fresh breakdown
                volume_ratio > self.volume_multiplier and  # Volume surge
                rsi < self.rsi_extreme and  # Extreme oversold but falling
                roc < -5 and  # Strong negative momentum
                adx > 35):  # Strong trend
                
                signal = -1.0  # Strong short signal
                confidence = 0.9
                reason = "Support breakdown with extreme volatility"
                
                # Boost confidence for cascading selloff
                if roc < -10 and bb_width > 0.1:
                    confidence = min(confidence + 0.1, 1.0)
                    reason += " + cascading selloff confirmed"
            
            # SCALE-IN OPPORTUNITY
            elif (atr_percentage > 2.5 and
                  current_close < support * 0.98 and  # Well below support
                  rsi < 15 and  # Ultra-extreme oversold
                  volume_ratio > 1.5):
                
                signal = -0.5  # Moderate short for scaling
                confidence = 0.7
                reason = "Scale-in opportunity in ongoing crash"
            
            # EXIT SIGNALS (for existing shorts)
            # Note: In practice, the orchestrator would track positions
            # This is a signal to close shorts, not go long
            elif (atr_percentage < 2.0 or  # Volatility normalizing
                  rsi_divergence or  # Bullish divergence detected
                  rsi > 30):  # RSI recovering
                
                signal = 0.0  # Neutral = close shorts
                confidence = 0.8
                reason = "Exit signal - volatility normalizing or divergence"
            
            # STAY OUT - Conditions not extreme enough
            elif atr_percentage < 2.5:
                signal = 0.0
                confidence = 0.5
                reason = "Insufficient volatility for crash strategy"
            
            # Calculate stop loss and target info
            stop_loss_distance = atr * self.atr_stop_multiplier
            stop_loss_price = current_close + stop_loss_distance
            
            return {
                'signal': round(signal, 3),
                'confidence': round(confidence, 3),
                'reason': reason,
                'atr_percentage': round(atr_percentage, 2),
                'rsi': round(rsi, 2),
                'volume_ratio': round(volume_ratio, 2),
                'support': round(support, 2),
                'roc': round(roc, 2),
                'adx': round(adx, 2),
                'stop_loss_price': round(stop_loss_price, 2),
                'stop_loss_distance_pct': round((stop_loss_distance / current_close) * 100, 2),
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
        return max(
            self.lookback_period,
            self.atr_period,
            self.rsi_period,
            50  # For ADX
        ) + 10
    
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        This strategy ONLY works in crash/panic conditions.
        
        Args:
            market_bias: Current market regime
            
        Returns:
            True only if market is in crash/panic
        """
        return market_bias in ['CRASH_PANIC', 'Crash', 'Panic']
    
    def get_risk_parameters(self) -> Dict[str, float]:
        """
        Get risk management parameters specific to crash conditions.
        
        Returns:
            Dictionary with risk parameters
        """
        return {
            'position_size_multiplier': 0.5,  # Half normal size
            'max_risk_per_trade': 0.005,  # 0.5% instead of 1%
            'stop_loss_atr_multiplier': self.atr_stop_multiplier,
            'trailing_stop_atr_multiplier': self.atr_trail_multiplier,
            'time_stop_hours': 48,  # Exit after 48 hours
            'profit_target_atr_multiplier': 3.0,  # 3x ATR profit target
            'scale_in_levels': 3,  # Allow 3 entries
            'scale_in_size_reduction': 0.7  # Each scale-in is 70% of previous
        }
