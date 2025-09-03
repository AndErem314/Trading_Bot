"""
Improved Market Regime Detector for Crypto Markets

This module implements an improved market regime detection using
multiple indicators with more appropriate thresholds for crypto markets.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional

# Fix numpy compatibility issue with pandas_ta
import numpy
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan

import pandas_ta as ta

logger = logging.getLogger(__name__)


class ImprovedMarketRegimeDetector:
    """
    Improved market regime detector with better logic for crypto markets.
    
    Regimes:
    - STRONG_BULLISH: Strong uptrend with momentum
    - BULLISH: Moderate uptrend
    - NEUTRAL_RANGING: Sideways/consolidation
    - BEARISH: Moderate downtrend
    - STRONG_BEARISH: Strong downtrend with momentum
    - CRASH_PANIC: Extreme bearish with high volatility
    """
    
    # Regime constants
    REGIME_CRASH_PANIC = "CRASH_PANIC"
    REGIME_STRONG_BULLISH = "STRONG_BULLISH"
    REGIME_BULLISH = "BULLISH"
    REGIME_STRONG_BEARISH = "STRONG_BEARISH"
    REGIME_BEARISH = "BEARISH"
    REGIME_NEUTRAL_RANGING = "NEUTRAL_RANGING"
    
    # Indicator parameters (adjusted for crypto)
    ADX_PERIOD = 14
    ATR_PERIOD = 14
    SMA_FAST = 20  # Faster SMA for crypto
    SMA_MEDIUM = 50
    SMA_SLOW = 100  # Using 100 instead of 200 for crypto
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Adjusted thresholds for crypto markets
    ADX_WEAK_THRESHOLD = 15  # Below this is ranging
    ADX_MODERATE_THRESHOLD = 20  # Moderate trend
    ADX_STRONG_THRESHOLD = 30  # Strong trend
    ADX_EXTREME_THRESHOLD = 40  # Very strong trend
    
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_EXTREME_OVERSOLD = 20
    RSI_EXTREME_OVERBOUGHT = 80
    
    ATR_PERCENTAGE_HIGH = 3.0  # High volatility for crypto
    ATR_PERCENTAGE_CRASH = 5.0  # Extreme volatility
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the improved regime detector.
        
        Args:
            data: OHLCV DataFrame with DatetimeIndex
        """
        self.data = data.copy()
        self.indicators = {}
        self._calculate_all_indicators()
    
    def _calculate_all_indicators(self) -> None:
        """Calculate all required indicators for regime detection."""
        try:
            # ADX for trend strength
            adx_result = ta.adx(
                self.data['high'], 
                self.data['low'], 
                self.data['close'], 
                length=self.ADX_PERIOD
            )
            self.indicators['ADX'] = adx_result[f'ADX_{self.ADX_PERIOD}']
            self.indicators['DI_PLUS'] = adx_result[f'DMP_{self.ADX_PERIOD}']
            self.indicators['DI_MINUS'] = adx_result[f'DMN_{self.ADX_PERIOD}']
            
            # Multiple SMAs for better trend detection
            self.indicators['SMA_20'] = ta.sma(self.data['close'], length=self.SMA_FAST)
            self.indicators['SMA_50'] = ta.sma(self.data['close'], length=self.SMA_MEDIUM)
            self.indicators['SMA_100'] = ta.sma(self.data['close'], length=self.SMA_SLOW)
            
            # RSI for momentum
            self.indicators['RSI'] = ta.rsi(self.data['close'], length=self.RSI_PERIOD)
            
            # MACD for trend confirmation
            macd_result = ta.macd(
                self.data['close'],
                fast=self.MACD_FAST,
                slow=self.MACD_SLOW,
                signal=self.MACD_SIGNAL
            )
            self.indicators['MACD'] = macd_result[f'MACD_{self.MACD_FAST}_{self.MACD_SLOW}_{self.MACD_SIGNAL}']
            self.indicators['MACD_SIGNAL'] = macd_result[f'MACDs_{self.MACD_FAST}_{self.MACD_SLOW}_{self.MACD_SIGNAL}']
            self.indicators['MACD_HISTOGRAM'] = macd_result[f'MACDh_{self.MACD_FAST}_{self.MACD_SLOW}_{self.MACD_SIGNAL}']
            
            # ATR for volatility
            self.indicators['ATR'] = ta.atr(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                length=self.ATR_PERIOD
            )
            
            # ATR percentage
            self.indicators['ATR_PERCENTAGE'] = (self.indicators['ATR'] / self.data['close']) * 100
            
            # Price rate of change (momentum)
            self.indicators['ROC_10'] = ta.roc(self.data['close'], length=10)
            self.indicators['ROC_20'] = ta.roc(self.data['close'], length=20)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
    
    def _calculate_trend_score(self, price, sma_20, sma_50, sma_100, di_plus, di_minus, macd, macd_hist, roc_10, roc_20):
        """
        Calculate a comprehensive trend score from -100 to 100.
        Positive = bullish, Negative = bearish, Near 0 = ranging
        """
        score = 0
        
        # SMA alignment scoring (0 to 30 points)
        if price > sma_20 > sma_50 > sma_100:
            score += 30  # Perfect bullish alignment
        elif price > sma_20 and sma_20 > sma_50:
            score += 20  # Good bullish alignment
        elif price > sma_20:
            score += 10  # Weak bullish
        elif price < sma_20 < sma_50 < sma_100:
            score -= 30  # Perfect bearish alignment
        elif price < sma_20 and sma_20 < sma_50:
            score -= 20  # Good bearish alignment
        elif price < sma_20:
            score -= 10  # Weak bearish
        
        # DI+/DI- scoring (0 to 20 points)
        di_diff = di_plus - di_minus
        if di_diff > 10:
            score += 20
        elif di_diff > 5:
            score += 10
        elif di_diff < -10:
            score -= 20
        elif di_diff < -5:
            score -= 10
        
        # MACD scoring (0 to 20 points)
        if macd > 0 and macd_hist > 0:
            score += 20
        elif macd > 0:
            score += 10
        elif macd < 0 and macd_hist < 0:
            score -= 20
        elif macd < 0:
            score -= 10
        
        # Momentum scoring (0 to 30 points)
        momentum_score = 0
        if roc_10 > 5:
            momentum_score += 15
        elif roc_10 > 2:
            momentum_score += 7
        elif roc_10 < -5:
            momentum_score -= 15
        elif roc_10 < -2:
            momentum_score -= 7
        
        if roc_20 > 10:
            momentum_score += 15
        elif roc_20 > 5:
            momentum_score += 7
        elif roc_20 < -10:
            momentum_score -= 15
        elif roc_20 < -5:
            momentum_score -= 7
        
        score += momentum_score
        
        return np.clip(score, -100, 100)
    
    def detect_market_regime(self) -> Tuple[str, Dict[str, float]]:
        """
        Detect current market regime using improved logic.
        
        Returns:
            Tuple of (regime_name, regime_metrics)
        """
        # Get current values
        idx = -1
        current_price = self.data['close'].iloc[idx]
        adx = self.indicators['ADX'].iloc[idx]
        di_plus = self.indicators['DI_PLUS'].iloc[idx]
        di_minus = self.indicators['DI_MINUS'].iloc[idx]
        sma_20 = self.indicators['SMA_20'].iloc[idx]
        sma_50 = self.indicators['SMA_50'].iloc[idx]
        sma_100 = self.indicators['SMA_100'].iloc[idx]
        rsi = self.indicators['RSI'].iloc[idx]
        macd_line = self.indicators['MACD'].iloc[idx]
        macd_hist = self.indicators['MACD_HISTOGRAM'].iloc[idx]
        atr_percentage = self.indicators['ATR_PERCENTAGE'].iloc[idx]
        roc_10 = self.indicators['ROC_10'].iloc[idx]
        roc_20 = self.indicators['ROC_20'].iloc[idx]
        
        # Check for sufficient data
        if pd.isna(adx) or pd.isna(sma_100) or pd.isna(macd_line) or pd.isna(rsi):
            return self.REGIME_NEUTRAL_RANGING, {"error": "Insufficient data"}
        
        # Calculate trend score
        trend_score = self._calculate_trend_score(
            current_price, sma_20, sma_50, sma_100,
            di_plus, di_minus, macd_line, macd_hist,
            roc_10, roc_20
        )
        
        # Crash detection
        is_crash = (
            trend_score < -50 and
            atr_percentage > self.ATR_PERCENTAGE_CRASH and
            rsi < self.RSI_EXTREME_OVERSOLD
        )
        
        # Determine regime based on trend score and ADX
        if is_crash:
            regime = self.REGIME_CRASH_PANIC
        elif adx < self.ADX_WEAK_THRESHOLD:
            # Low ADX = ranging market
            regime = self.REGIME_NEUTRAL_RANGING
        elif trend_score > 50 and adx > self.ADX_STRONG_THRESHOLD:
            regime = self.REGIME_STRONG_BULLISH
        elif trend_score > 30:
            regime = self.REGIME_BULLISH
        elif trend_score < -50 and adx > self.ADX_STRONG_THRESHOLD:
            regime = self.REGIME_STRONG_BEARISH
        elif trend_score < -30:
            regime = self.REGIME_BEARISH
        else:
            regime = self.REGIME_NEUTRAL_RANGING
        
        # Compile metrics
        regime_metrics = {
            'adx': float(round(adx, 2)),
            'trend_score': float(round(trend_score, 2)),
            'rsi': float(round(rsi, 2)),
            'atr_percentage': float(round(atr_percentage, 2)),
            'sma_20': float(round(sma_20, 2)),
            'sma_50': float(round(sma_50, 2)),
            'sma_100': float(round(sma_100, 2)),
            'current_price': float(round(current_price, 2)),
            'macd_line': float(round(macd_line, 4)),
            'macd_histogram': float(round(macd_hist, 4)),
            'di_plus': float(round(di_plus, 2)),
            'di_minus': float(round(di_minus, 2)),
            'roc_10': float(round(roc_10, 2)),
            'roc_20': float(round(roc_20, 2)),
            'is_crash': bool(is_crash)
        }
        
        return regime, regime_metrics
    
    def get_regime_history(self, lookback_periods: int = 100) -> pd.DataFrame:
        """
        Calculate regime history for backtesting and analysis.
        """
        results = []
        
        # Calculate starting index
        if lookback_periods >= len(self.data):
            start_idx = self.SMA_SLOW
        else:
            start_idx = max(self.SMA_SLOW, len(self.data) - lookback_periods)
        
        for i in range(start_idx, len(self.data)):
            # Get values at index i
            price = self.data['close'].iloc[i]
            adx = self.indicators['ADX'].iloc[i]
            di_plus = self.indicators['DI_PLUS'].iloc[i]
            di_minus = self.indicators['DI_MINUS'].iloc[i]
            sma_20 = self.indicators['SMA_20'].iloc[i]
            sma_50 = self.indicators['SMA_50'].iloc[i]
            sma_100 = self.indicators['SMA_100'].iloc[i]
            rsi = self.indicators['RSI'].iloc[i]
            macd_line = self.indicators['MACD'].iloc[i]
            macd_hist = self.indicators['MACD_HISTOGRAM'].iloc[i]
            atr_percentage = self.indicators['ATR_PERCENTAGE'].iloc[i]
            roc_10 = self.indicators['ROC_10'].iloc[i]
            roc_20 = self.indicators['ROC_20'].iloc[i]
            
            if pd.isna(adx) or pd.isna(sma_100) or pd.isna(macd_line) or pd.isna(rsi):
                continue
            
            # Calculate trend score
            trend_score = self._calculate_trend_score(
                price, sma_20, sma_50, sma_100,
                di_plus, di_minus, macd_line, macd_hist,
                roc_10, roc_20
            )
            
            # Crash detection
            is_crash = (
                trend_score < -50 and
                atr_percentage > self.ATR_PERCENTAGE_CRASH and
                rsi < self.RSI_EXTREME_OVERSOLD
            )
            
            # Determine regime
            if is_crash:
                regime = self.REGIME_CRASH_PANIC
            elif adx < self.ADX_WEAK_THRESHOLD:
                regime = self.REGIME_NEUTRAL_RANGING
            elif trend_score > 50 and adx > self.ADX_STRONG_THRESHOLD:
                regime = self.REGIME_STRONG_BULLISH
            elif trend_score > 30:
                regime = self.REGIME_BULLISH
            elif trend_score < -50 and adx > self.ADX_STRONG_THRESHOLD:
                regime = self.REGIME_STRONG_BEARISH
            elif trend_score < -30:
                regime = self.REGIME_BEARISH
            else:
                regime = self.REGIME_NEUTRAL_RANGING
            
            results.append({
                'timestamp': self.data.index[i],
                'regime': regime,
                'adx': adx,
                'trend_score': trend_score,
                'rsi': rsi,
                'atr_percentage': atr_percentage,
                'price': price
            })
        
        return pd.DataFrame(results).set_index('timestamp')
