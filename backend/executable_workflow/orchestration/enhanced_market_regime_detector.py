"""
Enhanced Market Regime Detector with Improved Logic

This module implements the refined market regime detection using
ADX, SMA alignment, MACD, and ATR indicators as specified.
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


class EnhancedMarketRegimeDetector:
    """
    Enhanced market regime detector using refined indicator-based rules.
    
    Regimes:
    - STRONG_BULLISH: ADX > 25 with bullish alignment
    - STRONG_BEARISH: ADX > 25 with bearish alignment
    - NEUTRAL_RANGING: ADX < 20 (weak/no trend)
    - CRASH_PANIC: ADX > 35 with extreme volatility and bearish trend
    """
    
    # Regime constants
    REGIME_CRASH_PANIC = "CRASH_PANIC"
    REGIME_STRONG_BULLISH = "STRONG_BULLISH"
    REGIME_STRONG_BEARISH = "STRONG_BEARISH"
    REGIME_NEUTRAL_RANGING = "NEUTRAL_RANGING"
    
    # Indicator parameters
    ADX_PERIOD = 14
    ATR_PERIOD = 14
    SMA_FAST = 50
    SMA_SLOW = 200
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Thresholds
    ADX_NEUTRAL_THRESHOLD = 20
    ADX_STRONG_THRESHOLD = 25
    ADX_CRASH_THRESHOLD = 35
    ATR_PERCENTAGE_CRASH = 2.5
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the enhanced regime detector.
        
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
            
            # SMAs for trend direction
            self.indicators['SMA_50'] = ta.sma(self.data['close'], length=self.SMA_FAST)
            self.indicators['SMA_200'] = ta.sma(self.data['close'], length=self.SMA_SLOW)
            
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
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
    
    def detect_market_regime(self) -> Tuple[str, Dict[str, float]]:
        """
        Detect current market regime using the refined logic.
        
        Returns:
            Tuple of (regime_name, regime_metrics)
        """
        # Get current values
        idx = -1
        current_price = self.data['close'].iloc[idx]
        adx = self.indicators['ADX'].iloc[idx]
        sma_50 = self.indicators['SMA_50'].iloc[idx]
        sma_200 = self.indicators['SMA_200'].iloc[idx]
        macd_line = self.indicators['MACD'].iloc[idx]
        atr_percentage = self.indicators['ATR_PERCENTAGE'].iloc[idx]
        
        # Check for sufficient data
        if pd.isna(adx) or pd.isna(sma_200) or pd.isna(macd_line):
            return self.REGIME_NEUTRAL_RANGING, {"error": "Insufficient data"}
        
        # Determine trend direction using SMA alignment
        sma_bullish = (current_price > sma_200) and (sma_50 > sma_200)
        sma_bearish = (current_price < sma_200) and (sma_50 < sma_200)
        
        # Confirm trend direction with MACD
        macd_bullish = macd_line > 0
        macd_bearish = macd_line < 0
        
        # Combined trend direction (both indicators must agree)
        trend_bullish = sma_bullish and macd_bullish
        trend_bearish = sma_bearish and macd_bearish
        
        # Crash detection
        is_crash = (adx > self.ADX_CRASH_THRESHOLD) and \
                   (atr_percentage > self.ATR_PERCENTAGE_CRASH) and \
                   trend_bearish
        
        # Determine final regime
        if is_crash:
            regime = self.REGIME_CRASH_PANIC
        elif adx > self.ADX_STRONG_THRESHOLD and trend_bullish:
            regime = self.REGIME_STRONG_BULLISH
        elif adx > self.ADX_STRONG_THRESHOLD and trend_bearish:
            regime = self.REGIME_STRONG_BEARISH
        elif adx < self.ADX_NEUTRAL_THRESHOLD:
            regime = self.REGIME_NEUTRAL_RANGING
        else:
            # Transitional state (ADX between 20-25 or mixed signals)
            regime = self.REGIME_NEUTRAL_RANGING
        
        # Compile metrics (convert numpy types to Python native types for JSON serialization)
        regime_metrics = {
            'adx': float(round(adx, 2)),
            'atr_percentage': float(round(atr_percentage, 2)),
            'sma_50': float(round(sma_50, 2)),
            'sma_200': float(round(sma_200, 2)),
            'current_price': float(round(current_price, 2)),
            'macd_line': float(round(macd_line, 4)),
            'sma_alignment': 'bullish' if sma_bullish else ('bearish' if sma_bearish else 'neutral'),
            'macd_alignment': 'bullish' if macd_bullish else 'bearish',
            'trend_direction': 'bullish' if trend_bullish else ('bearish' if trend_bearish else 'neutral'),
            'is_crash': bool(is_crash)
        }
        
        return regime, regime_metrics
    
    def get_regime_history(self, lookback_periods: int = 100) -> pd.DataFrame:
        """
        Calculate regime history for backtesting and analysis.
        
        Args:
            lookback_periods: Number of periods to calculate (if None, use all data)
            
        Returns:
            DataFrame with regime classifications
        """
        results = []
        
        # Calculate starting index - if lookback_periods is equal to data length, start from beginning
        if lookback_periods >= len(self.data):
            start_idx = self.SMA_SLOW  # Start from when we have enough data for SMA 200
        else:
            start_idx = max(self.SMA_SLOW, len(self.data) - lookback_periods)
        
        for i in range(start_idx, len(self.data)):
                
            # Get values at index i
            price = self.data['close'].iloc[i]
            adx = self.indicators['ADX'].iloc[i]
            sma_50 = self.indicators['SMA_50'].iloc[i]
            sma_200 = self.indicators['SMA_200'].iloc[i]
            macd_line = self.indicators['MACD'].iloc[i]
            atr_percentage = self.indicators['ATR_PERCENTAGE'].iloc[i]
            
            if pd.isna(adx) or pd.isna(sma_200) or pd.isna(macd_line):
                continue
            
            # Apply detection logic
            sma_bullish = (price > sma_200) and (sma_50 > sma_200)
            sma_bearish = (price < sma_200) and (sma_50 < sma_200)
            macd_bullish = macd_line > 0
            macd_bearish = macd_line < 0
            
            trend_bullish = sma_bullish and macd_bullish
            trend_bearish = sma_bearish and macd_bearish
            
            is_crash = (adx > self.ADX_CRASH_THRESHOLD) and \
                      (atr_percentage > self.ATR_PERCENTAGE_CRASH) and \
                      trend_bearish
            
            if is_crash:
                regime = self.REGIME_CRASH_PANIC
            elif adx > self.ADX_STRONG_THRESHOLD and trend_bullish:
                regime = self.REGIME_STRONG_BULLISH
            elif adx > self.ADX_STRONG_THRESHOLD and trend_bearish:
                regime = self.REGIME_STRONG_BEARISH
            elif adx < self.ADX_NEUTRAL_THRESHOLD:
                regime = self.REGIME_NEUTRAL_RANGING
            else:
                regime = self.REGIME_NEUTRAL_RANGING
            
            results.append({
                'timestamp': self.data.index[i],
                'regime': regime,
                'adx': adx,
                'atr_percentage': atr_percentage,
                'price': price
            })
        
        return pd.DataFrame(results).set_index('timestamp')
