"""
Ichimoku Cloud Breakout Strategy - Executable Implementation

This module provides an executable implementation of the Ichimoku Cloud
Breakout strategy that conforms to the TradingStrategy interface.

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


class IchimokuCloudBreakout(TradingStrategy):
    """
    Executable Ichimoku Cloud Breakout Strategy.
    
    This strategy uses the Ichimoku Cloud system to identify trend direction,
    support/resistance levels, and generate trading signals based on cloud
    breakouts and line crossovers.
    """
    
    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize the Ichimoku Cloud strategy.
        
        Args:
            data: OHLCV DataFrame
            config: Optional configuration with keys:
                - tenkan_period: Conversion line period (default: 9)
                - kijun_period: Base line period (default: 26)
                - senkou_b_period: Leading Span B period (default: 52)
                - displacement: Cloud displacement (default: 26)
                - chikou_period: Lagging span period (default: 26)
        """
        super().__init__(data, config)
        self.name = "Ichimoku Cloud Breakout"
        self.version = "1.0.0"
        
        # Strategy parameters
        self.tenkan_period = self.config.get('tenkan_period', 9)
        self.kijun_period = self.config.get('kijun_period', 26)
        self.senkou_b_period = self.config.get('senkou_b_period', 52)
        self.displacement = self.config.get('displacement', 26)
        self.chikou_period = self.config.get('chikou_period', 26)
        
        # Initialize indicators
        self.indicators = {}
        if self.has_sufficient_data():
            self._calculate_indicators()
    
    def _calculate_indicators(self) -> None:
        """Calculate Ichimoku Cloud indicators."""
        try:
            # Calculate Ichimoku using pandas_ta
            ichimoku_result = ta.ichimoku(
                self.data['high'],
                self.data['low'],
                self.data['close'],
                tenkan=self.tenkan_period,
                kijun=self.kijun_period,
                senkou=self.senkou_b_period
            )
            
            # Extract components (pandas_ta returns a tuple of dataframes)
            if isinstance(ichimoku_result, tuple):
                # Combine all dataframes
                ichimoku_df = pd.concat(ichimoku_result, axis=1)
            else:
                ichimoku_df = ichimoku_result
            
            # Map to our naming convention
            self.indicators['tenkan'] = ichimoku_df.iloc[:, 0]  # Conversion Line
            self.indicators['kijun'] = ichimoku_df.iloc[:, 1]   # Base Line
            self.indicators['senkou_a'] = ichimoku_df.iloc[:, 2]  # Leading Span A
            self.indicators['senkou_b'] = ichimoku_df.iloc[:, 3]  # Leading Span B
            self.indicators['chikou'] = self.data['close'].shift(-self.chikou_period)  # Lagging Span
            
            # Calculate cloud thickness and color
            self.indicators['cloud_thickness'] = abs(
                self.indicators['senkou_a'] - self.indicators['senkou_b']
            )
            self.indicators['cloud_color'] = np.where(
                self.indicators['senkou_a'] > self.indicators['senkou_b'],
                'green',  # Bullish cloud
                'red'     # Bearish cloud
            )
            
            # Price position relative to cloud
            current_price = self.data['close']
            self.indicators['price_vs_cloud'] = np.where(
                current_price > self.indicators['senkou_a'].shift(self.displacement),
                np.where(
                    current_price > self.indicators['senkou_b'].shift(self.displacement),
                    'above',  # Price above cloud
                    'inside'  # Price inside cloud
                ),
                np.where(
                    current_price < self.indicators['senkou_b'].shift(self.displacement),
                    'below',  # Price below cloud
                    'inside'  # Price inside cloud
                )
            )
            
            # TK Cross detection
            self.indicators['tk_cross'] = self._detect_tk_cross()
            
            # Future cloud sentiment
            self.indicators['future_cloud'] = self._analyze_future_cloud()
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating indicators: {e}")
            self.indicators = {}
    
    def _detect_tk_cross(self) -> pd.Series:
        """Detect Tenkan-Kijun crossovers."""
        tk_cross = pd.Series(0, index=self.data.index)
        
        tenkan = self.indicators['tenkan']
        kijun = self.indicators['kijun']
        
        # Bullish cross: Tenkan crosses above Kijun
        bullish = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
        tk_cross[bullish] = 1
        
        # Bearish cross: Tenkan crosses below Kijun
        bearish = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
        tk_cross[bearish] = -1
        
        return tk_cross
    
    def _analyze_future_cloud(self) -> pd.Series:
        """Analyze future cloud for trend prediction."""
        future_cloud = pd.Series('neutral', index=self.data.index)
        
        # Look at cloud 26 periods ahead
        future_senkou_a = self.indicators['senkou_a']
        future_senkou_b = self.indicators['senkou_b']
        
        # Future cloud sentiment
        future_cloud = np.where(
            future_senkou_a > future_senkou_b,
            'bullish',
            'bearish'
        )
        
        return pd.Series(future_cloud, index=self.data.index)
    
    def calculate_signal(self) -> Dict[str, float]:
        """
        Calculate trading signal based on Ichimoku Cloud strategy.
        
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
            tenkan = self.indicators['tenkan'].iloc[idx]
            kijun = self.indicators['kijun'].iloc[idx]
            
            # Get cloud values (displaced)
            cloud_idx = idx - self.displacement if idx - self.displacement >= 0 else 0
            senkou_a = self.indicators['senkou_a'].iloc[cloud_idx]
            senkou_b = self.indicators['senkou_b'].iloc[cloud_idx]
            
            cloud_top = max(senkou_a, senkou_b)
            cloud_bottom = min(senkou_a, senkou_b)
            cloud_color = self.indicators['cloud_color'][cloud_idx]
            price_vs_cloud = self.indicators['price_vs_cloud'].iloc[idx]
            tk_cross = self.indicators['tk_cross'].iloc[idx]
            cloud_thickness = self.indicators['cloud_thickness'].iloc[cloud_idx]
            future_cloud = self.indicators['future_cloud'].iloc[idx]
            
            # Check for NaN values
            if pd.isna(tenkan) or pd.isna(kijun) or pd.isna(senkou_a):
                return {'signal': 0, 'reason': 'Indicators not ready'}
            
            # Initialize signal
            signal = 0.0
            confidence = 0.0
            reason = "No signal"
            
            # STRONG BUY SIGNALS
            if (price_vs_cloud == 'above' and 
                cloud_color == 'green' and 
                tk_cross == 1 and
                tenkan > kijun):
                
                signal = 1.0
                confidence = 0.9
                reason = "Price above green cloud with bullish TK cross"
                
                # Boost confidence for thick cloud (strong support)
                if cloud_thickness > self.data['close'].iloc[idx] * 0.02:  # 2% thickness
                    confidence = min(confidence + 0.1, 1.0)
                    reason += " + strong cloud support"
            
            # MODERATE BUY SIGNALS
            elif (price_vs_cloud == 'above' and 
                  tenkan > kijun and 
                  current_close > tenkan):
                
                signal = 0.7
                confidence = 0.7
                reason = "Price above cloud with bullish alignment"
                
                # Adjust based on future cloud
                if future_cloud == 'bullish':
                    confidence = min(confidence + 0.1, 0.9)
                    reason += " + bullish future cloud"
            
            # CLOUD BREAKOUT BUY
            elif (self.indicators['price_vs_cloud'].iloc[idx-1] == 'inside' and
                  price_vs_cloud == 'above' and
                  cloud_color == 'green'):
                
                signal = 0.8
                confidence = 0.8
                reason = "Bullish cloud breakout"
            
            # STRONG SELL SIGNALS
            elif (price_vs_cloud == 'below' and 
                  cloud_color == 'red' and 
                  tk_cross == -1 and
                  tenkan < kijun):
                
                signal = -1.0
                confidence = 0.9
                reason = "Price below red cloud with bearish TK cross"
                
                # Boost confidence for thick cloud (strong resistance)
                if cloud_thickness > self.data['close'].iloc[idx] * 0.02:
                    confidence = min(confidence + 0.1, 1.0)
                    reason += " + strong cloud resistance"
            
            # MODERATE SELL SIGNALS
            elif (price_vs_cloud == 'below' and 
                  tenkan < kijun and 
                  current_close < tenkan):
                
                signal = -0.7
                confidence = 0.7
                reason = "Price below cloud with bearish alignment"
                
                # Adjust based on future cloud
                if future_cloud == 'bearish':
                    confidence = min(confidence + 0.1, 0.9)
                    reason += " + bearish future cloud"
            
            # CLOUD BREAKOUT SELL
            elif (self.indicators['price_vs_cloud'].iloc[idx-1] == 'inside' and
                  price_vs_cloud == 'below' and
                  cloud_color == 'red'):
                
                signal = -0.8
                confidence = 0.8
                reason = "Bearish cloud breakout"
            
            # NEUTRAL/INSIDE CLOUD
            elif price_vs_cloud == 'inside':
                signal = 0.0
                confidence = 0.3
                reason = "Price inside cloud - no clear direction"
            
            return {
                'signal': round(signal, 3),
                'confidence': round(confidence, 3),
                'reason': reason,
                'tenkan': round(tenkan, 2),
                'kijun': round(kijun, 2),
                'cloud_top': round(cloud_top, 2),
                'cloud_bottom': round(cloud_bottom, 2),
                'cloud_color': cloud_color,
                'price_vs_cloud': price_vs_cloud,
                'cloud_thickness': round(cloud_thickness, 2),
                'future_cloud': future_cloud,
                'price': round(current_close, 2)
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error calculating signal: {e}")
            return {'signal': 0, 'error': str(e)}
    
    def get_required_data_points(self) -> int:
        """
        Get minimum required data points.
        
        Returns:
            Maximum period plus displacement
        """
        return self.senkou_b_period + self.displacement + 10
    
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        Determine if strategy is suitable for current market regime.
        
        Ichimoku works well in trending markets.
        
        Args:
            market_bias: Current market regime
            
        Returns:
            True if market is suitable for trend following
        """
        # Strategy works best in trending markets
        suitable_regimes = ['Strong Bullish', 'Bullish', 'Bearish', 'Strong Bearish']
        # Can work in neutral with clear cloud breaks
        moderately_suitable = ['Neutral']
        
        if market_bias in suitable_regimes:
            return True
        elif market_bias in moderately_suitable:
            return True  # Reduced effectiveness
        else:
            # Not suitable for ranging or crash conditions
            return False
