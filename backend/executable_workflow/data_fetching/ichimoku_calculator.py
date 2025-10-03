Ichimoku Cloud Calculator

This module provides a comprehensive calculator for the Ichimoku Cloud (Ichimoku Kinko Hyo) indicator,
a versatile technical analysis tool that provides support/resistance levels, trend direction,
momentum readings, and trading signals.

Components:
- Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
- Kijun-sen (Base Line): (26-period high + 26-period low) / 2
- Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods forward
- Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods forward
- Chikou Span (Lagging Span): Close price shifted 26 periods backward
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)


class IchimokuCalculator:
    """
    A comprehensive calculator for Ichimoku Cloud indicators.

    This class provides methods to calculate all Ichimoku components with
    configurable periods and proper time shifting for leading/lagging spans.

    The Ichimoku Cloud is unique in that it projects support/resistance levels
    into the future and provides multiple confirmation signals for trend analysis.
    """
    
    def __init__(self):
        """Initialize the Ichimoku Calculator."""
        self.component_names = {
            'tenkan': 'tenkan_sen',
            'kijun': 'kijun_sen',
            'senkou_a': 'senkou_span_a',
            'senkou_b': 'senkou_span_b',
            'chikou': 'chikou_span'
        }
    
    def calculate_ichimoku(self, 
                          df: pd.DataFrame, 
                          tenkan_period: int = 9,
                          kijun_period: int = 26,
                          senkou_b_period: int = 52,
                          chikou_offset: int = 26,
                          senkou_offset: int = 26) -> pd.DataFrame:
        """
        Calculate all Ichimoku Cloud components.

        This method uses the Donchian channel approach (high/low midpoints)
        which is the traditional Ichimoku calculation method.

        Args:
            df: DataFrame with OHLCV data (must have 'high', 'low', 'close' columns)
            tenkan_period: Period for Tenkan-sen (Conversion Line), default 9
            kijun_period: Period for Kijun-sen (Base Line), default 26
            senkou_b_period: Period for Senkou Span B, default 52
            chikou_offset: Offset for Chikou Span (backward), default 26
            senkou_offset: Forward offset for Senkou spans, default 26

        Returns:
            DataFrame with original data plus Ichimoku components:
            - tenkan_sen: Conversion Line
            - kijun_sen: Base Line
            - senkou_span_a: Leading Span A (shifted forward)
            - senkou_span_b: Leading Span B (shifted forward)
            - chikou_span: Lagging Span (shifted backward)
            - cloud_top: Maximum of Senkou A and B
            - cloud_bottom: Minimum of Senkou A and B
            - cloud_thickness: Absolute difference between Senkou spans
            - SpanAaboveSpanB: Boolean indicating if Senkou Span A > Senkou Span B
            - SpanAbelowSpanB: Boolean indicating if Senkou Span A < Senkou Span B

        Raises:
            ValueError: If required columns are missing or parameters are invalid
        """
        # Validate inputs
        required_columns = ['high', 'low', 'close']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if tenkan_period <= 0 or kijun_period <= 0 or senkou_b_period <= 0:
            raise ValueError("All periods must be positive integers")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate Tenkan-sen (Conversion Line)
        # (Highest high + Lowest low) / 2 over tenkan_period
        high_tenkan = result_df['high'].rolling(window=tenkan_period, min_periods=1).max()
        low_tenkan = result_df['low'].rolling(window=tenkan_period, min_periods=1).min()
        result_df['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
        
        # Calculate Kijun-sen (Base Line)
        # (Highest high + Lowest low) / 2 over kijun_period
        high_kijun = result_df['high'].rolling(window=kijun_period, min_periods=1).max()
        low_kijun = result_df['low'].rolling(window=kijun_period, min_periods=1).min()
        result_df['kijun_sen'] = (high_kijun + low_kijun) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        # (Tenkan + Kijun) / 2, shifted forward by senkou_offset periods
        senkou_a_raw = (result_df['tenkan_sen'] + result_df['kijun_sen']) / 2
        result_df['senkou_span_a'] = senkou_a_raw.shift(senkou_offset)
        
        # Calculate Senkou Span B (Leading Span B)
        # (Highest high + Lowest low) / 2 over senkou_b_period, shifted forward
        high_senkou = result_df['high'].rolling(window=senkou_b_period, min_periods=1).max()
        low_senkou = result_df['low'].rolling(window=senkou_b_period, min_periods=1).min()
        senkou_b_raw = (high_senkou + low_senkou) / 2
        result_df['senkou_span_b'] = senkou_b_raw.shift(senkou_offset)
        
        # Calculate Chikou Span (Lagging Span)
        # Close price shifted backward by chikou_offset periods
        result_df['chikou_span'] = result_df['close'].shift(-chikou_offset)
        
        # Calculate cloud boundaries
        result_df['cloud_top'] = result_df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        result_df['cloud_bottom'] = result_df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        
        # Calculate cloud thickness (absolute difference)
        result_df['cloud_thickness'] = abs(result_df['senkou_span_a'] - result_df['senkou_span_b'])
        
        # Add cloud color for visualization (optional)
        result_df['cloud_color'] = np.where(
            result_df['senkou_span_a'] >= result_df['senkou_span_b'],
            'green',  # Bullish cloud
            'red'     # Bearish cloud
        )
        
        # Add SpanAaboveSpanB and SpanAbelowSpanB variables for strategy
        result_df['SpanAaboveSpanB'] = result_df['senkou_span_a'] > result_df['senkou_span_b']
        result_df['SpanAbelowSpanB'] = result_df['senkou_span_a'] < result_df['senkou_span_b']
        
        logger.info(f"Calculated Ichimoku indicators for {len(result_df)} data points")
        
        return result_df
    
    def get_cloud_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze Ichimoku Cloud to generate trading signals.

        This method evaluates multiple Ichimoku conditions to determine
        the overall market sentiment and potential trading opportunities.

        Args:
            df: DataFrame with Ichimoku indicators (from calculate_ichimoku)

        Returns:
            DataFrame with additional signal columns:
            - price_vs_cloud: Position relative to cloud ('above', 'inside', 'below')
            - tk_cross: Tenkan-Kijun cross signals (1: bullish, -1: bearish, 0: none)
            - cloud_breakout: Cloud breakout signals (1: bullish, -1: bearish, 0: none)
            - ichimoku_signal: Overall signal strength (-1 to 1)
            - signal_strength: Categorical strength ('strong', 'moderate', 'weak', 'neutral')
            - signal_description: Human-readable signal description
        """
        if not self._has_ichimoku_columns(df):
            raise ValueError("DataFrame must contain Ichimoku indicators. Run calculate_ichimoku first.")
        
        # Create a copy to avoid modifying the original
        signal_df = df.copy()
        
        # Determine price position relative to cloud
        signal_df['price_vs_cloud'] = self._calculate_price_vs_cloud(signal_df)
        
        # Detect Tenkan-Kijun crossovers
        signal_df['tk_cross'] = self._detect_tk_crossovers(signal_df)
        
        # Detect cloud breakouts
        signal_df['cloud_breakout'] = self._detect_cloud_breakouts(signal_df)
        
        # Calculate Chikou span confirmation
        signal_df['chikou_confirmation'] = self._calculate_chikou_confirmation(signal_df)
        
        # Calculate composite Ichimoku signal
        signal_df['ichimoku_signal'] = self._calculate_composite_signal(signal_df)
        
        # Categorize signal strength
        signal_df['signal_strength'] = self._categorize_signal_strength(signal_df['ichimoku_signal'])
        
        # Add human-readable descriptions
        signal_df['signal_description'] = self._generate_signal_descriptions(signal_df)
        
        return signal_df
    
    def _has_ichimoku_columns(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has required Ichimoku columns."""
        required = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
        return all(col in df.columns for col in required)
    
    def _calculate_price_vs_cloud(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price position relative to the cloud."""
        conditions = [
            df['close'] > df['cloud_top'],
            df['close'] < df['cloud_bottom'],
            (df['close'] >= df['cloud_bottom']) & (df['close'] <= df['cloud_top'])
        ]
        choices = ['above', 'below', 'inside']
        return pd.Series(np.select(conditions, choices, default='inside'), index=df.index)
    
    def _detect_tk_crossovers(self, df: pd.DataFrame) -> pd.Series:
        """Detect Tenkan-Kijun crossovers."""
        tk_cross = pd.Series(0, index=df.index)
        
        # Bullish cross: Tenkan crosses above Kijun
        bullish_cross = (
            (df['tenkan_sen'] > df['kijun_sen']) & 
            (df['tenkan_sen'].shift(1) <= df['kijun_sen'].shift(1))
        )
        tk_cross[bullish_cross] = 1
        
        # Bearish cross: Tenkan crosses below Kijun
        bearish_cross = (
            (df['tenkan_sen'] < df['kijun_sen']) & 
            (df['tenkan_sen'].shift(1) >= df['kijun_sen'].shift(1))
        )
        tk_cross[bearish_cross] = -1
        
        return tk_cross
    
    def _detect_cloud_breakouts(self, df: pd.DataFrame) -> pd.Series:
        """Detect price breakouts through the cloud."""
        breakout = pd.Series(0, index=df.index)
        
        # Shift price_vs_cloud for comparison
        prev_position = df['price_vs_cloud'].shift(1)
        curr_position = df['price_vs_cloud']
        
        # Bullish breakout: Price moves from inside/below to above cloud
        bullish_breakout = (
            (curr_position == 'above') & 
            (prev_position.isin(['inside', 'below']))
        )
        breakout[bullish_breakout] = 1
        
        # Bearish breakout: Price moves from inside/above to below cloud
        bearish_breakout = (
            (curr_position == 'below') & 
            (prev_position.isin(['inside', 'above']))
        )
        breakout[bearish_breakout] = -1
        
        return breakout
    
    def _calculate_chikou_confirmation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Chikou span confirmation signals."""
        # Chikou span is already shifted back, so we compare with historical price
        # For current index i, chikou[i] represents close[i] shifted back
        # We need to compare with price from chikou_offset periods ago
        
        chikou_confirm = pd.Series(0, index=df.index)
        
        # We can only calculate this after we have enough history
        chikou_offset = 26  # Standard offset
        
        for i in range(chikou_offset, len(df)):
            if pd.notna(df['chikou_span'].iloc[i]):
                # Chikou above historical price is bullish
                if df['chikou_span'].iloc[i] > df['close'].iloc[i - chikou_offset]:
                    chikou_confirm.iloc[i] = 1
                # Chikou below historical price is bearish
                elif df['chikou_span'].iloc[i] < df['close'].iloc[i - chikou_offset]:
                    chikou_confirm.iloc[i] = -1
        
        return chikou_confirm
    
    def _calculate_composite_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate composite Ichimoku signal based on multiple factors.

        Signal strength ranges from -1 (strong bearish) to 1 (strong bullish).
        """
        signal = pd.Series(0.0, index=df.index)
        
        for i in range(len(df)):
            score = 0.0
            weight_sum = 0.0
            
            # Price vs Cloud (weight: 0.3)
            if df['price_vs_cloud'].iloc[i] == 'above':
                score += 0.3
                weight_sum += 0.3
            elif df['price_vs_cloud'].iloc[i] == 'below':
                score -= 0.3
                weight_sum += 0.3
            else:  # inside
                weight_sum += 0.3  # No score but count the weight
            
            # SpanAaboveSpanB condition (weight: 0.2)
            if df['SpanAaboveSpanB'].iloc[i]:
                score += 0.2
                weight_sum += 0.2
            else:  # SpanAbelowSpanB
                score -= 0.2
                weight_sum += 0.2
            
            # Tenkan vs Kijun (weight: 0.2)
            if df['tenkan_sen'].iloc[i] > df['kijun_sen'].iloc[i]:
                score += 0.2
                weight_sum += 0.2
            else:
                score -= 0.2
                weight_sum += 0.2
            
            # TK Cross (weight: 0.15)
            if df['tk_cross'].iloc[i] != 0:
                score += df['tk_cross'].iloc[i] * 0.15
                weight_sum += 0.15
            
            # Cloud breakout (weight: 0.15)
            if df['cloud_breakout'].iloc[i] != 0:
                score += df['cloud_breakout'].iloc[i] * 0.15
                weight_sum += 0.15
            
            # Normalize to [-1, 1] range
            if weight_sum > 0:
                signal.iloc[i] = score / weight_sum
            
        return signal
    
    def _categorize_signal_strength(self, signal: pd.Series) -> pd.Series:
        """Categorize signal strength into descriptive categories."""
        conditions = [
            signal >= 0.7,
            (signal >= 0.3) & (signal < 0.7),
            (signal > -0.3) & (signal < 0.3),
            (signal <= -0.3) & (signal > -0.7),
            signal <= -0.7
        ]
        choices = ['strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish']
        return pd.Series(np.select(conditions, choices, default='neutral'), index=signal.index)
    
    def _generate_signal_descriptions(self, df: pd.DataFrame) -> pd.Series:
        """Generate human-readable signal descriptions."""
        descriptions = []
        
        for i in range(len(df)):
            desc_parts = []
            
            # Price position
            pos = df['price_vs_cloud'].iloc[i]
            desc_parts.append(f"Price {pos} cloud")
            
            # Span A vs Span B
            if df['SpanAaboveSpanB'].iloc[i]:
                desc_parts.append("Span A > Span B")
            else:
                desc_parts.append("Span A < Span B")
            
            # TK relationship
            if df['tenkan_sen'].iloc[i] > df['kijun_sen'].iloc[i]:
                desc_parts.append("Tenkan > Kijun")
            else:
                desc_parts.append("Tenkan < Kijun")
            
            # Special events
            if df['tk_cross'].iloc[i] == 1:
                desc_parts.append("Bullish TK cross")
            elif df['tk_cross'].iloc[i] == -1:
                desc_parts.append("Bearish TK cross")
            
            if df['cloud_breakout'].iloc[i] == 1:
                desc_parts.append("Bullish cloud breakout")
            elif df['cloud_breakout'].iloc[i] == -1:
                desc_parts.append("Bearish cloud breakout")
            
            descriptions.append("; ".join(desc_parts))
        
        return pd.Series(descriptions, index=df.index)
    
    def get_current_analysis(self, df: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """
        Get comprehensive analysis of the current Ichimoku state.

        Args:
            df: DataFrame with Ichimoku indicators and signals

        Returns:
            Dictionary with current analysis including:
            - All indicator values
            - Signal strength and description
            - Key levels and recommendations
        """
        if len(df) == 0:
            return {}
        
        # Get the latest row
        latest = df.iloc[-1]
        
        analysis = {
            # Current indicator values
            'close': float(latest['close']) if pd.notna(latest['close']) else None,
            'tenkan_sen': float(latest['tenkan_sen']) if pd.notna(latest['tenkan_sen']) else None,
            'kijun_sen': float(latest['kijun_sen']) if pd.notna(latest['kijun_sen']) else None,
            'senkou_span_a': float(latest['senkou_span_a']) if pd.notna(latest['senkou_span_a']) else None,
            'senkou_span_b': float(latest['senkou_span_b']) if pd.notna(latest['senkou_span_b']) else None,
            'chikou_span': float(latest['chikou_span']) if pd.notna(latest['chikou_span']) else None,
            
            # Cloud metrics
            'cloud_top': float(latest['cloud_top']) if pd.notna(latest['cloud_top']) else None,
            'cloud_bottom': float(latest['cloud_bottom']) if pd.notna(latest['cloud_bottom']) else None,
            'cloud_thickness': float(latest['cloud_thickness']) if pd.notna(latest['cloud_thickness']) else None,
            'cloud_color': latest['cloud_color'] if 'cloud_color' in latest else None,
            
            # Span A vs Span B conditions
            'SpanAaboveSpanB': bool(latest.get('SpanAaboveSpanB', False)),
            'SpanAbelowSpanB': bool(latest.get('SpanAbelowSpanB', False)),
            
            # Signals
            'price_vs_cloud': latest.get('price_vs_cloud', 'unknown'),
            'ichimoku_signal': float(latest.get('ichimoku_signal', 0)),
            'signal_strength': latest.get('signal_strength', 'neutral'),
            'signal_description': latest.get('signal_description', 'No signal'),
            
            # Key support/resistance levels
            'immediate_support': float(latest['kijun_sen']) if pd.notna(latest['kijun_sen']) else None,
            'immediate_resistance': float(latest['tenkan_sen']) if pd.notna(latest['tenkan_sen']) else None,
            'major_support': float(latest['cloud_top']) if latest.get('price_vs_cloud') == 'above' else float(latest['cloud_bottom']),
            'major_resistance': float(latest['cloud_bottom']) if latest.get('price_vs_cloud') == 'below' else float(latest['cloud_top']),
        }
        
        # Add trend strength assessment
        if analysis['cloud_thickness'] and analysis['close']:
            thickness_pct = (analysis['cloud_thickness'] / analysis['close']) * 100
            if thickness_pct > 5:
                analysis['trend_strength'] = 'strong'
            elif thickness_pct > 2:
                analysis['trend_strength'] = 'moderate'
            else:
                analysis['trend_strength'] = 'weak'
        
        return analysis


# Example usage
if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='1h')
    
    # Generate realistic OHLCV data with trend
    close_prices = 50000 + np.cumsum(np.random.normal(0, 100, len(dates)))
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'close': close_prices,
        'high': close_prices + np.random.uniform(0, 200, len(dates)),
        'low': close_prices - np.random.uniform(0, 200, len(dates)),
        'open': close_prices + np.random.uniform(-100, 100, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    })
    
    # Fix OHLC relationships
    sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)
    
    # Initialize calculator
    calculator = IchimokuCalculator()
    
    # Calculate Ichimoku indicators
    print("Calculating Ichimoku Cloud indicators...")
    ichimoku_data = calculator.calculate_ichimoku(
        sample_data,
        tenkan_period=9,
        kijun_period=26,
        senkou_b_period=52
    )
    
    # Generate signals
    print("\nGenerating trading signals...")
    signal_data = calculator.get_cloud_signals(ichimoku_data)
    
    # Display last 5 rows
    print("\nLast 5 data points with Ichimoku indicators:")
    display_columns = ['close', 'tenkan_sen', 'kijun_sen', 'SpanAaboveSpanB', 'SpanAbelowSpanB',
                      'price_vs_cloud', 'signal_strength']
    print(signal_data[display_columns].tail())
    
    # Get current analysis
    print("\nCurrent Ichimoku Analysis:")
    analysis = calculator.get_current_analysis(signal_data)
    for key, value in analysis.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    # Show signal statistics
    print("\nSignal Distribution:")
    print(signal_data['signal_strength'].value_counts())
    
    # Find strong signals
    strong_signals = signal_data[
        signal_data['signal_strength'].isin(['strong_bullish', 'strong_bearish'])
    ]
    print(f"\nFound {len(strong_signals)} strong signals out of {len(signal_data)} data points")