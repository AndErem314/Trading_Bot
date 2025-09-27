"""
Ichimoku Signal Detection Engine

This module provides a comprehensive signal detection system for Ichimoku Cloud patterns.
It detects various signal characteristics and combinations to identify trading opportunities
based on the relationships between price, cloud components, and Ichimoku lines.

Signal Characteristics:
- Price vs Cloud: Above/Below cloud positions
- Tenkan vs Kijun: Conversion/Base line relationships
- Chikou vs Price: Lagging span vs historical price
- Chikou vs Cloud: Lagging span vs historical cloud
- Chikou Crosses: Crossovers with Senkou Span B

All signals use closed bars only to avoid false positives from incomplete candles.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Enumeration of available Ichimoku signal types."""
    PRICE_ABOVE_CLOUD = "PriceAboveCloud"
    PRICE_BELOW_CLOUD = "PriceBelowCloud"
    TENKAN_ABOVE_KIJUN = "TenkanAboveKijun"
    TENKAN_BELOW_KIJUN = "TenkanBelowKijun"
    CHIKOU_ABOVE_PRICE = "ChikouAbovePrice"
    CHIKOU_BELOW_PRICE = "ChikouBelowPrice"
    CHIKOU_ABOVE_CLOUD = "ChikouAboveCloud"
    CHIKOU_BELOW_CLOUD = "ChikouBelowCloud"
    CHIKOU_CROSS_ABOVE_SENKOU_B = "ChikouCrossAboveSenkouB"
    CHIKOU_CROSS_BELOW_SENKOU_B = "ChikouCrossBelowSenkouB"


@dataclass
class SignalStrength:
    """Data class for signal strength assessment."""
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    active_signals: int
    total_signals: int
    description: str


class IchimokuSignalDetector:
    """
    Comprehensive signal detection engine for Ichimoku Cloud patterns.
    
    This class detects various signal characteristics based on the relationships
    between price, Ichimoku lines, and cloud components. All detections use
    closed bars only to ensure signal reliability.
    
    Attributes:
        chikou_offset (int): Standard offset for Chikou span (default 26)
        senkou_offset (int): Standard offset for Senkou spans (default 26)
    """
    
    def __init__(self, chikou_offset: int = 26, senkou_offset: int = 26):
        """
        Initialize the signal detector.
        
        Args:
            chikou_offset: Backward offset for Chikou span
            senkou_offset: Forward offset for Senkou spans
        """
        self.chikou_offset = chikou_offset
        self.senkou_offset = senkou_offset
        self.signal_columns = []
    
    def detect_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all Ichimoku signal characteristics.
        
        This method calculates boolean signals for each characteristic,
        using only completed bars (ignoring the current incomplete bar).
        
        Args:
            df: DataFrame with Ichimoku indicators (from IchimokuCalculator)
            
        Returns:
            DataFrame with additional boolean signal columns:
            - price_above_cloud: True when close > max(senkou_a, senkou_b)
            - price_below_cloud: True when close < min(senkou_a, senkou_b)
            - tenkan_above_kijun: True when tenkan > kijun
            - tenkan_below_kijun: True when tenkan < kijun
            - chikou_above_price: True when chikou > historical price
            - chikou_below_price: True when chikou < historical price
            - chikou_above_cloud: True when chikou > historical cloud top
            - chikou_below_cloud: True when chikou < historical cloud bottom
            - chikou_cross_above_senkou_b: True when chikou crosses above senkou_b
            - chikou_cross_below_senkou_b: True when chikou crosses below senkou_b
            
        Raises:
            ValueError: If required Ichimoku columns are missing
        """
        # Validate required columns
        required_columns = ['close', 'tenkan_sen', 'kijun_sen', 
                          'senkou_span_a', 'senkou_span_b', 'chikou_span']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create a copy to avoid modifying original
        signal_df = df.copy()
        
        # Use all rows except the last one (current incomplete bar)
        # This ensures we only work with closed bars
        closed_bars_mask = pd.Series(True, index=signal_df.index)
        if len(signal_df) > 0:
            closed_bars_mask.iloc[-1] = False
        
        # 1. Price vs Cloud signals
        signal_df['price_above_cloud'] = self._detect_price_above_cloud(signal_df, closed_bars_mask)
        signal_df['price_below_cloud'] = self._detect_price_below_cloud(signal_df, closed_bars_mask)
        
        # 2. Tenkan vs Kijun signals
        signal_df['tenkan_above_kijun'] = self._detect_tenkan_above_kijun(signal_df, closed_bars_mask)
        signal_df['tenkan_below_kijun'] = self._detect_tenkan_below_kijun(signal_df, closed_bars_mask)
        
        # 3. Chikou vs Price signals
        signal_df['chikou_above_price'] = self._detect_chikou_above_price(signal_df, closed_bars_mask)
        signal_df['chikou_below_price'] = self._detect_chikou_below_price(signal_df, closed_bars_mask)
        
        # 4. Chikou vs Cloud signals
        signal_df['chikou_above_cloud'] = self._detect_chikou_above_cloud(signal_df, closed_bars_mask)
        signal_df['chikou_below_cloud'] = self._detect_chikou_below_cloud(signal_df, closed_bars_mask)
        
        # 5. Chikou cross signals
        signal_df['chikou_cross_above_senkou_b'] = self._detect_chikou_cross_above_senkou_b(signal_df, closed_bars_mask)
        signal_df['chikou_cross_below_senkou_b'] = self._detect_chikou_cross_below_senkou_b(signal_df, closed_bars_mask)
        
        # Store signal column names
        self.signal_columns = [
            'price_above_cloud', 'price_below_cloud',
            'tenkan_above_kijun', 'tenkan_below_kijun',
            'chikou_above_price', 'chikou_below_price',
            'chikou_above_cloud', 'chikou_below_cloud',
            'chikou_cross_above_senkou_b', 'chikou_cross_below_senkou_b'
        ]
        
        logger.info(f"Detected all signals for {len(signal_df)} data points")
        
        return signal_df
    
    def _detect_price_above_cloud(self, df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Detect when price is above the cloud."""
        signal = pd.Series(False, index=df.index)
        
        # Cloud spans are already shifted forward by senkou_offset
        # So we compare current price with shifted cloud values
        cloud_top = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        signal[mask] = (df['close'] > cloud_top)[mask]
        
        return signal
    
    def _detect_price_below_cloud(self, df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Detect when price is below the cloud."""
        signal = pd.Series(False, index=df.index)
        
        cloud_bottom = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        signal[mask] = (df['close'] < cloud_bottom)[mask]
        
        return signal
    
    def _detect_tenkan_above_kijun(self, df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Detect when Tenkan is above Kijun."""
        signal = pd.Series(False, index=df.index)
        signal[mask] = (df['tenkan_sen'] > df['kijun_sen'])[mask]
        return signal
    
    def _detect_tenkan_below_kijun(self, df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Detect when Tenkan is below Kijun."""
        signal = pd.Series(False, index=df.index)
        signal[mask] = (df['tenkan_sen'] < df['kijun_sen'])[mask]
        return signal
    
    def _detect_chikou_above_price(self, df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """
        Detect when Chikou is above historical price.
        
        Chikou span is the current close shifted back by chikou_offset.
        We need to compare it with the price from chikou_offset periods ago.
        """
        signal = pd.Series(False, index=df.index)
        
        # For each point, we need to check if chikou > historical price
        # Current chikou value represents close price shifted back
        for i in range(self.chikou_offset, len(df)):
            if mask.iloc[i] and pd.notna(df['chikou_span'].iloc[i]):
                # Chikou at position i compares with price at position (i - chikou_offset)
                historical_price = df['close'].iloc[i - self.chikou_offset]
                signal.iloc[i] = df['chikou_span'].iloc[i] > historical_price
        
        return signal
    
    def _detect_chikou_below_price(self, df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Detect when Chikou is below historical price."""
        signal = pd.Series(False, index=df.index)
        
        for i in range(self.chikou_offset, len(df)):
            if mask.iloc[i] and pd.notna(df['chikou_span'].iloc[i]):
                historical_price = df['close'].iloc[i - self.chikou_offset]
                signal.iloc[i] = df['chikou_span'].iloc[i] < historical_price
        
        return signal
    
    def _detect_chikou_above_cloud(self, df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """
        Detect when Chikou is above historical cloud.
        
        We need to compare Chikou with the cloud from chikou_offset periods ago,
        but also account for the senkou forward shift.
        """
        signal = pd.Series(False, index=df.index)
        
        for i in range(self.chikou_offset, len(df)):
            if mask.iloc[i] and pd.notna(df['chikou_span'].iloc[i]):
                # Historical cloud position
                hist_idx = i - self.chikou_offset
                
                # Get historical cloud values (they were already shifted forward)
                # So we need to look at the unshifted values from that time
                if hist_idx - self.senkou_offset >= 0:
                    # Calculate what the cloud was at the historical point
                    hist_senkou_a = df['senkou_span_a'].iloc[hist_idx]
                    hist_senkou_b = df['senkou_span_b'].iloc[hist_idx]
                    
                    if pd.notna(hist_senkou_a) and pd.notna(hist_senkou_b):
                        hist_cloud_top = max(hist_senkou_a, hist_senkou_b)
                        signal.iloc[i] = df['chikou_span'].iloc[i] > hist_cloud_top
        
        return signal
    
    def _detect_chikou_below_cloud(self, df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Detect when Chikou is below historical cloud."""
        signal = pd.Series(False, index=df.index)
        
        for i in range(self.chikou_offset, len(df)):
            if mask.iloc[i] and pd.notna(df['chikou_span'].iloc[i]):
                hist_idx = i - self.chikou_offset
                
                if hist_idx - self.senkou_offset >= 0:
                    hist_senkou_a = df['senkou_span_a'].iloc[hist_idx]
                    hist_senkou_b = df['senkou_span_b'].iloc[hist_idx]
                    
                    if pd.notna(hist_senkou_a) and pd.notna(hist_senkou_b):
                        hist_cloud_bottom = min(hist_senkou_a, hist_senkou_b)
                        signal.iloc[i] = df['chikou_span'].iloc[i] < hist_cloud_bottom
        
        return signal
    
    def _detect_chikou_cross_above_senkou_b(self, df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Detect when Chikou crosses above Senkou Span B."""
        signal = pd.Series(False, index=df.index)
        
        for i in range(self.chikou_offset + 1, len(df)):
            if mask.iloc[i] and pd.notna(df['chikou_span'].iloc[i]):
                hist_idx = i - self.chikou_offset
                
                if hist_idx > 0 and hist_idx - self.senkou_offset >= 0:
                    # Current and previous Chikou values
                    curr_chikou = df['chikou_span'].iloc[i]
                    prev_chikou = df['chikou_span'].iloc[i-1]
                    
                    # Historical Senkou B values
                    curr_senkou_b = df['senkou_span_b'].iloc[hist_idx]
                    prev_senkou_b = df['senkou_span_b'].iloc[hist_idx-1]
                    
                    if all(pd.notna([curr_chikou, prev_chikou, curr_senkou_b, prev_senkou_b])):
                        # Check for crossover
                        signal.iloc[i] = (prev_chikou <= prev_senkou_b) and (curr_chikou > curr_senkou_b)
        
        return signal
    
    def _detect_chikou_cross_below_senkou_b(self, df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Detect when Chikou crosses below Senkou Span B."""
        signal = pd.Series(False, index=df.index)
        
        for i in range(self.chikou_offset + 1, len(df)):
            if mask.iloc[i] and pd.notna(df['chikou_span'].iloc[i]):
                hist_idx = i - self.chikou_offset
                
                if hist_idx > 0 and hist_idx - self.senkou_offset >= 0:
                    curr_chikou = df['chikou_span'].iloc[i]
                    prev_chikou = df['chikou_span'].iloc[i-1]
                    
                    curr_senkou_b = df['senkou_span_b'].iloc[hist_idx]
                    prev_senkou_b = df['senkou_span_b'].iloc[hist_idx-1]
                    
                    if all(pd.notna([curr_chikou, prev_chikou, curr_senkou_b, prev_senkou_b])):
                        # Check for crossunder
                        signal.iloc[i] = (prev_chikou >= prev_senkou_b) and (curr_chikou < curr_senkou_b)
        
        return signal
    
    def get_signal_combination(self, df: pd.DataFrame, 
                             required_signals: List[Union[str, SignalType]]) -> pd.Series:
        """
        Get combined signal based on required signal characteristics.
        
        This method returns True only when ALL specified signals are active.
        
        Args:
            df: DataFrame with signal columns (from detect_all_signals)
            required_signals: List of signal types that must all be True
                             Can be SignalType enums or string column names
            
        Returns:
            Boolean Series indicating where all required signals are True
            
        Example:
            # Bullish combination: Price above cloud AND Tenkan above Kijun
            bullish = detector.get_signal_combination(df, [
                SignalType.PRICE_ABOVE_CLOUD,
                SignalType.TENKAN_ABOVE_KIJUN
            ])
        """
        if not required_signals:
            return pd.Series(True, index=df.index)
        
        # Convert SignalType enums to column names
        signal_columns = []
        for signal in required_signals:
            if isinstance(signal, SignalType):
                # Convert enum to lowercase column name
                col_name = signal.value.lower()
                # Convert CamelCase to snake_case
                col_name = ''.join(['_' + c.lower() if c.isupper() else c for c in col_name]).lstrip('_')
            else:
                col_name = signal
            
            if col_name not in df.columns:
                raise ValueError(f"Signal column '{col_name}' not found in DataFrame")
            
            signal_columns.append(col_name)
        
        # Combine all signals with AND logic
        combined_signal = pd.Series(True, index=df.index)
        for col in signal_columns:
            combined_signal = combined_signal & df[col]
        
        return combined_signal
    
    def validate_signal_strength(self, signals: pd.DataFrame, 
                               lookback_periods: int = 10) -> SignalStrength:
        """
        Validate and assess the strength of current signals.
        
        This method evaluates signal quality based on:
        - Number of active signals
        - Consistency over recent periods
        - Conflicting signals check
        
        Args:
            signals: DataFrame with signal columns
            lookback_periods: Number of periods to check for consistency
            
        Returns:
            SignalStrength object with strength assessment
        """
        if len(signals) == 0:
            return SignalStrength(0, 0, 0, 0, "No data")
        
        # Get the latest signals (excluding current incomplete bar)
        latest_idx = -2 if len(signals) > 1 else -1
        latest_signals = signals.iloc[latest_idx]
        
        # Count active signals
        bullish_signals = [
            'price_above_cloud', 'tenkan_above_kijun', 
            'chikou_above_price', 'chikou_above_cloud',
            'chikou_cross_above_senkou_b'
        ]
        
        bearish_signals = [
            'price_below_cloud', 'tenkan_below_kijun',
            'chikou_below_price', 'chikou_below_cloud',
            'chikou_cross_below_senkou_b'
        ]
        
        # Count active signals
        active_bullish = sum(latest_signals.get(sig, False) for sig in bullish_signals if sig in latest_signals)
        active_bearish = sum(latest_signals.get(sig, False) for sig in bearish_signals if sig in latest_signals)
        
        # Check for conflicting signals
        has_conflicts = active_bullish > 0 and active_bearish > 0
        
        # Calculate signal consistency over lookback period
        lookback_start = max(0, latest_idx - lookback_periods + 1)
        recent_signals = signals.iloc[lookback_start:latest_idx+1]
        
        # Calculate consistency scores
        bullish_consistency = 0
        bearish_consistency = 0
        
        if len(recent_signals) > 0:
            for sig in bullish_signals:
                if sig in recent_signals.columns:
                    bullish_consistency += recent_signals[sig].sum() / len(recent_signals)
            
            for sig in bearish_signals:
                if sig in recent_signals.columns:
                    bearish_consistency += recent_signals[sig].sum() / len(recent_signals)
            
            bullish_consistency /= len(bullish_signals)
            bearish_consistency /= len(bearish_signals)
        
        # Determine overall strength and confidence
        if has_conflicts:
            strength = 0.3  # Weak due to conflicts
            confidence = 0.3
            description = "Conflicting signals detected"
        elif active_bullish >= 3:
            strength = min(0.5 + (active_bullish * 0.1), 1.0)
            confidence = bullish_consistency
            description = f"Strong bullish signals ({active_bullish}/5 active)"
        elif active_bearish >= 3:
            strength = min(0.5 + (active_bearish * 0.1), 1.0)
            confidence = bearish_consistency
            description = f"Strong bearish signals ({active_bearish}/5 active)"
        elif active_bullish > active_bearish:
            strength = 0.4 + (active_bullish * 0.1)
            confidence = bullish_consistency
            description = f"Moderate bullish signals ({active_bullish} active)"
        elif active_bearish > active_bullish:
            strength = 0.4 + (active_bearish * 0.1)
            confidence = bearish_consistency
            description = f"Moderate bearish signals ({active_bearish} active)"
        else:
            strength = 0.2
            confidence = 0.2
            description = "Neutral - no clear signal direction"
        
        return SignalStrength(
            strength=round(strength, 3),
            confidence=round(confidence, 3),
            active_signals=active_bullish + active_bearish,
            total_signals=len(bullish_signals) + len(bearish_signals),
            description=description
        )
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get a comprehensive summary of current signals.
        
        Args:
            df: DataFrame with signal columns
            
        Returns:
            Dictionary with signal summary including counts and descriptions
        """
        if len(df) == 0:
            return {"error": "No data available"}
        
        # Get latest completed bar
        latest_idx = -2 if len(df) > 1 else -1
        latest = df.iloc[latest_idx]
        
        summary = {
            "timestamp": df.index[latest_idx],
            "signals": {},
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0
        }
        
        # Categorize signals
        bullish_signals = {
            'price_above_cloud': 'Price Above Cloud',
            'tenkan_above_kijun': 'Tenkan Above Kijun',
            'chikou_above_price': 'Chikou Above Historical Price',
            'chikou_above_cloud': 'Chikou Above Historical Cloud',
            'chikou_cross_above_senkou_b': 'Chikou Crossed Above Senkou B'
        }
        
        bearish_signals = {
            'price_below_cloud': 'Price Below Cloud',
            'tenkan_below_kijun': 'Tenkan Below Kijun',
            'chikou_below_price': 'Chikou Below Historical Price',
            'chikou_below_cloud': 'Chikou Below Historical Cloud',
            'chikou_cross_below_senkou_b': 'Chikou Crossed Below Senkou B'
        }
        
        # Check each signal
        for sig_col, sig_name in bullish_signals.items():
            if sig_col in latest and latest[sig_col]:
                summary["signals"][sig_name] = "Active (Bullish)"
                summary["bullish_count"] += 1
            else:
                summary["signals"][sig_name] = "Inactive"
        
        for sig_col, sig_name in bearish_signals.items():
            if sig_col in latest and latest[sig_col]:
                summary["signals"][sig_name] = "Active (Bearish)"
                summary["bearish_count"] += 1
            else:
                summary["signals"][sig_name] = "Inactive"
        
        # Overall assessment
        if summary["bullish_count"] > summary["bearish_count"]:
            summary["overall_bias"] = "Bullish"
        elif summary["bearish_count"] > summary["bullish_count"]:
            summary["overall_bias"] = "Bearish"
        else:
            summary["overall_bias"] = "Neutral"
        
        return summary


# Example usage
if __name__ == "__main__":
    import logging
    from ichimoku_calculator import IchimokuCalculator
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='1h')
    
    # Generate trending OHLCV data
    trend = np.cumsum(np.random.normal(0.5, 1, len(dates)))  # Upward trend
    close_prices = 50000 + trend * 100
    
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
    
    # Calculate Ichimoku indicators
    print("Step 1: Calculating Ichimoku indicators...")
    calculator = IchimokuCalculator()
    ichimoku_data = calculator.calculate_ichimoku(sample_data)
    
    # Initialize signal detector
    print("\nStep 2: Initializing signal detector...")
    detector = IchimokuSignalDetector()
    
    # Detect all signals
    print("\nStep 3: Detecting all signals...")
    signal_data = detector.detect_all_signals(ichimoku_data)
    
    # Show signal summary
    print("\n" + "="*60)
    print("SIGNAL SUMMARY")
    print("="*60)
    summary = detector.get_signal_summary(signal_data)
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Overall Bias: {summary['overall_bias']}")
    print(f"Bullish Signals: {summary['bullish_count']}")
    print(f"Bearish Signals: {summary['bearish_count']}")
    print("\nIndividual Signals:")
    for signal, status in summary['signals'].items():
        print(f"  {signal}: {status}")
    
    # Example: Get specific signal combination
    print("\n" + "="*60)
    print("SIGNAL COMBINATIONS")
    print("="*60)
    
    # Strong bullish combination
    strong_bullish = detector.get_signal_combination(signal_data, [
        SignalType.PRICE_ABOVE_CLOUD,
        SignalType.TENKAN_ABOVE_KIJUN,
        SignalType.CHIKOU_ABOVE_PRICE
    ])
    
    print(f"\nStrong Bullish Signal (Price > Cloud + Tenkan > Kijun + Chikou > Price):")
    print(f"Active on {strong_bullish.sum()} out of {len(strong_bullish)} bars")
    
    # Strong bearish combination
    strong_bearish = detector.get_signal_combination(signal_data, [
        'price_below_cloud',
        'tenkan_below_kijun',
        'chikou_below_price'
    ])
    
    print(f"\nStrong Bearish Signal (Price < Cloud + Tenkan < Kijun + Chikou < Price):")
    print(f"Active on {strong_bearish.sum()} out of {len(strong_bearish)} bars")
    
    # Validate signal strength
    print("\n" + "="*60)
    print("SIGNAL STRENGTH VALIDATION")
    print("="*60)
    strength = detector.validate_signal_strength(signal_data)
    print(f"Signal Strength: {strength.strength}")
    print(f"Confidence: {strength.confidence}")
    print(f"Active Signals: {strength.active_signals}/{strength.total_signals}")
    print(f"Assessment: {strength.description}")
    
    # Show last few bars with signals
    print("\n" + "="*60)
    print("LAST 5 BARS WITH SIGNALS")
    print("="*60)
    display_cols = ['close', 'price_above_cloud', 'price_below_cloud', 
                   'tenkan_above_kijun', 'chikou_above_price']
    print(signal_data[display_cols].tail())