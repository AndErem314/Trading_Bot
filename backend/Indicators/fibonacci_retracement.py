"""
Fibonacci Retracement Calculator Module - UNIFIED VERSION
Uses unified_trading_data.db to calculate Fibonacci retracement levels and saves them to the unified database.

Fibonacci retracements are horizontal lines that indicate areas of support or resistance at the key Fibonacci levels
before the price continues in the original direction. These levels are created by drawing a trendline between
two extreme points and then dividing the vertical distance by the key Fibonacci ratios.

Key Fibonacci levels:
- 23.6% - Minor retracement level
- 38.2% - Moderate retracement level
- 50.0% - Halfway point (not a Fibonacci ratio but widely used)
- 61.8% - Golden ratio, major retracement level
- 78.6% - Deep retracement level

This module implements:
- Dynamic calculation of Fibonacci levels based on recent highs and lows
- Multiple lookback periods for different trend contexts
- Support/resistance level identification
- Signal generation when price approaches key levels
"""

import pandas as pd
import sqlite3
import numpy as np
from typing import Optional, Tuple


class FibonacciRetracementCalculator:
    """Calculates Fibonacci retracement levels using unified database."""
    
    def __init__(self, db_path: str = 'data/unified_trading_data.db'):
        # Ensure we use the correct database path relative to project root
        import os
        if not os.path.isabs(db_path) and not db_path.startswith('../'):
            # If running from Indicators subfolder, adjust path to project root
            current_dir = os.path.basename(os.getcwd())
            if current_dir == 'Indicators':
                db_path = '../../' + db_path
            elif current_dir == 'backend':
                db_path = '../' + db_path
        self.db_path = db_path
        
        # Fibonacci ratios
        self.fib_ratios = {
            'level_0': 0.0,      # 0% - High/Low point
            'level_23_6': 0.236,  # 23.6%
            'level_38_2': 0.382,  # 38.2%
            'level_50_0': 0.5,    # 50.0%
            'level_61_8': 0.618,  # 61.8% (Golden ratio)
            'level_78_6': 0.786,  # 78.6%
            'level_100': 1.0      # 100% - Complete retracement
        }
    
    def fetch_raw_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch raw OHLCV data from the unified database."""
        query = '''
            SELECT o.timestamp, o.open, o.high, o.low, o.close, o.volume
            FROM ohlcv_data o
            JOIN symbols s ON o.symbol_id = s.id
            JOIN timeframes t ON o.timeframe_id = t.id
            WHERE s.symbol = ? AND t.timeframe = ?
            ORDER BY o.timestamp ASC
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=(symbol, timeframe))

    def calculate_fibonacci_retracement(self, df: pd.DataFrame, lookback_period: int = 50) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels on data.
        
        Parameters:
        - df: DataFrame with OHLCV data
        - lookback_period: Period to look back for high/low calculation (default: 50)
        
        Returns:
        - DataFrame with Fibonacci retracement levels added
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Initialize Fibonacci level columns
        for level_name in self.fib_ratios.keys():
            df[level_name] = np.nan
        
        # Additional columns for analysis
        df['trend_direction'] = 'neutral'
        df['nearest_fib_level'] = np.nan
        df['fib_signal'] = 'hold'
        df['support_resistance'] = 'none'
        
        # Calculate rolling highs and lows
        rolling_high = df['high'].rolling(window=lookback_period, min_periods=1).max()
        rolling_low = df['low'].rolling(window=lookback_period, min_periods=1).min()
        
        for i in range(len(df)):
            swing_high = rolling_high.iloc[i]
            swing_low = rolling_low.iloc[i]
            current_price = df['close'].iloc[i]
            
            # Determine trend direction based on price position
            mid_point = (swing_high + swing_low) / 2
            if current_price > mid_point:
                trend_direction = 'uptrend'
                # For uptrend, calculate retracement from high to low
                price_range = swing_high - swing_low
                base_price = swing_high
                multiplier = -1  # Levels below the high
            else:
                trend_direction = 'downtrend'
                # For downtrend, calculate extension from low to high
                price_range = swing_high - swing_low
                base_price = swing_low
                multiplier = 1   # Levels above the low
            
            df.at[i, 'trend_direction'] = trend_direction
            
            # Calculate Fibonacci levels
            for level_name, ratio in self.fib_ratios.items():
                fib_level = base_price + (multiplier * price_range * ratio)
                df.at[i, level_name] = fib_level
            
            # Find nearest Fibonacci level
            nearest_level, nearest_distance = self._find_nearest_fib_level(
                current_price, df.iloc[i], list(self.fib_ratios.keys())
            )
            df.at[i, 'nearest_fib_level'] = nearest_distance
            
            # Generate signals based on proximity to key levels
            fib_signal = self._generate_fib_signal(
                current_price, df.iloc[i], trend_direction, nearest_level, nearest_distance
            )
            df.at[i, 'fib_signal'] = fib_signal
            
            # Identify support/resistance
            support_resistance = self._identify_support_resistance(
                current_price, df.iloc[i], trend_direction
            )
            df.at[i, 'support_resistance'] = support_resistance

        return df
    
    def _find_nearest_fib_level(self, current_price: float, row, level_names: list) -> Tuple[str, float]:
        """Find the nearest Fibonacci level to current price."""
        min_distance = float('inf')
        nearest_level = 'level_50_0'
        
        for level_name in level_names:
            level_price = row[level_name]
            if pd.notna(level_price):
                distance = abs(current_price - level_price) / current_price * 100
                if distance < min_distance:
                    min_distance = distance
                    nearest_level = level_name
        
        return nearest_level, min_distance
    
    def _generate_fib_signal(self, current_price: float, row, trend_direction: str, 
                           nearest_level: str, nearest_distance: float) -> str:
        """Generate trading signals based on Fibonacci levels."""
        # Strong signals when price is very close to key levels (within 1%)
        if nearest_distance < 1.0:
            if nearest_level in ['level_61_8', 'level_38_2']:  # Key retracement levels
                if trend_direction == 'uptrend':
                    return 'buy_support'  # Buying at support in uptrend
                else:
                    return 'sell_resistance'  # Selling at resistance in downtrend
            elif nearest_level == 'level_23_6':  # Minor retracement
                if trend_direction == 'uptrend':
                    return 'weak_buy'
                else:
                    return 'weak_sell'
            elif nearest_level == 'level_78_6':  # Deep retracement
                if trend_direction == 'uptrend':
                    return 'strong_buy'  # Last chance before trend reversal
                else:
                    return 'strong_sell'
        
        # Moderate signals when approaching key levels (within 2%)
        elif nearest_distance < 2.0:
            if nearest_level in ['level_61_8', 'level_50_0']:
                return 'approaching_key_level'
        
        return 'hold'
    
    def _identify_support_resistance(self, current_price: float, row, trend_direction: str) -> str:
        """Identify if current price is near support or resistance levels."""
        # Key levels that often act as support/resistance
        key_levels = ['level_38_2', 'level_50_0', 'level_61_8']
        
        for level_name in key_levels:
            level_price = row[level_name]
            if pd.notna(level_price):
                distance_pct = abs(current_price - level_price) / current_price * 100
                
                if distance_pct < 1.5:  # Within 1.5%
                    if current_price > level_price:
                        return f"{level_name}_support"
                    else:
                        return f"{level_name}_resistance"
        
        return 'none'

    def save_fibonacci_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated Fibonacci retracement data to the unified database."""
        if df.empty:
            print(f"[INFO] No Fibonacci data to save for {symbol} ({timeframe})")
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                inserted = 0
                skipped = 0
                
                for _, row in df.iterrows():
                    # Get OHLCV data ID for this timestamp
                    ohlcv_cursor = conn.execute('''
                        SELECT o.id FROM ohlcv_data o
                        JOIN symbols s ON o.symbol_id = s.id
                        JOIN timeframes t ON o.timeframe_id = t.id
                        WHERE s.symbol = ? AND t.timeframe = ? AND o.timestamp = ?
                    ''', (symbol, timeframe, row['timestamp']))
                    
                    ohlcv_result = ohlcv_cursor.fetchone()
                    if not ohlcv_result:
                        skipped += 1
                        continue
                    
                    ohlcv_id = ohlcv_result[0]
                    
                    # Insert or replace Fibonacci indicator data
                    conn.execute('''
                        INSERT OR REPLACE INTO fibonacci_retracement_indicator 
                        (ohlcv_id, level_0, level_23_6, level_38_2, level_50_0, level_61_8, 
                         level_78_6, level_100, trend_direction, nearest_fib_level, 
                         fib_signal, support_resistance)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ohlcv_id,
                        row['level_0'] if pd.notna(row['level_0']) else None,
                        row['level_23_6'] if pd.notna(row['level_23_6']) else None,
                        row['level_38_2'] if pd.notna(row['level_38_2']) else None,
                        row['level_50_0'] if pd.notna(row['level_50_0']) else None,
                        row['level_61_8'] if pd.notna(row['level_61_8']) else None,
                        row['level_78_6'] if pd.notna(row['level_78_6']) else None,
                        row['level_100'] if pd.notna(row['level_100']) else None,
                        row['trend_direction'],
                        row['nearest_fib_level'] if pd.notna(row['nearest_fib_level']) else None,
                        row['fib_signal'],
                        row['support_resistance']
                    ))
                    inserted += 1
                
                conn.commit()
                print(f"[INFO] Saved {inserted} Fibonacci indicators for {symbol} ({timeframe})")
                if skipped > 0:
                    print(f"[WARNING] Skipped {skipped} records (no matching OHLCV data)")
                    
        except Exception as e:
            print(f"[ERROR] Failed to save Fibonacci data for {symbol}: {e}")

    def analyze_fibonacci_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze current Fibonacci patterns and provide insights."""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        recent_data = df.tail(20)  # Last 20 periods
        
        # Count recent signals
        signal_counts = recent_data['fib_signal'].value_counts().to_dict()
        
        # Identify current key levels
        key_levels = {
            '23.6%': latest['level_23_6'],
            '38.2%': latest['level_38_2'],
            '50.0%': latest['level_50_0'],
            '61.8%': latest['level_61_8'],
            '78.6%': latest['level_78_6']
        }
        
        # Calculate distances to key levels
        current_price = latest['close'] if 'close' in df.columns else latest['level_50_0']
        level_distances = {}
        
        for level_name, level_price in key_levels.items():
            if pd.notna(level_price) and pd.notna(current_price):
                distance_pct = ((current_price - level_price) / level_price) * 100
                level_distances[level_name] = {
                    'price': level_price,
                    'distance_pct': distance_pct,
                    'above_below': 'above' if distance_pct > 0 else 'below'
                }
        
        analysis = {
            'current_signal': latest['fib_signal'],
            'trend_direction': latest['trend_direction'],
            'nearest_level_distance': latest['nearest_fib_level'],
            'support_resistance': latest['support_resistance'],
            'key_levels': key_levels,
            'level_distances': level_distances,
            'recent_signals': signal_counts,
            'price_analysis': self._analyze_price_position(latest, key_levels)
        }
        
        return analysis
    
    def _analyze_price_position(self, latest_data, key_levels: dict) -> dict:
        """Analyze current price position relative to Fibonacci levels."""
        current_price = latest_data.get('close', latest_data['level_50_0'])
        
        # Determine which levels are above and below current price
        levels_above = []
        levels_below = []
        
        for level_name, level_price in key_levels.items():
            if pd.notna(level_price):
                if level_price > current_price:
                    levels_above.append((level_name, level_price))
                elif level_price < current_price:
                    levels_below.append((level_name, level_price))
        
        # Sort levels
        levels_above.sort(key=lambda x: x[1])  # Closest resistance first
        levels_below.sort(key=lambda x: x[1], reverse=True)  # Closest support first
        
        # Identify immediate support and resistance
        immediate_resistance = levels_above[0] if levels_above else None
        immediate_support = levels_below[0] if levels_below else None
        
        return {
            'immediate_support': immediate_support,
            'immediate_resistance': immediate_resistance,
            'levels_above': levels_above,
            'levels_below': levels_below,
            'position_analysis': self._get_position_description(
                current_price, immediate_support, immediate_resistance
            )
        }
    
    def _get_position_description(self, current_price: float, support: tuple, resistance: tuple) -> str:
        """Get descriptive text for current price position."""
        if support and resistance:
            support_pct = ((current_price - support[1]) / support[1]) * 100
            resistance_pct = ((resistance[1] - current_price) / current_price) * 100
            return f"Between {support[0]} support ({support_pct:.1f}% above) and {resistance[0]} resistance ({resistance_pct:.1f}% below)"
        elif resistance:
            resistance_pct = ((resistance[1] - current_price) / current_price) * 100
            return f"Below all major levels, next resistance at {resistance[0]} ({resistance_pct:.1f}% above)"
        elif support:
            support_pct = ((current_price - support[1]) / support[1]) * 100
            return f"Above all major levels, last support at {support[0]} ({support_pct:.1f}% below)"
        else:
            return "Price outside calculated Fibonacci range"

    def get_fibonacci_signals(self, symbol: str, timeframe: str, limit: int = 10) -> pd.DataFrame:
        """Get recent Fibonacci signal history for a symbol."""
        query = '''
            SELECT o.timestamp, o.close, fib.level_38_2, fib.level_50_0, fib.level_61_8,
                   fib.fib_signal, fib.support_resistance, fib.trend_direction
            FROM ohlcv_data o
            JOIN symbols s ON o.symbol_id = s.id
            JOIN timeframes t ON o.timeframe_id = t.id
            JOIN fibonacci_retracement_indicator fib ON o.id = fib.ohlcv_id
            WHERE s.symbol = ? AND t.timeframe = ? AND fib.fib_signal != 'hold'
            ORDER BY o.timestamp DESC
            LIMIT ?
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=(symbol, timeframe, limit))


def main():
    """Main function to calculate and save Fibonacci retracement data for multiple symbols and timeframes."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
    timeframes = ['4h', '1d']

    calculator = FibonacciRetracementCalculator()

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[CALCULATING] Fibonacci Retracement for {symbol} - {timeframe.upper()}")
            df_raw = calculator.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            
            # Calculate Fibonacci retracement levels
            df_fib = calculator.calculate_fibonacci_retracement(df_raw)
            
            # Save to database
            calculator.save_fibonacci_data(df_fib, symbol, timeframe)
            
            # Print analysis for the latest data
            analysis = calculator.analyze_fibonacci_patterns(df_fib)
            if analysis:
                print(f"[ANALYSIS] Current signal: {analysis['current_signal']}")
                print(f"[ANALYSIS] Trend direction: {analysis['trend_direction']}")
                print(f"[ANALYSIS] Support/Resistance: {analysis['support_resistance']}")
                print(f"[ANALYSIS] Nearest level distance: {analysis['nearest_level_distance']:.2f}%")
                
                if analysis['price_analysis']['position_analysis']:
                    print(f"[ANALYSIS] Position: {analysis['price_analysis']['position_analysis']}")


if __name__ == '__main__':
    main()
