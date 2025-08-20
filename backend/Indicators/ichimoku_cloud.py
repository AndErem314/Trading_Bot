"""
Ichimoku Cloud Calculator module - UNIFIED VERSION
Uses unified_trading_data.db to calculate Ichimoku Cloud indicators and saves them to the unified database.

The Ichimoku Cloud, also known as Ichimoku Kinko Hyo, is a comprehensive indicator that defines support/resistance, trend direction, momentum, and provides trading signals.

This module implements:
- Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
- Kijun-sen (Base Line): (26-period high + 26-period low) / 2
- Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods forward
- Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods forward
- Chikou Span (Lagging Span): close price shifted 26 periods backward

Common trading signals:
- Bullish signals when current price is above the cloud
- Bearish signals when current price is below the cloud
"""
import pandas as pd
import sqlite3
import numpy as np
from typing import Optional


class IchimokuCloudCalculator:
    """Calculates Ichimoku Cloud indicators using unified database."""
    
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

    def calculate_ichimoku_cloud(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud indicators on data.
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - DataFrame with Ichimoku Cloud indicators added
        """
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        df['tenkan_sen'] = (df['high'].rolling(window=9, min_periods=1).max() + 
                           df['low'].rolling(window=9, min_periods=1).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        df['kijun_sen'] = (df['high'].rolling(window=26, min_periods=1).max() + 
                          df['low'].rolling(window=26, min_periods=1).min()) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods forward
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods forward
        df['senkou_span_b'] = ((df['high'].rolling(window=52, min_periods=1).max() + 
                               df['low'].rolling(window=52, min_periods=1).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): close price shifted 26 periods backward
        df['chikou_span'] = df['close'].shift(-26)
        
        # Determine Cloud Color
        df['cloud_color'] = np.where(df['senkou_span_a'] >= df['senkou_span_b'], 'green', 'red')
        
        # Determine Ichimoku Signal
        df['ichimoku_signal'] = 'neutral'
        
        # Strong bullish condition: All bullish factors aligned
        strong_bullish_condition = (
            (df['close'] > df['senkou_span_a']) &
            (df['close'] > df['senkou_span_b']) &
            (df['tenkan_sen'] > df['kijun_sen']) &
            (df['chikou_span'] > df['close'].shift(26)) &
            (df['cloud_color'] == 'green')
        )
        df.loc[strong_bullish_condition, 'ichimoku_signal'] = 'strong_bullish'
        
        # Bullish condition: Price above cloud
        bullish_condition = (
            (df['close'] > df['senkou_span_a']) &
            (df['close'] > df['senkou_span_b']) &
            (df['tenkan_sen'] > df['kijun_sen'])
        ) & ~strong_bullish_condition
        df.loc[bullish_condition, 'ichimoku_signal'] = 'bullish'
        
        # Strong bearish condition: All bearish factors aligned
        strong_bearish_condition = (
            (df['close'] < df['senkou_span_a']) &
            (df['close'] < df['senkou_span_b']) &
            (df['tenkan_sen'] < df['kijun_sen']) &
            (df['chikou_span'] < df['close'].shift(26)) &
            (df['cloud_color'] == 'red')
        )
        df.loc[strong_bearish_condition, 'ichimoku_signal'] = 'strong_bearish'
        
        # Bearish condition: Price below cloud
        bearish_condition = (
            (df['close'] < df['senkou_span_a']) &
            (df['close'] < df['senkou_span_b']) &
            (df['tenkan_sen'] < df['kijun_sen'])
        ) & ~strong_bearish_condition
        df.loc[bearish_condition, 'ichimoku_signal'] = 'bearish'

        return df

    def save_ichimoku_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated Ichimoku Cloud data to the unified database."""
        if df.empty:
            print(f"[INFO] No Ichimoku data to save for {symbol} ({timeframe})")
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
                    
                    # Insert or replace Ichimoku indicator data
                    conn.execute('''
                        INSERT OR REPLACE INTO ichimoku_indicator 
                        (ohlcv_id, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, 
                         chikou_span, cloud_color, ichimoku_signal)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ohlcv_id,
                        row['tenkan_sen'] if pd.notna(row['tenkan_sen']) else None,
                        row['kijun_sen'] if pd.notna(row['kijun_sen']) else None,
                        row['senkou_span_a'] if pd.notna(row['senkou_span_a']) else None,
                        row['senkou_span_b'] if pd.notna(row['senkou_span_b']) else None,
                        row['chikou_span'] if pd.notna(row['chikou_span']) else None,
                        row['cloud_color'],
                        row['ichimoku_signal']
                    ))
                    inserted += 1
                
                conn.commit()
                print(f"[INFO] Saved {inserted} Ichimoku indicators for {symbol} ({timeframe})")
                if skipped > 0:
                    print(f"[WARNING] Skipped {skipped} records (no matching OHLCV data)")
                    
        except Exception as e:
            print(f"[ERROR] Failed to save Ichimoku data for {symbol}: {e}")

    def analyze_ichimoku_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze current Ichimoku Cloud patterns and provide insights."""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        recent_data = df.tail(20)  # Last 20 periods for analysis
        
        # Determine price position relative to cloud
        if pd.notna(latest['senkou_span_a']) and pd.notna(latest['senkou_span_b']):
            cloud_top = max(latest['senkou_span_a'], latest['senkou_span_b'])
            cloud_bottom = min(latest['senkou_span_a'], latest['senkou_span_b'])
            
            if latest['close'] > cloud_top:
                price_vs_cloud = 'above_cloud'
            elif latest['close'] < cloud_bottom:
                price_vs_cloud = 'below_cloud'
            else:
                price_vs_cloud = 'inside_cloud'
        else:
            price_vs_cloud = 'unknown'
        
        # Check for recent signal changes
        recent_signals = recent_data['ichimoku_signal'].unique()
        signal_strength = self._calculate_signal_strength(latest)
        
        analysis = {
            'current_signal': latest['ichimoku_signal'],
            'cloud_color': latest['cloud_color'],
            'price_vs_cloud': price_vs_cloud,
            'tenkan_vs_kijun': 'above' if latest['tenkan_sen'] > latest['kijun_sen'] else 'below',
            'chikou_span_position': self._get_chikou_position(df, len(df) - 1),
            'signal_strength': signal_strength,
            'trend_consistency': len(recent_signals),
            'current_values': {
                'close': latest['close'],
                'tenkan_sen': latest['tenkan_sen'],
                'kijun_sen': latest['kijun_sen'],
                'senkou_span_a': latest['senkou_span_a'],
                'senkou_span_b': latest['senkou_span_b'],
                'chikou_span': latest['chikou_span']
            }
        }
        
        return analysis
    
    def _calculate_signal_strength(self, latest_data) -> str:
        """Calculate the strength of the current Ichimoku signal."""
        signal = latest_data['ichimoku_signal']
        
        if signal in ['strong_bullish', 'strong_bearish']:
            return 'strong'
        elif signal in ['bullish', 'bearish']:
            return 'moderate'
        else:
            return 'weak'
    
    def _get_chikou_position(self, df: pd.DataFrame, current_index: int) -> str:
        """Get Chikou span position relative to price."""
        if current_index >= 26:
            chikou_value = df.iloc[current_index]['chikou_span']
            price_26_ago = df.iloc[current_index - 26]['close']
            
            if pd.notna(chikou_value) and pd.notna(price_26_ago):
                return 'above_price' if chikou_value > price_26_ago else 'below_price'
        
        return 'insufficient_data'

    def get_cloud_crossovers(self, symbol: str, timeframe: str, limit: int = 10) -> pd.DataFrame:
        """Get recent cloud signal changes for a symbol."""
        query = '''
            SELECT o.timestamp, o.close, ich.tenkan_sen, ich.kijun_sen, 
                   ich.senkou_span_a, ich.senkou_span_b, ich.ichimoku_signal
            FROM ohlcv_data o
            JOIN symbols s ON o.symbol_id = s.id
            JOIN timeframes t ON o.timeframe_id = t.id
            JOIN ichimoku_indicator ich ON o.id = ich.ohlcv_id
            WHERE s.symbol = ? AND t.timeframe = ? AND ich.ichimoku_signal != 'neutral'
            ORDER BY o.timestamp DESC
            LIMIT ?
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=(symbol, timeframe, limit))


def main():
    """Main function to calculate and save Ichimoku Cloud data for multiple symbols and timeframes."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
    timeframes = ['4h', '1d']

    calculator = IchimokuCloudCalculator()

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[CALCULATING] Ichimoku Cloud for {symbol} - {timeframe.upper()}")
            df_raw = calculator.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            
            # Calculate Ichimoku Cloud
            df_ichimoku = calculator.calculate_ichimoku_cloud(df_raw)
            
            # Save to database
            calculator.save_ichimoku_data(df_ichimoku, symbol, timeframe)
            
            # Print analysis for the latest data
            analysis = calculator.analyze_ichimoku_patterns(df_ichimoku)
            if analysis:
                print(f"[ANALYSIS] Current signal: {analysis['current_signal']}")
                print(f"[ANALYSIS] Cloud color: {analysis['cloud_color']}")
                print(f"[ANALYSIS] Price vs cloud: {analysis['price_vs_cloud']}")
                print(f"[ANALYSIS] Tenkan vs Kijun: {analysis['tenkan_vs_kijun']}")
                print(f"[ANALYSIS] Signal strength: {analysis['signal_strength']}")


if __name__ == '__main__':
    main()
