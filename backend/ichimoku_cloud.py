"""
Ichimoku Cloud Calculator module.
Uses raw data from the database to calculate Ichimoku Cloud indicators and saves it to the database.

The Ichimoku Cloud, also known as Ichimoku Kinko Hyo, is a comprehensive indicator that defines support/resistance, trend direction, momentum, and provides trading signals.

This module implements:
- Tenkan-sen (Conversion Line)
- Kijun-sen (Base Line)
- Senkou Span A (Leading Span A)
- Senkou Span B (Leading Span B)
- Chikou Span (Lagging Span)

Common trading signals:
- Bullish signals when current price is above the cloud
- Bearish signals when current price is below the cloud
"""
import pandas as pd
import sqlite3
import numpy as np
from typing import Optional


class IchimokuCloudCalculator:
    """Calculates Ichimoku Cloud indicators from raw data."""
    
    def __init__(self, raw_db_path: str = 'data/raw_market_data.db', ichimoku_db_path: str = 'data/ichimoku_data.db'):
        # Ensure we use the correct database paths relative to project root
        import os
        if not os.path.isabs(raw_db_path) and not raw_db_path.startswith('../'):
            # If running from backend directory, adjust path to parent directory
            if os.path.basename(os.getcwd()) == 'backend':
                raw_db_path = '../' + raw_db_path
                ichimoku_db_path = '../' + ichimoku_db_path
        self.raw_db_path = raw_db_path
        self.ichimoku_db_path = ichimoku_db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with Ichimoku Cloud table if it doesn't exist."""
        with sqlite3.connect(self.ichimoku_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ichimoku_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    tenkan_sen REAL,
                    kijun_sen REAL,
                    senkou_span_a REAL,
                    senkou_span_b REAL,
                    chikou_span REAL,
                    cloud_color TEXT,
                    ichimoku_signal TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            conn.commit()
    
    def fetch_raw_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch raw OHLCV data from the database."""
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM raw_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp ASC
        '''
        with sqlite3.connect(self.raw_db_path) as conn:
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
        df['tenkan_sen'] = (df['high'].rolling(window=9, min_periods=1).max() + df['low'].rolling(window=9, min_periods=1).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        df['kijun_sen'] = (df['high'].rolling(window=26, min_periods=1).max() + df['low'].rolling(window=26, min_periods=1).min()) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods forward
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods forward
        df['senkou_span_b'] = ((df['high'].rolling(window=52, min_periods=1).max() + df['low'].rolling(window=52, min_periods=1).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): close price shifted 26 periods backward
        df['chikou_span'] = df['close'].shift(-26)
        
        # Determine Cloud Color
        df['cloud_color'] = np.where(df['senkou_span_a'] >= df['senkou_span_b'], 'green', 'red')
        
        # Determine Ichimoku Signal
        df['ichimoku_signal'] = 'neutral'
        bullish_condition = (
            (df['close'] > df['senkou_span_a']) &
            (df['close'] > df['senkou_span_b']) &
            (df['tenkan_sen'] > df['kijun_sen']) &
            (df['chikou_span'] > df['close'])
        )
        df.loc[bullish_condition, 'ichimoku_signal'] = 'bullish'
        
        bearish_condition = (
            (df['close'] < df['senkou_span_a']) &
            (df['close'] < df['senkou_span_b']) &
            (df['tenkan_sen'] < df['kijun_sen']) &
            (df['chikou_span'] < df['close'])
        )
        df.loc[bearish_condition, 'ichimoku_signal'] = 'bearish'

        return df

    def save_ichimoku_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated Ichimoku Cloud data to the database."""
        df_to_save = df[
            ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
             'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
             'cloud_color', 'ichimoku_signal']
        ].copy()
        df_to_save['symbol'] = symbol
        df_to_save['timeframe'] = timeframe
        
        # Reorder columns to match database schema
        df_to_save = df_to_save[
            ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 
             'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
             'cloud_color', 'ichimoku_signal']
        ]
        
        try:
            with sqlite3.connect(self.ichimoku_db_path) as conn:
                cursor = conn.cursor()
                inserted = 0
                for _, row in df_to_save.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO ichimoku_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume, 
                         tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span,
                         cloud_color, ichimoku_signal)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', tuple(row))
                    inserted += 1
                conn.commit()
                print(f"[INFO] Saved {inserted} Ichimoku Cloud records for {symbol} ({timeframe})")
        except Exception as e:
            print(f"[ERROR] Failed to save Ichimoku Cloud data for {symbol}: {e}")

    def analyze_ichimoku_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze current Ichimoku Cloud patterns and provide insights."""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        analysis = {
            'current_signal': latest['ichimoku_signal'],
            'cloud_color': latest['cloud_color'],
            'price_vs_cloud': 'above' if latest['close'] > max(latest['senkou_span_a'], latest['senkou_span_b']) else (
                'below' if latest['close'] < min(latest['senkou_span_a'], latest['senkou_span_b']) else 'inside'
            ),
            'tenkan_vs_kijun': 'above' if latest['tenkan_sen'] > latest['kijun_sen'] else 'below',
            'chikou_vs_price': 'above' if latest['chikou_span'] > latest['close'] else 'below',
            'recent_price': latest['close'],
            'senkou_spans': {
                'A': latest['senkou_span_a'],
                'B': latest['senkou_span_b']
            },
            'tenkan_kijun_values': {
                'tenkan_sen': latest['tenkan_sen'],
                'kijun_sen': latest['kijun_sen']
            }
        }
        
        return analysis

    def get_cloud_crossovers(self, symbol: str, timeframe: str, limit: int = 10) -> pd.DataFrame:
        """Get recent cloud crossover history for a symbol."""
        query = '''
            SELECT timestamp, close, senkou_span_a, senkou_span_b, ichimoku_signal
            FROM ichimoku_data
            WHERE symbol = ? AND timeframe = ? AND ichimoku_signal != 'neutral'
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        with sqlite3.connect(self.ichimoku_db_path) as conn:
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


if __name__ == '__main__':
    main()

