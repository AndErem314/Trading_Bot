"""
Gaussian Channel Calculator module.
Uses raw data from the database to calculate Gaussian Channel indicator and saves it to the database.
"""
import pandas as pd
import sqlite3
import numpy as np
from typing import Optional


class GaussianChannelCalculator:
    """Calculates Gaussian Channel indicator from raw data."""
    
    def __init__(self, raw_db_path: str = 'data/raw_market_data.db', gaussian_db_path: str = 'data/gaussian_channel_data.db'):
        self.raw_db_path = raw_db_path
        self.gaussian_db_path = gaussian_db_path
        self._init_indicators_table()
    
    def _init_indicators_table(self):
        """Initialize the gaussian_channel_data table in indicators database."""
        with sqlite3.connect(self.gaussian_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS gaussian_channel_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    gc_upper REAL,
                    gc_lower REAL,
                    gc_middle REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            conn.commit()
    
    def fetch_raw_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch raw OHLCV data from the raw database."""
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM raw_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp ASC
        '''
        with sqlite3.connect(self.raw_db_path) as conn:
            return pd.read_sql(query, conn, params=(symbol, timeframe))

    def calculate_gaussian_channel(self, df: pd.DataFrame, window: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Calculate Gaussian Channel on data."""
        df['gc_middle'] = df['close'].rolling(window=window, min_periods=1).mean()
        df['gc_std'] = df['close'].rolling(window=window, min_periods=1).std(ddof=0)
        df['gc_upper'] = df['gc_middle'] + (df['gc_std'] * std_dev)
        df['gc_lower'] = df['gc_middle'] - (df['gc_std'] * std_dev)
        return df

    def save_gaussian_channel_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated Gaussian Channel data to the indicators database."""
        df_to_save = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'gc_upper', 'gc_lower', 'gc_middle']].copy()
        df_to_save['symbol'] = symbol
        df_to_save['timeframe'] = timeframe
        
        # Reorder columns to match database schema
        df_to_save = df_to_save[['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'gc_upper', 'gc_lower', 'gc_middle']]
        
        try:
            with sqlite3.connect(self.gaussian_db_path) as conn:
                cursor = conn.cursor()
                inserted = 0
                for _, row in df_to_save.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO gaussian_channel_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume, gc_upper, gc_lower, gc_middle)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', tuple(row))
                    inserted += 1
                conn.commit()
                print(f"[INFO] Saved {inserted} Gaussian Channel records for {symbol} ({timeframe})")
        except Exception as e:
            print(f"[ERROR] Failed to save Gaussian Channel data for {symbol}: {e}")


def main():
    """Main function to calculate and save Gaussian Channel data for multiple symbols and timeframes."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
    timeframes = ['4h', '1d']

    calculator = GaussianChannelCalculator()

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[CALCULATING] Gaussian Channel for {symbol} - {timeframe.upper()}")
            df_raw = calculator.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            df_gc = calculator.calculate_gaussian_channel(df_raw)
            calculator.save_gaussian_channel_data(df_gc, symbol, timeframe)


if __name__ == '__main__':
    main()
