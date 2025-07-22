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
    
    def __init__(self, db_path: str = 'data/market_data.db'):
        self.db_path = db_path
    
    def fetch_raw_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch raw OHLCV data from the database."""
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM raw_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp ASC
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=(symbol, timeframe))

    def calculate_gaussian_channel(self, df: pd.DataFrame, window: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Calculate Gaussian Channel on data."""
        df['gc_middle'] = df['close'].rolling(window=window, min_periods=1).mean()
        df['gc_std'] = df['close'].rolling(window=window, min_periods=1).std(ddof=0)
        df['gc_upper'] = df['gc_middle'] + (df['gc_std'] * std_dev)
        df['gc_lower'] = df['gc_middle'] - (df['gc_std'] * std_dev)
        return df

    def save_gaussian_channel_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated Gaussian Channel data to the database."""
        df_to_save = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'gc_upper', 'gc_lower', 'gc_middle']].copy()
        df_to_save['symbol'] = symbol
        df_to_save['timeframe'] = timeframe
        try:
            with sqlite3.connect(self.db_path) as conn:
                df_to_save.to_sql('gaussian_channel_data', conn, if_exists='append', index=False)
                print(f"[INFO] Saved Gaussian Channel data for {symbol} ({timeframe})")
        except Exception as e:
            print(f"[ERROR] Failed to save Gaussian Channel data for {symbol}: {e}")


def main():
    """Main function to calculate and save Gaussian Channel data for multiple symbols and timeframes."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
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
