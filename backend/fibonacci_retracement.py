"""
Fibonacci Retracement Calculator Module
Calculates Fibonacci retracement levels based on high and low points.
"""

import pandas as pd
import sqlite3


class FibonacciRetracementCalculator:
    """Calculates Fibonacci retracement levels from raw data."""
    
    def __init__(self, raw_db_path: str = 'data/raw_market_data.db', fib_db_path: str = 'data/fibonacci_retracement_data.db'):
        self.raw_db_path = raw_db_path
        self.fib_db_path = fib_db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with Fibonacci Retracement table if it doesn't exist."""
        with sqlite3.connect(self.fib_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS fibonacci_retracement_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    level_23_6 REAL,
                    level_38_2 REAL,
                    level_50_0 REAL,
                    level_61_8 REAL,
                    level_76_4 REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            conn.commit()
    
    def fetch_raw_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch raw OHLCV data from the raw database."""
        query = '''
            SELECT timestamp, high, low
            FROM raw_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp ASC
        '''
        with sqlite3.connect(self.raw_db_path) as conn:
            return pd.read_sql(query, conn, params=(symbol, timeframe))

    def calculate_fibonacci_retracement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fibonacci retracement levels on data."""
        high = df['high'].max()
        low = df['low'].min()

        # Calculate Fibonacci levels
        df['level_23_6'] = high - (high - low) * 0.236
        df['level_38_2'] = high - (high - low) * 0.382
        df['level_50_0'] = high - (high - low) * 0.5
        df['level_61_8'] = high - (high - low) * 0.618
        df['level_76_4'] = high - (high - low) * 0.764

        return df

    def save_fibonacci_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated Fibonacci retracement data to the indicators database."""
        df_to_save = df[['timestamp', 'level_23_6', 'level_38_2', 'level_50_0', 'level_61_8', 'level_76_4']].copy()
        df_to_save['symbol'] = symbol
        df_to_save['timeframe'] = timeframe
        
        df_to_save = df_to_save[['symbol', 'timeframe', 'timestamp', 'level_23_6', 'level_38_2', 'level_50_0', 'level_61_8', 'level_76_4']]
        
        try:
            with sqlite3.connect(self.fib_db_path) as conn:
                cursor = conn.cursor()
                inserted = 0
                for _, row in df_to_save.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO fibonacci_retracement_data 
                        (symbol, timeframe, timestamp, level_23_6, level_38_2, level_50_0, level_61_8, level_76_4)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', tuple(row))
                    inserted += 1
                conn.commit()
                print(f"[INFO] Saved {inserted} Fibonacci retracement records for {symbol} ({timeframe})")
        except Exception as e:
            print(f"[ERROR] Failed to save Fibonacci retracement data for {symbol}: {e}")


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
            df_fib = calculator.calculate_fibonacci_retracement(df_raw)
            calculator.save_fibonacci_data(df_fib, symbol, timeframe)


if __name__ == '__main__':
    main()

