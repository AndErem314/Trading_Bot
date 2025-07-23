"""
Data fetcher module for getting raw market data and saving to database.
"""
import ccxt
import pandas as pd
import sqlite3
import os
import time
from datetime import datetime
from typing import Optional


class DataFetcher:
    """Fetches raw OHLCV data from exchange."""
    
    def __init__(self, exchange_name: str = 'binance'):
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': '',
            'secret': '',
            'timeout': 30000,
            'enableRateLimit': True,
            'sandbox': False,
        })
    
    def fetch_all_historical_ohlcv(self, symbol: str, timeframe: str, start_time: int, limit: int = 1000) -> pd.DataFrame:
        """Fetch all historical OHLCV data from start_time to current time."""
        all_data = []
        since = start_time
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                if not ohlcv:
                    break
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                all_data.append(df)
                since = df['timestamp'].iloc[-1] + 1
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                print(f"[ERROR] Failed to fetch historical data for {symbol}: {e}")
                break
        if all_data:
            full_df = pd.concat(all_data)
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], unit='ms')
            full_df.set_index('timestamp', inplace=True)
            return full_df
        return pd.DataFrame([])

    def fetch_recent_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data from exchange."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()


class DatabaseManager:
    """Manages database operations for raw market data."""
    
    def __init__(self, db_path: str = 'data/market_data.db'):
        self.db_path = db_path
        self.ensure_data_directory()
        self.init_database()
    
    def ensure_data_directory(self):
        """Ensure data directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def init_database(self):
        """Initialize database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Raw data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS raw_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            
            # Gaussian channel data table
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
    
    def save_raw_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save raw OHLCV data to database."""
        if df.empty:
            return
        
        # Prepare data for insertion
        df_copy = df.copy()
        df_copy.reset_index(inplace=True)
        df_copy['symbol'] = symbol
        df_copy['timeframe'] = timeframe
        
        # Reorder columns to match database schema
        df_copy = df_copy[['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df_copy.to_sql('raw_data', conn, if_exists='append', index=False)
                rows_inserted = len(df_copy)
                print(f"[INFO] Saved {rows_inserted} raw data records for {symbol} ({timeframe})")
        except sqlite3.IntegrityError:
            # Handle duplicate entries
            print(f"[INFO] Some data already exists for {symbol} ({timeframe}), skipping duplicates")
        except Exception as e:
            print(f"[ERROR] Failed to save raw data for {symbol}: {e}")
    
    def get_last_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """Get the last timestamp for a symbol/timeframe combination."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT MAX(timestamp) FROM raw_data WHERE symbol = ? AND timeframe = ?',
                    (symbol, timeframe)
                )
                result = cursor.fetchone()[0]
                if result:
                    return int(pd.to_datetime(result).timestamp() * 1000)
                return None
        except Exception as e:
            print(f"[ERROR] Failed to get last timestamp: {e}")
            return None



class RawDataCollector:
    """Main class for collecting and storing raw market data."""
    
    def __init__(self, exchange_name: str = 'binance'):
        self.data_fetcher = DataFetcher(exchange_name)
        self.db_manager = DatabaseManager()
    
    def collect_all_historical_data(self, symbol: str, timeframe: str, start_time: int):
        """Collect all historical data for a symbol and timeframe."""
        print(f"\n[FETCHING] All historical data for {symbol} - {timeframe.upper()} from {datetime.fromtimestamp(start_time/1000)}")
        df_full = self.data_fetcher.fetch_all_historical_ohlcv(symbol, timeframe, start_time)
        if df_full.empty:
            print(f"[INFO] No historical data for {symbol} ({timeframe})")
            return
        self.db_manager.save_raw_data(df_full, symbol, timeframe)
        print(f"[SUCCESS] Collected all historical data for {symbol} ({timeframe}) - {len(df_full)} records")

    def collect_recent_data(self, symbol: str, timeframe: str):
        """Collect recent data for a symbol and timeframe."""
        print(f"\n[FETCHING] Recent data for {symbol} - {timeframe.upper()}")
        fetch_since = self.db_manager.get_last_timestamp(symbol, timeframe)
        df_recent = self.data_fetcher.fetch_recent_ohlcv(symbol, timeframe, since=fetch_since)
        if df_recent.empty:
            print(f"[INFO] No new recent data for {symbol} ({timeframe})")
            return
        self.db_manager.save_raw_data(df_recent, symbol, timeframe)
        print(f"[SUCCESS] Collected recent data for {symbol} ({timeframe}) - {len(df_recent)} records")

    def collect_data(self, symbol: str, timeframe: str, start_time: Optional[int] = None):
        """Collect raw data for a symbol and timeframe."""
        print(f"\n[FETCHING] Raw data for {symbol} - {timeframe.upper()}")
        
        # Determine fetch start time
        if start_time is None:
            fetch_since = self.db_manager.get_last_timestamp(symbol, timeframe)
        else:
            fetch_since = start_time
        
        # Fetch data
        df_raw = self.data_fetcher.fetch_recent_ohlcv(symbol, timeframe, since=fetch_since)
        
        if df_raw.empty:
            print(f"[INFO] No new data for {symbol} ({timeframe})")
            return
        
        # Save to database
        self.db_manager.save_raw_data(df_raw, symbol, timeframe)
        print(f"[SUCCESS] Collected {len(df_raw)} records for {symbol} ({timeframe})")


def main():
    """Main function to collect raw data for multiple symbols and timeframes."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
    timeframes = ['4h', '1d']
    
    # Start from January 1, 2021
    start_time = int(datetime(2021, 1, 1).timestamp() * 1000)
    
    collector = RawDataCollector()
    
    for symbol in symbols:
        for timeframe in timeframes:
            collector.collect_data(symbol, timeframe, start_time=start_time)


if __name__ == '__main__':
    main()
