"""
Unified Data Manager for OHLCV data operations.
Handles all database operations for the unified_trading_data.db schema.
"""
import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging


class UnifiedDataManager:
    """Manages OHLCV data operations for the unified trading database."""
    
    def __init__(self, db_path: str = 'data/unified_trading_data.db'):
        self.db_path = db_path
        self.ensure_data_directory()
        self._symbol_cache = {}
        self._timeframe_cache = {}
        self._load_caches()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def ensure_data_directory(self):
        """Ensure data directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _load_caches(self):
        """Load symbol and timeframe caches from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load symbols
                cursor = conn.execute("SELECT id, symbol FROM symbols")
                self._symbol_cache = {symbol: id for id, symbol in cursor.fetchall()}
                
                # Load timeframes
                cursor = conn.execute("SELECT id, timeframe FROM timeframes")
                self._timeframe_cache = {timeframe: id for id, timeframe in cursor.fetchall()}
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to load caches: {e}")
    
    def get_or_create_symbol_id(self, symbol: str) -> int:
        """Get symbol ID, creating if it doesn't exist."""
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO symbols (symbol) VALUES (?)",
                    (symbol,)
                )
                
                # Get the ID (either newly created or existing)
                cursor = conn.execute("SELECT id FROM symbols WHERE symbol = ?", (symbol,))
                symbol_id = cursor.fetchone()[0]
                
                # Update cache
                self._symbol_cache[symbol] = symbol_id
                conn.commit()
                
                return symbol_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get/create symbol ID for {symbol}: {e}")
            raise
    
    def get_or_create_timeframe_id(self, timeframe: str) -> int:
        """Get timeframe ID, creating if it doesn't exist."""
        if timeframe in self._timeframe_cache:
            return self._timeframe_cache[timeframe]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO timeframes (timeframe) VALUES (?)",
                    (timeframe,)
                )
                
                # Get the ID (either newly created or existing)
                cursor = conn.execute("SELECT id FROM timeframes WHERE timeframe = ?", (timeframe,))
                timeframe_id = cursor.fetchone()[0]
                
                # Update cache
                self._timeframe_cache[timeframe] = timeframe_id
                conn.commit()
                
                return timeframe_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get/create timeframe ID for {timeframe}: {e}")
            raise
    
    def save_ohlcv_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, int]:
        """
        Save OHLCV data to the unified database.
        
        Args:
            df: DataFrame with OHLCV data (timestamp as index)
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            
        Returns:
            Dict with 'inserted', 'duplicates', 'errors' counts
        """
        if df.empty:
            return {'inserted': 0, 'duplicates': 0, 'errors': 0}
        
        # Get or create symbol and timeframe IDs
        symbol_id = self.get_or_create_symbol_id(symbol)
        timeframe_id = self.get_or_create_timeframe_id(timeframe)
        
        # Prepare data for insertion
        df_copy = df.copy()
        df_copy.reset_index(inplace=True)
        
        # Validate required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df_copy.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Prepare records for batch insert
        records = []
        for _, row in df_copy.iterrows():
            # Validate data integrity
            if any(pd.isna([row['open'], row['high'], row['low'], row['close'], row['volume']])):
                self.logger.warning(f"Skipping record with NaN values at {row['timestamp']}")
                continue
            
            if row['high'] < row['low'] or row['open'] < 0 or row['close'] < 0:
                self.logger.warning(f"Skipping invalid OHLCV data at {row['timestamp']}")
                continue
            
            # Convert timestamp to ISO string for SQLite
            timestamp_str = row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
            
            records.append((
                symbol_id,
                timeframe_id,
                timestamp_str,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ))
        
        if not records:
            self.logger.warning("No valid records to insert")
            return {'inserted': 0, 'duplicates': 0, 'errors': 0}
        
        # Batch insert with duplicate handling
        inserted_count = 0
        duplicate_count = 0
        error_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN TRANSACTION")
                
                for record in records:
                    try:
                        conn.execute('''
                            INSERT INTO ohlcv_data 
                            (symbol_id, timeframe_id, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', record)
                        inserted_count += 1
                    except sqlite3.IntegrityError:
                        # Duplicate entry (violates unique constraint)
                        duplicate_count += 1
                    except sqlite3.Error as e:
                        self.logger.error(f"Error inserting record: {e}")
                        error_count += 1
                
                conn.commit()
                
                self.logger.info(
                    f"OHLCV data for {symbol} ({timeframe}): "
                    f"{inserted_count} inserted, {duplicate_count} duplicates, {error_count} errors"
                )
                
        except sqlite3.Error as e:
            self.logger.error(f"Database transaction failed: {e}")
            raise
        
        return {
            'inserted': inserted_count,
            'duplicates': duplicate_count,
            'errors': error_count
        }
    
    def get_last_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        """Get the last timestamp for a symbol/timeframe combination."""
        try:
            symbol_id = self._symbol_cache.get(symbol)
            timeframe_id = self._timeframe_cache.get(timeframe)
            
            if symbol_id is None or timeframe_id is None:
                return None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT MAX(timestamp) FROM ohlcv_data 
                    WHERE symbol_id = ? AND timeframe_id = ?
                ''', (symbol_id, timeframe_id))
                
                result = cursor.fetchone()[0]
                if result:
                    return int(pd.to_datetime(result).timestamp() * 1000)
                return None
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get last timestamp for {symbol} ({timeframe}): {e}")
            return None
    
    def get_data_gaps(self, symbol: str, timeframe: str, start_time: int, end_time: int) -> List[Tuple[int, int]]:
        """
        Identify gaps in data for a given symbol/timeframe within a time range.
        
        Returns:
            List of (gap_start_timestamp, gap_end_timestamp) tuples
        """
        try:
            symbol_id = self._symbol_cache.get(symbol)
            timeframe_id = self._timeframe_cache.get(timeframe)
            
            if symbol_id is None or timeframe_id is None:
                return [(start_time, end_time)]  # No data exists, entire range is a gap
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT timestamp FROM ohlcv_data 
                    WHERE symbol_id = ? AND timeframe_id = ? 
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                ''', (symbol_id, timeframe_id, 
                     pd.to_datetime(start_time, unit='ms').isoformat(),
                     pd.to_datetime(end_time, unit='ms').isoformat()))
                
                timestamps = [int(pd.to_datetime(row[0]).timestamp() * 1000) for row in cursor.fetchall()]
                
                if not timestamps:
                    return [(start_time, end_time)]
                
                # Calculate timeframe interval in milliseconds
                timeframe_ms = self._timeframe_to_ms(timeframe)
                gaps = []
                
                # Check gap before first timestamp
                if timestamps[0] > start_time + timeframe_ms:
                    gaps.append((start_time, timestamps[0] - timeframe_ms))
                
                # Check gaps between timestamps
                for i in range(len(timestamps) - 1):
                    expected_next = timestamps[i] + timeframe_ms
                    if timestamps[i + 1] > expected_next:
                        gaps.append((expected_next, timestamps[i + 1] - timeframe_ms))
                
                # Check gap after last timestamp
                if timestamps[-1] < end_time - timeframe_ms:
                    gaps.append((timestamps[-1] + timeframe_ms, end_time))
                
                return gaps
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get data gaps for {symbol} ({timeframe}): {e}")
            return [(start_time, end_time)]
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        multipliers = {
            'm': 60 * 1000,          # minutes
            'h': 60 * 60 * 1000,     # hours
            'd': 24 * 60 * 60 * 1000, # days
            'w': 7 * 24 * 60 * 60 * 1000  # weeks
        }
        
        if timeframe[-1] in multipliers:
            return int(timeframe[:-1]) * multipliers[timeframe[-1]]
        else:
            # Default fallback for unknown formats
            return 60 * 60 * 1000  # 1 hour
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get a summary of all OHLCV data in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT 
                        s.symbol,
                        t.timeframe,
                        COUNT(*) as record_count,
                        MIN(o.timestamp) as earliest,
                        MAX(o.timestamp) as latest
                    FROM ohlcv_data o
                    JOIN symbols s ON o.symbol_id = s.id
                    JOIN timeframes t ON o.timeframe_id = t.id
                    GROUP BY s.symbol, t.timeframe
                    ORDER BY s.symbol, t.timeframe
                '''
                
                return pd.read_sql_query(query, conn)
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get data summary: {e}")
            return pd.DataFrame()
    
    def validate_data_integrity(self) -> Dict[str, int]:
        """Validate OHLCV data integrity and return statistics."""
        issues = {
            'invalid_ohlc': 0,
            'negative_prices': 0,
            'zero_volume': 0,
            'duplicate_timestamps': 0
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check for invalid OHLC relationships (high < low, etc.)
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM ohlcv_data 
                    WHERE high < low OR open < 0 OR high < 0 OR low < 0 OR close < 0
                ''')
                issues['invalid_ohlc'] = cursor.fetchone()[0]
                
                # Check for negative prices
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM ohlcv_data 
                    WHERE open < 0 OR high < 0 OR low < 0 OR close < 0
                ''')
                issues['negative_prices'] = cursor.fetchone()[0]
                
                # Check for zero volume (might be valid for some assets)
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM ohlcv_data WHERE volume = 0
                ''')
                issues['zero_volume'] = cursor.fetchone()[0]
                
                self.logger.info(f"Data integrity check completed: {issues}")
                
        except sqlite3.Error as e:
            self.logger.error(f"Data integrity validation failed: {e}")
        
        return issues
