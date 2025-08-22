"""
Data Manager for OHLCV data operations.
Handles all database operations for the per-symbol SQLite schema (e.g., data/trading_data_BTC.db).
"""
import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging


class DataManager:
    """Manages OHLCV data operations for the per-symbol trading database."""
    
    def __init__(self, db_path: str = 'data/trading_data_BTC.db'):
        import os
        # Resolve to absolute path relative to project root (parent of backend)
        if not os.path.isabs(db_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            db_path = os.path.abspath(os.path.join(project_root, db_path))
        self.db_path = db_path
        self.ensure_data_directory()
        self._symbol_cache = {}
        self._timeframe_cache = {}
        
        # Setup logging early so exception handlers can use it
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Ensure schema exists before any DB operations
        self._ensure_schema()
        
        # Load caches after logger is ready
        self._load_caches()
    
    def _ensure_schema(self):
        """Create required tables and indexes if they do not exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS timeframes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timeframe TEXT UNIQUE NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_id INTEGER NOT NULL,
                    timeframe_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                    FOREIGN KEY (timeframe_id) REFERENCES timeframes(id),
                    UNIQUE(symbol_id, timeframe_id, timestamp)
                );
                """)
                # Indicator tables (subset used by indicators)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS sma_indicator (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ohlcv_id INTEGER NOT NULL,
                    sma_50 REAL,
                    sma_200 REAL,
                    sma_ratio REAL,
                    price_vs_sma50 REAL,
                    price_vs_sma200 REAL,
                    trend_strength REAL,
                    sma_signal TEXT,
                    cross_signal TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
                    UNIQUE(ohlcv_id)
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS macd_indicator (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ohlcv_id INTEGER NOT NULL,
                    ema_12 REAL,
                    ema_26 REAL,
                    macd_line REAL,
                    signal_line REAL,
                    histogram REAL,
                    macd_signal TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
                    UNIQUE(ohlcv_id)
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS bollinger_bands_indicator (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ohlcv_id INTEGER NOT NULL,
                    bb_upper REAL,
                    bb_lower REAL,
                    bb_middle REAL,
                    bb_width REAL,
                    bb_percent REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
                    UNIQUE(ohlcv_id)
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS rsi_indicator (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ohlcv_id INTEGER NOT NULL,
                    rsi REAL,
                    rsi_sma_5 REAL,
                    rsi_sma_10 REAL,
                    overbought BOOLEAN,
                    oversold BOOLEAN,
                    trend_strength TEXT,
                    divergence_signal TEXT,
                    momentum_shift BOOLEAN,
                    support_resistance TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
                    UNIQUE(ohlcv_id)
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS ichimoku_indicator (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ohlcv_id INTEGER NOT NULL,
                    tenkan_sen REAL,
                    kijun_sen REAL,
                    senkou_span_a REAL,
                    senkou_span_b REAL,
                    chikou_span REAL,
                    cloud_color TEXT,
                    ichimoku_signal TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
                    UNIQUE(ohlcv_id)
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS parabolic_sar_indicator (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ohlcv_id INTEGER NOT NULL,
                    parabolic_sar REAL,
                    trend TEXT,
                    reversal_signal BOOLEAN,
                    signal_strength REAL,
                    acceleration_factor REAL,
                    sar_signal TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
                    UNIQUE(ohlcv_id)
                );
                """)
                
                # Ensure parabolic_sar_indicator has expected columns
                try:
                    cursor = conn.execute("PRAGMA table_info(parabolic_sar_indicator)")
                    existing_cols = {row[1] for row in cursor.fetchall()}
                    desired_cols = {
                        ('acceleration_factor', 'REAL'),
                        ('sar_signal', 'TEXT')
                    }
                    for col_name, col_type in desired_cols:
                        if col_name not in existing_cols:
                            conn.execute(f"ALTER TABLE parabolic_sar_indicator ADD COLUMN {col_name} {col_type}")
                except sqlite3.Error as e:
                    self.logger.warning(f"Could not extend parabolic_sar_indicator schema: {e}")
                
                conn.execute("""
                CREATE TABLE IF NOT EXISTS fibonacci_retracement_indicator (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ohlcv_id INTEGER NOT NULL,
                    level_23_6 REAL,
                    level_38_2 REAL,
                    level_50_0 REAL,
                    level_61_8 REAL,
                    level_76_4 REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
                    UNIQUE(ohlcv_id)
                );
                """)
                
                # Ensure fibonacci_retracement_indicator has extended columns used by the indicator module
                try:
                    cursor = conn.execute("PRAGMA table_info(fibonacci_retracement_indicator)")
                    existing_cols = {row[1] for row in cursor.fetchall()}
                    desired_cols = {
                        ('level_0', 'REAL'),
                        ('level_78_6', 'REAL'),
                        ('level_100', 'REAL'),
                        ('trend_direction', 'TEXT'),
                        ('nearest_fib_level', 'REAL'),
                        ('fib_signal', 'TEXT'),
                        ('support_resistance', 'TEXT')
                    }
                    for col_name, col_type in desired_cols:
                        if col_name not in existing_cols:
                            conn.execute(f"ALTER TABLE fibonacci_retracement_indicator ADD COLUMN {col_name} {col_type}")
                except sqlite3.Error as e:
                    self.logger.warning(f"Could not extend fibonacci_retracement_indicator schema: {e}")
                
                conn.execute("""
                CREATE TABLE IF NOT EXISTS gaussian_channel_indicator (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ohlcv_id INTEGER NOT NULL,
                    gc_upper REAL,
                    gc_lower REAL,
                    gc_middle REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
                    UNIQUE(ohlcv_id)
                );
                """)
                # Indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_timestamp ON ohlcv_data(symbol_id, timeframe_id, timestamp);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv_data(timestamp);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_data(symbol_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_timeframe ON ohlcv_data(timeframe_id);")
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to ensure schema: {e}")

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
        Save OHLCV data to the database.
        
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
