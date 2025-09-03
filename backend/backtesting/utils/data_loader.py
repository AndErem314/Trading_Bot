"""
Data Loader Utility

This module handles loading historical market data from SQLite databases
for backtesting purposes.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads historical market data from SQLite databases
    """
    
    def __init__(self, config: Dict[str, any]):
        """
        Initialize data loader with configuration
        
        Args:
            config: Data configuration from backtest config
        """
        self.config = config
        self.data_dir = "data"  # Base data directory
        
    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load market data for a specific symbol and timeframe
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '4h', '1h', '1d')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        # Convert symbol to database format (BTC/USDT -> BTC)
        db_symbol = symbol.split('/')[0]  # Extract base currency
        
        # Construct database path with correct naming convention
        db_path = os.path.join(self.data_dir, f"trading_data_{db_symbol}.db")
        
        if not os.path.exists(db_path):
            logger.error(f"Database not found: {db_path}")
            return pd.DataFrame()
        
        try:
            # Connect to database
            conn = sqlite3.connect(db_path)
            
            # Build query for the actual database schema
            query = """
            SELECT 
                o.timestamp,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume
            FROM ohlcv_data o
            JOIN timeframes t ON o.timeframe_id = t.id
            JOIN symbols s ON o.symbol_id = s.id
            WHERE t.timeframe = ?
            """
            
            # Add date filters if provided
            conditions = []
            params = [timeframe]  # Start with timeframe parameter
            
            if start_date:
                start_timestamp = pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%S')
                conditions.append("o.timestamp >= ?")
                params.append(start_timestamp)
                
            if end_date:
                end_timestamp = pd.to_datetime(end_date).strftime('%Y-%m-%dT%H:%M:%S')
                conditions.append("o.timestamp <= ?")
                params.append(end_timestamp)
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
                
            query += " ORDER BY o.timestamp"
            
            # Load data
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                logger.warning(f"No data found for {symbol} {timeframe} in specified range")
                return pd.DataFrame()
            
            # Convert timestamp to datetime (ISO format in database)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
            df.set_index('timestamp', inplace=True)
            
            # Ensure numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any NaN values
            df.dropna(inplace=True)
            
            logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def load_multiple_timeframes(
        self,
        symbol: str,
        timeframes: list,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple timeframes
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        data = {}
        
        for timeframe in timeframes:
            df = self.load_data(symbol, timeframe, start_date, end_date)
            if not df.empty:
                data[timeframe] = df
                
        return data
    
    def resample_data(
        self,
        data: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample data to a different timeframe
        
        Args:
            data: Original OHLCV data
            target_timeframe: Target timeframe (e.g., '4h', '1d')
            
        Returns:
            Resampled DataFrame
        """
        # Map timeframe to pandas frequency
        timeframe_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1w': '1W'
        }
        
        if target_timeframe not in timeframe_map:
            logger.error(f"Unknown timeframe: {target_timeframe}")
            return data
            
        freq = timeframe_map[target_timeframe]
        
        # Resample
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove NaN values
        resampled.dropna(inplace=True)
        
        return resampled
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicators to the data
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with added indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data is suitable for backtesting
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            True if data is valid
        """
        if data.empty:
            logger.error("Data is empty")
            return False
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False
            
        # Check for NaN values
        if data[required_columns].isna().any().any():
            logger.warning("Data contains NaN values")
            
        # Check for zero or negative prices
        price_columns = ['open', 'high', 'low', 'close']
        if (data[price_columns] <= 0).any().any():
            logger.error("Data contains zero or negative prices")
            return False
            
        # Check high/low consistency
        if (data['high'] < data['low']).any():
            logger.error("High prices are less than low prices")
            return False
            
        # Check OHLC consistency
        if ((data['high'] < data['open']) | (data['high'] < data['close'])).any():
            logger.error("High prices are less than open or close")
            return False
            
        if ((data['low'] > data['open']) | (data['low'] > data['close'])).any():
            logger.error("Low prices are greater than open or close")
            return False
            
        return True
