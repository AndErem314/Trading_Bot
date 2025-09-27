"""
OHLCV Data Fetcher for Cryptocurrency Exchanges

This module provides a robust class for fetching OHLCV (Open, High, Low, Close, Volume) data
from various cryptocurrency exchanges using the CCXT library.

Features:
- Support for multiple exchanges (Binance, Bybit, etc.)
- Focused timeframe support (15m, 1h, 4h, 1d)
- Data caching to minimize API calls
- Rate limiting and error handling
- Clean pandas DataFrame output with datetime index
"""

import ccxt
import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List, Any
from datetime import datetime, timedelta
import logging
import time
import pickle
import os
from pathlib import Path
import hashlib

# Configure logging
logger = logging.getLogger(__name__)


class OHLCVDataFetcher:
    """
    A class for fetching and managing OHLCV data from cryptocurrency exchanges.
    
    This class provides a unified interface for fetching historical candlestick data
    from various exchanges, with built-in caching, rate limiting, and error handling.
    
    Attributes:
        exchange_id (str): The ID of the exchange (e.g., 'binance', 'bybit')
        exchange: CCXT exchange instance
        cache_dir (Path): Directory for storing cached data
        rate_limit_delay (float): Delay between API calls in seconds
    """
    
    # Supported exchanges
    SUPPORTED_EXCHANGES = ['binance', 'bybit', 'okx', 'kucoin', 'kraken', 'coinbase']
    
    # Supported timeframes (focused set)
    VALID_TIMEFRAMES = ['15m', '1h', '4h', '1d']
    
    def __init__(self, exchange_id: str = 'binance', cache_dir: Optional[str] = None, 
                 api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize the OHLCV Data Fetcher.
        
        Args:
            exchange_id: The exchange to use (default: 'binance')
            cache_dir: Directory for caching data (default: './cache/ohlcv')
            api_key: API key for private endpoints (optional)
            api_secret: API secret for private endpoints (optional)
            
        Raises:
            ValueError: If the exchange is not supported
        """
        if exchange_id.lower() not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"Exchange '{exchange_id}' not supported. "
                           f"Supported exchanges: {', '.join(self.SUPPORTED_EXCHANGES)}")
        
        self.exchange_id = exchange_id.lower()
        
        # Initialize exchange
        exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,  # Enable built-in rate limiting
            'rateLimit': 1200,  # Default rate limit (can be adjusted per exchange)
        })
        
        # Set up cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./cache/ohlcv')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.rate_limit_delay = 0.1  # Default delay between requests
        self.last_request_time = 0
        
        logger.info(f"Initialized OHLCVDataFetcher for {self.exchange_id}")
    
    def fetch_data(self, symbol: str, timeframe: str, limit: int = 300, 
                   start_time: Optional[datetime] = None, 
                   end_time: Optional[datetime] = None,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe (must be one of: '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch (default: 300)
            start_time: Start time for historical data (optional)
            end_time: End time for historical data (optional)
            use_cache: Whether to use cached data if available (default: True)
            
        Returns:
            pd.DataFrame: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
                         and datetime index
            
        Raises:
            ValueError: If the timeframe is invalid or symbol is not found
            Exception: If there's an error fetching data from the exchange
        """
        # Validate inputs
        self._validate_inputs(symbol, timeframe, limit)
        
        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(symbol, timeframe, limit, start_time, end_time)
            if cached_data is not None:
                logger.info(f"Loaded {len(cached_data)} candles from cache for {symbol} {timeframe}")
                return cached_data
        
        # Fetch from exchange
        try:
            data = self._fetch_from_exchange(symbol, timeframe, limit, start_time, end_time)
            
            # Cache the data
            if use_cache and len(data) > 0:
                self._save_to_cache(data, symbol, timeframe, limit, start_time, end_time)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {str(e)}")
            raise
    
    def _validate_inputs(self, symbol: str, timeframe: str, limit: int) -> None:
        """Validate input parameters."""
        if timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe '{timeframe}'. "
                           f"Valid timeframes: {', '.join(self.VALID_TIMEFRAMES)}")
        
        if limit <= 0:
            raise ValueError("Limit must be a positive integer")
        
        # Check if symbol exists on exchange
        if not self.exchange.has['fetchOHLCV']:
            raise ValueError(f"Exchange {self.exchange_id} does not support OHLCV data")
    
    def _fetch_from_exchange(self, symbol: str, timeframe: str, limit: int,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch data from the exchange with rate limiting and error handling.
        
        Returns:
            pd.DataFrame: OHLCV data
        """
        # Rate limiting
        self._apply_rate_limit()
        
        try:
            # Convert datetime to milliseconds timestamp
            since = int(start_time.timestamp() * 1000) if start_time else None
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            if not ohlcv:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter by end_time if specified
            if end_time:
                df = df[df.index <= end_time]
            
            # Validate the data
            df = self.validate_data(df)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe} from {self.exchange_id}")
            
            return df
            
        except ccxt.BaseError as e:
            logger.error(f"Exchange error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def validate_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.
        
        Args:
            dataframe: Raw OHLCV DataFrame
            
        Returns:
            pd.DataFrame: Validated and cleaned DataFrame
        """
        if dataframe.empty:
            return dataframe
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove duplicates
        initial_len = len(dataframe)
        dataframe = dataframe[~dataframe.index.duplicated(keep='last')]
        if len(dataframe) < initial_len:
            logger.warning(f"Removed {initial_len - len(dataframe)} duplicate rows")
        
        # Sort by timestamp
        dataframe = dataframe.sort_index()
        
        # Check for NaN values
        nan_counts = dataframe[required_columns].isna().sum()
        if nan_counts.any():
            logger.warning(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
            # Forward fill NaN values
            dataframe = dataframe.fillna(method='ffill')
            # If still NaN (at the beginning), use backward fill
            dataframe = dataframe.fillna(method='bfill')
        
        # Validate OHLC relationships
        invalid_candles = (
            (dataframe['high'] < dataframe['low']) |
            (dataframe['high'] < dataframe['open']) |
            (dataframe['high'] < dataframe['close']) |
            (dataframe['low'] > dataframe['open']) |
            (dataframe['low'] > dataframe['close'])
        )
        
        if invalid_candles.any():
            logger.warning(f"Found {invalid_candles.sum()} invalid candles, fixing...")
            # Fix invalid candles
            dataframe.loc[invalid_candles, 'high'] = dataframe.loc[invalid_candles, 
                                                                   ['open', 'high', 'low', 'close']].max(axis=1)
            dataframe.loc[invalid_candles, 'low'] = dataframe.loc[invalid_candles, 
                                                                  ['open', 'high', 'low', 'close']].min(axis=1)
        
        # Ensure positive volume
        negative_volume = dataframe['volume'] < 0
        if negative_volume.any():
            logger.warning(f"Found {negative_volume.sum()} negative volume values, setting to 0")
            dataframe.loc[negative_volume, 'volume'] = 0
        
        return dataframe
    
    def _apply_rate_limit(self) -> None:
        """Apply rate limiting between API calls."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_key(self, symbol: str, timeframe: str, limit: int,
                      start_time: Optional[datetime], end_time: Optional[datetime]) -> str:
        """Generate a unique cache key for the data request."""
        key_parts = [
            self.exchange_id,
            symbol.replace('/', '_'),
            timeframe,
            str(limit),
            str(start_time) if start_time else 'None',
            str(end_time) if end_time else 'None'
        ]
        key_string = '_'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_from_cache(self, symbol: str, timeframe: str, limit: int,
                        start_time: Optional[datetime], end_time: Optional[datetime]) -> Optional[pd.DataFrame]:
        """Load data from cache if available and fresh."""
        cache_key = self._get_cache_key(symbol, timeframe, limit, start_time, end_time)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check cache age based on timeframe
        cache_age_limits = {
            '15m': 300,    # 5 minutes
            '1h': 1800,    # 30 minutes
            '4h': 7200,    # 2 hours
            '1d': 21600    # 6 hours
        }
        
        cache_age = time.time() - cache_file.stat().st_mtime
        max_age = cache_age_limits.get(timeframe, 300)
        
        if cache_age > max_age and not start_time:  # Only expire recent data
            logger.debug(f"Cache expired for {symbol} {timeframe}")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, symbol: str, timeframe: str, limit: int,
                      start_time: Optional[datetime], end_time: Optional[datetime]) -> None:
        """Save data to cache."""
        cache_key = self._get_cache_key(symbol, timeframe, limit, start_time, end_time)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached data for {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols from the exchange.
        
        Returns:
            List of available symbol strings
        """
        try:
            self.exchange.load_markets()
            return list(self.exchange.symbols)
        except Exception as e:
            logger.error(f"Error loading markets: {str(e)}")
            return []
    
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            symbol: Specific symbol to clear (optional)
            timeframe: Specific timeframe to clear (optional)
        """
        if not symbol and not timeframe:
            # Clear all cache
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
            logger.info("Cleared all cache")
        else:
            # Clear specific cache files
            pattern = '*'
            if symbol:
                pattern += f"*{symbol.replace('/', '_')}*"
            if timeframe:
                pattern += f"*{timeframe}*"
            
            for cache_file in self.cache_dir.glob(f"{pattern}.pkl"):
                cache_file.unlink()
            logger.info(f"Cleared cache for {symbol or 'all symbols'} {timeframe or 'all timeframes'}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the data fetcher
    fetcher = OHLCVDataFetcher(exchange_id='binance')
    
    # Example 1: Fetch BTC/USDT 1h data (last 300 candles)
    print("Example 1: Fetching BTC/USDT 1h data...")
    btc_1h_data = fetcher.fetch_data(
        symbol='BTC/USDT',
        timeframe='1h',
        limit=300
    )
    
    print(f"\nFetched {len(btc_1h_data)} candles")
    print(f"Date range: {btc_1h_data.index[0]} to {btc_1h_data.index[-1]}")
    print(f"\nLast 5 candles:")
    print(btc_1h_data.tail())
    
    # Example 2: Fetch ETH/USDT 4h data (last 200 candles)
    print("\n" + "="*60)
    print("Example 2: Fetching ETH/USDT 4h data...")
    eth_4h_data = fetcher.fetch_data(
        symbol='ETH/USDT',
        timeframe='4h',
        limit=200
    )
    
    print(f"\nFetched {len(eth_4h_data)} candles")
    print(f"Date range: {eth_4h_data.index[0]} to {eth_4h_data.index[-1]}")
    
    # Example 3: Fetch BTC/USDT daily data
    print("\n" + "="*60)
    print("Example 3: Fetching BTC/USDT daily data...")
    btc_daily_data = fetcher.fetch_data(
        symbol='BTC/USDT',
        timeframe='1d',
        limit=100
    )
    
    print(f"\nFetched {len(btc_daily_data)} candles")
    print(f"\nData statistics for the last candle:")
    last_candle = btc_daily_data.iloc[-1]
    print(f"- Date:   {btc_daily_data.index[-1]}")
    print(f"- Open:   ${last_candle['open']:,.2f}")
    print(f"- High:   ${last_candle['high']:,.2f}")
    print(f"- Low:    ${last_candle['low']:,.2f}")
    print(f"- Close:  ${last_candle['close']:,.2f}")
    print(f"- Volume: {last_candle['volume']:,.2f}")
    
    # Example 4: Fetch 15m data with custom limit
    print("\n" + "="*60)
    print("Example 4: Fetching BTC/USDT 15m data...")
    btc_15m_data = fetcher.fetch_data(
        symbol='BTC/USDT',
        timeframe='15m',
        limit=100
    )
    
    print(f"\nFetched {len(btc_15m_data)} candles")
    print(f"Time span: {(btc_15m_data.index[-1] - btc_15m_data.index[0]).total_seconds() / 3600:.1f} hours")
    
    # Example 5: Using different exchange (Bybit)
    print("\n" + "="*60)
    print("Example 5: Using Bybit exchange...")
    bybit_fetcher = OHLCVDataFetcher(exchange_id='bybit')
    
    btc_bybit_data = bybit_fetcher.fetch_data(
        symbol='BTC/USDT',
        timeframe='1h',
        limit=50
    )
    
    print(f"\nFetched {len(btc_bybit_data)} candles from Bybit")
    print(f"Latest close price: ${btc_bybit_data['close'].iloc[-1]:,.2f}")
    
    # Demonstrate cache usage
    print("\n" + "="*60)
    print("Example 6: Demonstrating cache...")
    
    # This will use cache since we already fetched it
    print("Fetching BTC/USDT 1h data again (should use cache)...")
    cached_data = fetcher.fetch_data(
        symbol='BTC/USDT',
        timeframe='1h',
        limit=300
    )
    print(f"Data fetched (from cache): {len(cached_data)} candles")