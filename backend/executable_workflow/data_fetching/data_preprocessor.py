"""
Data Preprocessor for OHLCV Data

This module provides comprehensive preprocessing capabilities for OHLCV data,
including gap detection, outlier removal, data validation, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A class for preprocessing and validating OHLCV data.
    
    This class provides methods for detecting gaps, handling missing data,
    removing outliers, and ensuring data integrity for trading applications.
    
    Attributes:
        outlier_zscore_threshold (float): Z-score threshold for outlier detection
        volume_outlier_quantile (float): Quantile threshold for volume outliers
        max_price_change_pct (float): Maximum allowed price change percentage
    """
    
    # Timeframe to timedelta mapping
    TIMEFRAME_DELTAS = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '2h': timedelta(hours=2),
        '4h': timedelta(hours=4),
        '6h': timedelta(hours=6),
        '8h': timedelta(hours=8),
        '12h': timedelta(hours=12),
        '1d': timedelta(days=1),
        '1w': timedelta(weeks=1),
    }
    
    def __init__(self, 
                 outlier_zscore_threshold: float = 3.0,
                 volume_outlier_quantile: float = 0.99,
                 max_price_change_pct: float = 20.0):
        """
        Initialize the DataPreprocessor.
        
        Args:
            outlier_zscore_threshold: Z-score threshold for price outlier detection
            volume_outlier_quantile: Upper quantile for volume outlier detection
            max_price_change_pct: Maximum allowed percentage price change between candles
        """
        self.outlier_zscore_threshold = outlier_zscore_threshold
        self.volume_outlier_quantile = volume_outlier_quantile
        self.max_price_change_pct = max_price_change_pct
    
    def clean_data(self, df: pd.DataFrame, timeframe: str = '1h') -> pd.DataFrame:
        """
        Main method to clean and preprocess OHLCV data.
        
        This method performs a complete data cleaning pipeline including:
        - Validation of data structure
        - Detection and filling of gaps
        - Outlier removal
        - Data integrity validation
        
        Args:
            df: DataFrame with OHLCV data and datetime index
            timeframe: The timeframe of the data (e.g., '15m', '1h', '4h', '1d')
            
        Returns:
            pd.DataFrame: Cleaned and validated OHLCV data
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Step 1: Ensure proper datetime index
        cleaned_df = self._ensure_datetime_index(cleaned_df)
        
        # Step 2: Sort by index (timestamp)
        cleaned_df = cleaned_df.sort_index()
        
        # Step 3: Remove duplicate timestamps
        initial_len = len(cleaned_df)
        cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='last')]
        if len(cleaned_df) < initial_len:
            logger.info(f"Removed {initial_len - len(cleaned_df)} duplicate timestamps")
        
        # Step 4: Validate OHLCV integrity
        cleaned_df = self.validate_ohlcv_integrity(cleaned_df)
        
        # Step 5: Detect and fill gaps
        gaps = self.detect_gaps(cleaned_df, timeframe)
        if gaps:
            logger.info(f"Found {len(gaps)} gaps in data")
            cleaned_df = self.fill_missing_data(cleaned_df, timeframe)
        
        # Step 6: Remove outliers
        cleaned_df = self._remove_outliers(cleaned_df)
        
        # Step 7: Add calculated features
        cleaned_df = self._add_calculated_features(cleaned_df)
        
        # Final validation
        if not self._validate_final_data(cleaned_df):
            logger.warning("Final data validation failed, returning original data")
            return df
        
        logger.info(f"Data preprocessing complete. Final shape: {cleaned_df.shape}")
        return cleaned_df
    
    def detect_gaps(self, df: pd.DataFrame, timeframe: str) -> List[Dict[str, Union[datetime, int]]]:
        """
        Detect gaps in the time series data.
        
        Args:
            df: DataFrame with datetime index
            timeframe: The expected timeframe (e.g., '15m', '1h', '4h', '1d')
            
        Returns:
            List of dictionaries containing gap information:
            - 'start': Start time of the gap
            - 'end': End time of the gap
            - 'missing_candles': Number of missing candles
        """
        if timeframe not in self.TIMEFRAME_DELTAS:
            logger.warning(f"Unknown timeframe '{timeframe}', using 1h as default")
            timeframe = '1h'
        
        expected_delta = self.TIMEFRAME_DELTAS[timeframe]
        gaps = []
        
        for i in range(1, len(df)):
            current_time = df.index[i]
            previous_time = df.index[i-1]
            actual_delta = current_time - previous_time
            
            # Allow for small variations (up to 1 second)
            if actual_delta > expected_delta + timedelta(seconds=1):
                missing_candles = int(actual_delta / expected_delta) - 1
                gaps.append({
                    'start': previous_time,
                    'end': current_time,
                    'missing_candles': missing_candles
                })
        
        return gaps
    
    def fill_missing_data(self, df: pd.DataFrame, timeframe: str = '1h') -> pd.DataFrame:
        """
        Fill missing candles in the data.
        
        This method creates a complete time series and fills missing values
        using forward fill for prices and zero for volume.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: The timeframe of the data
            
        Returns:
            pd.DataFrame: DataFrame with filled missing data
        """
        if timeframe not in self.TIMEFRAME_DELTAS:
            logger.warning(f"Unknown timeframe '{timeframe}', using 1h as default")
            timeframe = '1h'
        
        # Create a complete date range
        freq_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H',
            '12h': '12H', '1d': '1D', '1w': '1W'
        }
        
        freq = freq_map.get(timeframe, '1H')
        complete_index = pd.date_range(
            start=df.index[0],
            end=df.index[-1],
            freq=freq
        )
        
        # Reindex to include all timestamps
        filled_df = df.reindex(complete_index)
        
        # Count missing rows
        missing_count = filled_df.isnull().any(axis=1).sum()
        if missing_count > 0:
            logger.info(f"Filling {missing_count} missing candles")
            
            # Forward fill price data
            for col in ['open', 'high', 'low', 'close']:
                if col in filled_df.columns:
                    filled_df[col] = filled_df[col].fillna(method='ffill')
            
            # Fill volume with 0 for missing candles
            if 'volume' in filled_df.columns:
                filled_df['volume'] = filled_df['volume'].fillna(0)
        
        return filled_df
    
    def validate_ohlcv_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix OHLCV data integrity.
        
        This method ensures that:
        - High >= Low
        - High >= Open and Close
        - Low <= Open and Close
        - All values are positive
        - Volume is non-negative
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with corrected OHLCV values
        """
        df = df.copy()
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Fix negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            negative_mask = df[col] < 0
            if negative_mask.any():
                logger.warning(f"Found {negative_mask.sum()} negative values in '{col}', setting to NaN")
                df.loc[negative_mask, col] = np.nan
        
        # Fix OHLC relationships
        invalid_high_low = df['high'] < df['low']
        if invalid_high_low.any():
            logger.warning(f"Found {invalid_high_low.sum()} candles where high < low, swapping values")
            df.loc[invalid_high_low, ['high', 'low']] = df.loc[invalid_high_low, ['low', 'high']].values
        
        # Ensure high is the maximum of OHLC
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        
        # Ensure low is the minimum of OHLC
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Fix negative volume
        negative_volume = df['volume'] < 0
        if negative_volume.any():
            logger.warning(f"Found {negative_volume.sum()} negative volume values, setting to 0")
            df.loc[negative_volume, 'volume'] = 0
        
        # Handle NaN values
        nan_count = df[required_columns].isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values, forward filling")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has a proper datetime index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert the index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                # If there's a 'timestamp' or 'datetime' column, use it
                for col in ['timestamp', 'datetime', 'date']:
                    if col in df.columns:
                        df.index = pd.to_datetime(df[col])
                        df = df.drop(columns=[col])
                        break
                else:
                    raise ValueError(f"Cannot convert index to datetime: {e}")
        
        # Ensure timezone-aware datetime index is in UTC
        if df.index.tz is not None and df.index.tz != 'UTC':
            df.index = df.index.tz_convert('UTC')
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from OHLCV data.
        
        Uses multiple methods:
        - Z-score for price outliers
        - Percentage change threshold
        - Volume quantile threshold
        """
        df = df.copy()
        initial_len = len(df)
        
        # Method 1: Z-score for prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = z_scores > self.outlier_zscore_threshold
            if outliers.any():
                logger.info(f"Found {outliers.sum()} outliers in '{col}' using z-score method")
                df.loc[outliers, col] = np.nan
        
        # Method 2: Percentage change threshold
        for col in price_columns:
            pct_change = df[col].pct_change().abs()
            extreme_changes = pct_change > (self.max_price_change_pct / 100)
            if extreme_changes.any():
                logger.info(f"Found {extreme_changes.sum()} extreme price changes in '{col}'")
                # Mark the changed value as outlier, not the base
                df.loc[extreme_changes.shift(-1).fillna(False), col] = np.nan
        
        # Method 3: Volume outliers
        if 'volume' in df.columns:
            volume_threshold = df['volume'].quantile(self.volume_outlier_quantile)
            volume_outliers = df['volume'] > volume_threshold * 10  # 10x the 99th percentile
            if volume_outliers.any():
                logger.info(f"Found {volume_outliers.sum()} volume outliers")
                df.loc[volume_outliers, 'volume'] = volume_threshold
        
        # Fill NaN values created by outlier removal
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        final_len = len(df)
        if final_len < initial_len:
            logger.info(f"Removed {initial_len - final_len} rows due to outliers")
        
        return df
    
    def _add_calculated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated features to the DataFrame.
        
        Features added:
        - True Range (TR)
        - Average True Range (ATR)
        - Volume Profile indicators
        - Price spread
        - Candle body percentage
        """
        df = df.copy()
        
        # True Range
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Average True Range (14-period)
        df['atr_14'] = df['true_range'].rolling(window=14, min_periods=1).mean()
        
        # Volume Profile
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        # Price spread
        df['price_spread'] = (df['high'] - df['low']) / df['low'] * 100
        
        # Candle body percentage
        df['candle_body_pct'] = abs(df['close'] - df['open']) / df['open'] * 100
        
        # Trend indicator
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
        
        # Clean up temporary columns
        df = df.drop(columns=['high_low', 'high_close', 'low_close'], errors='ignore')
        
        logger.info("Added calculated features to DataFrame")
        return df
    
    def _validate_final_data(self, df: pd.DataFrame) -> bool:
        """
        Perform final validation checks on the preprocessed data.
        
        Returns:
            bool: True if data passes all validations, False otherwise
        """
        # Check if DataFrame is empty
        if df.empty:
            logger.error("DataFrame is empty after preprocessing")
            return False
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error("Missing required OHLCV columns")
            return False
        
        # Check for NaN values
        if df[required_columns].isnull().any().any():
            logger.error("NaN values found in OHLCV data after preprocessing")
            return False
        
        # Check data types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"Column '{col}' is not numeric")
                return False
        
        # Check index is sorted
        if not df.index.is_monotonic_increasing:
            logger.error("Index is not sorted in ascending order")
            return False
        
        return True
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax') -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        Normalize OHLCV data using specified method.
        
        Args:
            df: DataFrame with OHLCV data
            method: Normalization method ('minmax', 'zscore', or 'robust')
            
        Returns:
            Tuple of (normalized DataFrame, normalization parameters)
        """
        df = df.copy()
        params = {}
        
        # Only normalize price columns, not volume
        price_columns = ['open', 'high', 'low', 'close']
        
        if method == 'minmax':
            # Min-Max normalization (0-1 range)
            for col in price_columns:
                col_min = df[col].min()
                col_max = df[col].max()
                df[col] = (df[col] - col_min) / (col_max - col_min)
                params[col] = {'min': col_min, 'max': col_max}
                
        elif method == 'zscore':
            # Z-score normalization (standard normal distribution)
            for col in price_columns:
                col_mean = df[col].mean()
                col_std = df[col].std()
                df[col] = (df[col] - col_mean) / col_std
                params[col] = {'mean': col_mean, 'std': col_std}
                
        elif method == 'robust':
            # Robust scaling using median and IQR
            for col in price_columns:
                col_median = df[col].median()
                col_q1 = df[col].quantile(0.25)
                col_q3 = df[col].quantile(0.75)
                col_iqr = col_q3 - col_q1
                df[col] = (df[col] - col_median) / col_iqr
                params[col] = {'median': col_median, 'iqr': col_iqr}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Normalize volume separately (always min-max)
        vol_min = df['volume'].min()
        vol_max = df['volume'].max()
        if vol_max > vol_min:
            df['volume'] = (df['volume'] - vol_min) / (vol_max - vol_min)
        else:
            df['volume'] = 0
        params['volume'] = {'min': vol_min, 'max': vol_max}
        
        return df, params
    
    def denormalize_data(self, df: pd.DataFrame, params: Dict[str, Dict[str, float]], method: str = 'minmax') -> pd.DataFrame:
        """
        Denormalize data back to original scale.
        
        Args:
            df: Normalized DataFrame
            params: Normalization parameters from normalize_data()
            method: The normalization method that was used
            
        Returns:
            pd.DataFrame: Denormalized data
        """
        df = df.copy()
        
        for col, col_params in params.items():
            if col not in df.columns:
                continue
                
            if method == 'minmax' or col == 'volume':
                df[col] = df[col] * (col_params['max'] - col_params['min']) + col_params['min']
            elif method == 'zscore':
                df[col] = df[col] * col_params['std'] + col_params['mean']
            elif method == 'robust':
                df[col] = df[col] * col_params['iqr'] + col_params['median']
        
        return df