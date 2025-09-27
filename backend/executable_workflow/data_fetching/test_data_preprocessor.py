"""
Unit tests for DataPreprocessor class

This module contains comprehensive unit tests for the DataPreprocessor class,
covering various data validation scenarios including missing data, gaps,
outliers, and integrity checks.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h')
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, len(dates)),
            'high': np.random.uniform(100, 120, len(dates)),
            'low': np.random.uniform(80, 100, len(dates)),
            'close': np.random.uniform(90, 110, len(dates)),
            'volume': np.random.uniform(1000, 10000, len(dates))
        }, index=dates)
        
        # Fix OHLC relationships
        self.sample_data['high'] = self.sample_data[['open', 'high', 'close']].max(axis=1)
        self.sample_data['low'] = self.sample_data[['open', 'low', 'close']].min(axis=1)
    
    def test_clean_data_with_valid_data(self):
        """Test clean_data with valid OHLCV data."""
        result = self.preprocessor.clean_data(self.sample_data, timeframe='1h')
        
        # Check that data is returned
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        
        # Check that all required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, result.columns)
        
        # Check that calculated features are added
        self.assertIn('true_range', result.columns)
        self.assertIn('atr_14', result.columns)
        self.assertIn('volume_ratio', result.columns)
    
    def test_clean_data_with_empty_dataframe(self):
        """Test clean_data with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.preprocessor.clean_data(empty_df)
        self.assertTrue(result.empty)
    
    def test_detect_gaps_no_gaps(self):
        """Test detect_gaps when there are no gaps."""
        gaps = self.preprocessor.detect_gaps(self.sample_data, '1h')
        self.assertEqual(len(gaps), 0)
    
    def test_detect_gaps_with_gaps(self):
        """Test detect_gaps when there are gaps in the data."""
        # Create data with gaps
        df_with_gaps = self.sample_data.copy()
        # Remove some rows to create gaps
        df_with_gaps = df_with_gaps.drop(df_with_gaps.index[5:8])  # 3-hour gap
        df_with_gaps = df_with_gaps.drop(df_with_gaps.index[15:17])  # 2-hour gap
        
        gaps = self.preprocessor.detect_gaps(df_with_gaps, '1h')
        
        # Should detect 2 gaps
        self.assertEqual(len(gaps), 2)
        
        # First gap should have 3 missing candles
        self.assertEqual(gaps[0]['missing_candles'], 3)
        
        # Second gap should have 2 missing candles
        self.assertEqual(gaps[1]['missing_candles'], 2)
    
    def test_fill_missing_data(self):
        """Test fill_missing_data method."""
        # Create data with gaps
        df_with_gaps = self.sample_data.copy()
        df_with_gaps = df_with_gaps.drop(df_with_gaps.index[5:8])
        
        # Fill missing data
        filled_df = self.preprocessor.fill_missing_data(df_with_gaps, '1h')
        
        # Check that gaps are filled
        self.assertEqual(len(filled_df), len(self.sample_data))
        
        # Check that filled values are forward filled for prices
        # and 0 for volume
        filled_indices = ~filled_df.index.isin(df_with_gaps.index)
        self.assertTrue((filled_df.loc[filled_indices, 'volume'] == 0).all())
    
    def test_validate_ohlcv_integrity_with_invalid_data(self):
        """Test validate_ohlcv_integrity with invalid OHLC relationships."""
        # Create invalid data
        invalid_data = self.sample_data.copy()
        
        # Make some high values less than low
        invalid_data.iloc[0:5, invalid_data.columns.get_loc('high')] = 50
        invalid_data.iloc[0:5, invalid_data.columns.get_loc('low')] = 100
        
        # Add negative prices
        invalid_data.iloc[10:12, invalid_data.columns.get_loc('open')] = -10
        
        # Add negative volume
        invalid_data.iloc[15:17, invalid_data.columns.get_loc('volume')] = -1000
        
        # Validate and fix
        fixed_data = self.preprocessor.validate_ohlcv_integrity(invalid_data)
        
        # Check that all OHLC relationships are valid
        self.assertTrue((fixed_data['high'] >= fixed_data['low']).all())
        self.assertTrue((fixed_data['high'] >= fixed_data['open']).all())
        self.assertTrue((fixed_data['high'] >= fixed_data['close']).all())
        self.assertTrue((fixed_data['low'] <= fixed_data['open']).all())
        self.assertTrue((fixed_data['low'] <= fixed_data['close']).all())
        
        # Check no negative values
        self.assertTrue((fixed_data[['open', 'high', 'low', 'close']] >= 0).all().all())
        self.assertTrue((fixed_data['volume'] >= 0).all())
    
    def test_outlier_removal(self):
        """Test outlier removal functionality."""
        # Create data with outliers
        data_with_outliers = self.sample_data.copy()
        
        # Add price outliers
        data_with_outliers.iloc[5, data_with_outliers.columns.get_loc('high')] = 1000  # 10x normal
        data_with_outliers.iloc[10, data_with_outliers.columns.get_loc('low')] = 1  # Very low
        
        # Add volume outlier
        data_with_outliers.iloc[15, data_with_outliers.columns.get_loc('volume')] = 1000000
        
        # Clean data
        cleaned = self.preprocessor.clean_data(data_with_outliers, '1h')
        
        # Check that extreme values are removed or capped
        self.assertLess(cleaned['high'].max(), 1000)
        self.assertGreater(cleaned['low'].min(), 1)
        self.assertLess(cleaned['volume'].max(), 1000000)
    
    def test_normalize_data_minmax(self):
        """Test min-max normalization."""
        normalized, params = self.preprocessor.normalize_data(self.sample_data, method='minmax')
        
        # Check that all price values are between 0 and 1
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            self.assertGreaterEqual(normalized[col].min(), 0)
            self.assertLessEqual(normalized[col].max(), 1)
        
        # Check that parameters are stored
        for col in price_columns + ['volume']:
            self.assertIn(col, params)
            self.assertIn('min', params[col])
            self.assertIn('max', params[col])
    
    def test_normalize_data_zscore(self):
        """Test z-score normalization."""
        normalized, params = self.preprocessor.normalize_data(self.sample_data, method='zscore')
        
        # Check that normalized data has mean ~0 and std ~1
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            self.assertAlmostEqual(normalized[col].mean(), 0, places=5)
            self.assertAlmostEqual(normalized[col].std(), 1, places=5)
        
        # Check parameters
        for col in price_columns:
            self.assertIn('mean', params[col])
            self.assertIn('std', params[col])
    
    def test_normalize_data_robust(self):
        """Test robust normalization."""
        normalized, params = self.preprocessor.normalize_data(self.sample_data, method='robust')
        
        # Check that parameters are stored
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            self.assertIn('median', params[col])
            self.assertIn('iqr', params[col])
    
    def test_denormalize_data(self):
        """Test data denormalization."""
        # Normalize first
        normalized, params = self.preprocessor.normalize_data(self.sample_data, method='minmax')
        
        # Denormalize
        denormalized = self.preprocessor.denormalize_data(normalized, params, method='minmax')
        
        # Check that values are close to original
        pd.testing.assert_frame_equal(
            self.sample_data[['open', 'high', 'low', 'close', 'volume']],
            denormalized[['open', 'high', 'low', 'close', 'volume']],
            atol=1e-10
        )
    
    def test_datetime_index_conversion(self):
        """Test conversion of various datetime formats."""
        # Test with timestamp column
        df_with_timestamp = self.sample_data.copy()
        df_with_timestamp.index = range(len(df_with_timestamp))
        df_with_timestamp['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df_with_timestamp), freq='1h')
        
        cleaned = self.preprocessor.clean_data(df_with_timestamp, '1h')
        self.assertIsInstance(cleaned.index, pd.DatetimeIndex)
    
    def test_calculated_features(self):
        """Test that calculated features are correctly added."""
        cleaned = self.preprocessor.clean_data(self.sample_data, '1h')
        
        # Check for expected features
        expected_features = [
            'true_range', 'atr_14', 'volume_sma_20', 'volume_ratio',
            'price_spread', 'candle_body_pct', 'sma_20', 'price_vs_sma20'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, cleaned.columns)
            # Check that features don't contain NaN (except for initial periods)
            non_nan_count = cleaned[feature].notna().sum()
            self.assertGreater(non_nan_count, len(cleaned) * 0.5)  # At least 50% non-NaN
    
    def test_extreme_price_changes(self):
        """Test handling of extreme price changes."""
        # Create data with extreme changes
        extreme_data = self.sample_data.copy()
        
        # Add 50% price jump (above threshold)
        extreme_data.iloc[5, extreme_data.columns.get_loc('close')] = extreme_data.iloc[4, extreme_data.columns.get_loc('close')] * 1.5
        
        cleaned = self.preprocessor.clean_data(extreme_data, '1h')
        
        # Check that extreme changes are handled
        pct_changes = cleaned['close'].pct_change().abs()
        self.assertTrue((pct_changes[pct_changes.notna()] < 0.25).all())  # All changes < 25%
    
    def test_missing_columns(self):
        """Test behavior when required columns are missing."""
        # Create data missing 'volume'
        incomplete_data = self.sample_data.drop(columns=['volume'])
        
        with self.assertRaises(ValueError):
            self.preprocessor.validate_ohlcv_integrity(incomplete_data)
    
    def test_duplicate_timestamps(self):
        """Test handling of duplicate timestamps."""
        # Create data with duplicates
        dup_data = pd.concat([self.sample_data, self.sample_data.iloc[5:8]])
        dup_data = dup_data.sort_index()
        
        cleaned = self.preprocessor.clean_data(dup_data, '1h')
        
        # Check no duplicates in result
        self.assertEqual(len(cleaned.index.unique()), len(cleaned))
    
    def test_unsorted_data(self):
        """Test that unsorted data is properly sorted."""
        # Shuffle the data
        shuffled_data = self.sample_data.sample(frac=1)
        
        cleaned = self.preprocessor.clean_data(shuffled_data, '1h')
        
        # Check that index is sorted
        self.assertTrue(cleaned.index.is_monotonic_increasing)
    
    def test_different_timeframes(self):
        """Test preprocessing with different timeframes."""
        timeframes = ['15m', '1h', '4h', '1d']
        
        for tf in timeframes:
            # Create appropriate data for timeframe
            if tf == '15m':
                dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='15min')
            elif tf == '4h':
                dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='4h')
            elif tf == '1d':
                dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='1d')
            else:
                dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h')
            
            test_data = pd.DataFrame({
                'open': np.random.uniform(90, 110, len(dates)),
                'high': np.random.uniform(100, 120, len(dates)),
                'low': np.random.uniform(80, 100, len(dates)),
                'close': np.random.uniform(90, 110, len(dates)),
                'volume': np.random.uniform(1000, 10000, len(dates))
            }, index=dates)
            
            # Fix OHLC relationships
            test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
            test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
            
            # Process data
            result = self.preprocessor.clean_data(test_data, timeframe=tf)
            
            # Basic validation
            self.assertFalse(result.empty)
            self.assertIn('true_range', result.columns)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False)


if __name__ == '__main__':
    unittest.main()