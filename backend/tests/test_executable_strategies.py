"""
Unit Tests for Executable Trading Strategies

This module provides comprehensive tests for the executable strategy framework
to ensure signal generation works correctly.

"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_strategy_interface import TradingStrategy, ExampleSMACrossover
from strategies_executable import (
    BollingerBandsMeanReversion,
    RSIMomentumDivergence,
    MACDMomentumCrossover,
    SMAGoldenCross
)


class TestDataGenerator:
    """Helper class to generate test OHLCV data."""
    
    @staticmethod
    def generate_trending_data(periods=500, trend='up', volatility=0.02):
        """Generate trending OHLCV data."""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
        
        # Base price with trend
        if trend == 'up':
            base_prices = np.linspace(100, 150, periods)
        elif trend == 'down':
            base_prices = np.linspace(150, 100, periods)
        else:  # sideways
            base_prices = np.ones(periods) * 125
        
        # Add noise
        noise = np.random.normal(0, volatility * base_prices, periods)
        close_prices = base_prices + noise
        
        # Generate OHLC from close
        high_prices = close_prices + np.abs(np.random.normal(0, volatility * close_prices, periods))
        low_prices = close_prices - np.abs(np.random.normal(0, volatility * close_prices, periods))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Generate volume
        base_volume = 1000000
        volume = base_volume + np.random.normal(0, base_volume * 0.3, periods)
        volume = np.abs(volume)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
    
    @staticmethod
    def generate_ranging_data(periods=500, range_center=125, range_width=10):
        """Generate ranging/sideways OHLCV data."""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
        
        # Oscillating prices
        t = np.linspace(0, 4 * np.pi, periods)
        close_prices = range_center + range_width * np.sin(t)
        
        # Add some noise
        noise = np.random.normal(0, 1, periods)
        close_prices += noise
        
        # Generate OHLC
        high_prices = close_prices + np.abs(np.random.normal(0, 0.5, periods))
        low_prices = close_prices - np.abs(np.random.normal(0, 0.5, periods))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Generate volume
        volume = 1000000 + np.random.normal(0, 200000, periods)
        volume = np.abs(volume)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)


class TestTradingStrategyInterface(unittest.TestCase):
    """Test the base TradingStrategy interface."""
    
    def setUp(self):
        """Set up test data."""
        self.data_generator = TestDataGenerator()
        self.trending_data = self.data_generator.generate_trending_data()
        self.ranging_data = self.data_generator.generate_ranging_data()
    
    def test_interface_enforcement(self):
        """Test that abstract methods must be implemented."""
        with self.assertRaises(TypeError):
            # Should fail because abstract methods not implemented
            class IncompleteStrategy(TradingStrategy):
                pass
            
            strategy = IncompleteStrategy(self.trending_data)
    
    def test_data_validation(self):
        """Test data validation in base class."""
        # Missing columns should raise error
        bad_data = pd.DataFrame({'close': [100, 101, 102]})
        
        with self.assertRaises(ValueError):
            strategy = ExampleSMACrossover(bad_data)
    
    def test_has_sufficient_data(self):
        """Test sufficient data checking."""
        small_data = self.trending_data.iloc[:20]
        strategy = ExampleSMACrossover(small_data)
        
        # Default SMA periods are 10 and 30, need 31 points
        self.assertFalse(strategy.has_sufficient_data())
        
        sufficient_data = self.trending_data.iloc[:35]
        strategy = ExampleSMACrossover(sufficient_data)
        self.assertTrue(strategy.has_sufficient_data())


class TestBollingerBandsStrategy(unittest.TestCase):
    """Test Bollinger Bands Mean Reversion strategy."""
    
    def setUp(self):
        """Set up test data."""
        self.data_generator = TestDataGenerator()
        self.ranging_data = self.data_generator.generate_ranging_data()
        self.trending_data = self.data_generator.generate_trending_data()
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = BollingerBandsMeanReversion(self.ranging_data)
        
        self.assertEqual(strategy.name, "Bollinger Bands Mean Reversion")
        self.assertTrue(strategy.has_sufficient_data())
        self.assertTrue(hasattr(strategy, 'indicators'))
    
    def test_signal_generation_ranging_market(self):
        """Test signal generation in ranging market."""
        strategy = BollingerBandsMeanReversion(self.ranging_data)
        
        signal_data = strategy.calculate_signal()
        
        # Should return a dictionary with required keys
        self.assertIn('signal', signal_data)
        self.assertIn('confidence', signal_data)
        self.assertIn('reason', signal_data)
        
        # Signal should be between -1 and 1
        self.assertGreaterEqual(signal_data['signal'], -1)
        self.assertLessEqual(signal_data['signal'], 1)
    
    def test_market_regime_suitability(self):
        """Test market regime filtering."""
        strategy = BollingerBandsMeanReversion(self.ranging_data)
        
        # Should work well in ranging markets
        self.assertTrue(strategy.is_strategy_allowed('Neutral'))
        self.assertTrue(strategy.is_strategy_allowed('Ranging'))
        
        # Less suitable for strong trends
        self.assertFalse(strategy.is_strategy_allowed('Strong Bullish'))
        self.assertFalse(strategy.is_strategy_allowed('Strong Bearish'))
    
    def test_custom_parameters(self):
        """Test custom parameter configuration."""
        config = {
            'bb_length': 30,
            'bb_std': 2.5,
            'rsi_length': 21
        }
        
        strategy = BollingerBandsMeanReversion(self.ranging_data, config)
        
        self.assertEqual(strategy.bb_length, 30)
        self.assertEqual(strategy.bb_std, 2.5)
        self.assertEqual(strategy.rsi_length, 21)


class TestRSIMomentumStrategy(unittest.TestCase):
    """Test RSI Momentum Divergence strategy."""
    
    def setUp(self):
        """Set up test data."""
        self.data_generator = TestDataGenerator()
        self.trending_data = self.data_generator.generate_trending_data()
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = RSIMomentumDivergence(self.trending_data)
        
        self.assertEqual(strategy.name, "RSI Momentum Divergence Swing")
        self.assertTrue(strategy.has_sufficient_data())
    
    def test_divergence_detection(self):
        """Test divergence detection logic."""
        strategy = RSIMomentumDivergence(self.trending_data)
        
        # Divergence should be detected in indicators
        self.assertIn('divergence', strategy.indicators)
        
        # Check divergence values are valid
        divergence_values = strategy.indicators['divergence'].unique()
        valid_values = ['none', 'bullish', 'bearish']
        for val in divergence_values:
            self.assertIn(val, valid_values)
    
    def test_signal_with_oversold_conditions(self):
        """Test signal generation with oversold RSI."""
        # Create data that will likely have oversold conditions
        crash_data = self.data_generator.generate_trending_data(trend='down', volatility=0.05)
        strategy = RSIMomentumDivergence(crash_data)
        
        signal_data = strategy.calculate_signal()
        
        # Should have all required fields
        self.assertIn('signal', signal_data)
        self.assertIn('rsi', signal_data)
        self.assertIn('momentum_shift', signal_data)
        self.assertIn('trend_strength', signal_data)


class TestMACDStrategy(unittest.TestCase):
    """Test MACD Momentum Crossover strategy."""
    
    def setUp(self):
        """Set up test data."""
        self.data_generator = TestDataGenerator()
        self.trending_data = self.data_generator.generate_trending_data(trend='up')
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MACDMomentumCrossover(self.trending_data)
        
        self.assertEqual(strategy.name, "MACD Momentum Crossover")
        self.assertTrue(strategy.has_sufficient_data())
    
    def test_crossover_detection(self):
        """Test MACD crossover detection."""
        strategy = MACDMomentumCrossover(self.trending_data)
        
        # Should have crossover indicator
        self.assertIn('macd_cross', strategy.indicators)
        
        # Crossover values should be -1, 0, or 1
        cross_values = strategy.indicators['macd_cross'].unique()
        for val in cross_values:
            self.assertIn(val, [-1, 0, 1])
    
    def test_trending_market_signals(self):
        """Test signals in trending market."""
        strategy = MACDMomentumCrossover(self.trending_data)
        
        # MACD should work well in trending markets
        self.assertTrue(strategy.is_strategy_allowed('Bullish'))
        self.assertTrue(strategy.is_strategy_allowed('Strong Bullish'))
        
        signal_data = strategy.calculate_signal()
        self.assertIsNotNone(signal_data['signal'])


class TestSMAGoldenCrossStrategy(unittest.TestCase):
    """Test SMA Golden Cross strategy."""
    
    def setUp(self):
        """Set up test data."""
        self.data_generator = TestDataGenerator()
        # Need more data for 200-period SMA
        self.long_data = self.data_generator.generate_trending_data(periods=300)
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = SMAGoldenCross(self.long_data)
        
        self.assertEqual(strategy.name, "SMA Golden Cross")
        self.assertTrue(strategy.has_sufficient_data())
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        short_data = self.long_data.iloc[:100]  # Less than 200 periods
        strategy = SMAGoldenCross(short_data)
        
        self.assertFalse(strategy.has_sufficient_data())
        
        signal_data = strategy.calculate_signal()
        self.assertEqual(signal_data['signal'], 0)
        self.assertEqual(signal_data['reason'], 'Insufficient data')
    
    def test_golden_cross_detection(self):
        """Test golden/death cross detection."""
        # Create synthetic data with a crossover
        periods = 300
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
        
        # Create prices that will generate a golden cross
        prices = np.concatenate([
            np.linspace(100, 80, 150),  # Downtrend
            np.linspace(80, 120, 150)   # Uptrend
        ])
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.ones(periods) * 1000000
        }, index=dates)
        
        strategy = SMAGoldenCross(data)
        
        # Check if crossovers were detected
        crossovers = strategy.indicators['golden_cross']
        self.assertTrue((crossovers != 0).any())


class TestEnhancedOrchestrator(unittest.TestCase):
    """Test Enhanced Meta Strategy Orchestrator."""
    
    def setUp(self):
        """Set up test environment."""
        self.data_generator = TestDataGenerator()
        self.test_data = {
            'BTC/USDT': {
                'H1': self.data_generator.generate_trending_data()
            }
        }
    
    def test_strategy_registry(self):
        """Test strategy registry contains all strategies."""
        from enhanced_meta_strategy_orchestrator import EnhancedMetaStrategyOrchestrator
        
        expected_strategies = [
            'BollingerBandsMeanReversion',
            'RSIMomentumDivergence',
            'MACDMomentumCrossover',
            'SMAGoldenCross'
        ]
        
        for strategy_name in expected_strategies:
            self.assertIn(
                strategy_name,
                EnhancedMetaStrategyOrchestrator.STRATEGY_REGISTRY
            )
    
    def test_config_loading(self):
        """Test configuration loading."""
        from enhanced_meta_strategy_orchestrator import EnhancedMetaStrategyOrchestrator
        
        # Test with default config
        orchestrator = EnhancedMetaStrategyOrchestrator(
            db_connection_string="sqlite:///:memory:",
            symbols=['BTC/USDT']
        )
        
        self.assertIn('strategies', orchestrator.config)
        self.assertIn('signal_filters', orchestrator.config)


class TestSignalAggregation(unittest.TestCase):
    """Test signal aggregation and weighting."""
    
    def test_weighted_signal_calculation(self):
        """Test calculation of weighted composite signals."""
        # Create mock signals
        signals = {
            'BTC/USDT': {
                'strategy1': {
                    'signal': 0.8,
                    'weight': 0.7,
                    'weighted_signal': 0.56,
                    'confidence': 0.9
                },
                'strategy2': {
                    'signal': 0.4,
                    'weight': 0.3,
                    'weighted_signal': 0.12,
                    'confidence': 0.7
                }
            }
        }
        
        # Calculate expected composite
        total_weighted = 0.56 + 0.12
        total_weight = 0.7 + 0.3
        expected_signal = total_weighted / total_weight
        
        # Test composite calculation
        from enhanced_meta_strategy_orchestrator import EnhancedMetaStrategyOrchestrator
        
        orchestrator = EnhancedMetaStrategyOrchestrator(
            db_connection_string="sqlite:///:memory:",
            symbols=['BTC/USDT']
        )
        
        composite = orchestrator.calculate_composite_signals(signals)
        
        self.assertAlmostEqual(
            composite['BTC/USDT']['signal'],
            expected_signal,
            places=3
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_signal_generation(self):
        """Test complete signal generation flow."""
        from enhanced_meta_strategy_orchestrator import EnhancedMetaStrategyOrchestrator
        
        # Create test data
        data_generator = TestDataGenerator()
        test_data = data_generator.generate_trending_data()
        
        # Create orchestrator with mock data
        orchestrator = EnhancedMetaStrategyOrchestrator(
            db_connection_string="sqlite:///:memory:",
            symbols=['BTC/USDT']
        )
        
        # Mock data loading
        orchestrator.data_cache = {
            'BTC/USDT': {
                'H1': test_data
            }
        }
        
        # Initialize strategies
        orchestrator._initialize_strategies('H1')
        
        # Check strategies were initialized
        self.assertGreater(len(orchestrator.strategies['BTC/USDT']), 0)
        
        # Update weights
        orchestrator.update_strategy_weights('Bullish')
        
        # Generate signals
        signals = orchestrator.generate_signals()
        
        # Should have signals for BTC/USDT
        self.assertIn('BTC/USDT', signals)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False)


if __name__ == '__main__':
    run_tests()
