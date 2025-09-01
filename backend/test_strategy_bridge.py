"""
Test Strategy Bridge Integration

This script tests the strategy bridge to ensure correct functionality
with both old SQL-based descriptors and new executable strategies.

"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.strategy_bridge import StrategyBridge, UnifiedStrategyFactory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(periods: int = 200) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
    
    # Generate price data with trend and noise
    base_price = 50000
    trend = np.linspace(0, 1000, periods)
    noise = np.random.normal(0, 500, periods)
    prices = base_price + trend + noise
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.normal(0, 100, periods),
        'high': prices + np.abs(np.random.normal(0, 200, periods)),
        'low': prices - np.abs(np.random.normal(0, 200, periods)),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, periods)
    })
    
    data.set_index('timestamp', inplace=True)
    return data


def test_single_strategy(strategy_name: str, data: pd.DataFrame) -> dict:
    """Test a single strategy through the bridge."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {strategy_name}")
    logger.info(f"{'='*60}")
    
    results = {'strategy': strategy_name, 'success': True, 'errors': []}
    
    try:
        # Create bridge
        bridge = UnifiedStrategyFactory.create_strategy(strategy_name)
        logger.info(f"✓ Created bridge for {strategy_name}")
        
        # Get strategy info
        info = bridge.get_strategy_info()
        logger.info(f"✓ Retrieved strategy info")
        logger.info(f"  Description: {info['descriptor']['description'].get('type', 'N/A')}")
        
        # Initialize executable
        bridge.initialize_executable(data)
        logger.info(f"✓ Initialized executable strategy")
        
        # Get live signal
        signal = bridge.get_live_signal()
        logger.info(f"✓ Generated live signal: {signal['signal']} (confidence: {signal['confidence']:.2f})")
        logger.info(f"  Reason: {signal['reason']}")
        
        # Test market regime suitability
        for regime in ['Bullish', 'Bearish', 'Neutral', 'Ranging']:
            suitable = bridge.is_strategy_allowed(regime)
            logger.info(f"  {regime} market: {'✓' if suitable else '✗'}")
        
        # Try to get historical signals (may fail if DB not available)
        try:
            historical = bridge.get_historical_signals(limit=5)
            if not historical.empty:
                logger.info(f"✓ Retrieved {len(historical)} historical signals")
            else:
                logger.info("  No historical signals available")
        except Exception as e:
            logger.info(f"  Historical signals not available: {str(e)}")
        
        results['signal'] = signal
        results['info'] = info
        
    except Exception as e:
        logger.error(f"✗ Error testing {strategy_name}: {str(e)}")
        results['success'] = False
        results['errors'].append(str(e))
    
    return results


def test_all_strategies():
    """Test all available strategies."""
    # Generate sample data
    data = generate_sample_data(200)
    logger.info(f"Generated sample data: {len(data)} periods")
    logger.info(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Get all available strategies
    strategies = UnifiedStrategyFactory.get_available_strategies()
    logger.info(f"\nFound {len(strategies)} strategies to test")
    
    # Test each strategy
    all_results = []
    for strategy_name in strategies:
        result = test_single_strategy(strategy_name, data)
        all_results.append(result)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    successful = sum(1 for r in all_results if r['success'])
    logger.info(f"Total strategies tested: {len(all_results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(all_results) - successful}")
    
    # Show signal distribution
    signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    for result in all_results:
        if result['success'] and 'signal' in result:
            signal = result['signal']['signal']
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    logger.info(f"\nSignal distribution:")
    for signal, count in signal_counts.items():
        logger.info(f"  {signal}: {count}")
    
    # Save results
    with open('strategy_bridge_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to strategy_bridge_test_results.json")
    
    return all_results


def test_strategy_comparison():
    """Test signal comparison between SQL and executable approaches."""
    logger.info(f"\n{'='*60}")
    logger.info("TESTING SIGNAL COMPARISON")
    logger.info(f"{'='*60}")
    
    data = generate_sample_data(200)
    
    # Test with a few strategies
    test_strategies = [
        'RSI_Momentum_Divergence',
        'Bollinger_Bands_Mean_Reversion',
        'MACD_Momentum_Crossover'
    ]
    
    for strategy_name in test_strategies:
        logger.info(f"\nComparing signals for {strategy_name}")
        
        try:
            bridge = UnifiedStrategyFactory.create_strategy(strategy_name)
            comparison = bridge.compare_signals(data)
            
            logger.info("Comparison results:")
            logger.info(f"  Historical: {comparison.get('historical', {})}")
            logger.info(f"  Live: {comparison.get('live', {})}")
            logger.info(f"  Data info: {comparison.get('data_info', {})}")
            
        except Exception as e:
            logger.error(f"  Error comparing signals: {str(e)}")


def test_batch_processing():
    """Test batch processing of multiple strategies."""
    logger.info(f"\n{'='*60}")
    logger.info("TESTING BATCH PROCESSING")
    logger.info(f"{'='*60}")
    
    data = generate_sample_data(200)
    
    # Create all strategies at once
    strategies = UnifiedStrategyFactory.create_all_strategies(data)
    logger.info(f"Created {len(strategies)} strategy bridges")
    
    # Process signals in batch
    signals = {}
    for name, bridge in strategies.items():
        try:
            signal = bridge.get_live_signal()
            signals[name] = signal
            logger.info(f"{name}: {signal['signal']} ({signal['confidence']:.2f})")
        except Exception as e:
            logger.error(f"{name}: Error - {str(e)}")
    
    return signals


if __name__ == "__main__":
    logger.info("Starting Strategy Bridge Integration Tests")
    logger.info("=" * 80)
    
    # Run all tests
    test_all_strategies()
    test_strategy_comparison()
    test_batch_processing()
    
    logger.info("\nAll tests completed!")
