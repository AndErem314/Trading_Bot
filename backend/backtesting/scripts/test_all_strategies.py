#!/usr/bin/env python3
"""
Test all strategies for import and basic functionality issues
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import necessary modules
from backend.backtesting.utils.strategy_loader import StrategyLoader
from backend.backtesting.utils.data_loader import DataLoader
import pandas as pd
import numpy as np

def test_strategies():
    """Test all available strategies"""
    # Initialize strategy loader
    strategy_loader = StrategyLoader()
    
    # Get all available strategies
    strategies = strategy_loader.get_available_strategies()
    
    # Test data (minimal)
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='4h'),
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(95, 115, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }).set_index('timestamp')
    
    # Results
    results = {
        'success': [],
        'import_warning': [],
        'failed': []
    }
    
    logger.info(f"Testing {len(strategies)} strategies...")
    
    for strategy_name in strategies.keys():
        logger.info(f"\nTesting {strategy_name}...")
        
        try:
            # Load strategy with default parameters
            strategy = strategy_loader.load_strategy(strategy_name, {})
            
            # Test signal generation
            signals = strategy.generate_signals(test_data)
            
            # Validate output
            if isinstance(signals, pd.DataFrame) and len(signals) > 0:
                logger.info(f"✓ {strategy_name} - SUCCESS")
                results['success'].append(strategy_name)
                
                # Check for any import warnings in logs
                if "Could not import existing strategy" in str(strategy_loader.loaded_strategies):
                    results['import_warning'].append(strategy_name)
            else:
                logger.error(f"✗ {strategy_name} - Invalid output")
                results['failed'].append((strategy_name, "Invalid output"))
                
        except Exception as e:
            logger.error(f"✗ {strategy_name} - FAILED: {str(e)}")
            results['failed'].append((strategy_name, str(e)))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total strategies: {len(strategies)}")
    logger.info(f"Successful: {len(results['success'])}")
    logger.info(f"Import warnings: {len(results['import_warning'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    
    if results['failed']:
        logger.info("\nFailed strategies:")
        for name, error in results['failed']:
            logger.info(f"  - {name}: {error}")
    
    if results['import_warning']:
        logger.info("\nStrategies with import warnings (but working):")
        for name in results['import_warning']:
            logger.info(f"  - {name}")
    
    return results

if __name__ == "__main__":
    # Ensure numpy compatibility
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan
    sys.modules['numpy'].NaN = np.nan
    
    test_strategies()
