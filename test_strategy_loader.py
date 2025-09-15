#!/usr/bin/env python3
"""
Test script to verify strategy loader fixes
"""

import sys
import os
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backtesting.utils.strategy_loader import StrategyLoader
from backtesting.utils.data_loader import DataLoader
from backtesting.core.engine import BacktestEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_strategy_loading():
    """Test if strategies can be loaded properly"""
    
    # Initialize strategy loader
    strategy_loader = StrategyLoader()
    
    # Test parameters for MACD strategy
    macd_params = {
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'momentum_period': 14,
        'atr_period': 14,
        'volume_threshold': 1.5,
        'adx_period': 14,
        'adx_threshold': 25
    }
    
    print("\n" + "="*60)
    print("TESTING STRATEGY LOADER FIXES")
    print("="*60)
    
    try:
        # Try to load MACD strategy
        print("\n1. Loading MACD Momentum strategy...")
        strategy = strategy_loader.load_strategy('macd_momentum', macd_params)
        print(f"   ✓ Successfully loaded: {strategy.__class__.__name__}")
        print(f"   ✓ Using implementation: {'Executable' if 'MACDMomentumCrossover' in str(strategy.__class__) else 'Wrapper'}")
        
        # Load sample data to test signals
        print("\n2. Loading test data...")
        data_config = {
            'symbol': 'BTC/USDT',
            'timeframe': '4h',
            'start_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'end_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        data_loader = DataLoader(data_config)
        market_data = data_loader.load_data(
            data_config['symbol'],
            data_config['timeframe'],
            data_config['start_date'],
            data_config['end_date']
        )
        print(f"   ✓ Loaded {len(market_data)} candles")
        
        # Generate signals
        print("\n3. Generating signals...")
        signals = strategy.generate_signals(market_data)
        non_zero_signals = signals[signals['signal'] != 0]
        print(f"   ✓ Generated {len(non_zero_signals)} trading signals")
        
        if len(non_zero_signals) > 0:
            print("\n4. Signal quality check:")
            # Check for ADX filtering
            if hasattr(strategy, '_macd_momentum_signals'):
                print("   ✓ Using enhanced wrapper with ADX filtering")
            else:
                print("   ✓ Using executable strategy with full feature set")
            
            # Show signal distribution
            buy_signals = len(signals[signals['signal'] == 1])
            sell_signals = len(signals[signals['signal'] == -1])
            avg_strength = signals[signals['signal'] != 0]['strength'].mean()
            
            print(f"   - Buy signals: {buy_signals}")
            print(f"   - Sell signals: {sell_signals}")
            print(f"   - Average signal strength: {avg_strength:.3f}")
            
            # Quick backtest
            print("\n5. Running quick backtest...")
            engine = BacktestEngine(
                initial_capital=100000,
                commission=0.001,
                slippage=0.001
            )
            
            results = engine.run_backtest(signals, market_data, 'BTC/USDT')
            
            print(f"   - Total return: {results['total_return']:.2f}%")
            print(f"   - Number of trades: {results['total_trades']}")
            print(f"   - Win rate: {results['win_rate']:.2f}%")
            
        else:
            print("   ⚠ No trading signals generated - check strategy logic")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "="*60)

if __name__ == "__main__":
    test_strategy_loading()