#!/usr/bin/env python3
"""
Test script for MetaStrategyOrchestrator
Tests the complete trading cycle with real database data
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.meta_strategy_orchestrator import MetaStrategyOrchestrator


def main():
    """Run the meta strategy orchestrator test."""
    
    print("=" * 80)
    print("META STRATEGY ORCHESTRATOR TEST")
    print("=" * 80)
    
    # Database configuration
    # Using the BTC database since it's most likely to have data
    db_path = Path("data/trading_data_BTC.db")
    
    if not db_path.exists():
        print(f"\n❌ Database not found: {db_path}")
        print("Please ensure you have collected data first.")
        print("Run: python3 run_trading_bot.py --mode collect")
        return
    
    # Create connection string
    connection_string = f"sqlite:///{db_path}"
    
    print(f"\nUsing database: {db_path}")
    
    try:
        # Initialize orchestrator with just BTC to start
        print("\n1. Initializing MetaStrategyOrchestrator...")
        orchestrator = MetaStrategyOrchestrator(
            db_connection_string=connection_string,
            symbols=['BTC/USDT'],  # Start with just BTC
            lookback_period=500
        )
        
        # Setup all components
        print("\n2. Setting up components (fetching data, initializing strategies)...")
        orchestrator.setup()
        
        # Get current regimes
        print("\n3. Checking market regimes...")
        btc_regimes = orchestrator.get_current_regimes('BTC/USDT')
        print(f"BTC Regimes: {btc_regimes}")
        
        # Determine overall bias
        print("\n4. Determining overall market bias...")
        bias, metadata = orchestrator.determine_overall_bias('BTC/USDT')
        print(f"Overall Bias: {bias}")
        print(f"Regime Details: D1={metadata['D1']}, H4={metadata['H4']}, H1={metadata['H1']}")
        
        # Get strategy signals
        print("\n5. Getting strategy signals...")
        signals = orchestrator.get_strategy_signals('BTC/USDT')
        if signals:
            print("Strategy Signals:")
            for strategy, signal_data in signals.items():
                print(f"  {strategy}: {signal_data}")
        else:
            print("No signals available")
        
        # Run the main trading cycle
        print("\n6. Running trading cycle...")
        print("-" * 80)
        orchestrator.run(symbol='BTC/USDT', portfolio_value=100000.0)
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
