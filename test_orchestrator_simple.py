#!/usr/bin/env python3
"""
Simple Test for MetaStrategyOrchestrator
Tests with a basic strategy to demonstrate functionality
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import the simplified orchestrator
import pandas as pd
from backend.Indicators import MarketRegimeDetector
from backend.test_strategy import SimpleTestStrategy
from datetime import datetime
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMetaStrategyOrchestrator:
    """Simplified orchestrator for testing."""
    
    def __init__(self, db_connection_string, symbols):
        self.db_connection_string = db_connection_string
        self.symbols = symbols
        self.data_cache = {}
        self.regime_detectors = {}
        self.strategies = {}
        
        # Create engine
        self.engine = create_engine(
            self.db_connection_string,
            poolclass=QueuePool,
            pool_size=5
        )
        
        # Initialize structures
        for symbol in self.symbols:
            self.data_cache[symbol] = {}
            self.regime_detectors[symbol] = {}
            self.strategies[symbol] = {}
    
    @contextmanager
    def get_db_connection(self):
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    def fetch_data(self, symbol, timeframe_db):
        """Fetch data from database."""
        query = text("""
            SELECT 
                o.timestamp,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume
            FROM ohlcv_data o
            JOIN symbols s ON o.symbol_id = s.id
            JOIN timeframes t ON o.timeframe_id = t.id
            WHERE s.symbol = :symbol
            AND t.timeframe = :timeframe
            ORDER BY o.timestamp DESC
            LIMIT 500
        """)
        
        with self.get_db_connection() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={'symbol': symbol, 'timeframe': timeframe_db}
            )
        
        if df.empty:
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df.set_index('timestamp', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        
        return df
    
    def run_simple_test(self):
        """Run a simple test of the orchestrator functionality."""
        
        print("\n1. Fetching data...")
        # Fetch H4 data for BTC
        df = self.fetch_data('BTC/USDT', '4h')
        
        if df.empty:
            print("No data found!")
            return
        
        print(f"   ✓ Fetched {len(df)} rows of BTC/USDT 4h data")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        
        # Store in cache
        self.data_cache['BTC/USDT']['H4'] = df
        
        print("\n2. Initializing regime detector...")
        # Initialize regime detector
        detector = MarketRegimeDetector(
            df=df,
            asset_name='BTC'
        )
        self.regime_detectors['BTC/USDT']['H4'] = detector
        
        regime, metrics = detector.classify_regime()
        print(f"   ✓ Current regime: {regime}")
        print(f"   Current price: ${metrics.get('price', 0):,.2f}")
        print(f"   ATR: ${metrics.get('atr', 0):,.2f}")
        
        print("\n3. Initializing strategy...")
        # Initialize simple test strategy
        strategy = SimpleTestStrategy(
            data=df,
            symbol='BTC/USDT',
            timeframe='H4'
        )
        self.strategies['BTC/USDT']['test_strategy'] = strategy
        
        # Get signal
        signal = strategy.get_signal()
        print(f"   ✓ Strategy initialized")
        print(f"   Current signal: {signal:.2f} ({'LONG' if signal > 0 else 'SHORT' if signal < 0 else 'NEUTRAL'})")
        
        print("\n4. Simulating trading decision...")
        # Simple trading logic
        if abs(signal) >= 0.5:  # Signal threshold
            direction = "LONG" if signal > 0 else "SHORT"
            
            # Calculate position size (simplified)
            portfolio_value = 100000
            risk_per_trade = 0.01
            atr = metrics.get('atr', 0)
            price = metrics.get('price', 0)
            
            if atr > 0 and price > 0:
                stop_distance = 2 * atr
                risk_amount = portfolio_value * risk_per_trade
                position_value = risk_amount / (stop_distance / price)
                position_size = position_value / price
                
                print(f"\n   TRADE SIGNAL GENERATED!")
                print(f"   Direction: {direction}")
                print(f"   Position size: {position_size:.6f} BTC")
                print(f"   Entry price: ${price:,.2f}")
                print(f"   Stop loss: ${price - stop_distance if signal > 0 else price + stop_distance:,.2f}")
                print(f"   Risk amount: ${risk_amount:.2f}")
            else:
                print("   ⚠️  Cannot calculate position size (missing ATR or price)")
        else:
            print("   Signal too weak - no trade")
        
        print("\n✅ Test completed successfully!")


def main():
    """Run the simple orchestrator test."""
    
    print("=" * 80)
    print("SIMPLE META STRATEGY ORCHESTRATOR TEST")
    print("=" * 80)
    
    # Database configuration
    db_path = Path("data/trading_data_BTC.db")
    
    if not db_path.exists():
        print(f"\n❌ Database not found: {db_path}")
        return
    
    connection_string = f"sqlite:///{db_path}"
    
    try:
        # Create and run simple orchestrator
        orchestrator = SimpleMetaStrategyOrchestrator(
            db_connection_string=connection_string,
            symbols=['BTC/USDT']
        )
        
        orchestrator.run_simple_test()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
