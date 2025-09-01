#!/usr/bin/env python3
"""
Test script for MarketRegimeDetector with crypto assets.
Demonstrates loading data from SQLite DBs and regime classification.
"""

import sqlite3
import pandas as pd

from backend.Indicators import MarketRegimeDetector


def load_crypto_data(asset: str, limit: int = 1000) -> pd.DataFrame:
    """Load OHLCV data from SQLite database."""
    db_path = f"data/trading_data_{asset}.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            # First, get the symbol_id and timeframe_id for 1h data
            symbol_query = "SELECT id FROM symbols WHERE symbol = ? LIMIT 1"
            timeframe_query = "SELECT id FROM timeframes WHERE timeframe = '1h' LIMIT 1"
            
            # For crypto, symbol format is 'BTC/USDT'
            symbol_name = f"{asset}/USDT"
            symbol_result = conn.execute(symbol_query, (symbol_name,)).fetchone()
            timeframe_result = conn.execute(timeframe_query).fetchone()
            
            if not symbol_result or not timeframe_result:
                print(f"Could not find symbol {symbol_name} or 1h timeframe")
                return pd.DataFrame()
            
            symbol_id = symbol_result[0]
            timeframe_id = timeframe_result[0]
            
            # Query to get recent OHLCV data
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data
                WHERE symbol_id = ? AND timeframe_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol_id, timeframe_id, limit))
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
    except Exception as e:
        print(f"Error loading {asset} data: {e}")
        return pd.DataFrame()


def main():
    """Test MarketRegimeDetector functionality."""
    
    # Test 1: Basic regime detection for BTC
    print("=" * 80)
    print("Test 1: BTC Regime Detection")
    print("=" * 80)
    
    btc_df = load_crypto_data('BTC')
    if not btc_df.empty:
        detector = MarketRegimeDetector(btc_df, asset_name='BTC')
        
        # Verify DB connection
        db_connected = detector.verify_db_connection()
        print(f"DB Connection Verified: {db_connected}")
        
        # Get current regime
        regime, metrics = detector.classify_regime()
        print(f"\nCurrent Regime: {regime}")
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print("Failed to load BTC data")
    
    # Test 2: ETH with BTC as benchmark
    print("\n" + "=" * 80)
    print("Test 2: ETH Regime Detection with BTC Benchmark")
    print("=" * 80)
    
    eth_df = load_crypto_data('ETH')
    btc_df = load_crypto_data('BTC')
    
    if not eth_df.empty and not btc_df.empty:
        detector_eth = MarketRegimeDetector(
            eth_df, 
            asset_name='ETH',
            benchmark_df=btc_df
        )
        
        regime, metrics = detector_eth.classify_regime()
        print(f"\nCurrent Regime: {regime}")
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Create visualization
        try:
            fig, ax = detector_eth.plot_regime()
            fig.savefig('eth_regime_chart.png', dpi=150, bbox_inches='tight')
            print("\nRegime chart saved to 'eth_regime_chart.png'")
        except Exception as e:
            print(f"\nPlotting error: {e}")
    
    # Test 3: SOL detection
    print("\n" + "=" * 80)
    print("Test 3: SOL Regime Detection")
    print("=" * 80)
    
    sol_df = load_crypto_data('SOL')
    if not sol_df.empty:
        detector_sol = MarketRegimeDetector(sol_df, asset_name='SOL')
        
        regime, metrics = detector_sol.classify_regime()
        print(f"\nCurrent Regime: {regime}")
        print("\nKey Indicators:")
        print(f"  ADX: {metrics['adx']}")
        print(f"  RSI: {metrics['rsi']}")
        print(f"  ATR Z-Score: {metrics['atr_zscore']}")
        print(f"  Volatility Regime: {metrics['volatility_regime']}")


if __name__ == "__main__":
    main()
