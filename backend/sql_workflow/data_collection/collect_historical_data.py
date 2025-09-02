"""
Script to collect all historical data from August 1st, 2020 for selected trading pairs.

Uses a per-symbol SQLite database (data/trading_data_BTC.db) with a normalized schema.
"""
from .data_fetcher import DataCollector
from datetime import datetime
from typing import List


def collect_all_historical_data_for_all_pairs(start_date: str = '2020-08-01', symbols: List[str] = None):
    """
    Collect all historical data for all trading pairs since the specified start date.
    Uses the per-symbol database for streamlined data management.
    
    Args:
        start_date: Start date for historical data collection
        symbols: List of symbols to collect data for (defaults to all available)
    """
    # Trading pairs and timeframes
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # All supported pairs
    timeframes = ['1h', '4h', '1d']

    print(f"📊 Starting historical data collection from {start_date}")
    print(f"🎯 Symbols: {symbols}")
    print(f"⏰ Timeframes: {timeframes}")
    
    # Process each symbol with its own database
    for symbol in symbols:
        symbol_short = symbol.split('/')[0]  # Get BTC, ETH, or SOL
        db_path = f'data/trading_data_{symbol_short}.db'
        print(f"\n🗄️ Processing {symbol} using database: {db_path}")
        
        # Initialize data collector with symbol-specific database
        collector = DataCollector(db_path=db_path)
        
        # Collect historical data for this symbol
        collector.collect_historical_data([symbol], timeframes, start_date)
    
    print("\n✅ Historical data collection completed for all symbols!")
    print("📈 Use 'python run_trading_bot.py --status' to view data summary")


if __name__ == '__main__':
    collect_all_historical_data_for_all_pairs()
