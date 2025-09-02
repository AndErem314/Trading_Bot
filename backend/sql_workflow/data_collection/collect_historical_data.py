"""
Script to collect all historical data from August 1st, 2020 for selected trading pairs.

Uses a per-symbol SQLite database (data/trading_data_BTC.db) with a normalized schema.
"""
from .data_fetcher import DataCollector
from datetime import datetime


def collect_all_historical_data_for_all_pairs(start_date: str = '2020-08-01'):
    """
    Collect all historical data for all trading pairs since the specified start date.
    Uses the per-symbol database for streamlined data management.
    """
    # Trading pairs and timeframes
    symbols = ['BTC/USDT']  # Only collect Bitcoin data
    timeframes = ['4h', '1d']

    print(f"📊 Starting historical data collection from {start_date}")
    print(f"🎯 Symbols: {symbols}")
    print(f"⏰ Timeframes: {timeframes}")
    print(f"🗄️ Using database: data/trading_data_BTC.db")
    
    # Initialize data collector
    collector = DataCollector()
    
    # Collect historical data using the per-symbol system
    collector.collect_historical_data(symbols, timeframes, start_date)
    
    print("\n✅ Historical data collection completed!")
    print("📈 Use 'python run_trading_bot.py --status' to view data summary")


if __name__ == '__main__':
    collect_all_historical_data_for_all_pairs()
