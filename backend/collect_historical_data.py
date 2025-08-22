"""
Script to collect all historical data from August 1st, 2020 for all trading pairs.

USES UNIFIED TRADING DATABASE SYSTEM
"""
from data_fetcher import DataCollector
from datetime import datetime


def collect_all_historical_data_for_all_pairs(start_date: str = '2020-08-01'):
    """
    Collect all historical data for all trading pairs since the specified start date.
    Uses the unified database system for streamlined data management.
    """
    # Trading pairs and timeframes
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
    timeframes = ['4h', '1d']

    print(f"📊 Starting historical data collection from {start_date}")
    print(f"🎯 Symbols: {symbols}")
    print(f"⏰ Timeframes: {timeframes}")
    print(f"🗄️ Using unified database: data/unified_trading_data.db")
    
    # Initialize unified data collector
    collector = DataCollector()
    
    # Collect historical data using the unified system
    collector.collect_historical_data(symbols, timeframes, start_date)
    
    print("\n✅ Historical data collection completed!")
    print("📈 Use 'python ../scripts/unified_data_workflow.py --summary' to view data summary")


if __name__ == '__main__':
    collect_all_historical_data_for_all_pairs()
