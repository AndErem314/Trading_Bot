#!/usr/bin/env python3
"""
Unified OHLCV Data Collection Workflow
=====================================

Main script for collecting and managing OHLCV data using the unified database system.
Provides simple commands for common data operations.

Usage examples:
    python scripts/unified_data_workflow.py --update          # Update all existing data
    python scripts/unified_data_workflow.py --collect-from "2021-01-01"  # Collect historical data
    python scripts/unified_data_workflow.py --validate       # Validate and repair data
    python scripts/unified_data_workflow.py --summary        # Show data summary
"""

import sys
import os
import argparse
from datetime import datetime

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from unified_data_fetcher import UnifiedDataCollector
    from unified_data_manager import UnifiedDataManager
except ImportError:
    # Handle relative imports if running from backend directory
    from backend.unified_data_fetcher import UnifiedDataCollector
    from backend.unified_data_manager import UnifiedDataManager


# Configuration
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
TIMEFRAMES = ['4h', '1d']
EXCHANGE = 'binance'
DB_PATH = 'data/unified_trading_data.db'


def setup_collector():
    """Initialize the unified data collector."""
    return UnifiedDataCollector(EXCHANGE, DB_PATH)


def update_data():
    """Update all existing data with latest values."""
    print("🔄 Starting incremental data update...")
    collector = setup_collector()
    collector.update_all_data(SYMBOLS, TIMEFRAMES)
    print("✅ Data update completed!")


def collect_historical_data(start_date: str):
    """Collect historical data from specified start date."""
    print(f"📈 Starting historical data collection from {start_date}...")
    collector = setup_collector()
    collector.collect_historical_data(SYMBOLS, TIMEFRAMES, start_date)
    print("✅ Historical data collection completed!")


def validate_and_repair_data(start_date: str = "2020-01-01"):
    """Validate data integrity and fill gaps."""
    print("🔍 Starting data validation and repair...")
    collector = setup_collector()
    collector.validate_and_repair_data(SYMBOLS, TIMEFRAMES, start_date)
    print("✅ Data validation and repair completed!")


def show_data_summary():
    """Display a summary of all data in the database."""
    print("📊 Database Summary:")
    print("=" * 50)
    
    data_manager = UnifiedDataManager(DB_PATH)
    
    # Get data summary
    summary_df = data_manager.get_data_summary()
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        
        # Calculate totals
        total_records = summary_df['record_count'].sum()
        print(f"\n📈 Total OHLCV records: {total_records:,}")
        print(f"🔢 Unique symbols: {summary_df['symbol'].nunique()}")
        print(f"⏰ Unique timeframes: {summary_df['timeframe'].nunique()}")
    else:
        print("No data found in database.")
    
    # Data integrity check
    print("\n🔍 Data Integrity Check:")
    print("-" * 30)
    integrity_issues = data_manager.validate_data_integrity()
    for issue_type, count in integrity_issues.items():
        status = "✅" if count == 0 else "⚠️"
        print(f"{status} {issue_type}: {count:,} records")


def add_new_symbol(symbol: str, start_date: str = "2021-01-01"):
    """Add a new symbol to the database."""
    print(f"➕ Adding new symbol: {symbol}")
    
    collector = setup_collector()
    
    # Collect data for the new symbol
    results = collector.fetcher.bulk_collect_data(
        [symbol], TIMEFRAMES, 
        int(pd.to_datetime(start_date).timestamp() * 1000), 
        force_refresh=True
    )
    
    print(f"✅ Added {symbol} to database:")
    for timeframe, result in results[symbol].items():
        print(f"  {timeframe}: {result['inserted']} records")


def quick_update():
    """Quick update for recent data (last 7 days)."""
    print("⚡ Quick update - fetching last 7 days of data...")
    
    collector = setup_collector()
    
    # Get recent data for all symbols/timeframes
    results = collector.fetcher.bulk_collect_data(SYMBOLS, TIMEFRAMES)
    
    total_new_records = sum(
        result.get('inserted', 0) 
        for symbol_results in results.values() 
        for result in symbol_results.values()
    )
    
    print(f"✅ Quick update completed: {total_new_records} new records added")


def main():
    parser = argparse.ArgumentParser(
        description="Unified OHLCV Data Collection Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/unified_data_workflow.py --update
  python scripts/unified_data_workflow.py --collect-from "2021-01-01"
  python scripts/unified_data_workflow.py --validate
  python scripts/unified_data_workflow.py --summary
  python scripts/unified_data_workflow.py --add-symbol "ADA/USDT"
        """
    )
    
    parser.add_argument('--update', action='store_true',
                        help='Update all existing data with latest values')
    
    parser.add_argument('--collect-from', type=str, metavar='DATE',
                        help='Collect historical data from specified date (YYYY-MM-DD)')
    
    parser.add_argument('--validate', action='store_true',
                        help='Validate data integrity and fill gaps')
    
    parser.add_argument('--summary', action='store_true',
                        help='Show database summary and statistics')
    
    parser.add_argument('--add-symbol', type=str, metavar='SYMBOL',
                        help='Add a new trading symbol to database')
    
    parser.add_argument('--quick-update', action='store_true',
                        help='Quick update for recent data only')
    
    parser.add_argument('--start-date', type=str, default="2021-01-01",
                        help='Start date for historical data collection (default: 2021-01-01)')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    try:
        if args.summary:
            show_data_summary()
        
        if args.update:
            update_data()
        
        if args.collect_from:
            collect_historical_data(args.collect_from)
        
        if args.validate:
            validate_and_repair_data(args.start_date)
        
        if args.add_symbol:
            # Import pandas here since it's only needed for this function
            import pandas as pd
            add_new_symbol(args.add_symbol, args.start_date)
        
        if args.quick_update:
            quick_update()
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
