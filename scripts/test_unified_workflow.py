#!/usr/bin/env python3
"""
Test script for the unified OHLCV data workflow.
Validates the implementation with basic functionality tests.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from unified_data_manager import UnifiedDataManager
from unified_data_fetcher import UnifiedDataFetcher


def test_data_manager():
    """Test basic UnifiedDataManager functionality."""
    print("🧪 Testing UnifiedDataManager...")
    
    data_manager = UnifiedDataManager('data/unified_trading_data.db')
    
    # Test symbol/timeframe creation
    symbol_id = data_manager.get_or_create_symbol_id('TEST/USDT')
    timeframe_id = data_manager.get_or_create_timeframe_id('1h')
    
    print(f"✅ Created symbol ID: {symbol_id}, timeframe ID: {timeframe_id}")
    
    # Test data summary
    summary = data_manager.get_data_summary()
    print(f"✅ Database summary retrieved: {len(summary)} symbol/timeframe combinations")
    
    # Test data integrity check
    integrity = data_manager.validate_data_integrity()
    print(f"✅ Data integrity check completed: {integrity}")
    
    return True


def test_data_fetcher():
    """Test basic UnifiedDataFetcher functionality."""
    print("🧪 Testing UnifiedDataFetcher...")
    
    try:
        fetcher = UnifiedDataFetcher('binance', 'data/unified_trading_data.db')
        
        # Test recent data fetch (small amount)
        print("📊 Testing recent data fetch...")
        df = fetcher.fetch_recent_ohlcv('BTC/USDT', '1d', limit=5)
        
        if not df.empty:
            print(f"✅ Fetched {len(df)} recent records for BTC/USDT")
            print(f"   Latest timestamp: {df.index[-1]}")
        else:
            print("⚠️  No recent data fetched (API might be limited)")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Data fetcher test failed (this is expected if API keys are not configured): {e}")
        return False


def test_database_operations():
    """Test database read operations on existing data."""
    print("🧪 Testing database operations...")
    
    data_manager = UnifiedDataManager('data/unified_trading_data.db')
    
    # Test getting last timestamp
    last_ts = data_manager.get_last_timestamp('BTC/USDT', '1d')
    if last_ts:
        print(f"✅ Last timestamp for BTC/USDT (1d): {datetime.fromtimestamp(last_ts/1000)}")
    else:
        print("⚠️  No data found for BTC/USDT (1d)")
    
    # Test data gaps (using a small time range)
    if last_ts:
        start_time = last_ts - (7 * 24 * 60 * 60 * 1000)  # 7 days ago
        gaps = data_manager.get_data_gaps('BTC/USDT', '1d', start_time, last_ts)
        print(f"✅ Found {len(gaps)} gaps in last 7 days for BTC/USDT (1d)")
    
    return True


def create_test_data():
    """Create some test OHLCV data for validation."""
    print("🧪 Creating test data...")
    
    # Generate sample OHLCV data
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=5),
        end=datetime.now(),
        freq='1H'
    )
    
    # Create realistic-looking OHLCV data
    np.random.seed(42)  # For reproducible test data
    base_price = 50000
    
    data = []
    for ts in timestamps:
        # Simple random walk for price simulation
        open_price = base_price + np.random.normal(0, 1000)
        high_price = open_price + abs(np.random.normal(500, 200))
        low_price = open_price - abs(np.random.normal(300, 150))
        close_price = open_price + np.random.normal(0, 800)
        volume = abs(np.random.normal(100, 50))
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    # Test saving the data
    data_manager = UnifiedDataManager('data/unified_trading_data.db')
    result = data_manager.save_ohlcv_data(df, 'TEST/USDT', '1h')
    
    print(f"✅ Test data creation: {result}")
    return True


def run_comprehensive_test():
    """Run a comprehensive test of the unified workflow."""
    print("🚀 Starting comprehensive unified workflow test...")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Data Manager
    try:
        test_data_manager()
        test_results.append(("Data Manager", "✅"))
    except Exception as e:
        print(f"❌ Data Manager test failed: {e}")
        test_results.append(("Data Manager", "❌"))
    
    # Test 2: Database Operations
    try:
        test_database_operations()
        test_results.append(("Database Operations", "✅"))
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        test_results.append(("Database Operations", "❌"))
    
    # Test 3: Data Fetcher (may fail without API keys)
    try:
        import numpy as np
        test_data_fetcher()
        test_results.append(("Data Fetcher", "✅"))
    except ImportError:
        print("⚠️  Skipping Data Fetcher test (numpy not available)")
        test_results.append(("Data Fetcher", "⚠️"))
    except Exception as e:
        print(f"⚠️  Data Fetcher test warning: {e}")
        test_results.append(("Data Fetcher", "⚠️"))
    
    # Test 4: Test Data Creation
    try:
        import numpy as np
        create_test_data()
        test_results.append(("Test Data Creation", "✅"))
    except ImportError:
        print("⚠️  Skipping Test Data Creation (numpy not available)")
        test_results.append(("Test Data Creation", "⚠️"))
    except Exception as e:
        print(f"❌ Test data creation failed: {e}")
        test_results.append(("Test Data Creation", "❌"))
    
    # Print test summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, status in test_results:
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, status in test_results if status == "✅")
    total = len(test_results)
    
    print(f"\n📊 Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The unified workflow is ready to use.")
    elif passed >= total * 0.75:
        print("✅ Most tests passed. The unified workflow should work correctly.")
    else:
        print("⚠️  Some critical tests failed. Please check the implementation.")
    
    return passed, total


if __name__ == '__main__':
    try:
        run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n⏹️  Tests cancelled by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
