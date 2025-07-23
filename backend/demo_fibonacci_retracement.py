#!/usr/bin/env python3
"""
Fibonacci Retracement Demo Script
Demonstrates the new Fibonacci Retracement indicator functionality
"""

import sys
sys.path.append('backend')

from fibonacci_retracement import FibonacciRetracementCalculator

def main():
    print("=== FIBONACCI RETRACEMENT INDICATOR DEMO ===")
    print("Demonstrating the new Fibonacci Retracement indicator\n")
    
    # Initialize calculator
    calculator = FibonacciRetracementCalculator()
    
    # Test with BTC/USDT daily data
    symbol = 'BTC/USDT'
    timeframe = '1d'
    
    print(f"Fetching data for {symbol} ({timeframe})...")
    df_raw = calculator.fetch_raw_data(symbol, timeframe)
    
    if df_raw.empty:
        print(f"No data available for {symbol} ({timeframe})")
        return
    
    print(f"Calculating Fibonacci Retracement levels for {len(df_raw)} records...")
    df_fib = calculator.calculate_fibonacci_retracement(df_raw)
    
    # Show the calculated levels
    print(f"\n=== FIBONACCI RETRACEMENT LEVELS ===")
    print(f"Based on High: ${df_raw['high'].max():,.2f}")
    print(f"Based on Low:  ${df_raw['low'].min():,.2f}")
    print(f"Range: ${df_raw['high'].max() - df_raw['low'].min():,.2f}")
    print()
    
    # Show the latest Fibonacci levels
    latest = df_fib.iloc[-1]
    print("Current Fibonacci Retracement Levels:")
    print(f"├─ 23.6%: ${latest['level_23_6']:,.2f}")
    print(f"├─ 38.2%: ${latest['level_38_2']:,.2f}")
    print(f"├─ 50.0%: ${latest['level_50_0']:,.2f}")
    print(f"├─ 61.8%: ${latest['level_61_8']:,.2f}")
    print(f"└─ 76.4%: ${latest['level_76_4']:,.2f}")
    
    # Show some statistics
    print(f"\n=== FIBONACCI RETRACEMENT STATISTICS ===")
    print(f"Total Records: {len(df_fib):,}")
    print(f"All records have the same retracement levels (based on overall high/low)")
    
    # Calculate what percentage of current price relative to the range
    current_price = 118506.83  # Latest BTC price from previous data
    high = df_raw['high'].max()
    low = df_raw['low'].min()
    current_position = (high - current_price) / (high - low) * 100
    
    print(f"\nCurrent Price Analysis:")
    print(f"Current BTC Price: ${current_price:,.2f}")
    print(f"Position in Range: {current_position:.1f}% retracement from high")
    
    # Determine which Fibonacci level the current price is closest to
    levels = {
        '23.6%': latest['level_23_6'],
        '38.2%': latest['level_38_2'],
        '50.0%': latest['level_50_0'],
        '61.8%': latest['level_61_8'],
        '76.4%': latest['level_76_4']
    }
    
    closest_level = min(levels.items(), key=lambda x: abs(x[1] - current_price))
    print(f"Closest Fibonacci Level: {closest_level[0]} (${closest_level[1]:,.2f})")
    print(f"Distance: ${abs(closest_level[1] - current_price):,.2f}")
    
    print(f"\n=== DATABASE INFORMATION ===")
    print(f"Database Path: {calculator.fib_db_path}")
    print(f"Raw Data Source: {calculator.raw_db_path}")
    print("Fibonacci Retracement data successfully stored in dedicated database!")
    
    print(f"\n=== FIBONACCI RETRACEMENT DEMO COMPLETED ===")

if __name__ == '__main__':
    main()
