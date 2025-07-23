#!/usr/bin/env python3
"""
Parabolic SAR Demo Script
Demonstrates the new Parabolic SAR indicator functionality
"""

import sys
sys.path.append('backend')

from parabolic_sar import ParabolicSARCalculator

def main():
    print("=== PARABOLIC SAR INDICATOR DEMO ===")
    print("Demonstrating the new Parabolic SAR (Stop and Reverse) indicator\n")
    
    # Initialize calculator
    calculator = ParabolicSARCalculator()
    
    # Test with BTC/USDT daily data
    symbol = 'BTC/USDT'
    timeframe = '1d'
    
    print(f"Fetching data for {symbol} ({timeframe})...")
    df_raw = calculator.fetch_raw_data(symbol, timeframe)
    
    if df_raw.empty:
        print(f"No data available for {symbol} ({timeframe})")
        return
    
    print(f"Calculating Parabolic SAR for {len(df_raw)} records...")
    df_sar = calculator.calculate_parabolic_sar(df_raw)
    
    # Analyze patterns
    print("\n=== PARABOLIC SAR ANALYSIS ===")
    analysis = calculator.analyze_parabolic_sar_patterns(df_sar)
    
    print(f"Current Trend: {analysis['current_trend'].upper()}")
    print(f"Current Price: ${analysis['sar_values']['current_price']:,.2f}")
    print(f"Current SAR: ${analysis['current_sar']:,.2f}")
    print(f"Signal Strength: {analysis['signal_strength']:.2f}%")
    print(f"Price vs SAR: {analysis['price_vs_sar']}")
    print(f"Recent Reversals (last 20 periods): {analysis['recent_reversals']}")
    print(f"Trend Persistence: {analysis['trend_persistence']}")
    
    if analysis['last_reversal_date']:
        print(f"Last Reversal Date: {analysis['last_reversal_date']}")
    
    # Show recent reversal history
    print(f"\n=== RECENT REVERSAL HISTORY ({symbol}) ===")
    reversal_history = calculator.get_reversal_history(symbol, timeframe, 5)
    
    if not reversal_history.empty:
        print("Date                | Close Price | SAR Value | New Trend | Signal Strength")
        print("-" * 75)
        for _, row in reversal_history.iterrows():
            date = row['timestamp']
            close = row['close']
            sar = row['parabolic_sar']
            trend = row['trend']
            strength = row['signal_strength']
            print(f"{date} | ${close:>9,.2f} | ${sar:>8,.2f} | {trend:>8s} | {strength:>10.2f}%")
    else:
        print("No recent reversals found.")
    
    # Show some statistics
    print(f"\n=== PARABOLIC SAR STATISTICS ===")
    reversals = df_sar['reversal_signal'].sum()
    uptrend_periods = (df_sar['trend'] == 'up').sum()
    downtrend_periods = (df_sar['trend'] == 'down').sum()
    
    print(f"Total Records: {len(df_sar):,}")
    print(f"Total Reversals: {reversals:,}")
    print(f"Uptrend Periods: {uptrend_periods:,} ({uptrend_periods/len(df_sar)*100:.1f}%)")
    print(f"Downtrend Periods: {downtrend_periods:,} ({downtrend_periods/len(df_sar)*100:.1f}%)")
    print(f"Average Signal Strength: {df_sar['signal_strength'].mean():.2f}%")
    
    print(f"\n=== DATABASE INFORMATION ===")
    print(f"Database Path: {calculator.parabolic_sar_db_path}")
    print(f"Raw Data Source: {calculator.raw_db_path}")
    print("Parabolic SAR data successfully stored in dedicated database!")
    
    print(f"\n=== PARABOLIC SAR DEMO COMPLETED ===")

if __name__ == '__main__':
    main()
