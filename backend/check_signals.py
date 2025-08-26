#!/usr/bin/env python3
"""
Quick Signal Checker for RSI Momentum Divergence Strategy

This script quickly checks for current trading signals based on the strategy.
"""

import sqlite3
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.Strategies import RSIMomentumDivergenceSwingStrategy


def check_current_signals():
    """Check for current trading signals"""
    strategy = RSIMomentumDivergenceSwingStrategy()
    db_path = 'data/trading_data_BTC.db'
    
    print(f"\n{'='*80}")
    print(f"RSI MOMENTUM DIVERGENCE STRATEGY - SIGNAL CHECK")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Get the strategy's SQL query
    query = strategy.get_sql_query()
    
    # Execute query
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        signals = []
        for row in cursor.fetchall():
            signal_dict = dict(zip(columns, row))
            signals.append(signal_dict)
    
    if not signals:
        print("❌ No signals found in the recent data.")
        print("\nThis could mean:")
        print("  • The market conditions don't currently meet the strategy criteria")
        print("  • RSI is in neutral territory (between 30-70)")
        print("  • No momentum shifts or crossovers detected")
    else:
        print(f"✅ Found {len(signals)} signals!\n")
        
        # Show last 10 signals
        print("📊 RECENT SIGNALS (Most Recent First):")
        print("-" * 80)
        
        for i, signal in enumerate(signals[:10]):
            timestamp = signal['timestamp']
            price = float(signal['price'])
            rsi = float(signal['rsi'])
            signal_name = signal['signal_name']
            trend = signal.get('trend_strength', 'N/A')
            divergence = signal.get('divergence_signal', 'N/A')
            
            # Add emoji based on signal type
            if 'BUY' in signal_name:
                emoji = "🟢"
            elif 'SELL' in signal_name:
                emoji = "🔴"
            elif 'EXIT' in signal_name:
                emoji = "🟡"
            else:
                emoji = "⚪"
            
            print(f"\n{emoji} Signal #{i+1}:")
            print(f"   Time: {timestamp}")
            print(f"   Type: {signal_name}")
            print(f"   Price: ${price:,.2f}")
            print(f"   RSI: {rsi:.2f}")
            print(f"   Trend: {trend}")
            print(f"   Divergence: {divergence}")
    
    # Get current market status
    print(f"\n{'='*80}")
    print("CURRENT MARKET STATUS:")
    print("-" * 80)
    
    status_query = """
    SELECT 
        o.timestamp,
        o.close as price,
        r.rsi,
        r.rsi_sma_5,
        r.rsi_sma_10,
        r.overbought,
        r.oversold,
        r.trend_strength,
        r.divergence_signal,
        r.momentum_shift
    FROM rsi_indicator r
    JOIN ohlcv_data o ON r.ohlcv_id = o.id
    ORDER BY o.timestamp DESC
    LIMIT 1
    """
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(status_query)
        current = cursor.fetchone()
        
        if current:
            timestamp, price, rsi, rsi_sma_5, rsi_sma_10, overbought, oversold, trend, divergence, momentum = current
            
            print(f"Latest Data: {timestamp}")
            print(f"Price: ${price:,.2f}")
            print(f"RSI: {rsi:.2f}")
            print(f"RSI SMA 5: {rsi_sma_5:.2f}")
            print(f"RSI SMA 10: {rsi_sma_10:.2f}")
            print(f"Status: {'OVERBOUGHT' if overbought else 'OVERSOLD' if oversold else 'NEUTRAL'}")
            print(f"Trend: {trend}")
            print(f"Divergence: {divergence}")
            print(f"Momentum Shift: {'YES' if momentum else 'NO'}")
            
            # Provide analysis
            print(f"\n📈 ANALYSIS:")
            if rsi > 70:
                print("  ⚠️  RSI is in overbought territory - potential sell signal incoming")
            elif rsi < 30:
                print("  ⚠️  RSI is in oversold territory - potential buy signal incoming")
            else:
                print("  ✓ RSI is in neutral territory")
            
            if rsi_sma_5 > rsi_sma_10:
                print("  ✓ Short-term RSI trend is bullish (SMA 5 > SMA 10)")
            else:
                print("  ✓ Short-term RSI trend is bearish (SMA 5 < SMA 10)")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    check_current_signals()
