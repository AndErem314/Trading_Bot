#!/usr/bin/env python3
"""
RSI (Relative Strength Index) Indicator Module
Calculates 14-period RSI with overbought/oversold levels, trend analysis, and divergence detection.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime


def calculate_rsi(prices, period=14):
    """
    Calculate RSI (Relative Strength Index) for given prices.
    
    Args:
        prices (pd.Series): Price series (typically close prices)
        period (int): RSI period (default: 14)
    
    Returns:
        pd.Series: RSI values (0-100)
    """
    if len(prices) < period + 1:
        return pd.Series([np.nan] * len(prices), index=prices.index)
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate initial average gain and loss using SMA
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    
    # Use Wilder's smoothing method for subsequent values
    for i in range(period, len(gains)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gains.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + losses.iloc[i]) / period
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def detect_rsi_signals(rsi_values, prices):
    """
    Detect RSI-based trading signals and patterns.
    
    Args:
        rsi_values (pd.Series): RSI values
        prices (pd.Series): Price series for divergence analysis
    
    Returns:
        dict: Dictionary containing various RSI signals and analysis
    """
    if len(rsi_values) < 5:
        return {
            'overbought_signal': False,
            'oversold_signal': False,
            'trend_strength': 'neutral',
            'divergence_signal': 'none',
            'support_resistance': None,
            'momentum_shift': False
        }
    
    current_rsi = rsi_values.iloc[-1]
    prev_rsi = rsi_values.iloc[-2] if len(rsi_values) >= 2 else current_rsi
    
    # Overbought/Oversold signals
    overbought_signal = current_rsi > 70
    oversold_signal = current_rsi < 30
    
    # Trend strength analysis
    recent_rsi = rsi_values.tail(5).mean()
    if recent_rsi > 60:
        trend_strength = 'strong_bullish'
    elif recent_rsi > 50:
        trend_strength = 'bullish'
    elif recent_rsi < 40:
        trend_strength = 'strong_bearish'
    elif recent_rsi < 50:
        trend_strength = 'bearish'
    else:
        trend_strength = 'neutral'
    
    # Divergence detection (simplified)
    divergence_signal = detect_rsi_divergence(rsi_values.tail(20), prices.tail(20))
    
    # Support/Resistance levels
    support_resistance = detect_rsi_support_resistance(rsi_values.tail(50))
    
    # Momentum shift detection
    momentum_shift = abs(current_rsi - prev_rsi) > 5
    
    return {
        'overbought_signal': overbought_signal,
        'oversold_signal': oversold_signal,
        'trend_strength': trend_strength,
        'divergence_signal': divergence_signal,
        'support_resistance': support_resistance,
        'momentum_shift': momentum_shift
    }


def detect_rsi_divergence(rsi_values, prices):
    """
    Detect bullish/bearish divergences between RSI and price.
    
    Args:
        rsi_values (pd.Series): RSI values (last 20 periods)
        prices (pd.Series): Price values (last 20 periods)
    
    Returns:
        str: 'bullish', 'bearish', or 'none'
    """
    if len(rsi_values) < 10 or len(prices) < 10:
        return 'none'
    
    try:
        # Find recent highs and lows in both RSI and price
        price_recent_high = prices.tail(10).max()
        price_recent_low = prices.tail(10).min()
        rsi_recent_high = rsi_values.tail(10).max()
        rsi_recent_low = rsi_values.tail(10).min()
        
        # Compare with earlier periods
        price_earlier_high = prices.head(10).max()
        price_earlier_low = prices.head(10).min()
        rsi_earlier_high = rsi_values.head(10).max()
        rsi_earlier_low = rsi_values.head(10).min()
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (price_recent_low < price_earlier_low and 
            rsi_recent_low > rsi_earlier_low and
            rsi_recent_low < 40):
            return 'bullish'
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (price_recent_high > price_earlier_high and 
            rsi_recent_high < rsi_earlier_high and
            rsi_recent_high > 60):
            return 'bearish'
        
        return 'none'
    
    except Exception:
        return 'none'


def detect_rsi_support_resistance(rsi_values):
    """
    Detect key RSI support and resistance levels.
    
    Args:
        rsi_values (pd.Series): RSI values (last 50 periods)
    
    Returns:
        dict: Support and resistance levels
    """
    if len(rsi_values) < 20:
        return {'support': None, 'resistance': None}
    
    try:
        # Common RSI levels
        levels = [30, 40, 50, 60, 70]
        current_rsi = rsi_values.iloc[-1]
        
        # Find nearest support (below current RSI)
        support_levels = [level for level in levels if level < current_rsi]
        support = max(support_levels) if support_levels else 30
        
        # Find nearest resistance (above current RSI)
        resistance_levels = [level for level in levels if level > current_rsi]
        resistance = min(resistance_levels) if resistance_levels else 70
        
        return {
            'support': support,
            'resistance': resistance
        }
    
    except Exception:
        return {'support': None, 'resistance': None}


def calculate_rsi_for_symbol_timeframe(symbol, timeframe):
    """
    Calculate RSI for a specific symbol and timeframe from unified database.
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC/USDT')
        timeframe (str): Timeframe (e.g., '4h', '1d')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Connect to unified database
        import os
        db_path = 'data/trading_data_BTC.db'
        if not os.path.isabs(db_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            db_path = os.path.abspath(os.path.join(project_root, db_path))
        raw_conn = sqlite3.connect(db_path)
        
        # Read OHLCV data from unified database
        query = """
        SELECT o.timestamp, o.open, o.high, o.low, o.close, o.volume 
        FROM ohlcv_data o
        JOIN symbols s ON o.symbol_id = s.id
        JOIN timeframes t ON o.timeframe_id = t.id
        WHERE s.symbol = ? AND t.timeframe = ? 
        ORDER BY o.timestamp ASC
        """
        
        df = pd.read_sql_query(query, raw_conn, params=(symbol, timeframe))
        
        if df.empty:
            print(f"[WARNING] No data found for {symbol} {timeframe}")
            raw_conn.close()
            return False
        
        raw_conn.close()
        
        print(f"[INFO] Calculating RSI for {symbol} {timeframe} - {len(df)} candles")
        
        # Calculate RSI
        rsi_values = calculate_rsi(df['close'])
        
        # Create RSI dataframe
        rsi_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'symbol': symbol,
            'timeframe': timeframe,
            'rsi': rsi_values,
            'rsi_sma_5': rsi_values.rolling(window=5).mean(),
            'rsi_sma_10': rsi_values.rolling(window=10).mean()
        })
        
        # Add signal analysis for each row
        signals_list = []
        for i in range(len(df)):
            if i < 20:  # Need enough data for signal analysis
                signals = {
                    'overbought_signal': False,
                    'oversold_signal': False,
                    'trend_strength': 'neutral',
                    'divergence_signal': 'none',
                    'support_resistance': None,
                    'momentum_shift': False
                }
            else:
                # Get data up to current point for signal analysis
                current_rsi = rsi_values.iloc[:i+1]
                current_prices = df['close'].iloc[:i+1]
                signals = detect_rsi_signals(current_rsi, current_prices)
            
            signals_list.append(signals)
        
        # Add signal columns
        rsi_df['overbought'] = [s['overbought_signal'] for s in signals_list]
        rsi_df['oversold'] = [s['oversold_signal'] for s in signals_list]
        rsi_df['trend_strength'] = [s['trend_strength'] for s in signals_list]
        rsi_df['divergence_signal'] = [s['divergence_signal'] for s in signals_list]
        rsi_df['momentum_shift'] = [s['momentum_shift'] for s in signals_list]
        
        # Support/resistance levels (convert dict to string for storage)
        rsi_df['support_resistance'] = [
            f"S:{s['support_resistance']['support']},R:{s['support_resistance']['resistance']}" 
            if s['support_resistance'] and s['support_resistance']['support'] is not None 
            else None for s in signals_list
        ]
        
        # Save to unified database using RSI indicators table
        rsi_conn = sqlite3.connect(db_path)
        
        inserted = 0
        skipped = 0
        
        for _, row in rsi_df.iterrows():
            # Get OHLCV data ID for this timestamp
            ohlcv_cursor = rsi_conn.execute('''
                SELECT o.id FROM ohlcv_data o
                JOIN symbols s ON o.symbol_id = s.id
                JOIN timeframes t ON o.timeframe_id = t.id
                WHERE s.symbol = ? AND t.timeframe = ? AND o.timestamp = ?
            ''', (symbol, timeframe, row['timestamp']))
            
            ohlcv_result = ohlcv_cursor.fetchone()
            if not ohlcv_result:
                skipped += 1
                continue
            
            ohlcv_id = ohlcv_result[0]
            
            # Insert or replace RSI indicator data
            rsi_conn.execute('''
                INSERT OR REPLACE INTO rsi_indicator 
                (ohlcv_id, rsi, rsi_sma_5, rsi_sma_10, overbought, oversold, 
                 trend_strength, divergence_signal, momentum_shift, support_resistance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ohlcv_id,
                row['rsi'] if pd.notna(row['rsi']) else None,
                row['rsi_sma_5'] if pd.notna(row['rsi_sma_5']) else None,
                row['rsi_sma_10'] if pd.notna(row['rsi_sma_10']) else None,
                bool(row['overbought']),
                bool(row['oversold']),
                row['trend_strength'],
                row['divergence_signal'],
                bool(row['momentum_shift']),
                row['support_resistance']
            ))
            inserted += 1
        
        rsi_conn.commit()
        rsi_conn.close()
        
        print(f"[INFO] Saved {inserted} RSI indicators for {symbol} ({timeframe})")
        if skipped > 0:
            print(f"[WARNING] Skipped {skipped} records (no matching OHLCV data)")
        
        # Calculate statistics
        valid_rsi = rsi_df['rsi'].dropna()
        if len(valid_rsi) > 0:
            overbought_count = sum(rsi_df['overbought'])
            oversold_count = sum(rsi_df['oversold'])
            
            print(f"[SUCCESS] RSI calculated for {symbol} {timeframe}")
            print(f"  - Total records: {len(rsi_df)}")
            print(f"  - Average RSI: {valid_rsi.mean():.2f}")
            print(f"  - Current RSI: {valid_rsi.iloc[-1]:.2f}")
            print(f"  - Overbought signals: {overbought_count}")
            print(f"  - Oversold signals: {oversold_count}")
            print(f"  - Current trend: {rsi_df['trend_strength'].iloc[-1]}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to calculate RSI for {symbol} {timeframe}: {e}")
        return False


def calculate_rsi_all_symbols(symbols, timeframes):
    """
    Calculate RSI for all specified symbols and timeframes.
    
    Args:
        symbols (list): List of trading symbols
        timeframes (list): List of timeframes
    
    Returns:
        bool: True if all calculations successful
    """
    print("=== RSI (14-period) Calculation Started ===")
    
    success_count = 0
    total_count = len(symbols) * len(timeframes)
    
    for symbol in symbols:
        for timeframe in timeframes:
            if calculate_rsi_for_symbol_timeframe(symbol, timeframe):
                success_count += 1
            print()  # Empty line for readability
    
    print(f"=== RSI Calculation Completed ===")
    print(f"Success: {success_count}/{total_count}")
    
    return success_count == total_count


def main():
    """Main function for RSI calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RSI Indicator Calculator')
    parser.add_argument('--symbols', nargs='+', 
                       default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC'],
                       help='Symbols to process')
    parser.add_argument('--timeframes', nargs='+', 
                       default=['4h', '1d'],
                       help='Timeframes to process')
    
    args = parser.parse_args()
    
    print("RSI (Relative Strength Index) Calculator")
    print(f"Symbols: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")
    print()
    
    try:
        success = calculate_rsi_all_symbols(args.symbols, args.timeframes)
        
        if success:
            print("\n=== RSI CALCULATION COMPLETED SUCCESSFULLY ===")
        else:
            print("\n=== RSI CALCULATION COMPLETED WITH ERRORS ===")
            
    except KeyboardInterrupt:
        print("\n[INFO] RSI calculation interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] RSI calculation failed: {e}")


if __name__ == '__main__':
    main()
