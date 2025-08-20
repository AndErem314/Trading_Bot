"""
Parabolic SAR Calculator module - UNIFIED VERSION
Uses unified_trading_data.db to calculate Parabolic SAR (Stop and Reverse) indicator and saves it to the unified database.

Parabolic SAR is a trend-following indicator that provides potential reversal points in the price action.
It's designed to give traders an entry and exit point for trades.

This module implements:
- Calculation of Parabolic SAR with configurable step and maximum settings
- Detection of trend reversals based on Parabolic SAR
- Signal generation for trend changes
- Signal strength based on distance between price and SAR

The Parabolic SAR appears as dots above or below the price:
- Dots below price indicate an uptrend
- Dots above price indicate a downtrend
- When dots switch sides, it signals a potential trend reversal
"""

import pandas as pd
import sqlite3
import numpy as np
from typing import Optional


class ParabolicSARCalculator:
    """Calculates Parabolic SAR indicator using unified database."""
    
    def __init__(self, db_path: str = 'data/unified_trading_data.db'):
        # Ensure we use the correct database path relative to project root
        import os
        if not os.path.isabs(db_path) and not db_path.startswith('../'):
            # If running from Indicators subfolder, adjust path to project root
            current_dir = os.path.basename(os.getcwd())
            if current_dir == 'Indicators':
                db_path = '../../' + db_path
            elif current_dir == 'backend':
                db_path = '../' + db_path
        self.db_path = db_path
    
    def fetch_raw_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch raw OHLCV data from the unified database."""
        query = '''
            SELECT o.timestamp, o.open, o.high, o.low, o.close, o.volume
            FROM ohlcv_data o
            JOIN symbols s ON o.symbol_id = s.id
            JOIN timeframes t ON o.timeframe_id = t.id
            WHERE s.symbol = ? AND t.timeframe = ?
            ORDER BY o.timestamp ASC
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=(symbol, timeframe))

    def calculate_parabolic_sar(self, df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
        """
        Calculate Parabolic SAR on data.
        
        Parameters:
        - df: DataFrame with OHLCV data
        - step: Acceleration factor step (default: 0.02)
        - max_step: Maximum acceleration factor (default: 0.2)
        
        Returns:
        - DataFrame with Parabolic SAR indicators added
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Initialize columns
        df['parabolic_sar'] = np.nan
        df['trend'] = 'up'
        df['reversal_signal'] = False
        df['signal_strength'] = 0.0
        df['acceleration_factor'] = step

        # Initialize first values
        df.at[0, 'parabolic_sar'] = df['low'].iloc[0]  # Start with low as initial SAR
        df.at[0, 'trend'] = 'up'
        df.at[0, 'signal_strength'] = 0.0
        df.at[0, 'acceleration_factor'] = step
        
        # Initialize variables for calculation
        ep = df['high'].iloc[0]  # extreme point
        af = step  # acceleration factor
        sar = df['low'].iloc[0]  # parabolic SAR value
        trend = 'up'

        for i in range(1, len(df)):
            prior_sar = sar
            prior_trend = trend
            
            if prior_trend == 'up':
                # Calculate SAR for uptrend
                sar = prior_sar + af * (ep - prior_sar)
                
                # Ensure SAR doesn't exceed previous two lows for uptrend
                sar = min(sar, df['low'].iloc[i - 1])
                if i > 1:
                    sar = min(sar, df['low'].iloc[i - 2])
                
                # Check for trend reversal
                if df['low'].iloc[i] <= sar:
                    # Trend reversal from up to down
                    trend = 'down'
                    df.at[i, 'reversal_signal'] = True
                    ep = df['low'].iloc[i]  # New extreme point
                    af = step  # Reset acceleration factor
                    sar = max(df['high'].iloc[i-1], ep)  # Use previous high as new SAR
                else:
                    # Continue uptrend
                    trend = 'up'
                    df.at[i, 'reversal_signal'] = False
                    # Update extreme point and acceleration factor if new high
                    if df['high'].iloc[i] > ep:
                        ep = df['high'].iloc[i]
                        af = min(af + step, max_step)
            else:
                # Calculate SAR for downtrend
                sar = prior_sar + af * (ep - prior_sar)
                
                # Ensure SAR doesn't go below previous two highs for downtrend
                sar = max(sar, df['high'].iloc[i - 1])
                if i > 1:
                    sar = max(sar, df['high'].iloc[i - 2])
                
                # Check for trend reversal
                if df['high'].iloc[i] >= sar:
                    # Trend reversal from down to up
                    trend = 'up'
                    df.at[i, 'reversal_signal'] = True
                    ep = df['high'].iloc[i]  # New extreme point
                    af = step  # Reset acceleration factor
                    sar = min(df['low'].iloc[i-1], ep)  # Use previous low as new SAR
                else:
                    # Continue downtrend
                    trend = 'down'
                    df.at[i, 'reversal_signal'] = False
                    # Update extreme point and acceleration factor if new low
                    if df['low'].iloc[i] < ep:
                        ep = df['low'].iloc[i]
                        af = min(af + step, max_step)

            # Set values for current period
            df.at[i, 'parabolic_sar'] = sar
            df.at[i, 'trend'] = trend
            df.at[i, 'acceleration_factor'] = af
            
            # Calculate signal strength based on distance between price and SAR
            if trend == 'up':
                df.at[i, 'signal_strength'] = ((df['close'].iloc[i] - sar) / sar) * 100
            else:
                df.at[i, 'signal_strength'] = ((sar - df['close'].iloc[i]) / sar) * 100

        # Generate trading signals based on SAR
        df = self._generate_sar_signals(df)

        return df

    def _generate_sar_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Parabolic SAR."""
        df['sar_signal'] = 'hold'
        
        # Buy signals: Trend changes to up (reversal from down to up)
        buy_condition = (df['trend'] == 'up') & df['reversal_signal']
        df.loc[buy_condition, 'sar_signal'] = 'buy'
        
        # Sell signals: Trend changes to down (reversal from up to down)
        sell_condition = (df['trend'] == 'down') & df['reversal_signal']
        df.loc[sell_condition, 'sar_signal'] = 'sell'
        
        # Strong buy: Uptrend with high signal strength
        strong_buy_condition = (df['trend'] == 'up') & (df['signal_strength'] > 5) & ~df['reversal_signal']
        df.loc[strong_buy_condition, 'sar_signal'] = 'strong_buy'
        
        # Strong sell: Downtrend with high signal strength
        strong_sell_condition = (df['trend'] == 'down') & (df['signal_strength'] > 5) & ~df['reversal_signal']
        df.loc[strong_sell_condition, 'sar_signal'] = 'strong_sell'
        
        return df

    def save_parabolic_sar_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated Parabolic SAR data to the unified database."""
        if df.empty:
            print(f"[INFO] No Parabolic SAR data to save for {symbol} ({timeframe})")
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                inserted = 0
                skipped = 0
                
                for _, row in df.iterrows():
                    # Get OHLCV data ID for this timestamp
                    ohlcv_cursor = conn.execute('''
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
                    
                    # Insert or replace Parabolic SAR indicator data
                    conn.execute('''
                        INSERT OR REPLACE INTO parabolic_sar_indicator 
                        (ohlcv_id, parabolic_sar, trend, reversal_signal, signal_strength, 
                         acceleration_factor, sar_signal)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ohlcv_id,
                        row['parabolic_sar'] if pd.notna(row['parabolic_sar']) else None,
                        row['trend'],
                        bool(row['reversal_signal']),
                        row['signal_strength'] if pd.notna(row['signal_strength']) else None,
                        row['acceleration_factor'] if pd.notna(row['acceleration_factor']) else None,
                        row['sar_signal']
                    ))
                    inserted += 1
                
                conn.commit()
                print(f"[INFO] Saved {inserted} Parabolic SAR indicators for {symbol} ({timeframe})")
                if skipped > 0:
                    print(f"[WARNING] Skipped {skipped} records (no matching OHLCV data)")
                    
        except Exception as e:
            print(f"[ERROR] Failed to save Parabolic SAR data for {symbol}: {e}")

    def analyze_parabolic_sar_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze current Parabolic SAR patterns and provide insights."""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        recent_data = df.tail(20)  # Last 20 periods
        
        # Count recent reversals
        recent_reversals = recent_data['reversal_signal'].sum()
        
        # Find most recent reversal
        recent_reversal_date = None
        if recent_reversals > 0:
            reversal_data = recent_data[recent_data['reversal_signal'] == True]
            if not reversal_data.empty:
                recent_reversal_date = reversal_data.iloc[-1]['timestamp']
        
        # Calculate trend persistence
        trend_persistence = self._calculate_trend_persistence(recent_data)
        
        analysis = {
            'current_trend': latest['trend'],
            'current_sar': latest['parabolic_sar'],
            'current_signal': latest['sar_signal'],
            'signal_strength': latest['signal_strength'],
            'acceleration_factor': latest['acceleration_factor'],
            'price_vs_sar': self._get_price_sar_relationship(
                latest['close'], latest['parabolic_sar'], latest['trend']
            ),
            'recent_reversals': int(recent_reversals),
            'last_reversal_date': recent_reversal_date,
            'trend_persistence': trend_persistence,
            'current_values': {
                'close': latest['close'],
                'parabolic_sar': latest['parabolic_sar'],
                'distance_to_sar': abs(latest['close'] - latest['parabolic_sar'])
            }
        }
        
        return analysis
    
    def _get_price_sar_relationship(self, close_price: float, sar_value: float, trend: str) -> str:
        """Get descriptive text for price vs SAR relationship."""
        if trend == 'up':
            distance_pct = ((close_price - sar_value) / sar_value) * 100
            if distance_pct > 10:
                return f"Strong uptrend (price {distance_pct:.1f}% above SAR)"
            elif distance_pct > 3:
                return f"Moderate uptrend (price {distance_pct:.1f}% above SAR)"
            else:
                return f"Weak uptrend (price {distance_pct:.1f}% above SAR)"
        else:
            distance_pct = ((sar_value - close_price) / sar_value) * 100
            if distance_pct > 10:
                return f"Strong downtrend (price {distance_pct:.1f}% below SAR)"
            elif distance_pct > 3:
                return f"Moderate downtrend (price {distance_pct:.1f}% below SAR)"
            else:
                return f"Weak downtrend (price {distance_pct:.1f}% below SAR)"
    
    def _calculate_trend_persistence(self, recent_data: pd.DataFrame) -> str:
        """Calculate how persistent the current trend has been."""
        if recent_data.empty:
            return "unknown"
        
        current_trend = recent_data['trend'].iloc[-1]
        consecutive_periods = 0
        
        # Count consecutive periods with same trend
        for i in range(len(recent_data) - 1, -1, -1):
            if recent_data['trend'].iloc[i] == current_trend:
                consecutive_periods += 1
            else:
                break
        
        if consecutive_periods >= 15:
            return "very_persistent"
        elif consecutive_periods >= 10:
            return "persistent"
        elif consecutive_periods >= 5:
            return "moderate"
        else:
            return "weak"

    def get_sar_reversals(self, symbol: str, timeframe: str, limit: int = 10) -> pd.DataFrame:
        """Get recent SAR reversal history for a symbol."""
        query = '''
            SELECT o.timestamp, o.close, sar.parabolic_sar, sar.trend, 
                   sar.sar_signal, sar.signal_strength
            FROM ohlcv_data o
            JOIN symbols s ON o.symbol_id = s.id
            JOIN timeframes t ON o.timeframe_id = t.id
            JOIN parabolic_sar_indicator sar ON o.id = sar.ohlcv_id
            WHERE s.symbol = ? AND t.timeframe = ? AND sar.reversal_signal = 1
            ORDER BY o.timestamp DESC
            LIMIT ?
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=(symbol, timeframe, limit))


def main():
    """Main function to calculate and save Parabolic SAR data for multiple symbols and timeframes."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
    timeframes = ['4h', '1d']

    calculator = ParabolicSARCalculator()

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[CALCULATING] Parabolic SAR for {symbol} - {timeframe.upper()}")
            df_raw = calculator.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            
            # Calculate Parabolic SAR
            df_sar = calculator.calculate_parabolic_sar(df_raw)
            
            # Save to database
            calculator.save_parabolic_sar_data(df_sar, symbol, timeframe)
            
            # Print analysis for the latest data
            analysis = calculator.analyze_parabolic_sar_patterns(df_sar)
            if analysis:
                print(f"[ANALYSIS] Current trend: {analysis['current_trend']}")
                print(f"[ANALYSIS] Current signal: {analysis['current_signal']}")
                print(f"[ANALYSIS] Price vs SAR: {analysis['price_vs_sar']}")
                print(f"[ANALYSIS] Signal strength: {analysis['signal_strength']:.2f}%")
                print(f"[ANALYSIS] Trend persistence: {analysis['trend_persistence']}")
                if analysis['recent_reversals'] > 0:
                    print(f"[ANALYSIS] Recent reversals: {analysis['recent_reversals']}")


if __name__ == '__main__':
    main()
