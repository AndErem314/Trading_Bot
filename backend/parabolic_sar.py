"""
Parabolic SAR Calculator module.
Uses raw data from the database to calculate Parabolic SAR (Stop and Reverse) indicator and saves it to the database.

Parabolic SAR is a trend-following indicator that provides potential reversal points in the price action.

This module implements:
- Calculation of Parabolic SAR with default step and maximum settings
- Detection of trend reversals based on Parabolic SAR
"""

import pandas as pd
import sqlite3
import numpy as np
from typing import Optional


class ParabolicSARCalculator:
    """Calculates Parabolic SAR indicator from raw data."""
    
    def __init__(self, raw_db_path: str = 'data/raw_market_data.db', parabolic_sar_db_path: str = 'data/parabolic_sar_data.db'):
        self.raw_db_path = raw_db_path
        self.parabolic_sar_db_path = parabolic_sar_db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with Parabolic SAR table if it doesn't exist."""
        with sqlite3.connect(self.parabolic_sar_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS parabolic_sar_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    parabolic_sar REAL,
                    trend TEXT,
                    reversal_signal BOOLEAN,
                    signal_strength REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            conn.commit()
    
    def fetch_raw_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch raw OHLCV data from the raw database."""
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM raw_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp ASC
        '''
        with sqlite3.connect(self.raw_db_path) as conn:
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
        df['parabolic_sar'] = np.nan
        df['trend'] = 'up'
        df['reversal_signal'] = False
        df['signal_strength'] = 0.0

        # Initialize first values
        df.at[0, 'parabolic_sar'] = df['low'][0]  # Start with low as initial SAR
        df.at[0, 'trend'] = 'up'
        df.at[0, 'signal_strength'] = 0.0
        
        ep = df['high'][0]  # extreme point
        af = step  # acceleration factor
        sar = df['low'][0]  # parabolic SAR value

        for i in range(1, len(df)):
            prior_sar = sar
            prior_trend = df['trend'][i - 1]
            
            if prior_trend == 'up':
                sar = prior_sar + af * (ep - prior_sar)
                # Ensure SAR doesn't exceed previous two lows for uptrend
                sar = min(sar, df['low'][i - 1])
                if i > 1:
                    sar = min(sar, df['low'][i - 2])
                
                if df['low'][i] <= sar:
                    # Trend reversal from up to down
                    df.at[i, 'trend'] = 'down'
                    df.at[i, 'reversal_signal'] = True
                    ep = df['low'][i]
                    af = step
                    sar = df['high'][i - 1]  # Use previous high as new SAR
                else:
                    df.at[i, 'trend'] = 'up'
                    df.at[i, 'reversal_signal'] = False
                    if df['high'][i] > ep:
                        ep = df['high'][i]
                        af = min(af + step, max_step)
            else:
                sar = prior_sar + af * (ep - prior_sar)
                # Ensure SAR doesn't exceed previous two highs for downtrend
                sar = max(sar, df['high'][i - 1])
                if i > 1:
                    sar = max(sar, df['high'][i - 2])
                
                if df['high'][i] >= sar:
                    # Trend reversal from down to up
                    df.at[i, 'trend'] = 'up'
                    df.at[i, 'reversal_signal'] = True
                    ep = df['high'][i]
                    af = step
                    sar = df['low'][i - 1]  # Use previous low as new SAR
                else:
                    df.at[i, 'trend'] = 'down'
                    df.at[i, 'reversal_signal'] = False
                    if df['low'][i] < ep:
                        ep = df['low'][i]
                        af = min(af + step, max_step)

            df.at[i, 'parabolic_sar'] = sar
            
            # Calculate signal strength based on distance between price and SAR
            if df['trend'][i] == 'up':
                df.at[i, 'signal_strength'] = (df['close'][i] - sar) / sar * 100
            else:
                df.at[i, 'signal_strength'] = (sar - df['close'][i]) / sar * 100

        return df

    def save_parabolic_sar_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated Parabolic SAR data to the indicators database."""
        df_to_save = df[
            ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
             'parabolic_sar', 'trend', 'reversal_signal', 'signal_strength']
        ].copy()
        df_to_save['symbol'] = symbol
        df_to_save['timeframe'] = timeframe
        
        # Reorder columns to match database schema
        df_to_save = df_to_save[
            ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 
             'parabolic_sar', 'trend', 'reversal_signal', 'signal_strength']
        ]
        
        try:
            with sqlite3.connect(self.parabolic_sar_db_path) as conn:
                cursor = conn.cursor()
                inserted = 0
                for _, row in df_to_save.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO parabolic_sar_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume, 
                         parabolic_sar, trend, reversal_signal, signal_strength)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', tuple(row))
                    inserted += 1
                conn.commit()
                print(f"[INFO] Saved {inserted} Parabolic SAR records for {symbol} ({timeframe})")
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
        
        analysis = {
            'current_trend': latest['trend'],
            'current_sar': latest['parabolic_sar'],
            'signal_strength': latest['signal_strength'],
            'price_vs_sar': self._get_price_sar_relationship(latest['close'], latest['parabolic_sar'], latest['trend']),
            'recent_reversals': int(recent_reversals),
            'last_reversal_date': recent_reversal_date,
            'trend_persistence': self._calculate_trend_persistence(recent_data),
            'sar_values': {
                'current_price': latest['close'],
                'current_sar': latest['parabolic_sar'],
                'distance_pct': latest['signal_strength']
            }
        }
        
        return analysis
    
    def _get_price_sar_relationship(self, price: float, sar: float, trend: str) -> str:
        """Get descriptive text for price vs SAR relationship."""
        distance_pct = abs((price - sar) / sar * 100)
        
        if trend == 'up':
            if distance_pct > 5:
                return f"Strong bullish (price {distance_pct:.1f}% above SAR)"
            elif distance_pct > 2:
                return f"Moderate bullish (price {distance_pct:.1f}% above SAR)"
            else:
                return f"Weak bullish (price {distance_pct:.1f}% above SAR)"
        else:
            if distance_pct > 5:
                return f"Strong bearish (price {distance_pct:.1f}% below SAR)"
            elif distance_pct > 2:
                return f"Moderate bearish (price {distance_pct:.1f}% below SAR)"
            else:
                return f"Weak bearish (price {distance_pct:.1f}% below SAR)"
    
    def _calculate_trend_persistence(self, recent_data: pd.DataFrame) -> str:
        """Calculate how persistent the current trend has been."""
        if len(recent_data) < 5:
            return "insufficient_data"
        
        current_trend = recent_data['trend'].iloc[-1]
        trend_count = (recent_data['trend'] == current_trend).sum()
        trend_percentage = (trend_count / len(recent_data)) * 100
        
        if trend_percentage >= 80:
            return f"Very persistent {current_trend}trend ({trend_percentage:.0f}% consistency)"
        elif trend_percentage >= 60:
            return f"Persistent {current_trend}trend ({trend_percentage:.0f}% consistency)"
        else:
            return f"Volatile trend ({trend_percentage:.0f}% consistency)"

    def get_reversal_history(self, symbol: str, timeframe: str, limit: int = 10) -> pd.DataFrame:
        """Get recent reversal history for a symbol."""
        query = '''
            SELECT timestamp, close, parabolic_sar, trend, signal_strength
            FROM parabolic_sar_data
            WHERE symbol = ? AND timeframe = ? AND reversal_signal = 1
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        with sqlite3.connect(self.parabolic_sar_db_path) as conn:
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
            df_sar = calculator.calculate_parabolic_sar(df_raw)
            calculator.save_parabolic_sar_data(df_sar, symbol, timeframe)


if __name__ == '__main__':
    main()

