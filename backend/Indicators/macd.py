"""
MACD (Moving Average Convergence Divergence) Calculator module.
Uses raw data from the database to calculate MACD indicator and saves it to the database.

MACD is a trend-following momentum indicator that shows the relationship between two moving averages.
It consists of:
- MACD Line: 12-period EMA - 26-period EMA
- Signal Line: 9-period EMA of the MACD Line
- Histogram: MACD Line - Signal Line

Common trading signals:
- Bullish signal when MACD line crosses above signal line
- Bearish signal when MACD line crosses below signal line
- Divergence between MACD and price can indicate trend reversal
"""
import pandas as pd
import sqlite3
import numpy as np
from typing import Optional


class MACDCalculator:
    """Calculates MACD indicators from raw data."""
    
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

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    def calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate MACD indicators on data.
        
        Parameters:
        - df: DataFrame with OHLCV data
        - fast_period: Fast EMA period (default: 12)
        - slow_period: Slow EMA period (default: 26)
        - signal_period: Signal line EMA period (default: 9)
        
        Returns:
        - DataFrame with MACD indicators added
        """
        # Calculate EMAs
        df['ema_12'] = self.calculate_ema(df['close'], fast_period)
        df['ema_26'] = self.calculate_ema(df['close'], slow_period)
        
        # Calculate MACD line (Fast EMA - Slow EMA)
        df['macd_line'] = df['ema_12'] - df['ema_26']
        
        # Calculate Signal line (9-period EMA of MACD line)
        df['signal_line'] = self.calculate_ema(df['macd_line'], signal_period)
        
        # Calculate Histogram (MACD line - Signal line)
        df['histogram'] = df['macd_line'] - df['signal_line']
        
        # Determine MACD signals
        df['macd_signal'] = 'neutral'
        
        # Bullish signal: MACD line crosses above signal line
        df['macd_cross_above'] = (df['macd_line'] > df['signal_line']) & (df['macd_line'].shift(1) <= df['signal_line'].shift(1))
        # Bearish signal: MACD line crosses below signal line
        df['macd_cross_below'] = (df['macd_line'] < df['signal_line']) & (df['macd_line'].shift(1) >= df['signal_line'].shift(1))
        
        # Set signals based on crossovers
        df.loc[df['macd_cross_above'], 'macd_signal'] = 'bullish'
        df.loc[df['macd_cross_below'], 'macd_signal'] = 'bearish'
        
        # Additional trend confirmation signals
        # Stronger bullish: MACD line above signal line and both above zero
        strong_bullish = (df['macd_line'] > df['signal_line']) & (df['macd_line'] > 0) & (df['signal_line'] > 0)
        df.loc[strong_bullish & ~df['macd_cross_above'], 'macd_signal'] = 'strong_bullish'
        
        # Stronger bearish: MACD line below signal line and both below zero
        strong_bearish = (df['macd_line'] < df['signal_line']) & (df['macd_line'] < 0) & (df['signal_line'] < 0)
        df.loc[strong_bearish & ~df['macd_cross_below'], 'macd_signal'] = 'strong_bearish'

        return df

    def save_macd_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated MACD data to the unified database."""
        if df.empty:
            print(f"[INFO] No MACD data to save for {symbol} ({timeframe})")
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
                    
                    # Insert or replace MACD indicator data
                    conn.execute('''
                        INSERT OR REPLACE INTO macd_indicators 
                        (ohlcv_id, ema_12, ema_26, macd_line, signal_line, histogram, macd_signal)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ohlcv_id,
                        row['ema_12'],
                        row['ema_26'], 
                        row['macd_line'],
                        row['signal_line'],
                        row['histogram'],
                        row['macd_signal']
                    ))
                    inserted += 1
                
                conn.commit()
                print(f"[INFO] Saved {inserted} MACD indicators for {symbol} ({timeframe})")
                if skipped > 0:
                    print(f"[WARNING] Skipped {skipped} records (no matching OHLCV data)")
                    
        except Exception as e:
            print(f"[ERROR] Failed to save MACD data for {symbol}: {e}")

    def analyze_macd_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze current MACD patterns and provide insights."""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        recent_data = df.tail(20)  # Last 20 periods
        
        # Check for recent crossovers
        recent_bullish_crossovers = recent_data['macd_cross_above'].sum()
        recent_bearish_crossovers = recent_data['macd_cross_below'].sum()
        
        # Momentum analysis
        macd_momentum = 'increasing' if latest['histogram'] > df.iloc[-2]['histogram'] else 'decreasing'
        
        # Zero line analysis
        macd_position = 'above_zero' if latest['macd_line'] > 0 else 'below_zero'
        signal_position = 'above_zero' if latest['signal_line'] > 0 else 'below_zero'
        
        analysis = {
            'current_signal': latest['macd_signal'],
            'macd_line': latest['macd_line'],
            'signal_line': latest['signal_line'],
            'histogram': latest['histogram'],
            'momentum': macd_momentum,
            'macd_position': macd_position,
            'signal_position': signal_position,
            'recent_crossovers': {
                'bullish': int(recent_bullish_crossovers),
                'bearish': int(recent_bearish_crossovers)
            },
            'trend_strength': self._get_trend_strength(latest),
            'divergence_risk': self._check_divergence_risk(df.tail(10))
        }
        
        return analysis
    
    def _get_trend_strength(self, latest_data) -> str:
        """Determine trend strength based on MACD components."""
        macd_line = latest_data['macd_line']
        signal_line = latest_data['signal_line']
        histogram = latest_data['histogram']
        
        if macd_line > signal_line and macd_line > 0 and histogram > 0:
            return 'strong_bullish'
        elif macd_line > signal_line and histogram > 0:
            return 'moderate_bullish'
        elif macd_line < signal_line and macd_line < 0 and histogram < 0:
            return 'strong_bearish'
        elif macd_line < signal_line and histogram < 0:
            return 'moderate_bearish'
        else:
            return 'weak'
    
    def _check_divergence_risk(self, recent_df: pd.DataFrame) -> str:
        """Check for potential divergence patterns."""
        if len(recent_df) < 5:
            return 'insufficient_data'
        
        price_trend = recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]
        macd_trend = recent_df['macd_line'].iloc[-1] - recent_df['macd_line'].iloc[0]
        
        # Bullish divergence: price falling but MACD rising
        if price_trend < 0 and macd_trend > 0:
            return 'bullish_divergence'
        # Bearish divergence: price rising but MACD falling
        elif price_trend > 0 and macd_trend < 0:
            return 'bearish_divergence'
        else:
            return 'no_divergence'

    def get_macd_crossovers(self, symbol: str, timeframe: str, limit: int = 10) -> pd.DataFrame:
        """Get recent MACD crossover history for a symbol."""
        query = '''
            SELECT timestamp, close, macd_line, signal_line, histogram, macd_signal
            FROM macd_data
            WHERE symbol = ? AND timeframe = ? AND macd_signal IN ('bullish', 'bearish')
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        with sqlite3.connect(self.macd_db_path) as conn:
            return pd.read_sql(query, conn, params=(symbol, timeframe, limit))


def main():
    """Main function to calculate and save MACD data for multiple symbols and timeframes."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
    timeframes = ['4h', '1d']

    calculator = MACDCalculator()

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[CALCULATING] MACD for {symbol} - {timeframe.upper()}")
            df_raw = calculator.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            
            # Calculate MACD
            df_macd = calculator.calculate_macd(df_raw)
            
            # Save to database
            calculator.save_macd_data(df_macd, symbol, timeframe)
            
            # Print analysis for the latest data
            analysis = calculator.analyze_macd_patterns(df_macd)
            if analysis:
                print(f"[ANALYSIS] Current signal: {analysis['current_signal']}")
                print(f"[ANALYSIS] Trend strength: {analysis['trend_strength']}")
                print(f"[ANALYSIS] Momentum: {analysis['momentum']}")
                print(f"[ANALYSIS] MACD position: {analysis['macd_position']}")
                if analysis['divergence_risk'] != 'no_divergence':
                    print(f"[ANALYSIS] Divergence risk: {analysis['divergence_risk']}")


if __name__ == '__main__':
    main()
