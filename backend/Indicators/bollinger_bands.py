"""
Bollinger Bands Calculator module.
Uses raw data from the database to calculate Bollinger Bands indicator and saves it to the database.

Bollinger Bands are a volatility indicator that helps identify overbought and oversold conditions,
potential trend reversals, or breakouts.

A Bollinger Band consists of:
- Middle Band: A 20-period simple moving average (SMA)
- Upper Band: 20 SMA + 2 standard deviations
- Lower Band: 20 SMA - 2 standard deviations

When volatility increases, the bands widen. When volatility decreases, they contract.
"""
import pandas as pd
import sqlite3
import numpy as np
from typing import Optional




class BollingerBandsCalculator:
    """Calculates Bollinger Bands indicator from raw data."""
    
    def __init__(self, db_path: str = 'data/trading_data_BTC.db'):
        import os
        if not os.path.isabs(db_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            db_path = os.path.abspath(os.path.join(project_root, db_path))
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

    def calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands on data.
        
        Parameters:
        - df: DataFrame with OHLCV data
        - window: Period for moving average (default: 20)
        - std_dev: Number of standard deviations (default: 2)
        
        Returns:
        - DataFrame with Bollinger Bands indicators added
        """
        # Calculate the middle band (Simple Moving Average)
        df['bb_middle'] = df['close'].rolling(window=window, min_periods=1).mean()
        
        # Calculate the standard deviation
        df['bb_std'] = df['close'].rolling(window=window, min_periods=1).std(ddof=0)
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        
        # Calculate additional indicators
        # Bollinger Band Width (measures volatility)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Bollinger Band Percent (%B) - shows where price is relative to the bands
        # %B = (Close - Lower Band) / (Upper Band - Lower Band)
        # Values above 1 indicate price is above upper band
        # Values below 0 indicate price is below lower band
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Handle edge cases where bands might be equal (no volatility)
        df['bb_percent'] = df['bb_percent'].fillna(0.5)  # Default to middle if no volatility
        
        return df

    def save_bollinger_bands_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated Bollinger Bands data to the unified database."""
        if df.empty:
            print(f"[INFO] No Bollinger Bands data to save for {symbol} ({timeframe})")
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
                    
                    # Insert or replace Bollinger Bands indicator data
                    conn.execute('''
                        INSERT OR REPLACE INTO bollinger_bands_indicator 
                        (ohlcv_id, bb_upper, bb_lower, bb_middle, bb_width, bb_percent)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        ohlcv_id,
                        row['bb_upper'],
                        row['bb_lower'], 
                        row['bb_middle'],
                        row['bb_width'],
                        row['bb_percent']
                    ))
                    inserted += 1
                
                conn.commit()
                print(f"[INFO] Saved {inserted} Bollinger Bands indicators for {symbol} ({timeframe})")
                if skipped > 0:
                    print(f"[WARNING] Skipped {skipped} records (no matching OHLCV data)")
                    
        except Exception as e:
            print(f"[ERROR] Failed to save Bollinger Bands data for {symbol}: {e}")

    def get_bollinger_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.
        
        Common signals:
        - Price touching/crossing upper band: Potential sell signal (overbought)
        - Price touching/crossing lower band: Potential buy signal (oversold)
        - Bollinger Band squeeze: Low volatility, potential breakout coming
        - Bollinger Band expansion: High volatility, trend continuation
        """
        df = df.copy()
        
        # Initialize signal column
        df['bb_signal'] = 'hold'
        
        # Buy signals (oversold conditions)
        buy_condition = (
            (df['close'] <= df['bb_lower']) |  # Price at or below lower band
            (df['bb_percent'] <= 0.2)  # %B indicates oversold
        )
        df.loc[buy_condition, 'bb_signal'] = 'buy'
        
        # Sell signals (overbought conditions)
        sell_condition = (
            (df['close'] >= df['bb_upper']) |  # Price at or above upper band
            (df['bb_percent'] >= 0.8)  # %B indicates overbought
        )
        df.loc[sell_condition, 'bb_signal'] = 'sell'
        
        # Squeeze detection (low volatility)
        df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(window=20, min_periods=1).quantile(0.25)
        
        return df

    def analyze_bollinger_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze current Bollinger Bands patterns and provide insights."""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        recent_data = df.tail(20)  # Last 20 periods
        
        analysis = {
            'current_position': self._get_position_description(latest['bb_percent']),
            'volatility_state': self._get_volatility_state(latest['bb_width'], recent_data['bb_width'].mean()),
            'recent_signal': latest.get('bb_signal', 'hold'),
            'squeeze_active': latest.get('bb_squeeze', False),
            'price_vs_bands': {
                'close': latest['close'],
                'upper_band': latest['bb_upper'],
                'middle_band': latest['bb_middle'],
                'lower_band': latest['bb_lower'],
                'percent_b': latest['bb_percent']
            }
        }
        
        return analysis
    
    def _get_position_description(self, bb_percent: float) -> str:
        """Get descriptive text for current price position relative to bands."""
        if bb_percent >= 1:
            return "Above upper band (very overbought)"
        elif bb_percent >= 0.8:
            return "Near upper band (overbought)"
        elif bb_percent >= 0.6:
            return "Above middle (bullish)"
        elif bb_percent >= 0.4:
            return "Around middle (neutral)"
        elif bb_percent >= 0.2:
            return "Below middle (bearish)"
        elif bb_percent > 0:
            return "Near lower band (oversold)"
        else:
            return "Below lower band (very oversold)"
    
    def _get_volatility_state(self, current_width: float, avg_width: float) -> str:
        """Get descriptive text for current volatility state."""
        ratio = current_width / avg_width if avg_width > 0 else 1
        
        if ratio >= 1.5:
            return "High volatility (bands expanding)"
        elif ratio >= 1.2:
            return "Above average volatility"
        elif ratio >= 0.8:
            return "Normal volatility"
        elif ratio >= 0.6:
            return "Below average volatility"
        else:
            return "Low volatility (potential squeeze)"


def main():
    """Main function to calculate and save Bollinger Bands data for multiple symbols and timeframes."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
    timeframes = ['4h', '1d']

    calculator = BollingerBandsCalculator()

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[CALCULATING] Bollinger Bands for {symbol} - {timeframe.upper()}")
            df_raw = calculator.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            
            # Calculate Bollinger Bands
            df_bb = calculator.calculate_bollinger_bands(df_raw)
            
            # Add trading signals
            df_bb = calculator.get_bollinger_signals(df_bb)
            
            # Save to database
            calculator.save_bollinger_bands_data(df_bb, symbol, timeframe)
            
            # Print analysis for the latest data
            analysis = calculator.analyze_bollinger_patterns(df_bb)
            if analysis:
                print(f"[ANALYSIS] Current position: {analysis['current_position']}")
                print(f"[ANALYSIS] Volatility: {analysis['volatility_state']}")
                print(f"[ANALYSIS] Recent signal: {analysis['recent_signal']}")


if __name__ == '__main__':
    main()
