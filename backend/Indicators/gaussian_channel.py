"""
Gaussian Channel Indicator Calculator Module - UNIFIED VERSION
Uses unified_trading_data.db to calculate Gaussian Channel indicators and saves them to the unified database.

The Gaussian Channel is a volatility-based channel indicator that creates dynamic support and resistance levels.
It uses a moving average as the middle line and applies a Gaussian-based calculation to determine
the upper and lower channel bounds.

Key Features:
- Dynamic channel bands based on price volatility
- Smooth channel lines using Gaussian smoothing
- Support/resistance level identification
- Breakout and reversal signal detection
- Trend direction confirmation

This module implements:
- Gaussian Channel calculation with configurable period and smoothing
- Upper, middle (moving average), and lower channel bands
- Signal generation for trend analysis and breakout detection
- Pattern analysis for current market conditions
- Integration with unified database system

Common trading signals:
- Price breaking above upper channel: Potential bullish breakout
- Price breaking below lower channel: Potential bearish breakout
- Price returning to middle line: Trend reversal or consolidation
- Channel width analysis for volatility assessment
"""

import pandas as pd
import sqlite3
import numpy as np
from typing import Optional


class GaussianChannelCalculator:
    """Calculates Gaussian Channel indicators using unified database."""
    
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

    def calculate_gaussian_channel(self, df: pd.DataFrame, period: int = 20, smoothing: float = 2.0) -> pd.DataFrame:
        """
        Calculate Gaussian Channel indicators on data.
        
        Parameters:
        - df: DataFrame with OHLCV data
        - period: Period for moving average calculation (default: 20)
        - smoothing: Smoothing factor for channel width (default: 2.0)
        
        Returns:
        - DataFrame with Gaussian Channel indicators added
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Calculate the middle line (Simple Moving Average)
        df['gc_middle'] = df['close'].rolling(window=period, min_periods=1).mean()
        
        # Calculate the standard deviation for channel width
        df['price_std'] = df['close'].rolling(window=period, min_periods=1).std(ddof=0)
        
        # Calculate True Range for volatility adjustment
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate Average True Range for dynamic channel adjustment
        df['atr'] = df['true_range'].rolling(window=period, min_periods=1).mean()
        
        # Gaussian-based channel calculation
        # Using a combination of standard deviation and ATR for more responsive channels
        df['channel_width'] = (df['price_std'] + df['atr'] * 0.5) * smoothing
        
        # Calculate upper and lower channel bounds
        df['gc_upper'] = df['gc_middle'] + df['channel_width']
        df['gc_lower'] = df['gc_middle'] - df['channel_width']
        
        # Calculate additional metrics
        self._calculate_channel_metrics(df)
        
        # Generate trading signals
        df = self._generate_channel_signals(df)
        
        # Clean up temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3', 'true_range', 'price_std', 'atr', 'channel_width'], axis=1)
        
        return df
    
    def _calculate_channel_metrics(self, df: pd.DataFrame) -> None:
        """Calculate additional channel analysis metrics."""
        # Channel position (where price is relative to channel)
        channel_range = df['gc_upper'] - df['gc_lower']
        df['channel_position'] = np.where(
            channel_range > 0,
            (df['close'] - df['gc_lower']) / channel_range,
            0.5  # Default to middle if no range
        )
        
        # Channel width as percentage of middle line (volatility measure)
        df['channel_width_pct'] = np.where(
            df['gc_middle'] > 0,
            (df['gc_upper'] - df['gc_lower']) / df['gc_middle'] * 100,
            0
        )
        
        # Price distance from middle line (momentum indicator)
        df['distance_from_middle'] = ((df['close'] - df['gc_middle']) / df['gc_middle'] * 100).fillna(0)
    
    def _generate_channel_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Gaussian Channel."""
        df['gc_signal'] = 'hold'
        df['breakout_signal'] = 'none'
        
        # Breakout signals
        upper_breakout = df['close'] > df['gc_upper']
        lower_breakout = df['close'] < df['gc_lower']
        
        # Trend reversal signals (return to channel after breakout)
        upper_return = (df['close'].shift(1) > df['gc_upper'].shift(1)) & (df['close'] <= df['gc_upper'])
        lower_return = (df['close'].shift(1) < df['gc_lower'].shift(1)) & (df['close'] >= df['gc_lower'])
        
        # Signal assignment
        df.loc[upper_breakout, 'gc_signal'] = 'strong_buy'
        df.loc[lower_breakout, 'gc_signal'] = 'strong_sell'
        df.loc[upper_return, 'gc_signal'] = 'sell'
        df.loc[lower_return, 'gc_signal'] = 'buy'
        
        # Channel position based signals
        df.loc[df['channel_position'] >= 0.8, 'gc_signal'] = 'sell'  # Near upper channel
        df.loc[df['channel_position'] <= 0.2, 'gc_signal'] = 'buy'   # Near lower channel
        
        # Breakout signal classification
        df.loc[upper_breakout, 'breakout_signal'] = 'bullish_breakout'
        df.loc[lower_breakout, 'breakout_signal'] = 'bearish_breakout'
        df.loc[upper_return, 'breakout_signal'] = 'bearish_reversal'
        df.loc[lower_return, 'breakout_signal'] = 'bullish_reversal'
        
        return df

    def save_gaussian_channel_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated Gaussian Channel data to the unified database."""
        if df.empty:
            print(f"[INFO] No Gaussian Channel data to save for {symbol} ({timeframe})")
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
                    
                    # Insert or replace Gaussian Channel indicator data
                    conn.execute('''
                        INSERT OR REPLACE INTO gaussian_channel_indicator 
                        (ohlcv_id, gc_upper, gc_middle, gc_lower)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        ohlcv_id,
                        row['gc_upper'] if pd.notna(row['gc_upper']) else None,
                        row['gc_middle'] if pd.notna(row['gc_middle']) else None,
                        row['gc_lower'] if pd.notna(row['gc_lower']) else None
                    ))
                    inserted += 1
                
                conn.commit()
                print(f"[INFO] Saved {inserted} Gaussian Channel indicators for {symbol} ({timeframe})")
                if skipped > 0:
                    print(f"[WARNING] Skipped {skipped} records (no matching OHLCV data)")
                    
        except Exception as e:
            print(f"[ERROR] Failed to save Gaussian Channel data for {symbol}: {e}")

    def analyze_gaussian_channel_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze current Gaussian Channel patterns and provide insights."""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        recent_data = df.tail(20)  # Last 20 periods
        
        # Channel analysis
        current_position = latest.get('channel_position', 0.5)
        channel_width = latest.get('channel_width_pct', 0)
        distance_from_middle = latest.get('distance_from_middle', 0)
        
        # Count recent signals
        recent_breakouts = recent_data['breakout_signal'].value_counts().to_dict()
        
        # Trend analysis based on channel position
        trend_analysis = self._analyze_trend_strength(recent_data)
        
        analysis = {
            'current_signal': latest.get('gc_signal', 'hold'),
            'breakout_signal': latest.get('breakout_signal', 'none'),
            'channel_position': current_position,
            'channel_position_description': self._get_position_description(current_position),
            'channel_width_pct': channel_width,
            'volatility_state': self._get_volatility_state(channel_width, recent_data['channel_width_pct'].mean()),
            'distance_from_middle': distance_from_middle,
            'trend_analysis': trend_analysis,
            'recent_breakouts': recent_breakouts,
            'current_values': {
                'close': latest['close'],
                'gc_upper': latest['gc_upper'],
                'gc_middle': latest['gc_middle'],
                'gc_lower': latest['gc_lower']
            }
        }
        
        return analysis
    
    def _get_position_description(self, position: float) -> str:
        """Get descriptive text for current channel position."""
        if position >= 0.9:
            return "Very close to upper channel (potential reversal zone)"
        elif position >= 0.7:
            return "Near upper channel (overbought territory)"
        elif position >= 0.6:
            return "Above middle (bullish bias)"
        elif position >= 0.4:
            return "Around channel middle (neutral zone)"
        elif position >= 0.3:
            return "Below middle (bearish bias)"
        elif position >= 0.1:
            return "Near lower channel (oversold territory)"
        else:
            return "Very close to lower channel (potential reversal zone)"
    
    def _get_volatility_state(self, current_width: float, avg_width: float) -> str:
        """Get descriptive text for current volatility state."""
        if avg_width == 0:
            return "Unknown volatility"
        
        ratio = current_width / avg_width
        
        if ratio >= 1.5:
            return "High volatility (expanding channel)"
        elif ratio >= 1.2:
            return "Above average volatility"
        elif ratio >= 0.8:
            return "Normal volatility"
        elif ratio >= 0.6:
            return "Low volatility (contracting channel)"
        else:
            return "Very low volatility (potential breakout setup)"
    
    def _analyze_trend_strength(self, recent_data: pd.DataFrame) -> dict:
        """Analyze trend strength based on recent channel behavior."""
        if len(recent_data) < 5:
            return {'strength': 'insufficient_data', 'direction': 'unknown'}
        
        # Calculate average position over recent periods
        avg_position = recent_data['channel_position'].mean()
        position_trend = recent_data['channel_position'].iloc[-1] - recent_data['channel_position'].iloc[0]
        
        # Determine trend direction and strength
        if avg_position > 0.6 and position_trend > 0:
            return {'strength': 'strong', 'direction': 'bullish'}
        elif avg_position > 0.5 and position_trend > 0:
            return {'strength': 'moderate', 'direction': 'bullish'}
        elif avg_position < 0.4 and position_trend < 0:
            return {'strength': 'strong', 'direction': 'bearish'}
        elif avg_position < 0.5 and position_trend < 0:
            return {'strength': 'moderate', 'direction': 'bearish'}
        else:
            return {'strength': 'weak', 'direction': 'sideways'}

    def get_channel_signals(self, symbol: str, timeframe: str, limit: int = 10) -> pd.DataFrame:
        """Get recent Gaussian Channel signal history for a symbol."""
        query = '''
            SELECT o.timestamp, o.close, gc.gc_upper, gc.gc_middle, gc.gc_lower
            FROM ohlcv_data o
            JOIN symbols s ON o.symbol_id = s.id
            JOIN timeframes t ON o.timeframe_id = t.id
            JOIN gaussian_channel_indicator gc ON o.id = gc.ohlcv_id
            WHERE s.symbol = ? AND t.timeframe = ?
            ORDER BY o.timestamp DESC
            LIMIT ?
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=(symbol, timeframe, limit))


def main():
    """Main function to calculate and save Gaussian Channel data for multiple symbols and timeframes."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
    timeframes = ['4h', '1d']

    calculator = GaussianChannelCalculator()

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[CALCULATING] Gaussian Channel for {symbol} - {timeframe.upper()}")
            df_raw = calculator.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            
            # Calculate Gaussian Channel
            df_gc = calculator.calculate_gaussian_channel(df_raw)
            
            # Save to database
            calculator.save_gaussian_channel_data(df_gc, symbol, timeframe)
            
            # Print analysis for the latest data
            analysis = calculator.analyze_gaussian_channel_patterns(df_gc)
            if analysis:
                print(f"[ANALYSIS] Current signal: {analysis['current_signal']}")
                print(f"[ANALYSIS] Channel position: {analysis['channel_position_description']}")
                print(f"[ANALYSIS] Volatility: {analysis['volatility_state']}")
                print(f"[ANALYSIS] Trend: {analysis['trend_analysis']['direction']} ({analysis['trend_analysis']['strength']})")
                if analysis['breakout_signal'] != 'none':
                    print(f"[ANALYSIS] Breakout signal: {analysis['breakout_signal']}")


if __name__ == '__main__':
    main()
