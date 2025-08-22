"""
Simple Moving Average (SMA) Calculator module.
Uses raw data from the database to calculate SMA indicators and saves it to the database.

Simple Moving Averages are trend-following indicators that smooth out price data
by creating a constantly updated average price over a specific time period.

This module implements:
- SMA 50: 50-period simple moving average (short-term trend)
- SMA 200: 200-period simple moving average (long-term trend)
- Golden Cross: When SMA 50 crosses above SMA 200 (bullish signal)
- Death Cross: When SMA 50 crosses below SMA 200 (bearish signal)

Common trading signals:
- Price above both SMAs: Strong uptrend
- Price below both SMAs: Strong downtrend
- Golden Cross: Major bullish signal (buy)
- Death Cross: Major bearish signal (sell)
"""
import pandas as pd
import sqlite3
import numpy as np
from typing import Optional, Tuple




class SimpleMovingAverageCalculator:
    """Calculates Simple Moving Average indicators from unified database."""
    
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

    def calculate_sma(self, df: pd.DataFrame, sma_50_period: int = 50, sma_200_period: int = 200) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages on data.
        
        Parameters:
        - df: DataFrame with OHLCV data
        - sma_50_period: Period for short-term SMA (default: 50)
        - sma_200_period: Period for long-term SMA (default: 200)
        
        Returns:
        - DataFrame with SMA indicators added
        """
        # Calculate SMAs
        df['sma_50'] = df['close'].rolling(window=sma_50_period, min_periods=1).mean()
        df['sma_200'] = df['close'].rolling(window=sma_200_period, min_periods=1).mean()
        
        # Calculate SMA ratio (50/200) - shows relative strength of short vs long term trend
        df['sma_ratio'] = df['sma_50'] / df['sma_200']
        
        # Calculate price position relative to SMAs (as percentage)
        df['price_vs_sma50'] = ((df['close'] - df['sma_50']) / df['sma_50']) * 100
        df['price_vs_sma200'] = ((df['close'] - df['sma_200']) / df['sma_200']) * 100
        
        # Calculate trend strength based on SMA alignment and slope
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        # Generate SMA-based signals
        df = self._generate_sma_signals(df)
        
        # Detect crossovers
        df['cross_signal'] = self._detect_crossovers(df)
        
        return df

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate trend strength based on SMA alignment and slope.
        Returns values from -100 to +100:
        - Positive values indicate uptrend strength
        - Negative values indicate downtrend strength
        """
        # SMA alignment score (50 vs 200)
        alignment_score = ((df['sma_50'] - df['sma_200']) / df['sma_200']) * 100
        
        # SMA slope (rate of change over 10 periods)
        sma_50_slope = df['sma_50'].pct_change(10) * 100
        sma_200_slope = df['sma_200'].pct_change(10) * 100
        
        # Combine alignment and slope for overall trend strength
        trend_strength = (alignment_score + sma_50_slope + sma_200_slope) / 3
        
        # Cap values between -100 and +100
        return np.clip(trend_strength, -100, 100)

    def _generate_sma_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on SMA analysis."""
        df['sma_signal'] = 'hold'
        
        # Strong bullish conditions
        strong_bull_condition = (
            (df['close'] > df['sma_50']) & 
            (df['close'] > df['sma_200']) & 
            (df['sma_50'] > df['sma_200']) &
            (df['trend_strength'] > 10)
        )
        df.loc[strong_bull_condition, 'sma_signal'] = 'strong_buy'
        
        # Bullish conditions
        bull_condition = (
            (df['close'] > df['sma_50']) & 
            (df['sma_50'] > df['sma_200']) &
            (df['trend_strength'] > 2)
        ) & ~strong_bull_condition
        df.loc[bull_condition, 'sma_signal'] = 'buy'
        
        # Strong bearish conditions
        strong_bear_condition = (
            (df['close'] < df['sma_50']) & 
            (df['close'] < df['sma_200']) & 
            (df['sma_50'] < df['sma_200']) &
            (df['trend_strength'] < -10)
        )
        df.loc[strong_bear_condition, 'sma_signal'] = 'strong_sell'
        
        # Bearish conditions
        bear_condition = (
            (df['close'] < df['sma_50']) & 
            (df['sma_50'] < df['sma_200']) &
            (df['trend_strength'] < -2)
        ) & ~strong_bear_condition
        df.loc[bear_condition, 'sma_signal'] = 'sell'
        
        return df

    def _detect_crossovers(self, df: pd.DataFrame) -> pd.Series:
        """Detect Golden Cross and Death Cross signals."""
        cross_signal = pd.Series('none', index=df.index)
        
        # Calculate when SMA 50 crosses above/below SMA 200
        sma_50_above_200 = (df['sma_50'] > df['sma_200']).fillna(False)
        sma_50_above_200_prev = sma_50_above_200.shift(1).fillna(False)
        
        # Golden Cross: SMA 50 crosses above SMA 200
        golden_cross = (~sma_50_above_200_prev) & sma_50_above_200
        cross_signal.loc[golden_cross] = 'golden_cross'
        
        # Death Cross: SMA 50 crosses below SMA 200
        death_cross = sma_50_above_200_prev & (~sma_50_above_200)
        cross_signal.loc[death_cross] = 'death_cross'
        
        return cross_signal

    def save_sma_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save calculated SMA data to the unified database."""
        if df.empty:
            print(f"[INFO] No SMA data to save for {symbol} ({timeframe})")
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
                    
                    # Insert or replace SMA indicator data
                    conn.execute('''
                        INSERT OR REPLACE INTO sma_indicator 
                        (ohlcv_id, sma_50, sma_200, sma_ratio, price_vs_sma50, price_vs_sma200,
                         trend_strength, sma_signal, cross_signal)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ohlcv_id,
                        row['sma_50'],
                        row['sma_200'], 
                        row['sma_ratio'],
                        row['price_vs_sma50'],
                        row['price_vs_sma200'],
                        row['trend_strength'],
                        row['sma_signal'],
                        row['cross_signal']
                    ))
                    inserted += 1
                
                conn.commit()
                print(f"[INFO] Saved {inserted} SMA indicators for {symbol} ({timeframe})")
                if skipped > 0:
                    print(f"[WARNING] Skipped {skipped} records (no matching OHLCV data)")
                    
        except Exception as e:
            print(f"[ERROR] Failed to save SMA data for {symbol}: {e}")

    def analyze_sma_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze current SMA patterns and provide insights."""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        recent_data = df.tail(50)  # Last 50 periods for analysis
        
        # Check for recent crossovers
        recent_golden_cross = any(recent_data['cross_signal'] == 'golden_cross')
        recent_death_cross = any(recent_data['cross_signal'] == 'death_cross')
        
        # Find most recent crossover
        recent_cross_date = None
        if recent_golden_cross or recent_death_cross:
            cross_data = recent_data[recent_data['cross_signal'].isin(['golden_cross', 'death_cross'])]
            if not cross_data.empty:
                recent_cross_date = cross_data.iloc[-1]['timestamp']
        
        analysis = {
            'current_trend': self._get_trend_description(latest['trend_strength']),
            'sma_alignment': self._get_sma_alignment(latest['sma_50'], latest['sma_200']),
            'price_position': self._get_price_position(latest['close'], latest['sma_50'], latest['sma_200']),
            'current_signal': latest['sma_signal'],
            'recent_crossover': {
                'golden_cross': recent_golden_cross,
                'death_cross': recent_death_cross,
                'date': recent_cross_date
            },
            'trend_strength': latest['trend_strength'],
            'sma_values': {
                'close': latest['close'],
                'sma_50': latest['sma_50'],
                'sma_200': latest['sma_200'],
                'sma_ratio': latest['sma_ratio']
            },
            'price_vs_smas': {
                'vs_sma_50': latest['price_vs_sma50'],
                'vs_sma_200': latest['price_vs_sma200']
            }
        }
        
        return analysis
    
    def _get_trend_description(self, trend_strength: float) -> str:
        """Get descriptive text for trend strength."""
        if trend_strength >= 20:
            return "Very strong uptrend"
        elif trend_strength >= 10:
            return "Strong uptrend"
        elif trend_strength >= 2:
            return "Moderate uptrend"
        elif trend_strength >= -2:
            return "Sideways/neutral trend"
        elif trend_strength >= -10:
            return "Moderate downtrend"
        elif trend_strength >= -20:
            return "Strong downtrend"
        else:
            return "Very strong downtrend"
    
    def _get_sma_alignment(self, sma_50: float, sma_200: float) -> str:
        """Get SMA alignment description."""
        if sma_50 > sma_200:
            ratio = (sma_50 / sma_200 - 1) * 100
            if ratio > 5:
                return f"Strong bullish alignment (SMA50 {ratio:.1f}% above SMA200)"
            else:
                return f"Bullish alignment (SMA50 {ratio:.1f}% above SMA200)"
        else:
            ratio = (1 - sma_50 / sma_200) * 100
            if ratio > 5:
                return f"Strong bearish alignment (SMA50 {ratio:.1f}% below SMA200)"
            else:
                return f"Bearish alignment (SMA50 {ratio:.1f}% below SMA200)"
    
    def _get_price_position(self, close: float, sma_50: float, sma_200: float) -> str:
        """Get price position relative to SMAs."""
        above_50 = close > sma_50
        above_200 = close > sma_200
        
        if above_50 and above_200:
            return "Above both SMAs (bullish position)"
        elif above_50 and not above_200:
            return "Above SMA50, below SMA200 (mixed signals)"
        elif not above_50 and above_200:
            return "Below SMA50, above SMA200 (weakening)"
        else:
            return "Below both SMAs (bearish position)"

    def get_crossover_history(self, symbol: str, timeframe: str, limit: int = 10) -> pd.DataFrame:
        """Get recent crossover history for a symbol."""
        query = '''
            SELECT o.timestamp, o.close, sma.sma_50, sma.sma_200, sma.cross_signal, sma.sma_signal
            FROM ohlcv_data o
            JOIN symbols s ON o.symbol_id = s.id
            JOIN timeframes t ON o.timeframe_id = t.id
            JOIN sma_indicator sma ON o.id = sma.ohlcv_id
            WHERE s.symbol = ? AND t.timeframe = ? AND sma.cross_signal != 'none'
            ORDER BY o.timestamp DESC
            LIMIT ?
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=(symbol, timeframe, limit))


def main():
    """Main function to calculate and save SMA data for multiple symbols and timeframes."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
    timeframes = ['4h', '1d']

    calculator = SimpleMovingAverageCalculator()

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[CALCULATING] SMA for {symbol} - {timeframe.upper()}")
            df_raw = calculator.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            
            # Calculate SMAs
            df_sma = calculator.calculate_sma(df_raw)
            
            # Save to database
            calculator.save_sma_data(df_sma, symbol, timeframe)
            
            # Print analysis for the latest data
            analysis = calculator.analyze_sma_patterns(df_sma)
            if analysis:
                print(f"[ANALYSIS] Current trend: {analysis['current_trend']}")
                print(f"[ANALYSIS] SMA alignment: {analysis['sma_alignment']}")
                print(f"[ANALYSIS] Price position: {analysis['price_position']}")
                print(f"[ANALYSIS] Current signal: {analysis['current_signal']}")
                
                # Show recent crossovers
                if analysis['recent_crossover']['golden_cross']:
                    print(f"[SIGNAL] Recent Golden Cross detected!")
                elif analysis['recent_crossover']['death_cross']:
                    print(f"[SIGNAL] Recent Death Cross detected!")


if __name__ == '__main__':
    main()
