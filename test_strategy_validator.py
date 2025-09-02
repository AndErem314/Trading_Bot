"""
Test Script for Strategy Validator

This script demonstrates how to use the StrategyValidator class to compare
an RSI Momentum Divergence strategy implemented in both SQL and Python.

Author: Trading System QA
Date: 2025-09-02
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import sqlite3
import os

# Import our validator and base strategy
from strategy_validator import StrategyValidator, BaseExecutableStrategy


class RSIMomentumDivergence(BaseExecutableStrategy):
    """
    RSI Momentum Divergence Strategy
    
    This strategy generates signals based on RSI divergences and momentum.
    For demonstration purposes, this is a simplified implementation.
    """
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize the RSI Momentum Divergence strategy.
        
        Args:
            rsi_period: Period for RSI calculation (default: 14)
            oversold: RSI oversold threshold (default: 30)
            overbought: RSI overbought threshold (default: 70)
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        
        # Internal state for calculations
        self.price_history = []
        self.rsi_history = []
        self.signal_history = []
        
    def calculate_rsi(self, prices: list) -> float:
        """
        Calculate RSI for the given price history.
        
        Args:
            prices: List of closing prices
            
        Returns:
            RSI value
        """
        if len(prices) < self.rsi_period + 1:
            return 50.0  # Neutral RSI if not enough data
        
        # Calculate price changes
        deltas = np.diff(prices[-self.rsi_period-1:])
        gains = deltas.copy()
        losses = deltas.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def detect_divergence(self) -> int:
        """
        Detect bullish or bearish divergence.
        
        Returns:
            1 for bullish divergence, -1 for bearish divergence, 0 for no divergence
        """
        if len(self.price_history) < 10 or len(self.rsi_history) < 10:
            return 0
        
        # Look at recent price and RSI trends
        recent_prices = self.price_history[-10:]
        recent_rsi = self.rsi_history[-10:]
        
        # Find local minima and maxima
        price_min_idx = np.argmin(recent_prices[:5])
        price_min_idx2 = np.argmin(recent_prices[5:]) + 5
        
        price_max_idx = np.argmax(recent_prices[:5])
        price_max_idx2 = np.argmax(recent_prices[5:]) + 5
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (recent_prices[price_min_idx2] < recent_prices[price_min_idx] and
            recent_rsi[price_min_idx2] > recent_rsi[price_min_idx] and
            recent_rsi[-1] < self.oversold):
            return 1
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (recent_prices[price_max_idx2] > recent_prices[price_max_idx] and
            recent_rsi[price_max_idx2] < recent_rsi[price_max_idx] and
            recent_rsi[-1] > self.overbought):
            return -1
        
        return 0
    
    def calculate_signal(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate signal based on the current state and new data.
        
        This method is called internally by process_candle but can also
        be used for batch processing.
        
        Args:
            dataframe: DataFrame with OHLCV data
            
        Returns:
            Dictionary with signal information
        """
        if dataframe.empty:
            return {'signal': 0, 'rsi': 50.0, 'reason': 'No data'}
        
        # Get the latest candle
        latest = dataframe.iloc[-1]
        
        # Update price history
        self.price_history.append(latest['close'])
        if len(self.price_history) > 100:  # Keep only recent history
            self.price_history.pop(0)
        
        # Calculate RSI
        current_rsi = self.calculate_rsi(self.price_history)
        self.rsi_history.append(current_rsi)
        if len(self.rsi_history) > 100:
            self.rsi_history.pop(0)
        
        # Generate signal based on RSI levels and divergence
        signal = 0
        reason = "Hold"
        
        # Check for divergence
        divergence = self.detect_divergence()
        
        if divergence == 1:
            signal = 1
            reason = "Bullish divergence detected"
        elif divergence == -1:
            signal = -1
            reason = "Bearish divergence detected"
        elif current_rsi < self.oversold:
            signal = 1
            reason = f"RSI oversold ({current_rsi:.2f})"
        elif current_rsi > self.overbought:
            signal = -1
            reason = f"RSI overbought ({current_rsi:.2f})"
        
        return {
            'signal': signal,
            'rsi': current_rsi,
            'reason': reason,
            'timestamp': latest['timestamp']
        }
    
    def process_candle(self, candle: pd.Series) -> Optional[int]:
        """
        Process a single candle and return a signal.
        
        Args:
            candle: A pandas Series representing a single OHLCV candle
            
        Returns:
            Signal value (-1, 0, or 1) or None if no signal
        """
        # Convert Series to DataFrame for calculate_signal
        df = pd.DataFrame([candle])
        result = self.calculate_signal(df)
        return result['signal']


def create_test_database(db_path: str):
    """
    Create a test database with sample OHLCV data.
    
    Args:
        db_path: Path to the SQLite database
    """
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create OHLCV table
    cursor.execute("""
        CREATE TABLE ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE(symbol, timeframe, timestamp)
        )
    """)
    
    # Generate sample data
    print("Generating sample OHLCV data...")
    
    base_price = 50000  # Starting price for BTCUSDT
    start_date = datetime(2024, 1, 1)
    data = []
    
    for i in range(100):  # 100 hours of data
        timestamp = start_date + timedelta(hours=i)
        
        # Create realistic price movements with trends
        trend = np.sin(i * 0.1) * 1000  # Sine wave trend
        noise = np.random.normal(0, 200)  # Random noise
        
        close = base_price + trend + noise
        open_price = close + np.random.normal(0, 50)
        high = max(open_price, close) + abs(np.random.normal(0, 30))
        low = min(open_price, close) - abs(np.random.normal(0, 30))
        volume = np.random.uniform(100, 1000)
        
        data.append((
            'BTCUSDT', '1h', timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            open_price, high, low, close, volume
        ))
        
        base_price = close  # Update base price for next candle
    
    # Insert data
    cursor.executemany(
        "INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        data
    )
    
    # Create signals table for SQL strategy results
    cursor.execute("""
        CREATE TABLE strategy_signals AS
        WITH price_data AS (
            SELECT 
                timestamp,
                close,
                LAG(close, 1) OVER (ORDER BY timestamp) as prev_close,
                LAG(close, 14) OVER (ORDER BY timestamp) as close_14_ago
            FROM ohlcv
            WHERE symbol = 'BTCUSDT' AND timeframe = '1h'
        ),
        rsi_calc AS (
            SELECT 
                timestamp,
                close,
                CASE 
                    WHEN close_14_ago IS NULL THEN 50
                    ELSE 100 - (100 / (1 + 
                        (AVG(CASE WHEN close > prev_close THEN close - prev_close ELSE 0 END) OVER (ORDER BY timestamp ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)) /
                        (AVG(CASE WHEN close < prev_close THEN prev_close - close ELSE 0 END) OVER (ORDER BY timestamp ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) + 0.0001)
                    ))
                END as rsi
            FROM price_data
        )
        SELECT 
            timestamp,
            CASE 
                WHEN rsi < 30 THEN 1  -- Buy signal
                WHEN rsi > 70 THEN -1  -- Sell signal
                ELSE 0  -- Hold
            END as signal,
            rsi,
            close
        FROM rsi_calc
    """)
    
    conn.commit()
    conn.close()
    
    print(f"Test database created at: {db_path}")


def format_comparison_results(comparison_df: pd.DataFrame):
    """
    Format and display comparison results in a clear, readable format.
    
    Args:
        comparison_df: DataFrame with comparison results
    """
    print("\n" + "="*80)
    print("STRATEGY VALIDATION RESULTS")
    print("="*80)
    
    # Summary statistics
    total_signals = len(comparison_df)
    matching_signals = comparison_df['signals_match'].sum()
    match_percentage = (matching_signals / total_signals * 100) if total_signals > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Total Signals Compared: {total_signals}")
    print(f"  Matching Signals: {matching_signals}")
    print(f"  Mismatched Signals: {total_signals - matching_signals}")
    print(f"  Match Rate: {match_percentage:.2f}%")
    
    # Validation score
    validation_score = "PASS" if match_percentage >= 95 else "FAIL"
    score_color = "✅" if validation_score == "PASS" else "❌"
    print(f"\nValidation Score: {score_color} {validation_score} (Signals matched {match_percentage:.2f}% of the time)")
    
    # Show first few results
    print("\nFirst 10 Comparison Results:")
    print("-"*80)
    print(f"{'Timestamp':<20} {'SQL Signal':>12} {'Exec Signal':>12} {'Match':>8} {'Diff':>8}")
    print("-"*80)
    
    for idx, row in comparison_df.head(10).iterrows():
        match_symbol = "✓" if row['signals_match'] else "✗"
        print(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
              f"{row['sql_signal']:>12.0f} "
              f"{row['executable_signal']:>12.0f} "
              f"{match_symbol:>8} "
              f"{row['difference']:>8.0f}")
    
    # Show mismatches if any
    mismatches = comparison_df[~comparison_df['signals_match']]
    if len(mismatches) > 0:
        print("\n" + "-"*80)
        print(f"Signal Mismatches ({len(mismatches)} found):")
        print("-"*80)
        print(f"{'Timestamp':<20} {'SQL Signal':>12} {'Exec Signal':>12} {'Difference':>12}")
        print("-"*80)
        
        for idx, row in mismatches.head(5).iterrows():
            print(f"{row['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{row['sql_signal']:>12.0f} "
                  f"{row['executable_signal']:>12.0f} "
                  f"{row['difference']:>12.0f}")
        
        if len(mismatches) > 5:
            print(f"... and {len(mismatches) - 5} more mismatches")
    
    # Signal distribution
    print("\n" + "-"*80)
    print("Signal Distribution:")
    print("-"*80)
    
    sql_dist = comparison_df['sql_signal'].value_counts().sort_index()
    exec_dist = comparison_df['executable_signal'].value_counts().sort_index()
    
    print(f"{'Signal Type':<15} {'SQL Strategy':>15} {'Exec Strategy':>15}")
    print("-"*80)
    
    signal_names = {-1: "Sell (-1)", 0: "Hold (0)", 1: "Buy (1)"}
    for signal in [-1, 0, 1]:
        sql_count = sql_dist.get(signal, 0)
        exec_count = exec_dist.get(signal, 0)
        print(f"{signal_names[signal]:<15} {sql_count:>15} {exec_count:>15}")
    
    print("="*80)


def main():
    """
    Main function to run the strategy validation test.
    """
    print("RSI Momentum Divergence Strategy Validation Test")
    print("="*80)
    
    # Database path
    db_path = "test_trading.db"
    
    # Create test database with sample data
    create_test_database(db_path)
    
    # Define the SQL query for RSI strategy
    # This query mimics the RSI calculation and generates signals
    rsi_sql_query = """
    SELECT 
        timestamp,
        signal
    FROM strategy_signals
    ORDER BY timestamp
    """
    
    try:
        # Create validator instance
        print("\nInitializing Strategy Validator...")
        validator = StrategyValidator(db_path)
        
        # Run comparison
        print("\nRunning strategy comparison...")
        comparison_df = validator.compare_strategies(
            sql_query=rsi_sql_query,
            executable_strategy_class=RSIMomentumDivergence,
            symbol='BTCUSDT',
            timeframe='1h',
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # Format and display results
        format_comparison_results(comparison_df)
        
        # Save detailed results to CSV
        output_file = "strategy_comparison_results.csv"
        comparison_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Additional analysis
        print("\nAdditional Analysis:")
        print("-"*80)
        
        # Check for systematic differences
        if not comparison_df['signals_match'].all():
            avg_diff = comparison_df['difference'].mean()
            std_diff = comparison_df['difference'].std()
            print(f"Average signal difference: {avg_diff:.4f}")
            print(f"Standard deviation of differences: {std_diff:.4f}")
            
            # Check if mismatches follow a pattern
            mismatch_indices = comparison_df[~comparison_df['signals_match']].index
            if len(mismatch_indices) > 1:
                gaps = np.diff(mismatch_indices)
                avg_gap = np.mean(gaps)
                print(f"Average gap between mismatches: {avg_gap:.2f} candles")
        
        # Performance metrics
        print("\nPerformance Metrics:")
        print(f"Total execution time: Check logs above")
        print(f"Candles processed per second: {len(comparison_df) / 1:.2f}")  # Rough estimate
        
    except Exception as e:
        print(f"\nError during validation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"\nTest database cleaned up.")


if __name__ == "__main__":
    main()
