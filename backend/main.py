"""
Main runner script for the Trading Bot.
Coordinates data collection and technical indicator calculation using a per-symbol SQLite database.
"""
import sys
import argparse
from datetime import datetime

from data_fetcher import DataCollector
from Indicators import (
    SimpleMovingAverageCalculator,
    BollingerBandsCalculator,
    IchimokuCloudCalculator,
    MACDCalculator,
    calculate_rsi_for_symbol_timeframe,
    ParabolicSARCalculator,
    FibonacciRetracementCalculator,
    GaussianChannelCalculator
)


def collect_raw_data(symbols, timeframes, start_time=None):
    """Collect raw data for specified symbols and timeframes."""
    print("=== COLLECTING RAW DATA ===")
    collector = DataCollector()
    
    if start_time:
        # Collect historical data from specific start time
        start_date = datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d')
        collector.collect_historical_data(symbols, timeframes, start_date)
    else:
        # Update existing data with latest values
        collector.update_all_data(symbols, timeframes)


def calculate_all_indicators(symbols, timeframes):
    """Calculate all technical indicators for specified symbols and timeframes."""
    print("\n=== CALCULATING ALL TECHNICAL INDICATORS ===")
    
    # Initialize all calculators
    sma_calc = SimpleMovingAverageCalculator()
    bb_calc = BollingerBandsCalculator()
    ichimoku_calc = IchimokuCloudCalculator()
    macd_calc = MACDCalculator()
    sar_calc = ParabolicSARCalculator()
    fib_calc = FibonacciRetracementCalculator()
    gc_calc = GaussianChannelCalculator()
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[PROCESSING] {symbol} - {timeframe.upper()}")
            
            # Fetch raw data once for all indicators
            df_raw = sma_calc.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            
            # Calculate Simple Moving Averages
            print(f"  [CALCULATING] Simple Moving Averages...")
            df_sma = sma_calc.calculate_sma(df_raw.copy())
            sma_calc.save_sma_data(df_sma, symbol, timeframe)
            
            # Calculate Bollinger Bands
            print(f"  [CALCULATING] Bollinger Bands...")
            df_bb = bb_calc.calculate_bollinger_bands(df_raw.copy())
            bb_calc.save_bollinger_bands_data(df_bb, symbol, timeframe)
            
            # Calculate Ichimoku Cloud
            print(f"  [CALCULATING] Ichimoku Cloud...")
            df_ichimoku = ichimoku_calc.calculate_ichimoku_cloud(df_raw.copy())
            ichimoku_calc.save_ichimoku_data(df_ichimoku, symbol, timeframe)
            
            # Calculate MACD
            print(f"  [CALCULATING] MACD...")
            df_macd = macd_calc.calculate_macd(df_raw.copy())
            macd_calc.save_macd_data(df_macd, symbol, timeframe)
            
            # Calculate RSI
            print(f"  [CALCULATING] RSI...")
            calculate_rsi_for_symbol_timeframe(symbol, timeframe)
            
            # Calculate Parabolic SAR
            print(f"  [CALCULATING] Parabolic SAR...")
            df_sar = sar_calc.calculate_parabolic_sar(df_raw.copy())
            sar_calc.save_parabolic_sar_data(df_sar, symbol, timeframe)
            
            # Calculate Fibonacci Retracement
            print(f"  [CALCULATING] Fibonacci Retracement...")
            df_fib = fib_calc.calculate_fibonacci_retracement(df_raw.copy())
            fib_calc.save_fibonacci_data(df_fib, symbol, timeframe)
            
            # Calculate Gaussian Channel
            print(f"  [CALCULATING] Gaussian Channel...")
            df_gc = gc_calc.calculate_gaussian_channel(df_raw.copy())
            gc_calc.save_gaussian_channel_data(df_gc, symbol, timeframe)

            print(f"  [COMPLETED] All indicators calculated for {symbol} ({timeframe})")


def calculate_simple_moving_averages(symbols, timeframes):
    """Calculate Simple Moving Average indicators for specified symbols and timeframes."""
    print("\n=== CALCULATING SIMPLE MOVING AVERAGES ===")
    calculator = SimpleMovingAverageCalculator()
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[CALCULATING] SMA for {symbol} - {timeframe.upper()}")
            df_raw = calculator.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            df_sma = calculator.calculate_sma(df_raw)
            calculator.save_sma_data(df_sma, symbol, timeframe)


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='Trading Bot - Data Collection and Analysis')
    parser.add_argument('--mode', choices=['collect', 'calculate', 'all_indicators', 'both'], 
                       default='both', help='Mode of operation')
    parser.add_argument('--symbols', nargs='+', 
                       default=['BTC/USDT'],  # Only Bitcoin by default
                       help='Symbols to process')
    parser.add_argument('--timeframes', nargs='+', 
                       default=['4h', '1d'],
                       help='Timeframes to process')
    parser.add_argument('--start-date', type=str, default='2021-01-01',
                       help='Start date for data collection (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Parse start date
    start_time = int(datetime.strptime(args.start_date, '%Y-%m-%d').timestamp() * 1000)
    
    print(f"Trading Bot starting...")
    print(f"Mode: {args.mode}")
    print(f"Symbols: {args.symbols}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Start date: {args.start_date}")
    
    try:
        if args.mode in ['collect', 'both']:
            collect_raw_data(args.symbols, args.timeframes, start_time)
        
        if args.mode in ['calculate', 'both']:
            calculate_simple_moving_averages(args.symbols, args.timeframes)
            
        if args.mode == 'all_indicators':
            calculate_all_indicators(args.symbols, args.timeframes)
            
        print("\n=== COMPLETED SUCCESSFULLY ===")
        
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
