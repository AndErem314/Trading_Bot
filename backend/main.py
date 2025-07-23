"""
Main runner script for the Trading Bot.
Coordinates data collection and Gaussian Channel calculation.
"""
import sys
import argparse
from datetime import datetime

from data_fetcher import RawDataCollector
from gaussian_channel import GaussianChannelCalculator
from simple_moving_average import SimpleMovingAverageCalculator
from bollinger_bands import BollingerBandsCalculator
from ichimoku_cloud import IchimokuCloudCalculator
from macd import MACDCalculator
from rsi import calculate_rsi_for_symbol_timeframe
from parabolic_sar import ParabolicSARCalculator


def collect_raw_data(symbols, timeframes, start_time=None):
    """Collect raw data for specified symbols and timeframes."""
    print("=== COLLECTING RAW DATA ===")
    collector = RawDataCollector()
    
    for symbol in symbols:
        for timeframe in timeframes:
            collector.collect_data(symbol, timeframe, start_time=start_time)


def calculate_all_indicators(symbols, timeframes):
    """Calculate all technical indicators for specified symbols and timeframes."""
    print("\n=== CALCULATING ALL TECHNICAL INDICATORS ===")
    
    # Initialize all calculators
    gaussian_calc = GaussianChannelCalculator()
    sma_calc = SimpleMovingAverageCalculator()
    bb_calc = BollingerBandsCalculator()
    ichimoku_calc = IchimokuCloudCalculator()
    macd_calc = MACDCalculator()
    sar_calc = ParabolicSARCalculator()
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[PROCESSING] {symbol} - {timeframe.upper()}")
            
            # Fetch raw data once for all indicators
            df_raw = gaussian_calc.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            
            # Calculate Gaussian Channel
            print(f"  [CALCULATING] Gaussian Channel...")
            df_gc = gaussian_calc.calculate_gaussian_channel(df_raw.copy())
            gaussian_calc.save_gaussian_channel_data(df_gc, symbol, timeframe)
            
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

            print(f"  [COMPLETED] All indicators calculated for {symbol} ({timeframe})")


def calculate_gaussian_channels(symbols, timeframes):
    """Calculate Gaussian Channel indicators for specified symbols and timeframes."""
    print("\n=== CALCULATING GAUSSIAN CHANNELS ===")
    calculator = GaussianChannelCalculator()
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n[CALCULATING] Gaussian Channel for {symbol} - {timeframe.upper()}")
            df_raw = calculator.fetch_raw_data(symbol, timeframe)
            if df_raw.empty:
                print(f"[INFO] No raw data available for {symbol} ({timeframe})")
                continue
            df_gc = calculator.calculate_gaussian_channel(df_raw)
            calculator.save_gaussian_channel_data(df_gc, symbol, timeframe)


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='Trading Bot - Data Collection and Analysis')
    parser.add_argument('--mode', choices=['collect', 'calculate', 'all_indicators', 'both'], 
                       default='both', help='Mode of operation')
    parser.add_argument('--symbols', nargs='+', 
                       default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC'],
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
            calculate_gaussian_channels(args.symbols, args.timeframes)
            
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
