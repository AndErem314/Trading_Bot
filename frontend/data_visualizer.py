"""
Data visualization module for displaying raw market data.
Provides candlestick charts, volume plots, and other market data visualizations.
"""
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple


class DataVisualizer:
    """Visualizes raw market data from the database."""
    
    def __init__(self, db_path: str = 'data/market_data.db'):
        self.db_path = db_path
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 10
    
    def fetch_raw_data(self, symbol: str, timeframe: str, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch raw OHLCV data from the database."""
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM raw_data
            WHERE symbol = ? AND timeframe = ?
        '''
        params = [symbol, timeframe]
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp ASC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn, params=params)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    
    def plot_candlestick(self, df: pd.DataFrame, title: str = "Candlestick Chart") -> plt.Figure:
        """Create a candlestick chart with volume."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), 
                                      gridspec_kw={'height_ratios': [3, 1]}, 
                                      sharex=True)
        
        # Candlestick chart
        for i, (_, row) in enumerate(df.iterrows()):
            color = 'green' if row['close'] >= row['open'] else 'red'
            alpha = 0.8
            
            # Draw the wick (high-low line)
            ax1.plot([i, i], [row['low'], row['high']], 
                    color='black', linewidth=1, alpha=0.7)
            
            # Draw the body (open-close rectangle)
            height = abs(row['close'] - row['open'])
            bottom = min(row['open'], row['close'])
            
            rect = Rectangle((i - 0.4, bottom), 0.8, height, 
                           facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.5)
            ax1.add_patch(rect)
        
        # Format x-axis with dates
        ax1.set_xlim(-0.5, len(df) - 0.5)
        ax1.set_ylabel('Price (USDT)')
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        colors = ['green' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'red' 
                 for i in range(len(df))]
        ax2.bar(range(len(df)), df['volume'], color=colors, alpha=0.7, width=0.8)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        # Set x-axis labels
        step = max(1, len(df) // 10)  # Show ~10 labels
        xticks = range(0, len(df), step)
        xtick_labels = [df.iloc[i]['timestamp'].strftime('%Y-%m-%d') for i in xticks]
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xtick_labels, rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_line_chart(self, df: pd.DataFrame, title: str = "Price Chart") -> plt.Figure:
        """Create a simple line chart of closing prices."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.plot(range(len(df)), df['close'], linewidth=2, color='blue', alpha=0.8)
        ax.fill_between(range(len(df)), df['close'], alpha=0.3, color='blue')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Price (USDT)')
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        step = max(1, len(df) // 10)
        xticks = range(0, len(df), step)
        xtick_labels = [df.iloc[i]['timestamp'].strftime('%Y-%m-%d') for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_ohlc_summary(self, df: pd.DataFrame, title: str = "OHLC Summary") -> plt.Figure:
        """Create an OHLC summary chart."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        x = range(len(df))
        ax.plot(x, df['high'], label='High', alpha=0.7, color='green')
        ax.plot(x, df['low'], label='Low', alpha=0.7, color='red')
        ax.plot(x, df['open'], label='Open', alpha=0.7, color='blue')
        ax.plot(x, df['close'], label='Close', alpha=0.7, color='orange', linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Price (USDT)')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        step = max(1, len(df) // 10)
        xticks = range(0, len(df), step)
        xtick_labels = [df.iloc[i]['timestamp'].strftime('%Y-%m-%d') for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_volume_analysis(self, df: pd.DataFrame, title: str = "Volume Analysis") -> plt.Figure:
        """Create volume analysis charts."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Volume over time
        ax1.plot(range(len(df)), df['volume'], color='purple', alpha=0.7)
        ax1.set_title('Volume Over Time')
        ax1.set_ylabel('Volume')
        ax1.grid(True, alpha=0.3)
        
        # Volume histogram
        ax2.hist(df['volume'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax2.set_title('Volume Distribution')
        ax2.set_xlabel('Volume')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Price vs Volume scatter
        ax3.scatter(df['volume'], df['close'], alpha=0.6, color='blue')
        ax3.set_title('Price vs Volume')
        ax3.set_xlabel('Volume')
        ax3.set_ylabel('Price (USDT)')
        ax3.grid(True, alpha=0.3)
        
        # Volume moving average
        volume_ma = df['volume'].rolling(window=20).mean()
        ax4.plot(range(len(df)), df['volume'], alpha=0.5, color='purple', label='Volume')
        ax4.plot(range(len(df)), volume_ma, color='red', linewidth=2, label='MA(20)')
        ax4.set_title('Volume with Moving Average')
        ax4.set_ylabel('Volume')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def get_available_data(self) -> pd.DataFrame:
        """Get summary of available data in the database."""
        query = '''
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as record_count,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM raw_data
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        '''
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn)
    
    def display_data_summary(self):
        """Display a summary of available data."""
        summary = self.get_available_data()
        print("\n" + "="*80)
        print("AVAILABLE DATA SUMMARY")
        print("="*80)
        
        for _, row in summary.iterrows():
            print(f"Symbol: {row['symbol']:<12} | Timeframe: {row['timeframe']:<4} | "
                  f"Records: {row['record_count']:<8} | "
                  f"From: {row['start_date'][:10]} | To: {row['end_date'][:10]}")
        
        print("="*80)
    
    def visualize_symbol(self, symbol: str, timeframe: str, 
                        chart_type: str = 'candlestick',
                        days: Optional[int] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        """Visualize data for a specific symbol and timeframe."""
        
        # Fetch data
        end_date = None
        start_date = None
        if days:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            start_date = start_date.strftime('%Y-%m-%d')
            end_date = end_date.strftime('%Y-%m-%d')
        
        df = self.fetch_raw_data(symbol, timeframe, start_date, end_date)
        
        if df.empty:
            print(f"No data found for {symbol} ({timeframe})")
            return None
        
        # Create title
        period_str = f" - Last {days} days" if days else ""
        title = f"{symbol} ({timeframe.upper()}) Price Chart{period_str}"
        
        # Generate appropriate chart
        if chart_type == 'candlestick':
            fig = self.plot_candlestick(df, title)
        elif chart_type == 'line':
            fig = self.plot_line_chart(df, title)
        elif chart_type == 'ohlc':
            fig = self.plot_ohlc_summary(df, title)
        elif chart_type == 'volume':
            fig = self.plot_volume_analysis(df, title)
        else:
            print(f"Unknown chart type: {chart_type}")
            return None
        
        # Save if requested
        if save_path:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        return fig


def get_user_input():
    """Get user input for visualization parameters."""
    print("\n" + "="*60)
    print("   TRADING BOT DATA VISUALIZER")
    print("="*60)
    
    # Get available data
    visualizer = DataVisualizer()
    summary = visualizer.get_available_data()
    
    # Display available symbols
    print("\nAvailable trading pairs:")
    symbols = summary['symbol'].unique()
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}")
    
    # Get symbol selection
    while True:
        try:
            choice = input(f"\nSelect a trading pair (1-{len(symbols)}) or enter symbol directly: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(symbols):
                selected_symbol = symbols[int(choice) - 1]
                break
            elif choice in symbols:
                selected_symbol = choice
                break
            else:
                print(f"Invalid choice. Please select 1-{len(symbols)} or enter a valid symbol.")
        except ValueError:
            print("Invalid input. Please enter a number or symbol.")
    
    # Get available timeframes for selected symbol
    symbol_timeframes = summary[summary['symbol'] == selected_symbol]['timeframe'].tolist()
    print(f"\nAvailable timeframes for {selected_symbol}:")
    for i, tf in enumerate(symbol_timeframes, 1):
        print(f"{i}. {tf}")
    
    # Get timeframe selection
    while True:
        try:
            choice = input(f"\nSelect timeframe (1-{len(symbol_timeframes)}) or enter directly: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(symbol_timeframes):
                selected_timeframe = symbol_timeframes[int(choice) - 1]
                break
            elif choice in symbol_timeframes:
                selected_timeframe = choice
                break
            else:
                print(f"Invalid choice. Please select 1-{len(symbol_timeframes)} or enter a valid timeframe.")
        except ValueError:
            print("Invalid input. Please enter a number or timeframe.")
    
    # Get number of days
    while True:
        try:
            days_input = input("\nEnter number of days to visualize (or 'all' for all available data): ").strip().lower()
            if days_input == 'all' or days_input == '':
                selected_days = None
                break
            else:
                selected_days = int(days_input)
                if selected_days <= 0:
                    print("Please enter a positive number of days.")
                    continue
                break
        except ValueError:
            print("Invalid input. Please enter a number or 'all'.")
    
    # Get chart type
    chart_types = ['candlestick', 'line', 'ohlc', 'volume']
    print("\nAvailable chart types:")
    for i, chart_type in enumerate(chart_types, 1):
        print(f"{i}. {chart_type.capitalize()}")
    
    while True:
        try:
            choice = input(f"\nSelect chart type (1-{len(chart_types)}) [default: candlestick]: ").strip()
            if choice == '':
                selected_chart_type = 'candlestick'
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(chart_types):
                selected_chart_type = chart_types[int(choice) - 1]
                break
            elif choice.lower() in chart_types:
                selected_chart_type = choice.lower()
                break
            else:
                print(f"Invalid choice. Please select 1-{len(chart_types)} or enter a valid chart type.")
        except ValueError:
            print("Invalid input. Please enter a number or chart type.")
    
    # Ask about saving
    save_chart = input("\nSave chart to file? (y/n) [default: n]: ").strip().lower()
    save_path = None
    if save_chart in ['y', 'yes']:
        default_filename = f"{selected_symbol.replace('/', '_')}_{selected_timeframe}_{selected_chart_type}{'_' + str(selected_days) + 'd' if selected_days else '_all'}.png"
        filename = input(f"Enter filename [default: {default_filename}]: ").strip()
        if not filename:
            filename = default_filename
        save_path = f"charts/{filename}"
    
    return {
        'symbol': selected_symbol,
        'timeframe': selected_timeframe,
        'days': selected_days,
        'chart_type': selected_chart_type,
        'save_path': save_path
    }


def interactive_mode():
    """Interactive mode with user input."""
    visualizer = DataVisualizer()
    
    # Display data summary
    visualizer.display_data_summary()
    
    while True:
        try:
            # Get user preferences
            params = get_user_input()
            
            print(f"\nGenerating {params['chart_type']} chart for {params['symbol']} ({params['timeframe']})...")
            if params['days']:
                print(f"Time period: Last {params['days']} days")
            else:
                print("Time period: All available data")
            
            # Generate visualization
            fig = visualizer.visualize_symbol(
                symbol=params['symbol'],
                timeframe=params['timeframe'],
                chart_type=params['chart_type'],
                days=params['days'],
                save_path=params['save_path']
            )
            
            if fig:
                plt.show()
                plt.close(fig)
                print("✓ Chart displayed successfully!")
            else:
                print("✗ Failed to generate chart.")
            
            # Ask if user wants to continue
            continue_choice = input("\nGenerate another chart? (y/n) [default: n]: ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue_choice = input("Continue anyway? (y/n) [default: y]: ").strip().lower()
            if continue_choice in ['n', 'no']:
                break
    
    print("\nVisualization session complete!")


def batch_mode():
    """Batch mode for generating multiple charts automatically."""
    visualizer = DataVisualizer()
    
    # Display data summary
    visualizer.display_data_summary()
    
    # Get available symbols
    summary = visualizer.get_available_data()
    symbols = summary['symbol'].unique()
    
    print(f"\nAvailable symbols: {', '.join(symbols)}")
    
    # Get batch parameters
    days_input = input("\nEnter number of days for all charts (or 'all' for all data) [default: 90]: ").strip()
    if days_input.lower() == 'all' or days_input == '':
        days = 90 if days_input == '' else None
    else:
        try:
            days = int(days_input)
        except ValueError:
            print("Invalid input, using default 90 days.")
            days = 90
    
    chart_type = input("Enter chart type (candlestick/line/ohlc/volume) [default: candlestick]: ").strip().lower()
    if chart_type not in ['candlestick', 'line', 'ohlc', 'volume']:
        chart_type = 'candlestick'
    
    timeframes = ['1d', '4h']
    
    print(f"\nGenerating {chart_type} charts for all symbols...")
    print(f"Time period: {'Last ' + str(days) + ' days' if days else 'All available data'}")
    
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                print(f"\nGenerating chart for {symbol} ({timeframe})...")
                
                fig = visualizer.visualize_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    chart_type=chart_type,
                    days=days,
                    save_path=f"charts/{symbol.replace('/', '_')}_{timeframe}_{chart_type}{'_' + str(days) + 'd' if days else '_all'}.png"
                )
                
                if fig:
                    plt.close(fig)  # Don't show in batch mode, just save
                    print(f"✓ Chart saved for {symbol} ({timeframe})")
                else:
                    print(f"✗ Failed to generate chart for {symbol} ({timeframe})")
                    
            except Exception as e:
                print(f"✗ Error generating chart for {symbol} ({timeframe}): {e}")
    
    print("\nBatch generation complete!")


def main():
    """Main function with mode selection."""
    print("\n" + "="*60)
    print("   TRADING BOT DATA VISUALIZER")
    print("="*60)
    print("\nSelect mode:")
    print("1. Interactive mode (custom charts with user input)")
    print("2. Batch mode (generate charts for all symbols)")
    print("3. Quick demo (legacy mode)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3) [default: 1]: ").strip()
            if choice == '' or choice == '1':
                interactive_mode()
                break
            elif choice == '2':
                batch_mode()
                break
            elif choice == '3':
                # Legacy demo mode
                visualizer = DataVisualizer()
                visualizer.display_data_summary()
                
                # Updated to include all symbols
                symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
                timeframes = ['1d', '4h']
                
                days = 90  # Default demo period
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        print(f"\nGenerating demo chart for {symbol} ({timeframe})...")
                        
                        fig = visualizer.visualize_symbol(
                            symbol=symbol,
                            timeframe=timeframe,
                            chart_type='candlestick',
                            days=days,
                            save_path=f"charts/{symbol.replace('/', '_')}_{timeframe}_candlestick_{days}d.png"
                        )
                        
                        if fig:
                            plt.show()
                            plt.close(fig)
                
                print("\nDemo visualization complete!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


if __name__ == '__main__':
    main()
