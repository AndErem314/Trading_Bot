"""
Example Strategy Builder Usage

This script demonstrates how to use the StrategyBuilder to create and execute
multiple trading strategies from configurations. It shows the complete workflow
from data fetching to strategy execution and performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_builder import StrategyBuilder, Trade
from data_fetching import OHLCVDataFetcher, DataPreprocessor, IchimokuCalculator
from config import StrategyConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def fetch_and_prepare_data(symbol: str = 'BTC/USDT', timeframe: str = '1h', limit: int = 500):
    """Fetch and prepare data with Ichimoku indicators."""
    print(f"\n1. Fetching {symbol} {timeframe} data...")
    
    # Fetch OHLCV data
    fetcher = OHLCVDataFetcher(exchange_id='binance')
    raw_data = fetcher.fetch_data(symbol, timeframe, limit=limit)
    
    # Preprocess data
    print("2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.clean_data(raw_data, timeframe=timeframe)
    
    # Calculate Ichimoku indicators
    print("3. Calculating Ichimoku indicators...")
    calculator = IchimokuCalculator()
    ichimoku_data = calculator.calculate_ichimoku(clean_data)
    
    print(f"   Data range: {ichimoku_data.index[0]} to {ichimoku_data.index[-1]}")
    print(f"   Total bars: {len(ichimoku_data)}")
    
    return ichimoku_data

def run_strategy_analysis(config_file: str = 'strategies.yaml'):
    """Run analysis on all strategies from configuration file."""
    print("\n" + "="*60)
    print("STRATEGY BUILDER ANALYSIS")
    print("="*60)
    
    # Load configurations
    config_manager = StrategyConfigManager()
    strategies = config_manager.load_config(config_file)
    
    print(f"\nLoaded {len(strategies)} strategies from {config_file}")
    
    # Fetch data once
    data = fetch_and_prepare_data()
    
    # Create strategy builder
    builder = StrategyBuilder(max_pyramiding=1)
    
    # Results storage
    all_results = {}
    
    # Run each strategy
    for strategy_id, config in strategies.items():
        if not config.enabled:
            print(f"\nSkipping {config.name} (disabled)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running: {config.name}")
        print(f"{'='*60}")
        
        # Build strategy from configuration
        strategy_func = builder.build_strategy_from_config(config)
        
        # Execute strategy
        results = strategy_func(data)
        
        # Store results
        all_results[strategy_id] = {
            'config': config,
            'results': results
        }
        
        # Print summary
        print_strategy_summary(config, results)
    
    # Compare strategies
    compare_strategies(all_results)
    
    # Plot results
    plot_strategy_results(all_results, data)
    
    return all_results

def print_strategy_summary(config, results):
    """Print summary of strategy results."""
    metrics = results['metrics']
    trades = results['trades']
    
    print(f"\nStrategy: {config.name}")
    print(f"Buy Logic: {' {0} '.format(config.signal_conditions.buy_logic).join(config.signal_conditions.buy_conditions)}")
    print(f"Sell Logic: {' {0} '.format(config.signal_conditions.sell_logic).join(config.signal_conditions.sell_conditions)}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  Avg Win: ${metrics['avg_win']:.2f}")
    print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Total Return: {metrics['total_return']:.1f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.1f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Avg Holding: {metrics['avg_holding_periods']:.1f} periods")
    
    if trades:
        # Show recent trades
        print(f"\nLast 3 Trades:")
        for trade in trades[-3:]:
            print(f"  {trade.entry_time.strftime('%Y-%m-%d %H:%M')} -> "
                  f"{trade.exit_time.strftime('%Y-%m-%d %H:%M')}: "
                  f"{trade.pnl_pct:.1f}% ({trade.exit_reason})")

def compare_strategies(all_results):
    """Compare performance across all strategies."""
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_data = []
    for strategy_id, result_data in all_results.items():
        config = result_data['config']
        metrics = result_data['results']['metrics']
        
        comparison_data.append({
            'Strategy': config.name,
            'Trades': metrics['total_trades'],
            'Win Rate %': metrics['win_rate'],
            'Profit Factor': metrics['profit_factor'],
            'Total Return %': metrics['total_return'],
            'Max DD %': metrics['max_drawdown'],
            'Sharpe': metrics['sharpe_ratio']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by total return
    comparison_df = comparison_df.sort_values('Total Return %', ascending=False)
    
    print("\nStrategy Performance Ranking:")
    print(comparison_df.to_string(index=False))
    
    # Find best strategy by different metrics
    print("\nBest Strategies by Metric:")
    print(f"  Highest Return: {comparison_df.iloc[0]['Strategy']} ({comparison_df.iloc[0]['Total Return %']:.1f}%)")
    
    best_sharpe = comparison_df.loc[comparison_df['Sharpe'].idxmax()]
    print(f"  Best Sharpe: {best_sharpe['Strategy']} ({best_sharpe['Sharpe']:.2f})")
    
    best_pf = comparison_df.loc[comparison_df['Profit Factor'].idxmax()]
    print(f"  Best Profit Factor: {best_pf['Strategy']} ({best_pf['Profit Factor']:.2f})")
    
    lowest_dd = comparison_df.loc[comparison_df['Max DD %'].idxmin()]
    print(f"  Lowest Drawdown: {lowest_dd['Strategy']} ({lowest_dd['Max DD %']:.1f}%)")

def plot_strategy_results(all_results, data):
    """Plot strategy results including equity curves and signals."""
    print("\nGenerating performance plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Equity Curves
    ax1 = axes[0]
    for strategy_id, result_data in all_results.items():
        config = result_data['config']
        equity_curve = result_data['results']['equity_curve']
        
        # Convert to series with proper index
        equity_series = pd.Series(equity_curve, index=data.index[:len(equity_curve)])
        ax1.plot(equity_series.index, equity_series.values, label=config.name)
    
    ax1.set_title('Strategy Equity Curves', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Price with Entry/Exit Signals (for best strategy)
    ax2 = axes[1]
    
    # Find best strategy by return
    best_strategy_id = max(all_results.items(), 
                          key=lambda x: x[1]['results']['metrics']['total_return'])[0]
    best_result = all_results[best_strategy_id]
    
    # Plot price
    ax2.plot(data.index, data['close'], label='Close Price', color='black', alpha=0.7)
    
    # Plot cloud
    ax2.fill_between(data.index, data['cloud_top'], data['cloud_bottom'], 
                    where=data['cloud_color'] == 'green', 
                    color='green', alpha=0.2, label='Bullish Cloud')
    ax2.fill_between(data.index, data['cloud_top'], data['cloud_bottom'], 
                    where=data['cloud_color'] == 'red', 
                    color='red', alpha=0.2, label='Bearish Cloud')
    
    # Plot entry/exit points
    trades = best_result['results']['trades']
    for trade in trades[:20]:  # Limit to recent trades for clarity
        # Entry point
        ax2.scatter(trade.entry_time, trade.entry_price, 
                   color='green', marker='^', s=100, zorder=5)
        # Exit point
        exit_color = 'green' if trade.pnl > 0 else 'red'
        ax2.scatter(trade.exit_time, trade.exit_price, 
                   color=exit_color, marker='v', s=100, zorder=5)
    
    ax2.set_title(f'Best Strategy: {best_result["config"].name} - Entry/Exit Points', 
                 fontsize=14)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Drawdown Comparison
    ax3 = axes[2]
    for strategy_id, result_data in all_results.items():
        config = result_data['config']
        equity_curve = result_data['results']['equity_curve']
        
        # Calculate drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = ((equity_series - running_max) / running_max) * 100
        
        # Convert to series with proper index
        drawdown_series = pd.Series(drawdown.values, index=data.index[:len(drawdown)])
        ax3.fill_between(drawdown_series.index, 0, drawdown_series.values, 
                        alpha=0.3, label=config.name)
    
    ax3.set_title('Strategy Drawdowns', fontsize=14)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Invert y-axis for drawdown
    
    plt.tight_layout()
    plt.show()

def demonstrate_custom_exit_conditions():
    """Demonstrate custom exit conditions beyond simple signals."""
    print("\n" + "="*60)
    print("CUSTOM EXIT CONDITIONS DEMONSTRATION")
    print("="*60)
    
    # Create a custom strategy with Tenkan/Kijun cross exits
    from config import (
        StrategyConfig, SignalConditions, IchimokuParameters,
        RiskManagement, PositionSizing
    )
    
    # Strategy that exits on TK cross
    tk_cross_strategy = StrategyConfig(
        name="TK Cross Exit Strategy",
        description="Uses Tenkan/Kijun cross for exits instead of fixed targets",
        enabled=True,
        signal_conditions=SignalConditions(
            buy_conditions=["PriceAboveCloud", "TenkanAboveKijun"],
            sell_conditions=["TenkanBelowKijun"],  # Exit on TK bearish cross
            buy_logic="AND",
            sell_logic="OR"
        ),
        ichimoku_parameters=IchimokuParameters(),
        risk_management=RiskManagement(
            stop_loss_pct=3.0,  # Wider stop since we use TK cross
            take_profit_pct=999.0,  # Very high so it doesn't trigger
            trailing_stop=False
        ),
        position_sizing=PositionSizing(method="fixed", fixed_size=1.0),
        timeframe="1h",
        symbols=["BTC/USDT"]
    )
    
    # Cloud-based stop strategy
    cloud_stop_strategy = StrategyConfig(
        name="Cloud Stop Strategy",
        description="Uses cloud as dynamic stop loss",
        enabled=True,
        signal_conditions=SignalConditions(
            buy_conditions=["PriceAboveCloud", "ChikouAbovePrice"],
            sell_conditions=["PriceBelowCloud"],
            buy_logic="AND",
            sell_logic="OR"
        ),
        ichimoku_parameters=IchimokuParameters(),
        risk_management=RiskManagement(
            stop_loss_pct=5.0,  # Will be adjusted to cloud
            take_profit_pct=10.0,
            trailing_stop=True,
            trailing_stop_pct=2.0
        ),
        position_sizing=PositionSizing(method="risk_based"),
        timeframe="4h",
        symbols=["BTC/USDT"]
    )
    
    # Test both strategies
    data = fetch_and_prepare_data(timeframe='1h', limit=300)
    builder = StrategyBuilder()
    
    for strategy in [tk_cross_strategy, cloud_stop_strategy]:
        print(f"\nTesting: {strategy.name}")
        strategy_func = builder.build_strategy_from_config(strategy)
        results = strategy_func(data)
        print_strategy_summary(strategy, results)

def main():
    """Main function to run all demonstrations."""
    # Run strategy analysis on configured strategies
    all_results = run_strategy_analysis('strategies.yaml')
    
    # Demonstrate custom exit conditions
    demonstrate_custom_exit_conditions()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()