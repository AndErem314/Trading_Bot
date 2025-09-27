"""
Example usage of the professional reporting system.

This script demonstrates:
1. Generating comprehensive backtest reports
2. Creating executive summaries
3. Exporting results in different formats
"""

from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from backend.executable_workflow.reporting import ReportGenerator
from backend.data_fetching.data_fetcher import DataFetcher
from backend.executable_workflow.backtesting import IchimokuBacktester
from backend.executable_workflow.analytics import PerformanceAnalyzer


def run_backtest():
    """Run a sample backtest to generate results"""
    # Initialize components
    data_fetcher = DataFetcher()
    backtester = IchimokuBacktester(initial_capital=10000)
    
    # Fetch some data
    symbol = "BTC-USD"
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    data = data_fetcher.fetch_data(symbol, start_date, end_date)
    
    # Define strategy parameters
    strategy_params = {
        'ichimoku_params': {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52
        },
        'signal_config': {
            'entry_signals': ['cloud_breakout', 'tk_cross'],
            'exit_signals': ['stop_loss', 'take_profit', 'cloud_reversal']
        },
        'risk_params': {
            'stop_loss_percent': 0.02,
            'take_profit_percent': 0.04,
            'position_size': 0.95
        }
    }
    
    # Run backtest
    trades, equity_curve = backtester.backtest(
        data,
        ichimoku_params=strategy_params['ichimoku_params'],
        signal_config=strategy_params['signal_config'],
        risk_params=strategy_params['risk_params']
    )
    
    # Calculate metrics
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_all_metrics(trades, equity_curve)
    
    # Prepare results
    results = {
        'data': data,
        'trades': trades,
        'equity_curve': equity_curve,
        'metrics': {
            'performance_metrics': metrics.__dict__
        },
        'strategy_config': {
            'name': 'Ichimoku Cloud Strategy',
            'description': 'Trading BTC using Ichimoku cloud breakouts and TK cross signals',
            'parameters': strategy_params
        }
    }
    
    return results


def main():
    """Demonstrate the reporting system"""
    print("Running backtest...")
    results = run_backtest()
    
    print("\nGenerating reports...")
    
    # Initialize report generator
    reporter = ReportGenerator(output_dir="frontend/backtest_reports")
    
    # 1. Create executive summary
    print("\nExecutive Summary:")
    print("=" * 80)
    summary = reporter.create_executive_summary(results['metrics']['performance_metrics'])
    print(f"Performance Rating: {summary['performance_rating']['rating']}")
    print(f"Risk Level: {summary['risk_assessment']['risk_level']}")
    print("\nKey Metrics:")
    print(f"- Total Return: {summary['overview']['total_return']}")
    print(f"- Sharpe Ratio: {summary['overview']['sharpe_ratio']}")
    print(f"- Win Rate: {summary['overview']['win_rate']}")
    print(f"- Max Drawdown: {summary['overview']['max_drawdown']}")
    
    print("\nKey Insights:")
    for i, insight in enumerate(summary['key_insights'], 1):
        print(f"{i}. {insight}")
        
    # 2. Generate comprehensive report in all formats
    print("\nGenerating comprehensive reports...")
    report_paths = reporter.generate_backtest_report(
        results=results,
        format='all',
        filename_prefix='ichimoku_backtest'
    )
    
    print("\nReport files generated:")
    print("-" * 80)
    for format, path in report_paths.items():
        if format == 'csv':
            print(f"\nCSV Reports:")
            for csv_type, csv_path in path.items():
                print(f"- {csv_type}: {csv_path}")
        else:
            print(f"- {format}: {path}")
            
    print("\nAnalyzing report...")
    print("=" * 80)
    metrics = results['metrics']['performance_metrics']
    trades = results['trades']
    
    print("\nDetailed Statistics:")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Average Trade Return: {metrics.get('avg_trade_return', 0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    print(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    if not trades.empty:
        print("\nTrade Analysis:")
        print(f"Longest Winning Streak: {metrics.get('longest_win_streak', 0)}")
        print(f"Longest Losing Streak: {metrics.get('longest_loss_streak', 0)}")
        print(f"Average Win: ${metrics.get('avg_win', 0):.2f}")
        print(f"Average Loss: ${metrics.get('avg_loss', 0):.2f}")
        print(f"Largest Win: ${metrics.get('largest_win', 0):.2f}")
        print(f"Largest Loss: ${metrics.get('largest_loss', 0):.2f}")
        
        if 'duration_hours' in trades.columns:
            avg_duration = trades['duration_hours'].mean()
            print(f"\nAverage Trade Duration: {avg_duration:.1f} hours")
        
        if 'entry_signal' in trades.columns:
            print("\nEntry Signal Performance:")
            signal_stats = trades.groupby('entry_signal').agg({
                'pnl': ['count', 'mean', 'sum'],
                'return_pct': 'mean'
            })
            for signal in signal_stats.index:
                count = signal_stats.loc[signal, ('pnl', 'count')]
                avg_pnl = signal_stats.loc[signal, ('pnl', 'mean')]
                total_pnl = signal_stats.loc[signal, ('pnl', 'sum')]
                avg_return = signal_stats.loc[signal, ('return_pct', 'mean')]
                print(f"\n{signal}:")
                print(f"  Count: {count}")
                print(f"  Average P&L: ${avg_pnl:.2f}")
                print(f"  Total P&L: ${total_pnl:.2f}")
                print(f"  Average Return: {avg_return:.2%}")
    
    print("\nDone! Reports have been generated and saved.")


if __name__ == "__main__":
    main()