"""
Performance Analytics Example

This module demonstrates how to use the PerformanceAnalyzer with
backtest results from the IchimokuBacktester.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

from backend.executable_workflow.analytics import PerformanceAnalyzer
from backend.executable_workflow.backtesting import IchimokuBacktester
from backend.executable_workflow.data_fetching import OHLCVDataFetcher, DataPreprocessor
from backend.executable_workflow.indicators import IchimokuCalculator
from backend.executable_workflow.config.models import (
    StrategyConfig, SignalCombination, IchimokuParameters,
    RiskManagement, PositionSizing
)

logger = logging.getLogger(__name__)


class PerformanceAnalysisExample:
    """
    Example class demonstrating comprehensive performance analysis
    of trading strategies using the PerformanceAnalyzer.
    """
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.backtester = IchimokuBacktester()
        
    def run_complete_analysis_example(self):
        """
        Run a complete example showing backtest and performance analysis.
        """
        logger.info("Starting Performance Analysis Example")
        
        # 1. Generate sample backtest results
        backtest_results = self._generate_sample_backtest_results()
        
        # 2. Extract trade history and equity curve
        trades_df = backtest_results['trades']
        equity_curve = backtest_results['equity_curve']
        
        # 3. Calculate all performance metrics
        logger.info("\n" + "="*60)
        logger.info("CALCULATING PERFORMANCE METRICS")
        logger.info("="*60)
        
        metrics = self.analyzer.calculate_all_metrics(trades_df, equity_curve)
        
        # 4. Display key metrics
        self._display_key_metrics(metrics)
        
        # 5. Generate performance report
        logger.info("\n" + "="*60)
        logger.info("GENERATING PERFORMANCE REPORT")
        logger.info("="*60)
        
        report = self.analyzer.generate_performance_report()
        print(report)
        
        # 6. Create visualizations
        logger.info("\n" + "="*60)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("="*60)
        
        # Equity curve plot
        fig1 = self.analyzer.plot_equity_curve(
            save_path='equity_curve.png',
            show_drawdowns=True,
            show_trades=True
        )
        
        # Additional custom visualizations
        self._create_custom_visualizations(metrics)
        
        # 7. Analyze drawdowns
        logger.info("\n" + "="*60)
        logger.info("DRAWDOWN ANALYSIS")
        logger.info("="*60)
        
        dd_analysis = self.analyzer.get_drawdown_analysis()
        print("\nTop 5 Drawdown Periods:")
        print(dd_analysis.head())
        
        # 8. Compare different strategies
        logger.info("\n" + "="*60)
        logger.info("STRATEGY COMPARISON")
        logger.info("="*60)
        
        self._compare_multiple_strategies()
        
        return metrics
    
    def _generate_sample_backtest_results(self) -> Dict[str, Any]:
        """
        Generate sample backtest results for demonstration.
        In real usage, this would come from actual backtesting.
        """
        # Create sample trade history
        np.random.seed(42)
        n_trades = 50
        
        # Generate realistic trade data
        trade_dates = pd.date_range(
            start='2024-01-01', 
            periods=n_trades*2, 
            freq='4H'
        )
        
        trades_data = []
        for i in range(n_trades):
            entry_price = 50000 + np.random.normal(0, 1000)
            exit_price = entry_price * (1 + np.random.normal(0.001, 0.02))
            
            # Add some losing trades
            if np.random.random() < 0.4:  # 40% losing trades
                exit_price = entry_price * (1 - np.random.uniform(0.005, 0.015))
            
            net_pnl = (exit_price - entry_price) * 0.001  # Position size
            
            trades_data.append({
                'entry_time': trade_dates[i*2],
                'exit_time': trade_dates[i*2 + 1],
                'symbol': 'BTC/USDT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': 0.001,
                'net_pnl': net_pnl,
                'return_pct': (exit_price / entry_price - 1) * 100,
                'bars_held': np.random.randint(5, 50)
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        # Generate equity curve
        initial_capital = 10000
        equity_dates = pd.date_range(
            start='2024-01-01',
            end='2024-03-01',
            freq='1H'
        )
        
        # Simulate equity curve with realistic volatility
        returns = np.random.normal(0.0002, 0.01, len(equity_dates))
        equity_values = initial_capital * (1 + returns).cumprod()
        
        # Add some trends and drawdowns
        trend = np.linspace(0, 0.1, len(equity_dates))
        drawdown_start = len(equity_dates) // 3
        drawdown_end = drawdown_start + len(equity_dates) // 6
        equity_values[drawdown_start:drawdown_end] *= 0.92  # 8% drawdown
        
        equity_values = equity_values * (1 + trend)
        
        equity_curve = pd.DataFrame({
            'total_value': equity_values,
            'returns': pd.Series(equity_values).pct_change()
        }, index=equity_dates)
        
        return {
            'trades': trades_df,
            'equity_curve': equity_curve,
            'metrics': {}  # Placeholder
        }
    
    def _display_key_metrics(self, metrics: Any):
        """Display key performance metrics in a formatted way."""
        print("\n" + "="*60)
        print("KEY PERFORMANCE METRICS")
        print("="*60)
        
        # Returns
        print(f"\nRETURNS:")
        print(f"  Total Return:      {metrics.total_return:>10.2f}%")
        print(f"  Annualized Return: {metrics.annualized_return:>10.2f}%")
        print(f"  Monthly Average:   {metrics.monthly_returns.mean():>10.2f}%")
        
        # Risk metrics
        print(f"\nRISK:")
        print(f"  Max Drawdown:      {metrics.max_drawdown_pct:>10.2f}%")
        print(f"  Volatility (Ann):  {metrics.annualized_volatility:>10.2f}%")
        print(f"  Value at Risk:     {metrics.var_95:>10.2f}%")
        
        # Risk-adjusted returns
        print(f"\nRISK-ADJUSTED RETURNS:")
        print(f"  Sharpe Ratio:      {metrics.sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:     {metrics.sortino_ratio:>10.2f}")
        print(f"  Calmar Ratio:      {metrics.calmar_ratio:>10.2f}")
        
        # Trading performance
        print(f"\nTRADING PERFORMANCE:")
        print(f"  Win Rate:          {metrics.win_rate:>10.2f}%")
        print(f"  Profit Factor:     {metrics.profit_factor:>10.2f}")
        print(f"  Average Trade:     ${metrics.expectancy:>10.2f}")
        print(f"  Payoff Ratio:      {metrics.payoff_ratio:>10.2f}")
    
    def _create_custom_visualizations(self, metrics: Any):
        """Create additional custom visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Monthly Returns Distribution
        ax1 = axes[0, 0]
        metrics.monthly_returns.hist(bins=20, ax=ax1, alpha=0.7, color='blue')
        ax1.axvline(metrics.monthly_returns.mean(), color='red', 
                   linestyle='--', label=f'Mean: {metrics.monthly_returns.mean():.2f}%')
        ax1.set_title('Monthly Returns Distribution')
        ax1.set_xlabel('Monthly Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Win/Loss Distribution
        ax2 = axes[0, 1]
        win_loss_data = [metrics.winning_trades, metrics.losing_trades]
        labels = [f'Wins ({metrics.winning_trades})', 
                 f'Losses ({metrics.losing_trades})']
        colors = ['green', 'red']
        ax2.pie(win_loss_data, labels=labels, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Win/Loss Distribution')
        
        # 3. Rolling Sharpe Ratio
        ax3 = axes[1, 0]
        if metrics.rolling_sharpe is not None:
            rolling_sharpe = metrics.rolling_sharpe.dropna()
            ax3.plot(rolling_sharpe.index, rolling_sharpe.values, 
                    color='purple', linewidth=2)
            ax3.axhline(metrics.sharpe_ratio, color='red', linestyle='--', 
                       label=f'Overall: {metrics.sharpe_ratio:.2f}')
            ax3.set_title('Rolling Sharpe Ratio (60-day window)')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Profit Factor by Month
        ax4 = axes[1, 1]
        # This would require monthly trade data - simplified here
        months = ['Jan', 'Feb', 'Mar']
        pf_values = [1.5, 2.1, 1.8]  # Sample data
        bars = ax4.bar(months, pf_values, color=['red' if x < 1 else 'green' for x in pf_values])
        ax4.axhline(1, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Monthly Profit Factor')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Profit Factor')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300)
        plt.close()
        
        logger.info("Custom visualizations saved to performance_analysis.png")
    
    def _compare_multiple_strategies(self):
        """Compare performance metrics across multiple strategies."""
        # Simulate multiple strategy results
        strategies = {
            'Conservative': {
                'total_return': 15.2,
                'sharpe_ratio': 1.2,
                'max_drawdown': -8.5,
                'win_rate': 65,
                'profit_factor': 1.8
            },
            'Aggressive': {
                'total_return': 28.5,
                'sharpe_ratio': 0.9,
                'max_drawdown': -18.2,
                'win_rate': 52,
                'profit_factor': 1.5
            },
            'Balanced': {
                'total_return': 21.3,
                'sharpe_ratio': 1.5,
                'max_drawdown': -12.1,
                'win_rate': 58,
                'profit_factor': 1.7
            }
        }
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(strategies).T
        
        print("\nStrategy Comparison:")
        print(comparison_df.to_string())
        
        # Visualize comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        metrics_to_plot = [
            ('total_return', 'Total Return (%)'),
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('max_drawdown', 'Max Drawdown (%)'),
            ('profit_factor', 'Profit Factor')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            values = comparison_df[metric].values
            strategies_list = comparison_df.index.tolist()
            colors = ['green' if v > 0 else 'red' for v in values]
            
            bars = ax.bar(strategies_list, values, color=colors, alpha=0.7)
            ax.set_title(title)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('strategy_comparison.png', dpi=300)
        plt.close()
        
        logger.info("Strategy comparison saved to strategy_comparison.png")
    
    def demonstrate_metric_calculations(self):
        """
        Demonstrate and explain how each metric is calculated.
        """
        print("\n" + "="*60)
        print("PERFORMANCE METRICS FORMULAS AND DEFINITIONS")
        print("="*60)
        
        formulas = {
            "Total Return": "((Final Value - Initial Value) / Initial Value) × 100",
            "Annualized Return": "((1 + Total Return)^(1/Years) - 1) × 100",
            "Sharpe Ratio": "(Annualized Return - Risk Free Rate) / Annualized Volatility",
            "Sortino Ratio": "(Annualized Return - Risk Free Rate) / Downside Deviation",
            "Calmar Ratio": "Annualized Return / |Maximum Drawdown %|",
            "Information Ratio": "(Portfolio Return - Benchmark Return) / Tracking Error",
            "Win Rate": "(Winning Trades / Total Trades) × 100",
            "Profit Factor": "Gross Profit / Gross Loss",
            "Expectancy": "Net Profit / Total Trades",
            "Payoff Ratio": "Average Win / Average Loss",
            "Max Drawdown": "Maximum peak to trough decline",
            "Value at Risk (95%)": "5th percentile of returns distribution",
            "Recovery Factor": "Total Return / Maximum Drawdown",
            "Volatility": "Standard deviation of returns × √252 (annualized)"
        }
        
        for metric, formula in formulas.items():
            print(f"\n{metric}:")
            print(f"  Formula: {formula}")
        
        print("\n" + "-"*60)
        print("\nMETRIC INTERPRETATIONS:")
        print("-"*60)
        
        interpretations = {
            "Sharpe Ratio": "Higher is better. >1 is good, >2 is excellent",
            "Sortino Ratio": "Similar to Sharpe but focuses on downside risk",
            "Win Rate": "% of profitable trades. >50% is positive",
            "Profit Factor": ">1 means profitable. >1.5 is good, >2 is excellent",
            "Max Drawdown": "Lower is better. <20% is generally acceptable",
            "Calmar Ratio": "Higher is better. >1 is good, >3 is excellent"
        }
        
        for metric, interpretation in interpretations.items():
            print(f"\n{metric}: {interpretation}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create example instance
    example = PerformanceAnalysisExample()
    
    # Run complete analysis
    print("\n1. Running Complete Performance Analysis Example:")
    metrics = example.run_complete_analysis_example()
    
    # Show metric calculations
    print("\n2. Performance Metrics Explanations:")
    example.demonstrate_metric_calculations()
    
    print("\n\nPerformance Analysis Example Complete!")
    print("Check the generated files:")
    print("- equity_curve.png")
    print("- performance_analysis.png")
    print("- strategy_comparison.png")