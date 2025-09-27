"""
Example usage of the ParameterOptimizer for automated parameter testing.

This script demonstrates:
1. Grid search optimization
2. Walk-forward optimization
3. Statistical validation
4. Parameter selection
"""

import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from backend.executable_workflow.optimization.parameter_optimizer import (
    ParameterOptimizer, ParameterSpace, OptimizationResult
)
from backend.executable_workflow.visualization import ResultsVisualizer
from backend.data_fetching.data_fetcher import DataFetcher
from backend.executable_workflow.backtesting import IchimokuBacktester
from backend.executable_workflow.analytics import PerformanceAnalyzer

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def run_grid_search_example():
    """Demonstrate grid search optimization"""
    print("=" * 80)
    print("GRID SEARCH OPTIMIZATION EXAMPLE")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer(
        db_path="btc_optimization_results.db",
        min_sample_trades=30,
        significance_level=0.05,
        n_jobs=-1  # Use all CPUs
    )
    
    # Define custom parameter space
    param_space = ParameterSpace(
        tenkan_periods=[7, 8, 9, 10],
        kijun_periods=[22, 24, 26],
        senkou_b_periods=[48, 52, 56],
        signal_combinations=[
            ['cloud_breakout'],
            ['cloud_breakout', 'tk_cross'],
            ['cloud_breakout', 'tk_cross', 'price_momentum']
        ],
        stop_loss_percent=[0.02, 0.025],
        take_profit_percent=[0.03, 0.04]
    )
    
    print(f"Total parameter combinations: {param_space.get_total_combinations()}")
    
    # Run grid search
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    symbol = "BTC-USD"
    
    print(f"\nRunning grid search for {symbol} from {start_date} to {end_date}")
    print("This may take several minutes depending on your CPU...")
    
    results = optimizer.grid_search_parameters(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        parameter_space=param_space,
        parallel=True
    )
    
    # Display top 10 results
    print(f"\nOptimization complete! Found {len(results)} results.")
    print("\nTop 10 parameter sets by Sharpe ratio:")
    print("-" * 80)
    
    for i, result in enumerate(results[:10]):
        print(f"\nRank {i+1}:")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"  Total Return: {result.total_return:.2%}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Parameters:")
        print(f"    - Tenkan: {result.parameters['ichimoku_params']['tenkan_period']}")
        print(f"    - Kijun: {result.parameters['ichimoku_params']['kijun_period']}")
        print(f"    - Senkou B: {result.parameters['ichimoku_params']['senkou_b_period']}")
        print(f"    - Signals: {', '.join(result.parameters['signal_config']['entry_signals'])}")
        print(f"    - Stop Loss: {result.parameters['risk_params']['stop_loss_percent']:.1%}")
        print(f"    - Take Profit: {result.parameters['risk_params']['take_profit_percent']:.1%}")
    
    # Select best parameters
    best_params = optimizer.select_best_parameters(results)
    print("\n" + "=" * 80)
    print("BEST PARAMETERS (Multi-criteria selection):")
    print("=" * 80)
    print(f"Selection Score: {best_params['selection_score']:.3f}")
    print(f"Expected Sharpe Ratio: {best_params['expected_performance']['sharpe_ratio']:.3f}")
    print(f"Expected Return: {best_params['expected_performance']['total_return']:.2%}")
    print(f"Expected Max Drawdown: {best_params['expected_performance']['max_drawdown']:.2%}")
    
    return results, best_params


def run_walk_forward_optimization():
    """Demonstrate walk-forward optimization"""
    print("\n" * 2)
    print("=" * 80)
    print("WALK-FORWARD OPTIMIZATION EXAMPLE")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer(
        db_path="btc_walk_forward_results.db",
        min_sample_trades=20,  # Lower threshold for shorter windows
        significance_level=0.05
    )
    
    # Use smaller parameter space for faster execution
    param_space = ParameterSpace(
        tenkan_periods=[8, 9, 10],
        kijun_periods=[24, 26],
        senkou_b_periods=[52, 56],
        signal_combinations=[
            ['cloud_breakout', 'tk_cross']
        ],
        stop_loss_percent=[0.025],
        take_profit_percent=[0.04]
    )
    
    # Run walk-forward optimization
    start_date = "2022-01-01"
    end_date = "2024-01-01"
    symbol = "BTC-USD"
    
    print(f"\nRunning walk-forward optimization for {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print("Window: 252 days training, 63 days testing, 21 days step")
    print("This will take some time...\n")
    
    wf_results = optimizer.walk_forward_optimization(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        window_size_days=252,  # 1 year training
        test_size_days=63,     # 3 months testing
        step_days=21,          # 1 month step
        parameter_space=param_space
    )
    
    # Analyze results
    analysis = wf_results['analysis']
    
    print("\n" + "=" * 80)
    print("WALK-FORWARD ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Number of windows tested: {len(wf_results['windows'])}")
    print(f"Mean out-of-sample Sharpe: {analysis['mean_out_sample_sharpe']:.3f}")
    print(f"Std out-of-sample Sharpe: {analysis['std_out_sample_sharpe']:.3f}")
    print(f"Mean out-of-sample return: {analysis['mean_out_sample_return']:.2%}")
    print(f"Mean stability score: {analysis['mean_stability_score']:.3f}")
    print(f"Consistency ratio: {analysis['consistency_ratio']:.2%}")
    
    print("\nMost frequently selected parameters:")
    freq_params = analysis['most_frequent_parameters']
    print(f"Frequency: {freq_params['frequency']:.2%}")
    print(f"Parameters: {freq_params['parameters']}")
    
    # Plot walk-forward results
    plot_walk_forward_results(wf_results)
    
    return wf_results


def run_statistical_validation(best_params: dict):
    """Demonstrate statistical validation of parameters"""
    print("\n" * 2)
    print("=" * 80)
    print("STATISTICAL VALIDATION EXAMPLE")
    print("=" * 80)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer()
    
    # Validate on out-of-sample data
    test_start = "2024-01-01"
    test_end = "2024-06-01"
    symbol = "BTC-USD"
    
    print(f"\nValidating parameters on out-of-sample period:")
    print(f"Test period: {test_start} to {test_end}")
    
    validation_results = optimizer.validate_optimization_results(
        symbol=symbol,
        parameters=best_params['parameters'],
        test_start_date=test_start,
        test_end_date=test_end,
        n_monte_carlo=1000
    )
    
    print(f"\nValidation Result: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
    
    # Display statistical tests
    if 'statistical_tests' in validation_results:
        tests = validation_results['statistical_tests']
        
        if 'returns_t_test' in tests:
            t_test = tests['returns_t_test']
            print(f"\nT-test for positive returns:")
            print(f"  T-statistic: {t_test['t_statistic']:.3f}")
            print(f"  P-value: {t_test['p_value']:.4f}")
            print(f"  Significant: {t_test['is_significant']}")
        
        if 'monte_carlo_test' in tests:
            mc_test = tests['monte_carlo_test']
            print(f"\nMonte Carlo permutation test:")
            print(f"  Actual Sharpe: {mc_test['actual_sharpe']:.3f}")
            print(f"  Random mean Sharpe: {mc_test['mc_mean_sharpe']:.3f}")
            print(f"  Percentile: {mc_test['percentile']:.2%}")
            print(f"  Significant: {mc_test['is_significant']}")
        
        if 'stability_check' in tests:
            stability = tests['stability_check']
            print(f"\nParameter stability check:")
            print(f"  Is stable: {stability['is_stable']}")
            print(f"  Sharpe CV: {stability['sharpe_cv']:.3f}")
            print(f"  Return CV: {stability['return_cv']:.3f}")
            print(f"  All windows profitable: {stability['all_profitable']}")
            print(f"  Number of windows: {stability['n_windows']}")
    
    # Run final backtest and visualize
    if validation_results['result'].sharpe_ratio != -999:
        run_final_backtest_visualization(symbol, best_params['parameters'], test_start, test_end)
    
    return validation_results


def plot_walk_forward_results(wf_results: dict):
    """Create visualization of walk-forward optimization results"""
    # Extract data for plotting
    window_results = wf_results['window_results']
    
    in_sample_sharpes = [w['train_result'].sharpe_ratio for w in window_results]
    out_sample_sharpes = [w['test_result'].sharpe_ratio for w in window_results]
    stability_scores = [w['stability_score'] for w in window_results]
    window_ids = list(range(len(window_results)))
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: In-sample vs Out-of-sample Sharpe ratios
    ax1 = axes[0]
    ax1.plot(window_ids, in_sample_sharpes, 'b-', label='In-sample', linewidth=2)
    ax1.plot(window_ids, out_sample_sharpes, 'r--', label='Out-of-sample', linewidth=2)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_xlabel('Window ID')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Walk-Forward Sharpe Ratios: In-sample vs Out-of-sample')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stability scores
    ax2 = axes[1]
    ax2.bar(window_ids, stability_scores, color='green', alpha=0.7)
    ax2.axhline(y=0.7, color='red', linestyle='--', label='Min stability threshold')
    ax2.set_xlabel('Window ID')
    ax2.set_ylabel('Stability Score')
    ax2.set_title('Parameter Stability Scores by Window')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Out-of-sample returns
    out_sample_returns = [w['test_result'].total_return for w in window_results]
    ax3 = axes[2]
    colors = ['green' if r > 0 else 'red' for r in out_sample_returns]
    ax3.bar(window_ids, out_sample_returns, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Window ID')
    ax3.set_ylabel('Total Return')
    ax3.set_title('Out-of-Sample Returns by Window')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("frontend/backtest_results/walk_forward")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "walk_forward_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def run_final_backtest_visualization(symbol: str, params: dict, start_date: str, end_date: str):
    """Run final backtest with best parameters and create visualizations"""
    print("\n" + "=" * 80)
    print("RUNNING FINAL BACKTEST WITH VISUALIZATION")
    print("=" * 80)
    
    # Fetch data
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_data(symbol, start_date, end_date)
    
    # Run backtest
    backtester = IchimokuBacktester(initial_capital=10000)
    trades_df, equity_curve = backtester.backtest(
        data,
        ichimoku_params=params['ichimoku_params'],
        signal_config=params['signal_config'],
        risk_params=params['risk_params']
    )
    
    # Calculate metrics
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.calculate_all_metrics(trades_df, equity_curve)
    
    print(f"\nFinal Performance Metrics:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Win Rate: {metrics.win_rate:.2%}")
    print(f"  Total Trades: {metrics.total_trades}")
    
    # Create visualizations
    visualizer = ResultsVisualizer()
    
    # Prepare results for visualization
    results = {
        'data': data,
        'trades': trades_df,
        'ichimoku_data': backtester.ichimoku_data,
        'equity_curve': equity_curve,
        'returns': equity_curve['returns'],
        'metrics': {
            'equity_curve': equity_curve,
            'trades': trades_df,
            'performance_metrics': metrics.__dict__,
            'returns': equity_curve['returns']
        },
        'strategy_config': {
            'name': 'Optimized Ichimoku Strategy',
            'parameters': params
        }
    }
    
    # Generate HTML report
    html_path = visualizer.generate_html_report(results, output_dir="frontend/backtest_results/optimized")
    print(f"\nHTML report generated: {html_path}")


def main():
    """Run complete parameter optimization workflow"""
    print("\n" + "=" * 80)
    print("AUTOMATED PARAMETER TESTING FRAMEWORK DEMO")
    print("=" * 80)
    
    try:
        # Step 1: Grid Search Optimization
        grid_results, best_params = run_grid_search_example()
        
        # Step 2: Walk-Forward Optimization
        wf_results = run_walk_forward_optimization()
        
        # Step 3: Statistical Validation
        validation_results = run_statistical_validation(best_params)
        
        # Step 4: Generate optimization report
        print("\n" + "=" * 80)
        print("GENERATING OPTIMIZATION REPORT")
        print("=" * 80)
        
        optimizer = ParameterOptimizer(db_path="btc_optimization_results.db")
        report_df = optimizer.generate_optimization_report("BTC-USD")
        
        # Save report
        report_path = Path("frontend/backtest_results") / "optimization_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\nOptimization report saved to: {report_path}")
        
        print("\n" + "=" * 80)
        print("PARAMETER OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print("\nSummary:")
        print(f"- Tested {len(grid_results)} parameter combinations")
        print(f"- Best Sharpe ratio: {best_params['expected_performance']['sharpe_ratio']:.3f}")
        print(f"- Walk-forward consistency: {wf_results['analysis']['consistency_ratio']:.2%}")
        print(f"- Statistical validation: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()