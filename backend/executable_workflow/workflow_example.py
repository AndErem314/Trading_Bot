"""
Workflow Example - Demonstration of Ichimoku Trading System Usage

This script demonstrates various ways to use the workflow controller
for backtesting, optimization, and analysis.

Examples included:
1. Quick backtest with default settings
2. Custom backtest with specific parameters
3. Parameter optimization
4. Multiple symbol analysis
5. Strategy comparison
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from main_workflow_controller import (
    IchimokuWorkflowController,
    WorkflowConfig,
    run_quick_backtest
)


def example_1_quick_backtest():
    """Example 1: Quick backtest with minimal configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Quick Backtest")
    print("="*60)
    
    # Method 1: Using convenience function
    results = run_quick_backtest(
        symbol='ETH/USDT',
        timeframe='4h',
        lookback_days=180
    )
    
    print(f"\nQuick backtest completed!")
    print(f"Total return: {results.performance_metrics['returns']['total_return']:.2%}")
    print(f"Sharpe ratio: {results.performance_metrics['risk']['sharpe_ratio']:.2f}")


def example_2_custom_backtest():
    """Example 2: Custom backtest with detailed configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Backtest")
    print("="*60)
    
    # Create custom configuration
    config = WorkflowConfig(
        # Data settings
        exchange='binance',
        symbol='BTC/USDT',
        timeframe='1h',
        start_date='2024-01-01',
        end_date='2024-06-30',
        
        # Trading settings
        initial_capital=50000.0,
        position_size_pct=0.8,  # Use 80% of capital
        commission=0.0005,      # 0.05% commission
        slippage=0.001,         # 0.1% slippage
        
        # Output settings
        output_dir='./results/custom_backtest',
        generate_report=True,
        report_formats=['html', 'json'],
        show_plots=True
    )
    
    # Create controller and run
    controller = IchimokuWorkflowController(config)
    results = controller.execute_full_workflow()
    
    # Display results
    print(f"\nCustom backtest completed!")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Total trades: {len(results.backtest_results['trades'])}")
    print(f"Final balance: ${results.backtest_results['final_balance']:,.2f}")
    print(f"Win rate: {results.performance_metrics['trades']['win_rate']:.2%}")


def example_3_parameter_optimization():
    """Example 3: Parameter optimization to find best settings."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Parameter Optimization")
    print("="*60)
    
    config = WorkflowConfig(
        symbol='BTC/USDT',
        timeframe='4h',
        lookback_days=365,
        
        # Enable optimization
        enable_optimization=True,
        optimization_metric='sharpe_ratio',
        n_jobs=-1,  # Use all CPU cores
        
        # Output
        output_dir='./results/optimization',
        generate_report=True,
        report_formats=['html', 'json']
    )
    
    controller = IchimokuWorkflowController(config)
    results = controller.execute_full_workflow()
    
    # Display optimization results
    if results.optimization_results:
        best_params = results.optimization_results['best_params']
        print(f"\nOptimization completed!")
        print(f"Best parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nPerformance with optimized parameters:")
        print(f"  Sharpe ratio: {results.performance_metrics['risk']['sharpe_ratio']:.2f}")
        print(f"  Total return: {results.performance_metrics['returns']['total_return']:.2%}")


def example_4_multiple_symbols():
    """Example 4: Analyze multiple symbols."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Multiple Symbol Analysis")
    print("="*60)
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT']
    results_summary = {}
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        config = WorkflowConfig(
            symbol=symbol,
            timeframe='1d',
            lookback_days=365,
            generate_report=False,  # Skip individual reports
            show_plots=False
        )
        
        try:
            controller = IchimokuWorkflowController(config)
            results = controller.execute_full_workflow()
            
            results_summary[symbol] = {
                'total_return': results.performance_metrics['returns']['total_return'],
                'sharpe_ratio': results.performance_metrics['risk']['sharpe_ratio'],
                'max_drawdown': results.performance_metrics['risk']['max_drawdown'],
                'win_rate': results.performance_metrics['trades']['win_rate'],
                'total_trades': len(results.backtest_results['trades'])
            }
            
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            results_summary[symbol] = None
    
    # Display comparative results
    print("\n" + "-"*60)
    print("COMPARATIVE RESULTS:")
    print("-"*60)
    print(f"{'Symbol':<12} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>8}")
    print("-"*60)
    
    for symbol, metrics in results_summary.items():
        if metrics:
            print(f"{symbol:<12} "
                  f"{metrics['total_return']:>10.2%} "
                  f"{metrics['sharpe_ratio']:>8.2f} "
                  f"{metrics['max_drawdown']:>8.2%} "
                  f"{metrics['win_rate']:>8.2%} "
                  f"{metrics['total_trades']:>8}")


def example_5_strategy_comparison():
    """Example 5: Compare different strategy configurations."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Strategy Configuration Comparison")
    print("="*60)
    
    # Base configuration
    base_config = {
        'symbol': 'BTC/USDT',
        'timeframe': '4h',
        'lookback_days': 180,
        'initial_capital': 10000,
        'generate_report': False,
        'show_plots': False
    }
    
    # Different strategy configurations to test
    strategy_configs = {
        'Conservative': {
            'stop_loss': 0.01,      # 1% stop loss
            'take_profit': 0.03,    # 3% take profit
            'position_size': 0.5    # 50% position size
        },
        'Standard': {
            'stop_loss': 0.02,      # 2% stop loss
            'take_profit': 0.06,    # 6% take profit
            'position_size': 0.95   # 95% position size
        },
        'Aggressive': {
            'stop_loss': 0.03,      # 3% stop loss
            'take_profit': 0.10,    # 10% take profit
            'position_size': 1.0    # 100% position size
        }
    }
    
    results_comparison = {}
    
    for strategy_name, params in strategy_configs.items():
        print(f"\nTesting {strategy_name} strategy...")
        
        # Create config with strategy-specific parameters
        config = WorkflowConfig(**base_config)
        config.position_size_pct = params['position_size']
        
        # Note: In a real scenario, you would pass these parameters
        # to the strategy configuration. For this example, we're
        # using the default strategy with different position sizes.
        
        controller = IchimokuWorkflowController(config)
        results = controller.execute_full_workflow()
        
        results_comparison[strategy_name] = {
            'total_return': results.performance_metrics['returns']['total_return'],
            'sharpe_ratio': results.performance_metrics['risk']['sharpe_ratio'],
            'max_drawdown': results.performance_metrics['risk']['max_drawdown'],
            'profit_factor': results.performance_metrics['trades']['profit_factor'],
            'trades': len(results.backtest_results['trades'])
        }
    
    # Display comparison
    print("\n" + "-"*70)
    print("STRATEGY COMPARISON:")
    print("-"*70)
    print(f"{'Strategy':<15} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'PF':>8} {'Trades':>8}")
    print("-"*70)
    
    for strategy, metrics in results_comparison.items():
        print(f"{strategy:<15} "
              f"{metrics['total_return']:>10.2%} "
              f"{metrics['sharpe_ratio']:>8.2f} "
              f"{metrics['max_drawdown']:>8.2%} "
              f"{metrics['profit_factor']:>8.2f} "
              f"{metrics['trades']:>8}")


def example_6_session_management():
    """Example 6: Demonstrate session save/load functionality."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Session Management")
    print("="*60)
    
    # Run a backtest
    config = WorkflowConfig(
        symbol='ETH/USDT',
        timeframe='1h',
        lookback_days=90
    )
    
    controller = IchimokuWorkflowController(config)
    results = controller.execute_full_workflow()
    
    # Save session
    session_name = f"example_session_{datetime.now().strftime('%Y%m%d')}"
    controller.save_session(session_name)
    print(f"\nSession saved as: {session_name}")
    
    # Create new controller and load session
    new_controller = IchimokuWorkflowController()
    session_path = f"./results/{session_name}_session.pkl"
    
    try:
        new_controller.load_session(session_path)
        print("Session loaded successfully!")
        
        # Access loaded results
        loaded_results = new_controller.results
        print(f"\nLoaded results:")
        print(f"  Symbol: {new_controller.config.symbol}")
        print(f"  Total return: {loaded_results.performance_metrics['returns']['total_return']:.2%}")
        
    except Exception as e:
        print(f"Error loading session: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("ICHIMOKU TRADING SYSTEM - WORKFLOW EXAMPLES")
    print("="*80)
    
    examples = [
        ("Quick Backtest", example_1_quick_backtest),
        ("Custom Backtest", example_2_custom_backtest),
        ("Parameter Optimization", example_3_parameter_optimization),
        ("Multiple Symbols", example_4_multiple_symbols),
        ("Strategy Comparison", example_5_strategy_comparison),
        ("Session Management", example_6_session_management)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print("0. Run all examples")
    print("q. Quit")
    
    while True:
        choice = input("\nSelect example to run (0-6 or q): ").strip().lower()
        
        if choice == 'q':
            print("\nExiting examples. Goodbye!")
            break
        elif choice == '0':
            # Run all examples
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"\nError in {name}: {e}")
                input("\nPress Enter to continue to next example...")
            break
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(examples):
                    examples[idx][1]()  # Run the selected example
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number or 'q'.")
            except Exception as e:
                print(f"\nError running example: {e}")


if __name__ == "__main__":
    # You can run individual examples directly or use the main menu
    main()
    
    # Or uncomment below to run specific examples:
    # example_1_quick_backtest()
    # example_2_custom_backtest()
    # example_3_parameter_optimization()
    # example_4_multiple_symbols()
    # example_5_strategy_comparison()
    # example_6_session_management()