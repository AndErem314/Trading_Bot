"""
Run Combined Strategies Backtest

This script runs backtesting for multiple strategies combined together
on the same trading pair (e.g., BTC/USDT).
"""

import sys
import os
import argparse
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtesting.core.engine import BacktestEngine
from backtesting.core.metrics import PerformanceMetrics
from backtesting.core.strategy_combiner import StrategyCombiner, CombinationMethod
from backtesting.optimization.optimizer import ParameterOptimizer
from backtesting.analysis.llm_analyzer import LLMAnalyzer
from backtesting.utils.data_loader import DataLoader
from backtesting.utils.strategy_loader import StrategyLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CombinedStrategyBacktest:
    """
    Main class for running combined strategy backtests
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the combined strategy backtester
        
        Args:
            config_path: Path to backtest configuration file
        """
        # Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load optimization ranges
        opt_config_path = os.path.join(
            os.path.dirname(config_path),
            'optimization_ranges.yaml'
        )
        with open(opt_config_path, 'r') as f:
            self.optimization_config = yaml.safe_load(f)
            
        # Initialize components
        self.data_loader = DataLoader(self.config['backtest']['data'])
        self.strategy_loader = StrategyLoader()
        self.engine = BacktestEngine(
            initial_capital=self.config['backtest']['initial_capital'],
            commission=self.config['backtest']['commission'],
            slippage=self.config['backtest']['slippage']
        )
        
        # Setup output directory
        self.output_dir = Path(self.config['output']['output_directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_optimized_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """
        Load optimized parameters for a strategy if available
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Optimized parameters or defaults
        """
        # Check if optimized parameters exist
        params_file = self.output_dir / strategy_name / f"{strategy_name}_best_params.json"
        
        if params_file.exists():
            logger.info(f"Loading optimized parameters for {strategy_name}")
            with open(params_file, 'r') as f:
                data = json.load(f)
                return data['best_parameters']
        else:
            # Load default parameters
            logger.info(f"No optimized parameters found for {strategy_name}, using defaults")
            strategy_config_path = 'backend/config/strategy_config.json'
            with open(strategy_config_path, 'r') as f:
                strategy_config = json.load(f)
            return strategy_config['strategies'].get(strategy_name, {}).get('parameters', {})
    
    def generate_strategy_signals(
        self,
        strategies: List[str],
        market_data: pd.DataFrame,
        use_optimized: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate signals for multiple strategies
        
        Args:
            strategies: List of strategy names
            market_data: Market OHLCV data
            use_optimized: Whether to use optimized parameters
            
        Returns:
            Dictionary mapping strategy names to signal DataFrames
        """
        strategy_signals = {}
        
        for strategy_name in strategies:
            try:
                # Load parameters
                if use_optimized:
                    parameters = self.load_optimized_parameters(strategy_name)
                else:
                    # Use defaults
                    strategy_config_path = 'backend/config/strategy_config.json'
                    with open(strategy_config_path, 'r') as f:
                        strategy_config = json.load(f)
                    parameters = strategy_config['strategies'].get(strategy_name, {}).get('parameters', {})
                
                # Load strategy
                strategy = self.strategy_loader.load_strategy(strategy_name, parameters)
                
                # Generate signals
                signals = strategy.generate_signals(market_data)
                strategy_signals[strategy_name] = signals
                
                logger.info(f"Generated signals for {strategy_name}")
                
            except Exception as e:
                logger.error(f"Error generating signals for {strategy_name}: {e}")
                
        return strategy_signals
    
    def run_combined_backtest(
        self,
        strategies: List[str],
        combination_method: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None,
        use_optimized: bool = True
    ) -> Dict[str, Any]:
        """
        Run backtest with combined strategies
        
        Args:
            strategies: List of strategy names to combine
            combination_method: Method for combining signals
            weights: Optional strategy weights
            use_optimized: Whether to use optimized parameters
            
        Returns:
            Backtest results
        """
        logger.info(f"Running combined backtest for {strategies} using {combination_method}")
        
        # Load market data
        symbol = self.config['backtest']['data']['symbol']
        timeframe = self.config['backtest']['data']['timeframe']
        start_date = self.config['backtest']['data']['start_date']
        end_date = self.config['backtest']['data']['end_date']
        
        market_data = self.data_loader.load_data(symbol, timeframe, start_date, end_date)
        
        if market_data.empty:
            logger.error(f"No data available for {symbol} {timeframe}")
            return {}
        
        # Generate signals for each strategy
        strategy_signals = self.generate_strategy_signals(strategies, market_data, use_optimized)
        
        if not strategy_signals:
            logger.error("No strategy signals generated")
            return {}
        
        # Initialize combiner
        combiner = StrategyCombiner(
            combination_method=CombinationMethod[combination_method.upper()],
            min_strategies_agree=max(2, len(strategies) // 2),  # At least half must agree
            signal_threshold=0.5
        )
        
        # Add strategies with weights
        if weights:
            for strategy_name in strategies:
                combiner.add_strategy(strategy_name, weights.get(strategy_name, 1.0))
        else:
            # Equal weights
            for strategy_name in strategies:
                combiner.add_strategy(strategy_name, 1.0)
        
        # Combine signals
        combined_signals = combiner.combine_signals(strategy_signals, market_data)
        
        # Run backtest with combined signals
        results = self.engine.run_backtest(combined_signals, market_data, symbol)
        
        # Calculate metrics
        metrics = PerformanceMetrics.generate_complete_metrics(
            results['equity_curve'],
            results['trades']
        )
        
        # Add strategy agreement analysis
        agreement_stats = combiner.analyze_strategy_agreement(strategy_signals)
        
        # Combine all results
        full_results = {
            **results,
            **metrics,
            'strategies': strategies,
            'combination_method': combination_method,
            'weights': weights or {s: 1.0 for s in strategies},
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'strategy_agreement': agreement_stats
        }
        
        # Save results
        self._save_combined_results(strategies, combination_method, full_results)
        
        return full_results
    
    def compare_combination_methods(
        self,
        strategies: List[str],
        use_optimized: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare different combination methods
        
        Args:
            strategies: List of strategies to combine
            use_optimized: Whether to use optimized parameters
            
        Returns:
            Results for each combination method
        """
        logger.info(f"Comparing combination methods for {strategies}")
        
        methods = ["majority_vote", "weighted_average", "unanimous", "any_signal", "score_based"]
        comparison_results = {}
        
        for method in methods:
            try:
                results = self.run_combined_backtest(
                    strategies=strategies,
                    combination_method=method,
                    use_optimized=use_optimized
                )
                comparison_results[method] = results
                logger.info(f"Completed {method}: Return={results.get('total_return_pct', 0):.2f}%")
            except Exception as e:
                logger.error(f"Error with {method}: {e}")
        
        # Create comparison summary
        summary = self._create_comparison_summary(comparison_results)
        
        # Save comparison
        comparison_name = "_".join(strategies)
        comparison_file = self.output_dir / f"combination_comparison_{comparison_name}.json"
        with open(comparison_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return comparison_results
    
    def optimize_combination_weights(
        self,
        strategies: List[str],
        method: str = "weighted_average"
    ) -> Dict[str, float]:
        """
        Optimize weights for strategy combination
        
        Args:
            strategies: List of strategies
            method: Combination method
            
        Returns:
            Optimal weights
        """
        logger.info(f"Optimizing combination weights for {strategies}")
        
        # Define weight optimization function
        def evaluate_weights(weight_values):
            # Normalize weights
            total = sum(weight_values)
            weights = {strategies[i]: w/total for i, w in enumerate(weight_values)}
            
            # Run backtest with these weights
            results = self.run_combined_backtest(
                strategies=strategies,
                combination_method=method,
                weights=weights,
                use_optimized=True
            )
            
            # Return metric to optimize
            return results.get('sharpe_ratio', 0)
        
        # Use simple grid search for weights
        best_score = -float('inf')
        best_weights = None
        
        # Try different weight combinations
        weight_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        for weights in self._generate_weight_combinations(len(strategies), weight_options):
            score = evaluate_weights(weights)
            if score > best_score:
                best_score = score
                best_weights = weights
        
        # Convert to dictionary
        optimal_weights = {strategies[i]: w for i, w in enumerate(best_weights)}
        
        logger.info(f"Optimal weights: {optimal_weights}")
        return optimal_weights
    
    def _generate_weight_combinations(self, n_strategies: int, options: List[float]):
        """Generate weight combinations that sum to approximately 1"""
        import itertools
        
        # Generate all combinations
        for combo in itertools.product(options, repeat=n_strategies):
            # Only use combinations that sum between 0.9 and 1.1
            if 0.9 <= sum(combo) <= 1.1:
                yield combo
    
    def _save_combined_results(
        self,
        strategies: List[str],
        method: str,
        results: Dict[str, Any]
    ):
        """Save combined strategy results"""
        # Create directory for combination
        combo_name = f"combined_{'_'.join(strategies)}_{method}"
        combo_dir = self.output_dir / combo_name
        combo_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_file = combo_dir / "metrics.json"
        metrics_to_save = {k: v for k, v in results.items() 
                          if k not in ['trades', 'equity_curve', 'monthly_returns']}
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Save trades
        if 'trades' in results and not results['trades'].empty:
            trades_file = combo_dir / "trades.csv"
            results['trades'].to_csv(trades_file, index=False)
        
        # Save equity curve
        if 'equity_curve' in results and not results['equity_curve'].empty:
            equity_file = combo_dir / "equity_curve.csv"
            results['equity_curve'].to_csv(equity_file, index=False)
        
        # Save performance summary
        summary_file = combo_dir / "summary.txt"
        summary = PerformanceMetrics.create_performance_summary(results)
        
        # Add combination details
        summary += f"\n\nCOMBINATION DETAILS\n"
        summary += f"Strategies: {', '.join(strategies)}\n"
        summary += f"Method: {method}\n"
        summary += f"Weights: {json.dumps(results.get('weights', {}), indent=2)}\n"
        
        if 'strategy_agreement' in results and results['strategy_agreement']:
            summary += f"\nSTRATEGY AGREEMENT\n"
            summary += f"Average Agreement: {results['strategy_agreement'].get('avg_agreement', 0):.1f}%\n"
            
        with open(summary_file, 'w') as f:
            f.write(summary)
    
    def _create_comparison_summary(self, comparison_results: Dict[str, Dict]) -> Dict:
        """Create summary of combination method comparison"""
        summary = {
            'methods': {},
            'best_method': None,
            'best_sharpe': -float('inf')
        }
        
        for method, results in comparison_results.items():
            if results:
                method_summary = {
                    'total_return': results.get('total_return_pct', 0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'max_drawdown': results.get('max_drawdown', 0),
                    'win_rate': results.get('win_rate', 0),
                    'total_trades': results.get('total_trades', 0)
                }
                summary['methods'][method] = method_summary
                
                # Track best method by Sharpe ratio
                if method_summary['sharpe_ratio'] > summary['best_sharpe']:
                    summary['best_sharpe'] = method_summary['sharpe_ratio']
                    summary['best_method'] = method
        
        return summary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run combined strategies backtest')
    parser.add_argument('strategies', nargs='+', help='List of strategies to combine')
    parser.add_argument('--method', default='weighted_average',
                       choices=['majority_vote', 'weighted_average', 'unanimous', 'any_signal', 'score_based'],
                       help='Combination method')
    parser.add_argument('--compare', action='store_true', help='Compare all combination methods')
    parser.add_argument('--optimize-weights', action='store_true', help='Optimize strategy weights')
    parser.add_argument('--use-defaults', action='store_true', 
                       help='Use default parameters instead of optimized ones')
    parser.add_argument('--weights', type=str, default=None,
                       help='JSON string of strategy weights')
    parser.add_argument('--config', default='backend/backtesting/config/backtest_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--symbol', choices=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                       default='BTC/USDT', help='Trading pair to test')
    
    args = parser.parse_args()
    
    # Update config with selected symbol
    backtester = CombinedStrategyBacktest(args.config)
    backtester.config['backtest']['data']['symbol'] = args.symbol
    
    logger.info(f"Testing combined strategies on {args.symbol}")
    
    if args.compare:
        # Compare all combination methods
        results = backtester.compare_combination_methods(
            strategies=args.strategies,
            use_optimized=not args.use_defaults
        )
        
        # Print comparison summary
        print("\n" + "="*60)
        print(f"COMBINATION METHOD COMPARISON for {args.symbol}")
        print("="*60)
        
        for method, result in results.items():
            if result:
                print(f"\n{method.upper()}:")
                print(f"  Total Return: {result.get('total_return_pct', 0):.2f}%")
                print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown: {result.get('max_drawdown', 0):.2f}%")
                print(f"  Win Rate: {result.get('win_rate', 0):.1f}%")
    
    elif args.optimize_weights:
        # Optimize weights for combination
        optimal_weights = backtester.optimize_combination_weights(
            strategies=args.strategies,
            method=args.method
        )
        
        print("\n" + "="*60)
        print(f"OPTIMAL WEIGHTS for {args.symbol}")
        print("="*60)
        for strategy, weight in optimal_weights.items():
            print(f"{strategy}: {weight:.2%}")
    
    else:
        # Run single combined backtest
        weights = None
        if args.weights:
            weights = json.loads(args.weights)
        
        results = backtester.run_combined_backtest(
            strategies=args.strategies,
            combination_method=args.method,
            weights=weights,
            use_optimized=not args.use_defaults
        )
        
        # Print results
        if results:
            print("\n" + "="*60)
            print(f"COMBINED STRATEGY BACKTEST COMPLETE: {args.symbol}")
            print(f"Strategies: {', '.join(args.strategies)}")
            print(f"Combination Method: {args.method}")
            print("="*60)
            print(f"Total Return: {results.get('total_return_pct', 0):.2f}%")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
            print(f"Win Rate: {results.get('win_rate', 0):.2f}%")
            print(f"Total Trades: {results.get('total_trades', 0)}")
            
            if 'strategy_agreement' in results and results['strategy_agreement']:
                print(f"\nStrategy Agreement: {results['strategy_agreement'].get('avg_agreement', 0):.1f}%")
            print("="*60)


if __name__ == '__main__':
    main()
