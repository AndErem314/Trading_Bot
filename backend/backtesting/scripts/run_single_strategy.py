"""
Run Single Strategy Backtest with Parameter Optimization

This script runs backtesting for an individual strategy, optionally
optimizing its parameters and generating comprehensive reports.
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
from typing import Dict, List, Optional, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtesting.core.engine import BacktestEngine
from backtesting.core.metrics import PerformanceMetrics
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


class SingleStrategyBacktest:
    """
    Main class for running single strategy backtests
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the backtester with configuration
        
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
        
    def run_backtest(
        self,
        strategy_name: str,
        parameters: Dict[str, Any],
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run backtest for a single strategy with given parameters
        
        Args:
            strategy_name: Name of the strategy
            parameters: Strategy parameters
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info(f"Running backtest for {strategy_name}")
        
        # Load market data
        symbol = self.config['backtest']['data']['symbol']
        timeframe = self.config['backtest']['data']['timeframe']
        start_date = self.config['backtest']['data']['start_date']
        end_date = self.config['backtest']['data']['end_date']
        
        market_data = self.data_loader.load_data(symbol, timeframe, start_date, end_date)
        
        if market_data.empty:
            logger.error(f"No data available for {symbol} {timeframe}")
            return {}
        
        # Load and initialize strategy
        strategy = self.strategy_loader.load_strategy(strategy_name, parameters)
        
        # Generate strategy signals
        signals = strategy.generate_signals(market_data)
        
        # Run backtest
        results = self.engine.run_backtest(signals, market_data, symbol)
        
        # Calculate additional metrics
        metrics = PerformanceMetrics.generate_complete_metrics(
            results['equity_curve'],
            results['trades']
        )
        
        # Combine results
        full_results = {
            **results,
            **metrics,
            'strategy_name': strategy_name,
            'parameters': parameters,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date
        }
        
        # Save results if requested
        if save_results:
            self._save_results(strategy_name, full_results)
            
        return full_results
    
    def optimize_strategy(
        self,
        strategy_name: str,
        optimization_method: str = 'grid_search'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters
        
        Args:
            strategy_name: Name of the strategy to optimize
            optimization_method: Method to use (grid_search, random_search, bayesian)
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting {optimization_method} optimization for {strategy_name}")
        
        # Get parameter ranges for this strategy
        param_ranges = self.optimization_config['strategies'].get(strategy_name, {}).get('parameters', {})
        
        if not param_ranges:
            logger.error(f"No optimization ranges defined for {strategy_name}")
            return {}
        
        # Create optimizer
        optimizer = ParameterOptimizer(
            backtest_function=lambda name, params: self.run_backtest(name, params, save_results=False),
            optimization_config=self.optimization_config['optimization']
        )
        
        # Run optimization
        result = optimizer.optimize_strategy(
            strategy_name=strategy_name,
            parameter_ranges=param_ranges,
            objective_metric=self.config['performance_criteria']['primary_metric'],
            method=optimization_method,
            constraints=self.config['performance_criteria']['constraints']
        )
        
        # Save optimization results
        optimizer.save_optimization_results(
            result,
            str(self.output_dir / strategy_name)
        )
        
        # Run final backtest with best parameters
        logger.info("Running final backtest with optimized parameters")
        final_results = self.run_backtest(
            strategy_name,
            result.best_parameters,
            save_results=True
        )
        
        return {
            'optimization_result': result,
            'final_backtest': final_results
        }
    
    def analyze_with_llm(
        self,
        strategy_name: str,
        results: Dict[str, Any],
        provider: str = "auto"
    ) -> str:
        """
        Analyze results using LLM
        
        Args:
            strategy_name: Name of the strategy
            results: Backtest results
            provider: LLM provider to use
            
        Returns:
            Analysis report
        """
        try:
            analyzer = LLMAnalyzer(provider=provider)
            
            # Prepare data for analysis
            performance_metrics = {
                k: v for k, v in results.items()
                if isinstance(v, (int, float)) and not k.startswith('_')
            }
            
            # Get current parameters
            current_params = results.get('parameters', {})
            
            # Get optimization ranges
            opt_ranges = self.optimization_config['strategies'].get(strategy_name, {}).get('parameters', {})
            
            # Get trade history and market data
            trade_history = results.get('trades', pd.DataFrame())
            market_data = self.data_loader.load_data(
                results['symbol'],
                results['timeframe'],
                results['start_date'],
                results['end_date']
            )
            
            # Generate analysis
            analysis = analyzer.analyze_strategy_performance(
                strategy_name=strategy_name,
                performance_metrics=performance_metrics,
                current_parameters=current_params,
                optimization_ranges=opt_ranges,
                trade_history=trade_history,
                market_data=market_data
            )
            
            # Generate report
            report = analyzer.generate_optimization_report([analysis])
            
            # Save report
            report_path = self.output_dir / strategy_name / f"{strategy_name}_llm_analysis.md"
            report_path.parent.mkdir(exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report)
                
            logger.info(f"LLM analysis saved to {report_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return "LLM analysis unavailable"
    
    def _save_results(self, strategy_name: str, results: Dict[str, Any]):
        """Save backtest results to disk"""
        # Create strategy directory
        strategy_dir = self.output_dir / strategy_name
        strategy_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_file = strategy_dir / f"{strategy_name}_metrics.json"
        metrics_to_save = {k: v for k, v in results.items() 
                          if k not in ['trades', 'equity_curve', 'monthly_returns']}
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Save trades
        if 'trades' in results and not results['trades'].empty:
            trades_file = strategy_dir / f"{strategy_name}_trades.csv"
            results['trades'].to_csv(trades_file, index=False)
        
        # Save equity curve
        if 'equity_curve' in results and not results['equity_curve'].empty:
            equity_file = strategy_dir / f"{strategy_name}_equity_curve.csv"
            results['equity_curve'].to_csv(equity_file, index=False)
        
        # Save performance summary
        summary_file = strategy_dir / f"{strategy_name}_summary.txt"
        summary = PerformanceMetrics.create_performance_summary(results)
        with open(summary_file, 'w') as f:
            f.write(summary)
            
        logger.info(f"Results saved to {strategy_dir}")
    
    def generate_report(self, strategy_name: str, results: Dict[str, Any]):
        """Generate comprehensive HTML report"""
        # This would generate charts and HTML report
        # For now, just log that it would be generated
        logger.info(f"Would generate HTML report for {strategy_name}")
        # TODO: Implement visualization module


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run single strategy backtest')
    parser.add_argument('strategy', help='Strategy name (e.g., bollinger_bands)')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--method', default='grid_search', 
                       choices=['grid_search', 'random_search', 'bayesian'],
                       help='Optimization method')
    parser.add_argument('--analyze', action='store_true', help='Run LLM analysis')
    parser.add_argument('--llm', default='auto', 
                       choices=['auto', 'gemini', 'openai'],
                       help='LLM provider for analysis')
    parser.add_argument('--config', default='backend/backtesting/config/backtest_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--params', type=str, default=None,
                       help='JSON string of parameters (if not optimizing)')
    
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = SingleStrategyBacktest(args.config)
    
    if args.optimize:
        # Run optimization
        results = backtester.optimize_strategy(args.strategy, args.method)
        final_results = results['final_backtest']
    else:
        # Use provided parameters or defaults
        if args.params:
            parameters = json.loads(args.params)
        else:
            # Load default parameters from strategy config
            strategy_config_path = 'backend/config/strategy_config.json'
            with open(strategy_config_path, 'r') as f:
                strategy_config = json.load(f)
            parameters = strategy_config['strategies'].get(args.strategy, {}).get('parameters', {})
        
        # Run single backtest
        final_results = backtester.run_backtest(args.strategy, parameters)
    
    # Run LLM analysis if requested
    if args.analyze and final_results:
        backtester.analyze_with_llm(args.strategy, final_results, args.llm)
    
    # Print summary
    if final_results:
        print("\n" + "="*60)
        print(f"BACKTEST COMPLETE: {args.strategy}")
        print("="*60)
        print(f"Total Return: {final_results.get('total_return_pct', 0):.2f}%")
        print(f"Sharpe Ratio: {final_results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {final_results.get('max_drawdown', 0):.2f}%")
        print(f"Win Rate: {final_results.get('win_rate', 0):.2f}%")
        print(f"Total Trades: {final_results.get('total_trades', 0)}")
        print("="*60)


if __name__ == '__main__':
    main()
