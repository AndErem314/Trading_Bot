#!/usr/bin/env python3
"""
Run Strategy with Optimized Parameters

This script runs a trading strategy using pre-optimized parameters
loaded from the optimized_strategies.yaml configuration file.
"""

import sys
import os
import argparse
import yaml
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtesting.scripts.run_single_strategy import SingleStrategyBacktest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedStrategyRunner:
    """Run strategies with optimized parameters"""
    
    def __init__(self):
        """Initialize the optimized strategy runner"""
        # Load configurations
        config_dir = Path(__file__).parent.parent / 'config'
        
        # Load main config
        with open(config_dir / 'backtest_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load optimized parameters
        with open(config_dir / 'optimized_strategies.yaml', 'r') as f:
            self.optimized_config = yaml.safe_load(f)
            
        # Initialize backtester
        self.backtester = SingleStrategyBacktest(str(config_dir / 'backtest_config.yaml'))
    
    def get_optimized_parameters(self, strategy_name: str, preset: str = None) -> dict:
        """
        Get optimized parameters for a strategy
        
        Args:
            strategy_name: Name of the strategy
            preset: Parameter preset to use ('best_return', 'conservative', 'balanced', or None for default)
            
        Returns:
            Dictionary of optimized parameters
        """
        if strategy_name not in self.optimized_config['strategies']:
            raise ValueError(f"No optimized parameters found for strategy: {strategy_name}")
        
        strategy_config = self.optimized_config['strategies'][strategy_name]
        optimized_params = strategy_config['optimized_parameters']
        
        # Determine which preset to use
        if preset is None:
            preset = strategy_config.get('default', 'default')
        
        if preset not in optimized_params:
            raise ValueError(f"Preset '{preset}' not found for strategy: {strategy_name}")
        
        # Get parameters and remove metadata
        params = optimized_params[preset].copy()
        params.pop('expected_return', None)  # Remove non-parameter fields
        
        return params
    
    def list_strategies(self):
        """List all strategies with optimized parameters"""
        print("\nAvailable Strategies with Optimized Parameters:")
        print("=" * 60)
        
        for strategy_name, config in self.optimized_config['strategies'].items():
            print(f"\n{strategy_name}:")
            print(f"  Description: {config['description']}")
            print(f"  Available presets:")
            
            for preset_name, preset_params in config['optimized_parameters'].items():
                expected_return = preset_params.get('expected_return', 'Not tested')
                print(f"    - {preset_name}: Expected return {expected_return}")
                
            default_preset = config.get('default', 'default')
            print(f"  Default preset: {default_preset}")
    
    def run_strategy(self, strategy_name: str, preset: str = None, 
                     symbol: str = None, analyze: bool = True):
        """
        Run a strategy with optimized parameters
        
        Args:
            strategy_name: Name of the strategy
            preset: Parameter preset to use
            symbol: Trading symbol (defaults to config)
            analyze: Whether to run LLM analysis
        """
        # Get optimized parameters
        try:
            parameters = self.get_optimized_parameters(strategy_name, preset)
        except ValueError as e:
            logger.error(str(e))
            return
        
        # Add symbol if specified
        if symbol:
            parameters['symbol'] = symbol
        else:
            symbol = self.config['backtest']['data']['symbol']
        
        # Log the parameters being used
        preset_name = preset or self.optimized_config['strategies'][strategy_name].get('default', 'default')
        logger.info(f"Running {strategy_name} with '{preset_name}' preset")
        logger.info(f"Parameters: {parameters}")
        
        # Run backtest
        results = self.backtester.run_backtest(strategy_name, parameters)
        
        # Display results
        if results:
            self._display_results(strategy_name, preset_name, results)
            
            # Run analysis if requested
            if analyze and results.get('total_trades', 0) > 0:
                logger.info("Running LLM analysis...")
                analysis = self.backtester.analyze_with_llm(strategy_name, results)
                print("\nLLM Analysis:")
                print("=" * 60)
                print(analysis)
    
    def _display_results(self, strategy_name: str, preset: str, results: dict):
        """Display backtest results"""
        print(f"\n{'=' * 60}")
        print(f"BACKTEST RESULTS: {strategy_name} ({preset} preset)")
        print(f"{'=' * 60}")
        print(f"Symbol: {results.get('symbol', 'N/A')}")
        print(f"Period: {results.get('start_date', 'N/A')} to {results.get('end_date', 'Present')}")
        print(f"Total Return: {results.get('total_return', 0):.2f}%")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        print(f"Win Rate: {results.get('win_rate', 0):.2%}")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        
        if results.get('total_trades', 0) > 0:
            print(f"Average Win: ${results.get('avg_win', 0):.2f}")
            print(f"Average Loss: ${results.get('avg_loss', 0):.2f}")
            print(f"Best Trade: ${results.get('best_trade', 0):.2f}")
            print(f"Worst Trade: ${results.get('worst_trade', 0):.2f}")
        
        print(f"Final Equity: ${results.get('final_equity', 0):.2f}")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description='Run strategy with optimized parameters')
    parser.add_argument('strategy', nargs='?', help='Strategy name to run')
    parser.add_argument('--preset', '-p', help='Parameter preset (best_return, conservative, balanced)')
    parser.add_argument('--symbol', '-s', help='Trading symbol (e.g., BTC/USDT)')
    parser.add_argument('--list', '-l', action='store_true', help='List available strategies')
    parser.add_argument('--no-analyze', action='store_true', help='Skip LLM analysis')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = OptimizedStrategyRunner()
    
    # List strategies if requested
    if args.list or not args.strategy:
        runner.list_strategies()
        if not args.strategy:
            print("\nUsage: python run_optimized_strategy.py <strategy_name> [options]")
        return
    
    # Run the strategy
    runner.run_strategy(
        strategy_name=args.strategy,
        preset=args.preset,
        symbol=args.symbol,
        analyze=not args.no_analyze
    )


if __name__ == "__main__":
    main()
