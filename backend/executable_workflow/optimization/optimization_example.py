"""
LLM Optimization Example

This module demonstrates how to use the LLM-integrated optimizer
for Ichimoku trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from datetime import datetime
import json

from backend.executable_workflow.optimization import (
    LLMOptimizer,
    IchimokuParameters,
    SignalCombination,
    RiskParameters
)
from backend.executable_workflow.backtesting import IchimokuBacktester
from backend.executable_workflow.data_fetching import OHLCVDataFetcher, DataPreprocessor
from backend.executable_workflow.indicators import IchimokuCalculator
from backend.executable_workflow.analytics import PerformanceAnalyzer
from backend.executable_workflow.config.models import (
    StrategyConfig, 
    PositionSizing, 
    RiskManagement,
    IchimokuParameters as ConfigIchimokuParameters,
    SignalCombination as ConfigSignalCombination
)

logger = logging.getLogger(__name__)


class OptimizationExample:
    """
    Example class demonstrating LLM-integrated optimization
    for Ichimoku strategies.
    """
    
    def __init__(self):
        self.optimizer = LLMOptimizer(provider="auto")
        self.backtester = IchimokuBacktester()
        self.analyzer = PerformanceAnalyzer()
        self.fetcher = OHLCVDataFetcher()
        self.preprocessor = DataPreprocessor()
        self.ichimoku_calc = IchimokuCalculator()
        
    def run_complete_optimization_example(self):
        """Run a complete optimization example with all features."""
        logger.info("Starting LLM Optimization Example")
        
        # 1. Prepare data
        logger.info("\n" + "="*60)
        logger.info("1. PREPARING DATA")
        logger.info("="*60)
        
        data = self._prepare_sample_data()
        
        # 2. Define initial parameters
        logger.info("\n" + "="*60)
        logger.info("2. INITIAL PARAMETERS")
        logger.info("="*60)
        
        initial_params = {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_span_b_period': 52,
            'displacement': 26,
            'signal_weight_price': 0.4,
            'signal_weight_cross': 0.3,
            'signal_weight_chikou': 0.3,
            'min_signal_strength': 0.6,
            'stop_loss_multiplier': 2.0,
            'take_profit_multiplier': 3.0,
            'position_size_pct': 2.0
        }
        
        logger.info(f"Initial parameters: {json.dumps(initial_params, indent=2)}")
        
        # 3. Run initial backtest
        logger.info("\n" + "="*60)
        logger.info("3. INITIAL BACKTEST")
        logger.info("="*60)
        
        initial_results = self._run_backtest(data, initial_params)
        self._display_results("Initial Results", initial_results)
        
        # 4. Analyze current performance with LLM
        logger.info("\n" + "="*60)
        logger.info("4. LLM ANALYSIS")
        logger.info("="*60)
        
        analysis = self.optimizer.analyze_backtest_results({
            'metrics': initial_results,
            'parameters': initial_params,
            'strategy_name': 'Ichimoku Strategy'
        })
        
        logger.info("LLM Analysis Summary:")
        if 'summary' in analysis:
            logger.info(analysis['summary'])
        else:
            logger.info(json.dumps(analysis, indent=2))
        
        # 5. Generate optimization suggestions
        logger.info("\n" + "="*60)
        logger.info("5. OPTIMIZATION SUGGESTIONS")
        logger.info("="*60)
        
        suggestions = self.optimizer.generate_optimization_suggestions(
            initial_params,
            initial_results
        )
        
        logger.info(f"Generated {len(suggestions)} parameter suggestions")
        for i, sugg in enumerate(suggestions[:3]):
            logger.info(f"\nSuggestion {i+1}:")
            self._display_param_changes(initial_params, sugg)
        
        # 6. Run parameter optimization
        logger.info("\n" + "="*60)
        logger.info("6. RUNNING OPTIMIZATION")
        logger.info("="*60)
        
        # Define backtest wrapper function
        def backtest_wrapper(params):
            try:
                results = self._run_backtest(data, params)
                return results
            except Exception as e:
                logger.error(f"Backtest failed: {e}")
                return {'sharpe_ratio': -100, 'total_return': -100}
        
        # Run different optimization methods
        optimization_results = {}
        
        # a) Bayesian Optimization
        logger.info("\na) Running Bayesian Optimization...")
        self.optimizer.optimization_method = "bayesian"
        bayesian_result = self.optimizer.run_parameter_optimization(
            backtest_wrapper,
            initial_params,
            n_iterations=20
        )
        optimization_results['bayesian'] = bayesian_result
        
        # b) Genetic Algorithm
        logger.info("\nb) Running Genetic Algorithm...")
        self.optimizer.optimization_method = "genetic"
        genetic_result = self.optimizer.run_parameter_optimization(
            backtest_wrapper,
            initial_params,
            n_iterations=20
        )
        optimization_results['genetic'] = genetic_result
        
        # c) Grid Search
        logger.info("\nc) Running Grid Search...")
        self.optimizer.optimization_method = "grid"
        grid_result = self.optimizer.run_parameter_optimization(
            backtest_wrapper,
            initial_params,
            n_iterations=20
        )
        optimization_results['grid'] = grid_result
        
        # 7. Compare optimization results
        logger.info("\n" + "="*60)
        logger.info("7. OPTIMIZATION RESULTS COMPARISON")
        logger.info("="*60)
        
        self._compare_optimization_results(optimization_results)
        
        # 8. Validate best parameters
        logger.info("\n" + "="*60)
        logger.info("8. PARAMETER VALIDATION")
        logger.info("="*60)
        
        # Find best result
        best_method = max(optimization_results.keys(), 
                         key=lambda x: optimization_results[x].optimized_metrics.get('sharpe_ratio', 0))
        best_result = optimization_results[best_method]
        
        logger.info(f"Best optimization method: {best_method}")
        
        # Validation function
        def validation_wrapper(params, seed=None):
            # Add some randomness to simulate different market conditions
            np.random.seed(seed)
            noise_factor = 1.0 + np.random.normal(0, 0.05)
            
            results = self._run_backtest(data, params)
            # Add noise to simulate variability
            for key in ['sharpe_ratio', 'total_return', 'max_drawdown']:
                if key in results:
                    results[key] *= noise_factor
            
            return results
        
        validation_results = self.optimizer.validate_optimized_parameters(
            best_result.optimized_params,
            validation_wrapper,
            n_runs=5
        )
        
        self._display_validation_results(validation_results)
        
        # 9. Generate final report
        logger.info("\n" + "="*60)
        logger.info("9. FINAL OPTIMIZATION REPORT")
        logger.info("="*60)
        
        self._generate_optimization_report(initial_params, initial_results, 
                                         best_result, validation_results)
        
        return best_result
    
    def demonstrate_llm_features(self):
        """Demonstrate specific LLM features."""
        logger.info("\n" + "="*60)
        logger.info("LLM FEATURE DEMONSTRATION")
        logger.info("="*60)
        
        # Sample metrics for demonstration
        sample_metrics = {
            'sharpe_ratio': 0.8,
            'total_return': 15.5,
            'max_drawdown': -18.5,
            'win_rate': 45.2,
            'profit_factor': 1.3,
            'total_trades': 156
        }
        
        sample_params = {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_span_b_period': 52,
            'stop_loss_multiplier': 2.0,
            'position_size_pct': 2.0
        }
        
        # 1. Get LLM analysis of poor performance
        logger.info("\n1. Analyzing Poor Performance:")
        poor_analysis = self.optimizer.analyze_backtest_results({
            'metrics': sample_metrics,
            'parameters': sample_params,
            'strategy_name': 'Ichimoku Strategy'
        })
        logger.info(f"Analysis: {poor_analysis.get('summary', 'No summary available')}")
        
        # 2. Modify metrics to show good performance
        good_metrics = sample_metrics.copy()
        good_metrics.update({
            'sharpe_ratio': 1.8,
            'total_return': 45.2,
            'max_drawdown': -8.5,
            'win_rate': 62.5,
            'profit_factor': 2.1
        })
        
        logger.info("\n2. Analyzing Good Performance:")
        good_analysis = self.optimizer.analyze_backtest_results({
            'metrics': good_metrics,
            'parameters': sample_params,
            'strategy_name': 'Ichimoku Strategy'
        })
        logger.info(f"Analysis: {good_analysis.get('summary', 'No summary available')}")
        
        # 3. Get specific optimization suggestions
        logger.info("\n3. Getting Optimization Suggestions:")
        suggestions = self.optimizer.generate_optimization_suggestions(
            sample_params, sample_metrics
        )
        
        logger.info(f"Received {len(suggestions)} suggestions:")
        for i, sugg in enumerate(suggestions[:3]):
            logger.info(f"\nSuggestion {i+1}:")
            for param, value in sugg.items():
                if param in sample_params and value != sample_params[param]:
                    logger.info(f"  {param}: {sample_params[param]} → {value}")
    
    def _prepare_sample_data(self) -> pd.DataFrame:
        """Prepare sample data for optimization."""
        # Generate synthetic OHLCV data for demonstration
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')
        n_bars = len(dates)
        
        # Generate realistic price movement
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, n_bars)
        prices = 50000 * (1 + returns).cumprod()
        
        # Add trend
        trend = np.linspace(0, 0.2, n_bars)
        prices = prices * (1 + trend)
        
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = prices * (1 + np.random.normal(0, 0.001, n_bars))
        data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, 0.002, n_bars)))
        data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, 0.002, n_bars)))
        data['volume'] = np.random.uniform(100, 1000, n_bars)
        
        # Calculate Ichimoku indicators
        ichimoku_data = self.ichimoku_calc.calculate(data)
        
        return ichimoku_data
    
    def _run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
        """Run a backtest with given parameters."""
        # Create strategy configuration
        strategy_config = StrategyConfig(
            name="Optimized Ichimoku",
            strategy_id="ichimoku_opt_001",
            symbols=["BTC/USDT"],
            timeframe="1h",
            ichimoku_params=ConfigIchimokuParameters(
                tenkan_period=int(params.get('tenkan_period', 9)),
                kijun_period=int(params.get('kijun_period', 26)),
                senkou_span_b_period=int(params.get('senkou_span_b_period', 52)),
                displacement=int(params.get('displacement', 26))
            ),
            buy_signals=ConfigSignalCombination(
                conditions=["PriceAboveCloud", "TenkanAboveKijun"],
                combination_type="AND",
                min_strength=params.get('min_signal_strength', 0.6)
            ),
            sell_signals=ConfigSignalCombination(
                conditions=["PriceBelowCloud", "TenkanBelowKijun"],
                combination_type="AND",
                min_strength=params.get('min_signal_strength', 0.6)
            ),
            risk_management=RiskManagement(
                stop_loss_type="atr",
                stop_loss_value=params.get('stop_loss_multiplier', 2.0),
                take_profit_type="atr",
                take_profit_value=params.get('take_profit_multiplier', 3.0)
            ),
            position_sizing=PositionSizing(
                method="fixed",
                risk_per_trade_pct=params.get('position_size_pct', 2.0),
                fixed_size=0.001
            )
        )
        
        # Run simplified backtest (for demonstration)
        # In real usage, this would call the actual backtester
        # results = self.backtester.run_backtest(strategy_config, data)
        
        # Simulate results based on parameters
        base_sharpe = 1.0
        base_return = 20.0
        base_drawdown = -10.0
        base_win_rate = 50.0
        
        # Adjust based on parameters (simplified simulation)
        period_factor = (params.get('tenkan_period', 9) / 9) * 0.8 + 0.2
        risk_factor = 1.0 / params.get('stop_loss_multiplier', 2.0)
        signal_factor = params.get('min_signal_strength', 0.6)
        
        results = {
            'sharpe_ratio': base_sharpe * period_factor * risk_factor,
            'total_return': base_return * period_factor * (1 + risk_factor),
            'max_drawdown': base_drawdown * (1.5 - risk_factor),
            'win_rate': base_win_rate * signal_factor,
            'profit_factor': 1.5 * period_factor * risk_factor,
            'total_trades': int(150 / period_factor),
            'avg_win': 100 * risk_factor,
            'avg_loss': -50 / risk_factor
        }
        
        return results
    
    def _display_results(self, title: str, results: Dict[str, float]):
        """Display backtest results."""
        logger.info(f"\n{title}:")
        logger.info(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
        logger.info(f"  Total Return: {results.get('total_return', 0):.2f}%")
        logger.info(f"  Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        logger.info(f"  Win Rate: {results.get('win_rate', 0):.2f}%")
        logger.info(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
        logger.info(f"  Total Trades: {results.get('total_trades', 0)}")
    
    def _display_param_changes(self, original: Dict[str, Any], optimized: Dict[str, Any]):
        """Display parameter changes."""
        for param, new_value in optimized.items():
            if param in original:
                old_value = original[param]
                if old_value != new_value:
                    change = ((new_value - old_value) / old_value * 100) if old_value != 0 else 0
                    logger.info(f"  {param}: {old_value} → {new_value} ({change:+.1f}%)")
    
    def _compare_optimization_results(self, results: Dict[str, Any]):
        """Compare results from different optimization methods."""
        comparison_data = []
        
        for method, result in results.items():
            comparison_data.append({
                'Method': method.capitalize(),
                'Original Sharpe': f"{result.original_metrics.get('sharpe_ratio', 0):.3f}",
                'Optimized Sharpe': f"{result.optimized_metrics.get('sharpe_ratio', 0):.3f}",
                'Improvement': f"{result.improvement_percentage:.1f}%",
                'Best Iteration': result.best_iteration,
                'Confidence': f"{result.confidence_score:.1f}%"
            })
        
        # Create comparison table
        df = pd.DataFrame(comparison_data)
        logger.info("\nOptimization Methods Comparison:")
        logger.info(df.to_string(index=False))
        
        # Show best parameters from each method
        logger.info("\nOptimized Parameters by Method:")
        for method, result in results.items():
            logger.info(f"\n{method.capitalize()}:")
            for param in ['tenkan_period', 'kijun_period', 'stop_loss_multiplier']:
                if param in result.optimized_params:
                    logger.info(f"  {param}: {result.optimized_params[param]}")
    
    def _display_validation_results(self, validation: Dict[str, Any]):
        """Display validation results."""
        logger.info("\nValidation Results:")
        logger.info(f"  Runs: {validation['validation_runs']}")
        logger.info(f"  Mean Sharpe: {validation['mean_sharpe']:.3f} (±{validation['std_sharpe']:.3f})")
        logger.info(f"  Mean Return: {validation['mean_return']:.2f}% (±{validation['std_return']:.2f}%)")
        logger.info(f"  Mean Drawdown: {validation['mean_max_drawdown']:.2f}%")
        logger.info(f"  Worst Drawdown: {validation['worst_drawdown']:.2f}%")
        logger.info(f"  Consistency Score: {validation['consistency_score']:.1f}/100")
        
        if 'llm_assessment' in validation:
            logger.info(f"\nLLM Assessment:")
            logger.info(validation['llm_assessment'][:500] + "..." if len(validation['llm_assessment']) > 500 else validation['llm_assessment'])
    
    def _generate_optimization_report(self, 
                                    initial_params: Dict[str, Any],
                                    initial_results: Dict[str, float],
                                    best_result: Any,
                                    validation: Dict[str, Any]):
        """Generate final optimization report."""
        logger.info("\nOPTIMIZATION SUMMARY REPORT")
        logger.info("="*60)
        
        logger.info("\n1. PERFORMANCE IMPROVEMENT:")
        logger.info(f"   Sharpe Ratio: {initial_results.get('sharpe_ratio', 0):.3f} → {best_result.optimized_metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"   Total Return: {initial_results.get('total_return', 0):.2f}% → {best_result.optimized_metrics.get('total_return', 0):.2f}%")
        logger.info(f"   Max Drawdown: {initial_results.get('max_drawdown', 0):.2f}% → {best_result.optimized_metrics.get('max_drawdown', 0):.2f}%")
        logger.info(f"   Overall Improvement: {best_result.improvement_percentage:.1f}%")
        
        logger.info("\n2. KEY PARAMETER CHANGES:")
        self._display_param_changes(initial_params, best_result.optimized_params)
        
        logger.info("\n3. VALIDATION CONFIDENCE:")
        logger.info(f"   Consistency Score: {validation['consistency_score']:.1f}/100")
        logger.info(f"   Parameter Stability: {'High' if validation['std_sharpe'] < 0.2 else 'Moderate' if validation['std_sharpe'] < 0.5 else 'Low'}")
        logger.info(f"   Ready for Live Trading: {'Yes' if validation['consistency_score'] > 80 else 'Needs more validation'}")
        
        logger.info("\n4. RECOMMENDATIONS:")
        if best_result.llm_suggestions:
            logger.info(f"   {best_result.llm_suggestions}")
        else:
            logger.info("   - Continue monitoring performance")
            logger.info("   - Consider paper trading before live deployment")
            logger.info("   - Regularly re-optimize based on market conditions")
        
        logger.info("\n" + "="*60)
        logger.info("Report generated successfully!")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create example instance
    example = OptimizationExample()
    
    # Run complete optimization example
    print("\n1. Running Complete Optimization Example:")
    best_result = example.run_complete_optimization_example()
    
    # Demonstrate LLM features
    print("\n2. Demonstrating LLM Features:")
    example.demonstrate_llm_features()
    
    print("\n✓ LLM Optimization Example Complete!")
    print("\nKey Features Demonstrated:")
    print("- AI-powered backtest analysis")
    print("- Parameter optimization suggestions")
    print("- Bayesian optimization")
    print("- Genetic algorithms")
    print("- Grid search")
    print("- Parameter validation")
    print("- Multi-objective optimization")
    print("- LLM-generated insights")