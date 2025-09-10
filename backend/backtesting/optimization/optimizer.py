"""
Strategy Parameter Optimization Module

This module provides various optimization methods to find optimal parameters
for trading strategies including grid search and Bayesian optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from tqdm import tqdm

# Optional imports for advanced optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    strategy_name: str
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    convergence_plot: Optional[Any] = None
    total_iterations: int = 0
    optimization_time: float = 0.0
    parameter_importance: Optional[Dict[str, float]] = None


class ParameterOptimizer:
    """
    Main parameter optimization class supporting multiple optimization methods
    """
    
    def __init__(
        self,
        backtest_function: Callable,
        optimization_config: Dict[str, Any],
        n_jobs: int = -1
    ):
        """
        Initialize the optimizer
        
        Args:
            backtest_function: Function that runs backtest and returns metrics
            optimization_config: Configuration for optimization from YAML
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.backtest_function = backtest_function
        self.optimization_config = optimization_config
        self.n_jobs = n_jobs if n_jobs > 0 else None
        
        # Track optimization history
        self.optimization_history = []
        self.best_score = -float('inf')
        self.best_parameters = {}
        
    def optimize_strategy(
        self,
        strategy_name: str,
        parameter_ranges: Dict[str, Dict],
        objective_metric: str = "sharpe_ratio",
        method: str = "grid_search",
        constraints: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """
        Optimize parameters for a given strategy
        
        Args:
            strategy_name: Name of the strategy to optimize
            parameter_ranges: Parameter ranges to search
            objective_metric: Metric to optimize (e.g., sharpe_ratio)
            method: Optimization method (grid_search, random_search, bayesian)
            constraints: Optional constraints on metrics
            
        Returns:
            OptimizationResult with best parameters and history
        """
        logger.info(f"Starting {method} optimization for {strategy_name}")
        start_time = datetime.now()
        
        # Reset tracking
        self.optimization_history = []
        self.best_score = -float('inf')
        self.best_parameters = {}
        
        # Choose optimization method
        if method == "grid_search":
            self._grid_search(strategy_name, parameter_ranges, objective_metric, constraints)
        elif method == "random_search":
            self._random_search(strategy_name, parameter_ranges, objective_metric, constraints)
        elif method == "bayesian":
            if SKOPT_AVAILABLE:
                self._bayesian_optimization(strategy_name, parameter_ranges, objective_metric, constraints)
            else:
                logger.warning("scikit-optimize not installed. Falling back to grid search.")
                self._grid_search(strategy_name, parameter_ranges, objective_metric, constraints)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate optimization time
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Analyze parameter importance if we have enough data
        parameter_importance = None
        if len(self.optimization_history) > 10:
            parameter_importance = self._calculate_parameter_importance(
                parameter_ranges.keys()
            )
        
        return OptimizationResult(
            strategy_name=strategy_name,
            best_parameters=self.best_parameters,
            best_score=self.best_score,
            optimization_history=self.optimization_history,
            total_iterations=len(self.optimization_history),
            optimization_time=optimization_time,
            parameter_importance=parameter_importance
        )
    
    def _grid_search(
        self,
        strategy_name: str,
        parameter_ranges: Dict[str, Dict],
        objective_metric: str,
        constraints: Optional[Dict[str, float]]
    ):
        """Perform grid search optimization"""
        # Generate parameter grid
        param_grid = self._generate_parameter_grid(parameter_ranges)
        
        if not param_grid:
            logger.error(f"Failed to generate parameter grid for {strategy_name}")
            logger.error(f"Parameter ranges: {parameter_ranges}")
            return
        
        logger.info(f"Grid search: Testing {len(param_grid)} parameter combinations")
        
        # Test each combination
        if self.n_jobs:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for params in param_grid:
                    future = executor.submit(
                        self._evaluate_parameters,
                        strategy_name,
                        params,
                        objective_metric,
                        constraints
                    )
                    futures.append((future, params))
                
                # Process results as they complete
                for future, params in tqdm(futures, desc="Grid Search Progress"):
                    try:
                        score, metrics = future.result()
                        self._update_best(params, score, metrics)
                    except Exception as e:
                        logger.error(f"Error evaluating parameters {params}: {e}")
        else:
            # Sequential execution
            for params in tqdm(param_grid, desc="Grid Search Progress"):
                try:
                    score, metrics = self._evaluate_parameters(
                        strategy_name, params, objective_metric, constraints
                    )
                    self._update_best(params, score, metrics)
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {e}")
    
    def _random_search(
        self,
        strategy_name: str,
        parameter_ranges: Dict[str, Dict],
        objective_metric: str,
        constraints: Optional[Dict[str, float]]
    ):
        """Perform random search optimization"""
        n_iter = self.optimization_config.get('random_search', {}).get('n_iter', 100)
        random_state = self.optimization_config.get('random_search', {}).get('random_state', 42)
        
        np.random.seed(random_state)
        logger.info(f"Random search: Testing {n_iter} random parameter combinations")
        
        for i in tqdm(range(n_iter), desc="Random Search Progress"):
            # Generate random parameters
            params = self._generate_random_parameters(parameter_ranges)
            
            try:
                score, metrics = self._evaluate_parameters(
                    strategy_name, params, objective_metric, constraints
                )
                self._update_best(params, score, metrics)
            except Exception as e:
                logger.error(f"Error evaluating parameters {params}: {e}")
    
    def _bayesian_optimization(
        self,
        strategy_name: str,
        parameter_ranges: Dict[str, Dict],
        objective_metric: str,
        constraints: Optional[Dict[str, float]]
    ):
        """Perform Bayesian optimization using scikit-optimize"""
        # Convert parameter ranges to skopt format
        dimensions = []
        param_names = []
        
        for param_name, param_config in parameter_ranges.items():
            param_type = param_config.get('type', 'float')
            param_names.append(param_name)
            
            if param_type == 'int':
                dimensions.append(Integer(param_config['min'], param_config['max'], name=param_name))
            elif param_type == 'float':
                dimensions.append(Real(param_config['min'], param_config['max'], name=param_name))
            elif param_type == 'bool':
                dimensions.append(Categorical([True, False], name=param_name))
            elif param_type == 'list':
                # Handle list-type parameters (e.g., fib_levels which are lists)
                # Convert lists to tuples for hashability or use indices
                options = param_config['options']
                if options and isinstance(options[0], list):
                    # Create indices for the options
                    dimensions.append(Categorical(range(len(options)), name=param_name + '_idx'))
                else:
                    dimensions.append(Categorical(options, name=param_name))
        
        # Track which parameters are list indices
        list_param_indices = {}
        for param_name, param_config in parameter_ranges.items():
            if param_config.get('type') == 'list' and isinstance(param_config['options'][0], list):
                list_param_indices[param_name + '_idx'] = (param_name, param_config['options'])
        
        # Define objective function for Bayesian optimization
        @use_named_args(dimensions)
        def objective(**params):
            try:
                # Convert indices back to actual list values
                converted_params = {}
                for key, value in params.items():
                    if key in list_param_indices:
                        orig_name, options = list_param_indices[key]
                        converted_params[orig_name] = options[value]
                    else:
                        converted_params[key] = value
                
                score, metrics = self._evaluate_parameters(
                    strategy_name, converted_params, objective_metric, constraints
                )
                self._update_best(converted_params, score, metrics)
                # Return negative score because skopt minimizes
                # Handle infinite or very large values from evaluation
                if np.isinf(score):
                    # If score is -inf (constraints not met), return a large positive value for minimization
                    # Otherwise, if it's +inf, also return a large positive value.
                    return 1e9  # A large finite number indicating a very bad result
                return -score
            except Exception as e:
                logger.error(f"Error in Bayesian optimization: {e}")
                # Return a large finite number if an exception occurs during evaluation
                return 1e9
        
        # Run Bayesian optimization
        n_calls = self.optimization_config.get('bayesian', {}).get('n_calls', 100)
        n_initial_points = self.optimization_config.get('bayesian', {}).get('n_initial_points', 20)
        
        logger.info(f"Bayesian optimization: {n_calls} iterations with {n_initial_points} initial points")
        
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=self.optimization_config.get('bayesian', {}).get('acq_func', 'EI'),
            random_state=42
        )
        
        # Extract best parameters - need to reconstruct param names from dimensions
        best_params = {}
        for i, dim in enumerate(dimensions):
            best_params[dim.name] = result.x[i]
        
        # Convert indices back to actual list values for best params
        converted_best_params = {}
        for key, value in best_params.items():
            if key in list_param_indices:
                orig_name, options = list_param_indices[key]
                converted_best_params[orig_name] = options[value]
            elif key.endswith('_idx'):  # Skip index parameters
                continue
            else:
                converted_best_params[key] = value
        
        self.best_parameters = converted_best_params
        self.best_score = -result.fun
    
    def _generate_parameter_grid(self, parameter_ranges: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for grid search"""
        param_values = {}
        
        for param_name, param_config in parameter_ranges.items():
            param_type = param_config.get('type', 'float')
            
            if param_type in ['int', 'float']:
                min_val = param_config['min']
                max_val = param_config['max']
                step = param_config.get('step', 1)
                
                if param_type == 'int':
                    values = list(range(int(min_val), int(max_val) + 1, int(step)))
                else:
                    values = np.arange(min_val, max_val + step/2, step).tolist()
                    
                param_values[param_name] = values
                
            elif param_type == 'bool':
                param_values[param_name] = param_config.get('options', [True, False])
                
            elif param_type == 'list':
                param_values[param_name] = param_config['options']
        
        # Generate all combinations
        keys = param_values.keys()
        values = param_values.values()
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        # Limit combinations if too many
        max_combinations = self.optimization_config.get('grid_search', {}).get('max_combinations', 1000)
        if len(combinations) > max_combinations and not self.optimization_config.get('grid_search', {}).get('exhaustive', False):
            logger.warning(f"Grid has {len(combinations)} combinations. Randomly sampling {max_combinations}")
            np.random.shuffle(combinations)
            combinations = combinations[:max_combinations]
        
        return combinations
    
    def _generate_random_parameters(self, parameter_ranges: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate random parameters within specified ranges"""
        params = {}
        
        for param_name, param_config in parameter_ranges.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'int':
                params[param_name] = np.random.randint(param_config['min'], param_config['max'] + 1)
            elif param_type == 'float':
                params[param_name] = np.random.uniform(param_config['min'], param_config['max'])
            elif param_type == 'bool':
                params[param_name] = np.random.choice(param_config.get('options', [True, False]))
            elif param_type == 'list':
                # For list type, we need to handle lists of lists (like fib_levels)
                options = param_config['options']
                # Use random.choice from standard library which can handle any object type
                import random
                params[param_name] = random.choice(options)
        
        return params
    
    def _evaluate_parameters(
        self,
        strategy_name: str,
        parameters: Dict[str, Any],
        objective_metric: str,
        constraints: Optional[Dict[str, float]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a set of parameters and return the objective score
        
        Returns:
            Tuple of (objective_score, all_metrics)
        """
        # Run backtest with given parameters
        results = self.backtest_function(strategy_name, parameters)
        
        # Extract metrics
        metrics = {
            'total_return': results.get('total_return', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'max_drawdown': results.get('max_drawdown', 100),
            'win_rate': results.get('win_rate', 0),
            'profit_factor': results.get('profit_factor', 0),
            'total_trades': results.get('total_trades', 0)
        }
        
        # Check constraints
        if constraints:
            if metrics['total_trades'] < constraints.get('min_trades', 0):
                return -float('inf'), metrics
            if metrics['max_drawdown'] > constraints.get('max_drawdown', 100):
                return -float('inf'), metrics
            if metrics['win_rate'] < constraints.get('min_win_rate', 0):
                return -float('inf'), metrics
            if metrics['sharpe_ratio'] < constraints.get('min_sharpe_ratio', -float('inf')):
                return -float('inf'), metrics
        
        # Get objective score
        score = metrics.get(objective_metric, 0)
        
        # For drawdown, we want to minimize (so negate)
        if 'drawdown' in objective_metric:
            score = -score
        
        # Record in history
        self.optimization_history.append({
            'parameters': parameters.copy(),
            'score': score,
            'metrics': metrics.copy(),
            'timestamp': datetime.now().isoformat()
        })
        
        return score, metrics
    
    def _update_best(self, parameters: Dict[str, Any], score: float, metrics: Dict[str, float]):
        """Update best parameters if current score is better"""
        if score > self.best_score:
            self.best_score = score
            self.best_parameters = parameters.copy()
            logger.info(f"New best score: {score:.4f} with parameters: {parameters}")
    
    def _calculate_parameter_importance(self, parameter_names: List[str]) -> Dict[str, float]:
        """
        Calculate relative importance of each parameter based on optimization history
        
        Uses variance in objective score for different parameter values
        """
        if len(self.optimization_history) < 10:
            return {}
        
        importance = {}
        
        for param_name in parameter_names:
            # Group scores by parameter value
            param_scores = {}
            for record in self.optimization_history:
                param_value = record['parameters'].get(param_name)
                score = record['score']
                
                if param_value not in param_scores:
                    param_scores[param_value] = []
                param_scores[param_value].append(score)
            
            # Calculate variance across different parameter values
            mean_scores = [np.mean(scores) for scores in param_scores.values()]
            importance[param_name] = np.var(mean_scores) if len(mean_scores) > 1 else 0
        
        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance
    
    def _convert_numpy_types(self, obj):
        """Helper to convert numpy types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(elem) for elem in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    def save_optimization_results(self, result: OptimizationResult, output_dir: str):
        """Save optimization results to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best parameters
        params_file = os.path.join(output_dir, f"{result.strategy_name}_best_params.json")
        with open(params_file, 'w') as f:
            json.dump({
                'strategy_name': result.strategy_name,
                'best_parameters': self._convert_numpy_types(result.best_parameters),
                'best_score': result.best_score,
                'optimization_time': result.optimization_time,
                'total_iterations': result.total_iterations
            }, f, indent=2)
        
        # Save optimization history
        history_file = os.path.join(output_dir, f"{result.strategy_name}_optimization_history.json")
        with open(history_file, 'w') as f:
            json.dump(self._convert_numpy_types(result.optimization_history), f, indent=2)
        
        # Save parameter importance if available
        if result.parameter_importance:
            importance_file = os.path.join(output_dir, f"{result.strategy_name}_param_importance.json")
            with open(importance_file, 'w') as f:
                json.dump(result.parameter_importance, f, indent=2)
        
        logger.info(f"Optimization results saved to {output_dir}")
    
    def generate_optimization_report(self, result: OptimizationResult) -> str:
        """Generate a text report of optimization results"""
        report = f"""
Optimization Report for {result.strategy_name}
{'=' * 60}

Best Parameters Found:
{json.dumps(result.best_parameters, indent=2)}

Performance Metrics:
- Best Score ({self.optimization_config.get('performance_criteria', {}).get('primary_metric', 'sharpe_ratio')}): {result.best_score:.4f}
- Total Iterations: {result.total_iterations}
- Optimization Time: {result.optimization_time:.2f} seconds

"""
        
        if result.parameter_importance:
            report += "Parameter Importance:\n"
            for param, importance in sorted(result.parameter_importance.items(), key=lambda x: x[1], reverse=True):
                report += f"  - {param}: {importance:.2%}\n"
            report += "\n"
        
        # Add convergence information
        if result.optimization_history:
            scores = [h['score'] for h in result.optimization_history]
            report += f"Score Evolution:\n"
            report += f"  - Initial: {scores[0]:.4f}\n"
            report += f"  - Final: {scores[-1]:.4f}\n"
            report += f"  - Best: {max(scores):.4f}\n"
            report += f"  - Mean: {np.mean(scores):.4f}\n"
            report += f"  - Std Dev: {np.std(scores):.4f}\n"
        
        return report


class MultiObjectiveOptimizer(ParameterOptimizer):
    """
    Extended optimizer for multi-objective optimization
    """
    
    def optimize_multi_objective(
        self,
        strategy_name: str,
        parameter_ranges: Dict[str, Dict],
        objectives: List[Dict[str, Any]],
        method: str = "grid_search"
    ) -> OptimizationResult:
        """
        Optimize for multiple objectives with weighted importance
        
        Args:
            strategy_name: Name of the strategy
            parameter_ranges: Parameter search space
            objectives: List of objectives with metric, weight, and direction
            method: Optimization method
            
        Returns:
            OptimizationResult with Pareto-optimal solutions
        """
        # Create composite objective function
        def composite_objective(metrics: Dict[str, float]) -> float:
            score = 0.0
            for obj in objectives:
                metric = obj['metric']
                weight = obj['weight']
                direction = obj.get('direction', 'maximize')
                
                value = metrics.get(metric, 0)
                if direction == 'minimize':
                    value = -value
                    
                score += weight * value
            
            return score
        
        # Modify evaluation to use composite objective
        original_evaluate = self._evaluate_parameters
        
        def multi_evaluate(strategy_name, parameters, objective_metric, constraints):
            _, metrics = original_evaluate(strategy_name, parameters, objective_metric, constraints)
            score = composite_objective(metrics)
            return score, metrics
        
        self._evaluate_parameters = multi_evaluate
        
        # Run optimization
        result = self.optimize_strategy(
            strategy_name=strategy_name,
            parameter_ranges=parameter_ranges,
            objective_metric="composite",
            method=method
        )
        
        # Restore original evaluate function
        self._evaluate_parameters = original_evaluate
        
        return result
