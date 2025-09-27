"""
Automated Parameter Testing Framework for Ichimoku Trading Strategies

This module provides comprehensive parameter optimization with:
- Grid search across parameter space
- Walk-forward optimization with rolling windows
- Out-of-sample testing validation
- Statistical significance testing
- Overfitting mitigation techniques
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import itertools
import json
import sqlite3
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from backend.data_fetching.data_fetcher import DataFetcher
from backend.executable_workflow.backtesting import IchimokuBacktester
from backend.executable_workflow.analytics import PerformanceAnalyzer
from backend.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """Defines the parameter search space for optimization"""
    tenkan_periods: List[int] = field(default_factory=lambda: list(range(7, 13)))
    kijun_periods: List[int] = field(default_factory=lambda: list(range(22, 27)))
    senkou_b_periods: List[int] = field(default_factory=lambda: list(range(48, 57)))
    signal_combinations: List[List[str]] = field(default_factory=lambda: [
        ['cloud_breakout'],
        ['tk_cross'],
        ['cloud_breakout', 'tk_cross'],
        ['cloud_breakout', 'price_momentum'],
        ['cloud_breakout', 'tk_cross', 'price_momentum'],
        ['cloud_breakout', 'tk_cross', 'chikou_confirmation']
    ])
    stop_loss_percent: List[float] = field(default_factory=lambda: [0.02, 0.025, 0.03])
    take_profit_percent: List[float] = field(default_factory=lambda: [0.03, 0.04, 0.05])
    
    def get_total_combinations(self) -> int:
        """Calculate total number of parameter combinations"""
        return (len(self.tenkan_periods) * 
                len(self.kijun_periods) * 
                len(self.senkou_b_periods) * 
                len(self.signal_combinations) * 
                len(self.stop_loss_percent) * 
                len(self.take_profit_percent))


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward optimization window"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int


@dataclass
class OptimizationResult:
    """Stores results from parameter optimization"""
    parameters: Dict[str, Any]
    in_sample_metrics: Dict[str, float]
    out_of_sample_metrics: Optional[Dict[str, float]] = None
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    stability_score: float = 0.0
    statistical_significance: Optional[Dict[str, float]] = None
    optimization_date: datetime = field(default_factory=datetime.now)
    window_id: Optional[int] = None
    is_robust: bool = False


class ParameterOptimizer:
    """
    Advanced parameter optimization framework with walk-forward analysis,
    cross-validation, and statistical significance testing.
    """
    
    def __init__(self, 
                 db_path: str = "optimization_results.db",
                 min_sample_trades: int = 30,
                 significance_level: float = 0.05,
                 n_jobs: int = -1):
        """
        Initialize the parameter optimizer.
        
        Args:
            db_path: Path to SQLite database for storing results
            min_sample_trades: Minimum number of trades for statistical validity
            significance_level: P-value threshold for statistical tests
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.db_path = db_path
        self.min_sample_trades = min_sample_trades
        self.significance_level = significance_level
        self.n_jobs = n_jobs if n_jobs > 0 else None
        
        self.data_fetcher = DataFetcher()
        self.analyzer = PerformanceAnalyzer()
        
        # Initialize results database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for storing optimization results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                optimization_date TIMESTAMP,
                symbol TEXT,
                parameters TEXT,
                in_sample_sharpe REAL,
                in_sample_return REAL,
                in_sample_drawdown REAL,
                in_sample_win_rate REAL,
                out_sample_sharpe REAL,
                out_sample_return REAL,
                out_sample_drawdown REAL,
                out_sample_win_rate REAL,
                stability_score REAL,
                is_robust BOOLEAN,
                window_id INTEGER,
                statistical_tests TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def grid_search_parameters(self, 
                             symbol: str,
                             start_date: str,
                             end_date: str,
                             parameter_space: Optional[ParameterSpace] = None,
                             parallel: bool = True) -> List[OptimizationResult]:
        """
        Perform grid search optimization across parameter space.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for backtesting
            end_date: End date for backtesting
            parameter_space: Parameter search space (uses default if None)
            parallel: Whether to use parallel processing
            
        Returns:
            List of optimization results sorted by performance
        """
        if parameter_space is None:
            parameter_space = ParameterSpace()
            
        # Fetch data
        data = self.data_fetcher.fetch_data(symbol, start_date, end_date)
        logger.info(f"Data fetched: {len(data)} candles from {start_date} to {end_date}")
        
        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_space)
        logger.info(f"Total parameter combinations to test: {len(param_combinations)}")
        
        # Run optimization
        if parallel and self.n_jobs != 1:
            results = self._parallel_grid_search(data, param_combinations)
        else:
            results = self._sequential_grid_search(data, param_combinations)
            
        # Sort by Sharpe ratio
        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        
        # Save to database
        self._save_results_to_db(symbol, results)
        
        return results
        
    def _generate_parameter_combinations(self, space: ParameterSpace) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from parameter space"""
        combinations = []
        
        for params in itertools.product(
            space.tenkan_periods,
            space.kijun_periods,
            space.senkou_b_periods,
            space.signal_combinations,
            space.stop_loss_percent,
            space.take_profit_percent
        ):
            combinations.append({
                'ichimoku_params': {
                    'tenkan_period': params[0],
                    'kijun_period': params[1],
                    'senkou_b_period': params[2]
                },
                'signal_config': {
                    'entry_signals': params[3],
                    'exit_signals': ['stop_loss', 'take_profit']
                },
                'risk_params': {
                    'stop_loss_percent': params[4],
                    'take_profit_percent': params[5],
                    'position_size': 0.95
                }
            })
            
        return combinations
        
    def _run_single_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> OptimizationResult:
        """Run a single backtest with given parameters"""
        try:
            # Create backtester with parameters
            backtester = IchimokuBacktester(
                initial_capital=10000,
                position_size=params['risk_params']['position_size']
            )
            
            # Run backtest
            trades, equity_curve = backtester.backtest(
                data,
                ichimoku_params=params['ichimoku_params'],
                signal_config=params['signal_config'],
                risk_params=params['risk_params']
            )
            
            # Calculate metrics if we have enough trades
            if len(trades) < self.min_sample_trades:
                return OptimizationResult(
                    parameters=params,
                    in_sample_metrics={},
                    sharpe_ratio=-999,  # Penalize insufficient trades
                    total_return=0,
                    max_drawdown=1,
                    win_rate=0,
                    profit_factor=0
                )
                
            # Calculate performance metrics
            metrics = self.analyzer.calculate_all_metrics(trades, equity_curve)
            
            return OptimizationResult(
                parameters=params,
                in_sample_metrics={
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'total_return': metrics.total_return,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'total_trades': metrics.total_trades
                },
                sharpe_ratio=metrics.sharpe_ratio,
                total_return=metrics.total_return,
                max_drawdown=metrics.max_drawdown,
                win_rate=metrics.win_rate,
                profit_factor=metrics.profit_factor
            )
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return OptimizationResult(
                parameters=params,
                in_sample_metrics={},
                sharpe_ratio=-999,
                total_return=0,
                max_drawdown=1,
                win_rate=0,
                profit_factor=0
            )
            
    def _parallel_grid_search(self, data: pd.DataFrame, param_combinations: List[Dict]) -> List[OptimizationResult]:
        """Run grid search in parallel"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_params = {
                executor.submit(self._run_single_backtest, data, params): params 
                for params in param_combinations
            }
            
            for i, future in enumerate(as_completed(future_to_params)):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{len(param_combinations)} backtests")
                        
                except Exception as e:
                    logger.error(f"Error in parallel execution: {e}")
                    
        return results
        
    def _sequential_grid_search(self, data: pd.DataFrame, param_combinations: List[Dict]) -> List[OptimizationResult]:
        """Run grid search sequentially"""
        results = []
        
        for i, params in enumerate(param_combinations):
            result = self._run_single_backtest(data, params)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{len(param_combinations)} backtests")
                
        return results
        
    def walk_forward_optimization(self,
                                symbol: str,
                                start_date: str,
                                end_date: str,
                                window_size_days: int = 252,  # 1 year
                                test_size_days: int = 63,      # 3 months
                                step_days: int = 21,           # 1 month
                                parameter_space: Optional[ParameterSpace] = None) -> Dict[str, Any]:
        """
        Perform walk-forward optimization with rolling windows.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            window_size_days: Training window size in days
            test_size_days: Test window size in days
            step_days: Step size for rolling window
            parameter_space: Parameter search space
            
        Returns:
            Dictionary containing optimization results and analysis
        """
        # Fetch full data
        data = self.data_fetcher.fetch_data(symbol, start_date, end_date)
        logger.info(f"Starting walk-forward optimization for {symbol}")
        
        # Generate walk-forward windows
        windows = self._generate_walk_forward_windows(
            data, window_size_days, test_size_days, step_days
        )
        logger.info(f"Generated {len(windows)} walk-forward windows")
        
        # Results storage
        window_results = []
        best_params_by_window = []
        
        for window in windows:
            logger.info(f"Processing window {window.window_id + 1}/{len(windows)}")
            
            # Get train and test data
            train_data = data[
                (data.index >= window.train_start) & 
                (data.index <= window.train_end)
            ].copy()
            
            test_data = data[
                (data.index >= window.test_start) & 
                (data.index <= window.test_end)
            ].copy()
            
            # Run grid search on training data
            train_results = self.grid_search_parameters(
                symbol,
                window.train_start.strftime('%Y-%m-%d'),
                window.train_end.strftime('%Y-%m-%d'),
                parameter_space,
                parallel=True
            )
            
            # Get best parameters from training
            best_params = train_results[0].parameters if train_results else None
            
            if best_params:
                # Test on out-of-sample data
                test_result = self._run_single_backtest(test_data, best_params)
                test_result.window_id = window.window_id
                
                # Calculate stability score
                stability_score = self._calculate_stability_score(
                    train_results[0], test_result
                )
                test_result.stability_score = stability_score
                
                # Store results
                window_results.append({
                    'window_id': window.window_id,
                    'train_result': train_results[0],
                    'test_result': test_result,
                    'best_params': best_params,
                    'stability_score': stability_score
                })
                
                best_params_by_window.append(best_params)
                
        # Analyze results across all windows
        analysis = self._analyze_walk_forward_results(window_results)
        
        return {
            'windows': windows,
            'window_results': window_results,
            'analysis': analysis,
            'best_stable_parameters': self._select_most_stable_parameters(window_results)
        }
        
    def _generate_walk_forward_windows(self,
                                     data: pd.DataFrame,
                                     window_size_days: int,
                                     test_size_days: int,
                                     step_days: int) -> List[WalkForwardWindow]:
        """Generate walk-forward optimization windows"""
        windows = []
        
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_date = start_date
        window_id = 0
        
        while current_date + timedelta(days=window_size_days + test_size_days) <= end_date:
            train_start = current_date
            train_end = current_date + timedelta(days=window_size_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_size_days)
            
            windows.append(WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                window_id=window_id
            ))
            
            current_date += timedelta(days=step_days)
            window_id += 1
            
        return windows
        
    def _calculate_stability_score(self, 
                                 train_result: OptimizationResult,
                                 test_result: OptimizationResult) -> float:
        """
        Calculate stability score comparing in-sample and out-of-sample performance.
        Higher scores indicate more stable parameters.
        """
        # Compare key metrics
        sharpe_diff = abs(train_result.sharpe_ratio - test_result.sharpe_ratio)
        return_diff = abs(train_result.total_return - test_result.total_return)
        dd_diff = abs(train_result.max_drawdown - test_result.max_drawdown)
        win_rate_diff = abs(train_result.win_rate - test_result.win_rate)
        
        # Normalize differences (lower is better)
        sharpe_score = 1 / (1 + sharpe_diff)
        return_score = 1 / (1 + return_diff * 10)  # Scale return difference
        dd_score = 1 / (1 + dd_diff * 10)
        win_rate_score = 1 / (1 + win_rate_diff * 10)
        
        # Weighted average (higher is more stable)
        stability_score = (
            0.4 * sharpe_score +
            0.3 * return_score +
            0.2 * dd_score +
            0.1 * win_rate_score
        )
        
        return stability_score
        
    def validate_optimization_results(self,
                                    symbol: str,
                                    parameters: Dict[str, Any],
                                    test_start_date: str,
                                    test_end_date: str,
                                    n_monte_carlo: int = 1000) -> Dict[str, Any]:
        """
        Validate optimization results with statistical significance testing.
        
        Args:
            symbol: Trading symbol
            parameters: Parameters to validate
            test_start_date: Start date for validation
            test_end_date: End date for validation
            n_monte_carlo: Number of Monte Carlo simulations
            
        Returns:
            Validation results including statistical tests
        """
        # Fetch test data
        test_data = self.data_fetcher.fetch_data(symbol, test_start_date, test_end_date)
        
        # Run backtest with optimized parameters
        result = self._run_single_backtest(test_data, parameters)
        
        if result.sharpe_ratio == -999:  # Insufficient trades
            return {
                'is_valid': False,
                'reason': 'Insufficient trades for validation',
                'result': result
            }
            
        # Statistical significance tests
        statistical_tests = {}
        
        # 1. T-test for returns being significantly positive
        backtester = IchimokuBacktester()
        trades, equity_curve = backtester.backtest(
            test_data,
            ichimoku_params=parameters['ichimoku_params'],
            signal_config=parameters['signal_config'],
            risk_params=parameters['risk_params']
        )
        
        returns = equity_curve['returns'].dropna()
        
        if len(returns) > 30:
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            statistical_tests['returns_t_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < self.significance_level and t_stat > 0
            }
        
        # 2. Monte Carlo permutation test
        mc_sharpe_ratios = []
        for _ in range(n_monte_carlo):
            # Shuffle returns to create random strategy
            shuffled_returns = returns.sample(frac=1).reset_index(drop=True)
            mc_sharpe = self._calculate_sharpe_ratio(shuffled_returns)
            mc_sharpe_ratios.append(mc_sharpe)
            
        # Calculate percentile of actual Sharpe ratio
        actual_sharpe = result.sharpe_ratio
        percentile = (np.array(mc_sharpe_ratios) < actual_sharpe).mean()
        
        statistical_tests['monte_carlo_test'] = {
            'actual_sharpe': actual_sharpe,
            'mc_mean_sharpe': np.mean(mc_sharpe_ratios),
            'mc_std_sharpe': np.std(mc_sharpe_ratios),
            'percentile': percentile,
            'is_significant': percentile > (1 - self.significance_level)
        }
        
        # 3. Check for parameter stability (rolling window performance)
        stability_check = self._check_parameter_stability(
            test_data, parameters, window_size=63  # 3 months
        )
        statistical_tests['stability_check'] = stability_check
        
        # Overall validation
        is_valid = (
            statistical_tests.get('returns_t_test', {}).get('is_significant', False) and
            statistical_tests.get('monte_carlo_test', {}).get('is_significant', False) and
            stability_check['is_stable']
        )
        
        return {
            'is_valid': is_valid,
            'result': result,
            'statistical_tests': statistical_tests,
            'parameters': parameters
        }
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio from returns series"""
        if len(returns) == 0:
            return 0
        return np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
        
    def _check_parameter_stability(self,
                                 data: pd.DataFrame,
                                 parameters: Dict[str, Any],
                                 window_size: int) -> Dict[str, Any]:
        """Check if parameters perform consistently across rolling windows"""
        rolling_sharpes = []
        rolling_returns = []
        
        for i in range(0, len(data) - window_size, window_size // 2):
            window_data = data.iloc[i:i + window_size].copy()
            
            try:
                result = self._run_single_backtest(window_data, parameters)
                if result.sharpe_ratio != -999:
                    rolling_sharpes.append(result.sharpe_ratio)
                    rolling_returns.append(result.total_return)
            except:
                continue
                
        if len(rolling_sharpes) < 3:
            return {'is_stable': False, 'reason': 'Insufficient rolling windows'}
            
        # Calculate coefficient of variation for stability
        sharpe_cv = np.std(rolling_sharpes) / (np.mean(rolling_sharpes) + 1e-10)
        return_cv = np.std(rolling_returns) / (np.mean(rolling_returns) + 1e-10)
        
        # Check if all windows are profitable
        all_profitable = all(r > 0 for r in rolling_returns)
        
        is_stable = sharpe_cv < 0.5 and return_cv < 0.5 and all_profitable
        
        return {
            'is_stable': is_stable,
            'sharpe_cv': sharpe_cv,
            'return_cv': return_cv,
            'all_profitable': all_profitable,
            'n_windows': len(rolling_sharpes),
            'mean_sharpe': np.mean(rolling_sharpes),
            'mean_return': np.mean(rolling_returns)
        }
        
    def select_best_parameters(self,
                             optimization_results: List[OptimizationResult],
                             min_stability_score: float = 0.7) -> Dict[str, Any]:
        """
        Select best parameters considering multiple criteria.
        
        Args:
            optimization_results: List of optimization results
            min_stability_score: Minimum stability score threshold
            
        Returns:
            Best parameters with justification
        """
        # Filter by stability score if available
        stable_results = [
            r for r in optimization_results 
            if r.stability_score >= min_stability_score
        ] if any(r.stability_score > 0 for r in optimization_results) else optimization_results
        
        if not stable_results:
            logger.warning("No results meet stability criteria, using all results")
            stable_results = optimization_results
            
        # Multi-criteria scoring
        scored_results = []
        
        for result in stable_results:
            # Calculate composite score
            score = (
                0.3 * self._normalize_metric(result.sharpe_ratio, 
                                            [r.sharpe_ratio for r in stable_results]) +
                0.2 * self._normalize_metric(result.total_return,
                                            [r.total_return for r in stable_results]) +
                0.2 * self._normalize_metric(1 - result.max_drawdown,
                                            [1 - r.max_drawdown for r in stable_results]) +
                0.15 * self._normalize_metric(result.win_rate,
                                             [r.win_rate for r in stable_results]) +
                0.15 * self._normalize_metric(result.stability_score,
                                             [r.stability_score for r in stable_results])
            )
            
            scored_results.append((score, result))
            
        # Sort by composite score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        best_result = scored_results[0][1]
        
        return {
            'parameters': best_result.parameters,
            'expected_performance': {
                'sharpe_ratio': best_result.sharpe_ratio,
                'total_return': best_result.total_return,
                'max_drawdown': best_result.max_drawdown,
                'win_rate': best_result.win_rate,
                'stability_score': best_result.stability_score
            },
            'selection_score': scored_results[0][0],
            'selection_method': 'multi-criteria composite scoring',
            'alternatives': [
                {
                    'parameters': r[1].parameters,
                    'score': r[0]
                } for r in scored_results[1:4]  # Top 3 alternatives
            ]
        }
        
    def _normalize_metric(self, value: float, all_values: List[float]) -> float:
        """Normalize metric to 0-1 range"""
        if not all_values:
            return 0
        min_val = min(all_values)
        max_val = max(all_values)
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
        
    def _analyze_walk_forward_results(self, window_results: List[Dict]) -> Dict[str, Any]:
        """Analyze walk-forward optimization results"""
        out_sample_sharpes = [w['test_result'].sharpe_ratio for w in window_results]
        out_sample_returns = [w['test_result'].total_return for w in window_results]
        stability_scores = [w['stability_score'] for w in window_results]
        
        # Parameter frequency analysis
        param_counts = {}
        for w in window_results:
            param_key = json.dumps(w['best_params'], sort_keys=True)
            param_counts[param_key] = param_counts.get(param_key, 0) + 1
            
        most_frequent_params = max(param_counts.items(), key=lambda x: x[1])
        
        return {
            'mean_out_sample_sharpe': np.mean(out_sample_sharpes),
            'std_out_sample_sharpe': np.std(out_sample_sharpes),
            'mean_out_sample_return': np.mean(out_sample_returns),
            'mean_stability_score': np.mean(stability_scores),
            'consistency_ratio': len([s for s in out_sample_sharpes if s > 0]) / len(out_sample_sharpes),
            'most_frequent_parameters': {
                'parameters': json.loads(most_frequent_params[0]),
                'frequency': most_frequent_params[1] / len(window_results)
            }
        }
        
    def _select_most_stable_parameters(self, window_results: List[Dict]) -> Dict[str, Any]:
        """Select parameters that perform most consistently across windows"""
        # Group results by parameters
        param_performance = {}
        
        for w in window_results:
            param_key = json.dumps(w['best_params'], sort_keys=True)
            if param_key not in param_performance:
                param_performance[param_key] = {
                    'sharpes': [],
                    'returns': [],
                    'stability_scores': []
                }
            
            param_performance[param_key]['sharpes'].append(w['test_result'].sharpe_ratio)
            param_performance[param_key]['returns'].append(w['test_result'].total_return)
            param_performance[param_key]['stability_scores'].append(w['stability_score'])
            
        # Score each parameter set
        best_params = None
        best_score = -np.inf
        
        for param_key, performance in param_performance.items():
            if len(performance['sharpes']) < 2:  # Need multiple occurrences
                continue
                
            # Calculate consistency metrics
            mean_sharpe = np.mean(performance['sharpes'])
            sharpe_consistency = 1 / (np.std(performance['sharpes']) + 0.1)
            mean_stability = np.mean(performance['stability_scores'])
            frequency = len(performance['sharpes']) / len(window_results)
            
            # Composite score
            score = (
                0.3 * mean_sharpe +
                0.3 * sharpe_consistency +
                0.2 * mean_stability +
                0.2 * frequency
            )
            
            if score > best_score:
                best_score = score
                best_params = json.loads(param_key)
                
        return best_params
        
    def _save_results_to_db(self, symbol: str, results: List[OptimizationResult]):
        """Save optimization results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in results[:100]:  # Save top 100 results
            cursor.execute('''
                INSERT INTO optimization_results (
                    optimization_date, symbol, parameters,
                    in_sample_sharpe, in_sample_return, in_sample_drawdown, in_sample_win_rate,
                    out_sample_sharpe, out_sample_return, out_sample_drawdown, out_sample_win_rate,
                    stability_score, is_robust, window_id, statistical_tests
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.optimization_date,
                symbol,
                json.dumps(result.parameters),
                result.in_sample_metrics.get('sharpe_ratio', 0),
                result.in_sample_metrics.get('total_return', 0),
                result.in_sample_metrics.get('max_drawdown', 0),
                result.in_sample_metrics.get('win_rate', 0),
                result.out_of_sample_metrics.get('sharpe_ratio', 0) if result.out_of_sample_metrics else 0,
                result.out_of_sample_metrics.get('total_return', 0) if result.out_of_sample_metrics else 0,
                result.out_of_sample_metrics.get('max_drawdown', 0) if result.out_of_sample_metrics else 0,
                result.out_of_sample_metrics.get('win_rate', 0) if result.out_of_sample_metrics else 0,
                result.stability_score,
                result.is_robust,
                result.window_id,
                json.dumps(result.statistical_significance) if result.statistical_significance else None
            ))
            
        conn.commit()
        conn.close()
        
    def generate_optimization_report(self, 
                                   symbol: str,
                                   optimization_type: str = 'grid_search') -> pd.DataFrame:
        """Generate optimization report from database results"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM optimization_results 
            WHERE symbol = ? 
            ORDER BY in_sample_sharpe DESC
            LIMIT 50
        '''
        
        df = pd.read_sql_query(query, conn, params=[symbol])
        conn.close()
        
        # Parse JSON fields
        df['parameters'] = df['parameters'].apply(json.loads)
        df['statistical_tests'] = df['statistical_tests'].apply(
            lambda x: json.loads(x) if x else {}
        )
        
        return df