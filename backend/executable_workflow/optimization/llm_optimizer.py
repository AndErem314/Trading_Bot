"""
LLM-Integrated Optimization System for Ichimoku Strategies

This module provides AI-powered optimization for Ichimoku trading strategies
using both Bayesian optimization and genetic algorithms, with LLM analysis
for intelligent parameter selection and strategy improvement suggestions.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from itertools import product
import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
from dotenv import load_dotenv

# Import optimization algorithms
try:
    from bayes_opt import BayesianOptimization
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    warnings.warn("bayesian-optimization not installed. Install with: pip install bayesian-optimization")

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    warnings.warn("DEAP not installed. Install with: pip install deap")

# Load environment variables
load_dotenv()

# Import LLM providers
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    strategy_name: str
    original_params: Dict[str, Any]
    optimized_params: Dict[str, Any]
    original_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    improvement_percentage: float
    optimization_method: str
    llm_suggestions: Optional[str] = None
    confidence_score: float = 0.0
    optimization_history: Optional[List[Dict]] = None
    best_iteration: Optional[int] = None


@dataclass
class IchimokuParameters:
    """Ichimoku indicator parameters"""
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_span_b_period: int = 52
    displacement: int = 26
    chikou_displacement: int = 26


@dataclass
class SignalCombination:
    """Signal combination parameters"""
    use_price_above_cloud: bool = True
    use_tenkan_kijun_cross: bool = True
    use_chikou_confirmation: bool = True
    use_cloud_twist: bool = False
    min_signal_strength: float = 0.6
    signal_weight_price: float = 0.4
    signal_weight_cross: float = 0.3
    signal_weight_chikou: float = 0.3


@dataclass
class RiskParameters:
    """Risk management parameters"""
    stop_loss_type: str = "atr"  # "fixed", "atr", "cloud"
    stop_loss_multiplier: float = 2.0
    take_profit_multiplier: float = 3.0
    position_size_pct: float = 2.0
    max_drawdown_pct: float = 20.0
    trailing_stop_enabled: bool = False
    trailing_stop_activation: float = 1.5
    trailing_stop_distance: float = 0.5


class LLMOptimizer:
    """
    LLM-Integrated Optimizer for Ichimoku trading strategies.
    
    Combines AI-powered analysis with algorithmic optimization methods
    to find optimal parameter combinations.
    """
    
    def __init__(self, 
                 provider: str = "auto",
                 gemini_api_key: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 optimization_method: str = "bayesian"):
        """
        Initialize the LLM Optimizer.
        
        Args:
            provider: LLM provider ("gemini", "openai", or "auto")
            gemini_api_key: Gemini API key (if None, reads from env)
            openai_api_key: OpenAI API key (if None, reads from env)
            optimization_method: "bayesian", "genetic", or "grid"
        """
        self.provider = provider
        self.optimization_method = optimization_method
        
        # Initialize LLM
        self._initialize_llm(gemini_api_key, openai_api_key)
        
        # Parameter bounds for optimization
        self.parameter_bounds = {
            # Ichimoku parameters
            'tenkan_period': (5, 20),
            'kijun_period': (20, 50),
            'senkou_span_b_period': (40, 120),
            'displacement': (20, 35),
            
            # Signal weights
            'signal_weight_price': (0.1, 0.6),
            'signal_weight_cross': (0.1, 0.6),
            'signal_weight_chikou': (0.1, 0.6),
            'min_signal_strength': (0.4, 0.8),
            
            # Risk parameters
            'stop_loss_multiplier': (1.0, 4.0),
            'take_profit_multiplier': (1.5, 5.0),
            'position_size_pct': (0.5, 5.0)
        }
        
        # Track optimization history
        self.optimization_history = []
        self.best_params = None
        self.best_score = -np.inf
    
    def _initialize_llm(self, gemini_api_key: Optional[str], openai_api_key: Optional[str]):
        """Initialize LLM provider"""
        self.llm_client = None
        self.active_provider = None
        
        if self.provider == "auto":
            # Try Gemini first
            if GEMINI_AVAILABLE:
                api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
                if api_key:
                    try:
                        genai.configure(api_key=api_key)
                        self.llm_client = genai.GenerativeModel('gemini-1.5-pro')
                        self.active_provider = "gemini"
                        logger.info("Using Gemini for LLM optimization")
                    except Exception as e:
                        logger.warning(f"Failed to initialize Gemini: {e}")
            
            # Try OpenAI if Gemini fails
            if not self.llm_client and OPENAI_AVAILABLE:
                api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
                if api_key:
                    try:
                        self.llm_client = OpenAI(api_key=api_key)
                        self.active_provider = "openai"
                        logger.info("Using OpenAI for LLM optimization")
                    except Exception as e:
                        logger.warning(f"Failed to initialize OpenAI: {e}")
        
        elif self.provider == "gemini" and GEMINI_AVAILABLE:
            api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.llm_client = genai.GenerativeModel('gemini-1.5-pro')
                self.active_provider = "gemini"
        
        elif self.provider == "openai" and OPENAI_AVAILABLE:
            api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
            if api_key:
                self.llm_client = OpenAI(api_key=api_key)
                self.active_provider = "openai"
        
        if not self.llm_client:
            logger.warning("No LLM provider available. Using rule-based optimization only.")
    
    def analyze_backtest_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze backtest results using LLM to identify optimization opportunities.
        
        Args:
            results: Backtest results including metrics, trades, and parameters
            
        Returns:
            Analysis dictionary with insights and suggestions
        """
        if not self.llm_client:
            return self._rule_based_analysis(results)
        
        # Prepare analysis prompt
        prompt = self._create_analysis_prompt(results)
        
        try:
            if self.active_provider == "gemini":
                response = self.llm_client.generate_content(prompt)
                analysis_text = response.text
            else:  # OpenAI
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert in algorithmic trading and Ichimoku strategy optimization."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                analysis_text = response.choices[0].message.content
            
            # Parse LLM response
            analysis = self._parse_llm_analysis(analysis_text)
            logger.info(f"LLM analysis completed using {self.active_provider}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._rule_based_analysis(results)
    
    def generate_optimization_suggestions(self, 
                                        current_params: Dict[str, Any],
                                        performance_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Generate parameter optimization suggestions based on current performance.
        
        Args:
            current_params: Current strategy parameters
            performance_metrics: Current performance metrics
            
        Returns:
            List of suggested parameter combinations
        """
        suggestions = []
        
        # LLM-based suggestions
        if self.llm_client:
            llm_suggestions = self._get_llm_parameter_suggestions(current_params, performance_metrics)
            suggestions.extend(llm_suggestions)
        
        # Rule-based suggestions
        rule_suggestions = self._get_rule_based_suggestions(current_params, performance_metrics)
        suggestions.extend(rule_suggestions)
        
        # Remove duplicates
        unique_suggestions = []
        seen = set()
        for sugg in suggestions:
            key = tuple(sorted(sugg.items()))
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(sugg)
        
        return unique_suggestions[:10]  # Return top 10 suggestions
    
    def run_parameter_optimization(self, 
                                 backtest_function: callable,
                                 initial_params: Optional[Dict[str, Any]] = None,
                                 n_iterations: int = 50) -> OptimizationResult:
        """
        Run parameter optimization using the specified method.
        
        Args:
            backtest_function: Function that takes parameters and returns performance score
            initial_params: Initial parameter values
            n_iterations: Number of optimization iterations
            
        Returns:
            OptimizationResult with optimized parameters and metrics
        """
        logger.info(f"Starting {self.optimization_method} optimization with {n_iterations} iterations")
        
        # Reset optimization tracking
        self.optimization_history = []
        self.best_params = initial_params or {}
        self.best_score = -np.inf
        
        # Run optimization based on method
        if self.optimization_method == "bayesian":
            if not BAYESIAN_OPT_AVAILABLE:
                logger.warning("Bayesian optimization not available, falling back to grid search")
                self.optimization_method = "grid"
            else:
                optimized_params = self._run_bayesian_optimization(backtest_function, n_iterations)
        
        elif self.optimization_method == "genetic":
            if not DEAP_AVAILABLE:
                logger.warning("Genetic algorithm not available, falling back to grid search")
                self.optimization_method = "grid"
            else:
                optimized_params = self._run_genetic_optimization(backtest_function, n_iterations)
        
        else:  # grid search
            optimized_params = self._run_grid_search(backtest_function, n_iterations)
        
        # Get final metrics
        original_metrics = backtest_function(self.best_params) if isinstance(self.best_params, dict) else {}
        optimized_metrics = backtest_function(optimized_params)
        
        # Calculate improvement
        original_score = self._calculate_optimization_score(original_metrics)
        optimized_score = self._calculate_optimization_score(optimized_metrics)
        improvement = ((optimized_score - original_score) / abs(original_score)) * 100 if original_score != 0 else 0
        
        # Get LLM analysis of results
        llm_suggestions = None
        if self.llm_client:
            analysis = self.analyze_backtest_results({
                'original_params': self.best_params,
                'optimized_params': optimized_params,
                'original_metrics': original_metrics,
                'optimized_metrics': optimized_metrics,
                'optimization_history': self.optimization_history
            })
            llm_suggestions = analysis.get('summary', '')
        
        return OptimizationResult(
            strategy_name="Ichimoku Strategy",
            original_params=self.best_params,
            optimized_params=optimized_params,
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            improvement_percentage=improvement,
            optimization_method=self.optimization_method,
            llm_suggestions=llm_suggestions,
            confidence_score=self._calculate_confidence_score(optimized_metrics),
            optimization_history=self.optimization_history,
            best_iteration=self._get_best_iteration()
        )
    
    def validate_optimized_parameters(self, 
                                    params: Dict[str, Any],
                                    validation_function: callable,
                                    n_runs: int = 5) -> Dict[str, Any]:
        """
        Validate optimized parameters with multiple runs or walk-forward analysis.
        
        Args:
            params: Parameters to validate
            validation_function: Function to run validation backtest
            n_runs: Number of validation runs
            
        Returns:
            Validation results including stability metrics
        """
        logger.info(f"Validating parameters with {n_runs} runs")
        
        validation_results = []
        for i in range(n_runs):
            result = validation_function(params, seed=i)
            validation_results.append(result)
        
        # Calculate stability metrics
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in validation_results]
        returns = [r.get('total_return', 0) for r in validation_results]
        max_drawdowns = [r.get('max_drawdown', 0) for r in validation_results]
        
        stability_metrics = {
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown': np.max(max_drawdowns),
            'consistency_score': self._calculate_consistency_score(validation_results),
            'validation_runs': n_runs,
            'all_results': validation_results
        }
        
        # Get LLM assessment
        if self.llm_client:
            assessment = self._get_llm_validation_assessment(params, stability_metrics)
            stability_metrics['llm_assessment'] = assessment
        
        return stability_metrics
    
    # Private optimization methods
    
    def _run_bayesian_optimization(self, backtest_function: callable, n_iterations: int) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        
        def objective_wrapper(**kwargs):
            """Wrapper to match BayesianOptimization interface"""
            score = backtest_function(kwargs)
            self._update_optimization_history(kwargs, score)
            return self._calculate_optimization_score(score) if isinstance(score, dict) else score
        
        # Create optimizer
        optimizer = BayesianOptimization(
            f=objective_wrapper,
            pbounds=self.parameter_bounds,
            random_state=42,
            verbose=2
        )
        
        # Run optimization
        optimizer.maximize(
            init_points=min(10, n_iterations // 5),
            n_iter=n_iterations - min(10, n_iterations // 5)
        )
        
        # Get best parameters
        best_params = optimizer.max['params']
        
        # Convert float parameters to appropriate types
        for key in ['tenkan_period', 'kijun_period', 'senkou_span_b_period', 'displacement']:
            if key in best_params:
                best_params[key] = int(round(best_params[key]))
        
        return best_params
    
    def _run_genetic_optimization(self, backtest_function: callable, n_iterations: int) -> Dict[str, Any]:
        """Run genetic algorithm optimization"""
        
        # Create fitness and individual classes
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Create toolbox
        toolbox = base.Toolbox()
        
        # Define attributes
        param_names = list(self.parameter_bounds.keys())
        for i, (param, (low, high)) in enumerate(self.parameter_bounds.items()):
            if param in ['tenkan_period', 'kijun_period', 'senkou_span_b_period', 'displacement']:
                toolbox.register(f"attr_{i}", random.randint, low, high)
            else:
                toolbox.register(f"attr_{i}", random.uniform, low, high)
        
        # Create individual and population
        toolbox.register("individual", tools.initCycle, creator.Individual,
                        [getattr(toolbox, f"attr_{i}") for i in range(len(param_names))], n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Define genetic operators
        def evaluate(individual):
            params = dict(zip(param_names, individual))
            score = backtest_function(params)
            self._update_optimization_history(params, score)
            return (self._calculate_optimization_score(score) if isinstance(score, dict) else score,)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run genetic algorithm
        population = toolbox.population(n=min(50, n_iterations))
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution
        for gen in range(n_iterations // len(population)):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Select next generation
            population[:] = toolbox.select(offspring, k=len(population))
        
        # Get best individual
        best_ind = tools.selBest(population, k=1)[0]
        best_params = dict(zip(param_names, best_ind))
        
        return best_params
    
    def _run_grid_search(self, backtest_function: callable, n_iterations: int) -> Dict[str, Any]:
        """Run grid search optimization"""
        
        # Create parameter grid
        param_grid = {}
        for param, (low, high) in self.parameter_bounds.items():
            if param in ['tenkan_period', 'kijun_period', 'senkou_span_b_period', 'displacement']:
                param_grid[param] = np.linspace(low, high, min(5, n_iterations // 10), dtype=int)
            else:
                param_grid[param] = np.linspace(low, high, min(5, n_iterations // 10))
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        # Sample if too many combinations
        if len(combinations) > n_iterations:
            combinations = random.sample(combinations, n_iterations)
        
        # Test each combination
        best_score = -np.inf
        best_params = {}
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            score = backtest_function(params)
            self._update_optimization_history(params, score)
            
            score_value = self._calculate_optimization_score(score) if isinstance(score, dict) else score
            
            if score_value > best_score:
                best_score = score_value
                best_params = params.copy()
        
        return best_params
    
    def _calculate_optimization_score(self, metrics: Union[Dict[str, float], float]) -> float:
        """Calculate optimization objective score"""
        if isinstance(metrics, (int, float)):
            return float(metrics)
        
        # Multi-objective optimization score
        sharpe = metrics.get('sharpe_ratio', 0)
        returns = metrics.get('total_return', 0)
        max_dd = metrics.get('max_drawdown', 0)
        win_rate = metrics.get('win_rate', 0)
        
        # Penalize negative returns heavily
        if returns < 0:
            return -1000
        
        # Weighted score
        score = (
            0.4 * sharpe +
            0.2 * (returns / 100) +  # Normalize returns
            0.2 * (1 - abs(max_dd) / 100) +  # Normalize drawdown
            0.2 * (win_rate / 100)  # Normalize win rate
        )
        
        return score
    
    def _update_optimization_history(self, params: Dict[str, Any], score: Any):
        """Update optimization history"""
        score_value = self._calculate_optimization_score(score) if isinstance(score, dict) else score
        
        self.optimization_history.append({
            'iteration': len(self.optimization_history),
            'params': params.copy(),
            'score': score_value,
            'metrics': score if isinstance(score, dict) else {'score': score}
        })
        
        if score_value > self.best_score:
            self.best_score = score_value
            self.best_params = params.copy()
    
    def _get_best_iteration(self) -> Optional[int]:
        """Get iteration number with best score"""
        if not self.optimization_history:
            return None
        
        best_idx = max(range(len(self.optimization_history)), 
                      key=lambda i: self.optimization_history[i]['score'])
        return self.optimization_history[best_idx]['iteration']
    
    def _calculate_confidence_score(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence score for optimization results"""
        confidence = 50.0  # Base confidence
        
        # Adjust based on metrics
        if metrics.get('sharpe_ratio', 0) > 1.5:
            confidence += 15
        if metrics.get('total_return', 0) > 20:
            confidence += 10
        if metrics.get('max_drawdown', 0) > -15:
            confidence += 10
        if metrics.get('win_rate', 0) > 55:
            confidence += 10
        if metrics.get('total_trades', 0) > 100:
            confidence += 5
        
        return min(confidence, 95.0)
    
    def _calculate_consistency_score(self, results: List[Dict[str, float]]) -> float:
        """Calculate consistency score across validation runs"""
        if len(results) < 2:
            return 0.0
        
        sharpe_cv = np.std([r.get('sharpe_ratio', 0) for r in results]) / (np.mean([r.get('sharpe_ratio', 0) for r in results]) + 0.001)
        return_cv = np.std([r.get('total_return', 0) for r in results]) / (np.mean([r.get('total_return', 0) for r in results]) + 0.001)
        
        # Lower CV is better (more consistent)
        consistency = 100 * (1 - min(sharpe_cv + return_cv, 1.0))
        
        return consistency
    
    # LLM prompt creation methods
    
    def _create_analysis_prompt(self, results: Dict[str, Any]) -> str:
        """Create detailed prompt for LLM analysis"""
        
        prompt = f"""
You are an expert in Ichimoku trading strategy optimization. Analyze these backtest results and provide specific optimization recommendations.

CURRENT STRATEGY PERFORMANCE:
{json.dumps(results.get('metrics', {}), indent=2)}

CURRENT ICHIMOKU PARAMETERS:
{json.dumps(results.get('parameters', {}), indent=2)}

TRADE STATISTICS:
- Total Trades: {results.get('metrics', {}).get('total_trades', 0)}
- Win Rate: {results.get('metrics', {}).get('win_rate', 0):.2f}%
- Average Win: ${results.get('metrics', {}).get('avg_win', 0):.2f}
- Average Loss: ${results.get('metrics', {}).get('avg_loss', 0):.2f}
- Profit Factor: {results.get('metrics', {}).get('profit_factor', 0):.2f}

OPTIMIZATION OBJECTIVE:
1. Maximize Sharpe Ratio (risk-adjusted returns)
2. Minimize Maximum Drawdown
3. Maintain Win Rate above 50%
4. Ensure sufficient trade frequency

Please provide your analysis in JSON format:
{{
    "parameter_adjustments": {{
        "tenkan_period": {{
            "current": 9,
            "suggested": 12,
            "reasoning": "Explanation"
        }},
        // ... other parameters
    }},
    "signal_optimization": {{
        "add_signals": ["signal_name"],
        "remove_signals": ["signal_name"],
        "adjust_weights": {{
            "signal_name": 0.4
        }}
    }},
    "risk_management": {{
        "stop_loss_suggestion": "Specific suggestion",
        "position_sizing": "Specific suggestion"
    }},
    "market_conditions": {{
        "best_performing": "Description of ideal conditions",
        "avoid_conditions": "Description of conditions to avoid"
    }},
    "confidence_level": 85,
    "expected_improvement": {{
        "sharpe_ratio": "+0.3",
        "max_drawdown": "-5%",
        "win_rate": "+5%"
    }},
    "summary": "Brief summary of key recommendations"
}}
"""
        return prompt
    
    def _parse_llm_analysis(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback parsing
                return {
                    'summary': response_text,
                    'confidence_level': 70,
                    'parse_error': True
                }
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                'summary': response_text[:500],
                'confidence_level': 50,
                'parse_error': True,
                'error': str(e)
            }
    
    def _get_llm_parameter_suggestions(self, 
                                     current_params: Dict[str, Any],
                                     metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get parameter suggestions from LLM"""
        
        prompt = f"""
Based on these Ichimoku strategy metrics, suggest 3 different parameter combinations to test:

CURRENT PARAMETERS:
{json.dumps(current_params, indent=2)}

CURRENT PERFORMANCE:
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
- Total Return: {metrics.get('total_return', 0):.2f}%
- Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
- Win Rate: {metrics.get('win_rate', 0):.2f}%

PARAMETER BOUNDS:
{json.dumps(self.parameter_bounds, indent=2)}

Provide exactly 3 parameter sets in JSON format:
[
    {{"tenkan_period": 12, "kijun_period": 26, ...}},
    {{"tenkan_period": 9, "kijun_period": 30, ...}},
    {{"tenkan_period": 15, "kijun_period": 35, ...}}
]

Focus on:
1. One conservative set (reduce risk)
2. One aggressive set (increase returns)
3. One balanced set (optimize Sharpe ratio)
"""
        
        try:
            if self.active_provider == "gemini":
                response = self.llm_client.generate_content(prompt)
                response_text = response.text
            else:  # OpenAI
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an Ichimoku trading expert. Provide only valid JSON responses."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5
                )
                response_text = response.choices[0].message.content
            
            # Parse response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                suggestions = json.loads(response_text[json_start:json_end])
                
                # Validate and clean suggestions
                valid_suggestions = []
                for sugg in suggestions:
                    if isinstance(sugg, dict):
                        # Ensure all required parameters are present
                        cleaned_sugg = current_params.copy()
                        cleaned_sugg.update(sugg)
                        
                        # Validate bounds
                        for param, value in cleaned_sugg.items():
                            if param in self.parameter_bounds:
                                low, high = self.parameter_bounds[param]
                                cleaned_sugg[param] = max(low, min(high, value))
                        
                        valid_suggestions.append(cleaned_sugg)
                
                return valid_suggestions[:3]
                
        except Exception as e:
            logger.error(f"Failed to get LLM suggestions: {e}")
        
        return []
    
    def _get_rule_based_suggestions(self, 
                                  current_params: Dict[str, Any],
                                  metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate rule-based parameter suggestions"""
        suggestions = []
        
        # Base parameters
        base = current_params.copy()
        
        # Suggestion 1: Reduce risk if drawdown is high
        if metrics.get('max_drawdown', 0) < -15:
            risk_reduced = base.copy()
            risk_reduced['stop_loss_multiplier'] = max(1.0, base.get('stop_loss_multiplier', 2.0) - 0.5)
            risk_reduced['position_size_pct'] = max(0.5, base.get('position_size_pct', 2.0) - 0.5)
            risk_reduced['min_signal_strength'] = min(0.8, base.get('min_signal_strength', 0.6) + 0.1)
            suggestions.append(risk_reduced)
        
        # Suggestion 2: Increase aggression if Sharpe is low
        if metrics.get('sharpe_ratio', 0) < 1.0:
            aggressive = base.copy()
            aggressive['take_profit_multiplier'] = min(5.0, base.get('take_profit_multiplier', 3.0) + 0.5)
            aggressive['min_signal_strength'] = max(0.4, base.get('min_signal_strength', 0.6) - 0.1)
            suggestions.append(aggressive)
        
        # Suggestion 3: Adjust Ichimoku periods based on win rate
        if metrics.get('win_rate', 0) < 45:
            period_adjusted = base.copy()
            # Try slower periods for better signals
            period_adjusted['tenkan_period'] = min(20, base.get('tenkan_period', 9) + 2)
            period_adjusted['kijun_period'] = min(50, base.get('kijun_period', 26) + 4)
            suggestions.append(period_adjusted)
        elif metrics.get('win_rate', 0) > 65:
            # Try faster periods for more trades
            period_adjusted = base.copy()
            period_adjusted['tenkan_period'] = max(5, base.get('tenkan_period', 9) - 2)
            period_adjusted['kijun_period'] = max(20, base.get('kijun_period', 26) - 4)
            suggestions.append(period_adjusted)
        
        # Suggestion 4: Optimize signal weights
        if len(suggestions) < 3:
            weight_optimized = base.copy()
            total_weight = (base.get('signal_weight_price', 0.4) + 
                          base.get('signal_weight_cross', 0.3) + 
                          base.get('signal_weight_chikou', 0.3))
            
            # Normalize and adjust based on performance
            if metrics.get('win_rate', 0) > 55:
                # Current weights are working, make small adjustments
                weight_optimized['signal_weight_price'] = 0.35
                weight_optimized['signal_weight_cross'] = 0.35
                weight_optimized['signal_weight_chikou'] = 0.30
            else:
                # Try different weight distribution
                weight_optimized['signal_weight_price'] = 0.50
                weight_optimized['signal_weight_cross'] = 0.30
                weight_optimized['signal_weight_chikou'] = 0.20
            
            suggestions.append(weight_optimized)
        
        return suggestions
    
    def _get_llm_validation_assessment(self, 
                                     params: Dict[str, Any],
                                     stability_metrics: Dict[str, Any]) -> str:
        """Get LLM assessment of validation results"""
        
        prompt = f"""
Assess the stability and reliability of these optimized Ichimoku parameters based on validation results:

OPTIMIZED PARAMETERS:
{json.dumps(params, indent=2)}

VALIDATION RESULTS ({stability_metrics['validation_runs']} runs):
- Mean Sharpe Ratio: {stability_metrics['mean_sharpe']:.3f} (std: {stability_metrics['std_sharpe']:.3f})
- Mean Return: {stability_metrics['mean_return']:.2f}% (std: {stability_metrics['std_return']:.2f}%)
- Mean Max Drawdown: {stability_metrics['mean_max_drawdown']:.2f}%
- Worst Drawdown: {stability_metrics['worst_drawdown']:.2f}%
- Consistency Score: {stability_metrics['consistency_score']:.1f}/100

Provide a brief assessment of:
1. Parameter stability and robustness
2. Risk of overfitting
3. Recommended confidence level for live trading
4. Any concerns or adjustments needed
"""
        
        try:
            if self.active_provider == "gemini":
                response = self.llm_client.generate_content(prompt)
                return response.text
            else:  # OpenAI
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a quantitative trading expert assessing strategy robustness."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to get LLM validation assessment: {e}")
            return "Unable to generate LLM assessment due to error."
    
    def _rule_based_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based analysis when LLM is not available"""
        
        metrics = results.get('metrics', {})
        analysis = {
            'summary': "",
            'recommendations': [],
            'confidence_level': 60
        }
        
        # Analyze Sharpe ratio
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < 0.5:
            analysis['recommendations'].append("Sharpe ratio is low. Consider reducing position size or tightening stop losses.")
        elif sharpe > 1.5:
            analysis['recommendations'].append("Good Sharpe ratio. Current risk-adjusted returns are strong.")
        
        # Analyze drawdown
        max_dd = metrics.get('max_drawdown', 0)
        if max_dd < -20:
            analysis['recommendations'].append("High drawdown detected. Implement stricter risk management.")
        
        # Analyze win rate
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 40:
            analysis['recommendations'].append("Low win rate. Consider adjusting entry signals or Ichimoku periods.")
        elif win_rate > 60:
            analysis['recommendations'].append("High win rate achieved. Ensure sufficient profit per trade.")
        
        # Create summary
        analysis['summary'] = f"Rule-based analysis: {len(analysis['recommendations'])} recommendations generated. "
        analysis['summary'] += f"Overall performance is {'strong' if sharpe > 1 else 'needs improvement'}."
        
        return analysis


# Prompt templates for different optimization scenarios
OPTIMIZATION_PROMPTS = {
    'conservative': """
    Focus on reducing risk while maintaining positive returns:
    - Prioritize lower drawdowns over higher returns
    - Increase signal confirmation requirements
    - Suggest tighter stop losses
    - Recommend smaller position sizes
    """,
    
    'aggressive': """
    Focus on maximizing returns with acceptable risk:
    - Allow higher drawdowns for better returns
    - Suggest parameters for catching larger moves
    - Optimize for higher profit targets
    - Balance trade frequency with quality
    """,
    
    'balanced': """
    Focus on optimal risk-adjusted returns:
    - Maximize Sharpe ratio
    - Balance win rate with profit factor
    - Optimize signal weights for consistency
    - Ensure stable performance across market conditions
    """,
    
    'market_adaptive': """
    Create parameters that adapt to different market conditions:
    - Identify trending vs ranging market parameters
    - Suggest volatility-based adjustments
    - Recommend dynamic position sizing
    - Optimize for various timeframes
    """
}


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("LLM Optimizer for Ichimoku Strategies")
    print("-" * 50)
    print("Features:")
    print("- AI-powered parameter analysis")
    print("- Bayesian optimization")
    print("- Genetic algorithms")
    print("- Multi-objective optimization")
    print("- Automated validation")
    print("\nSupported LLMs: Gemini and OpenAI")
    print("Optimization methods: Bayesian, Genetic, Grid Search")