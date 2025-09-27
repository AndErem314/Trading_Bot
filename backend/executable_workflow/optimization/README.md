# Optimization Module

This module provides comprehensive optimization tools for Ichimoku trading strategies, including both automated parameter testing and LLM-guided optimization.

## Components

### 1. ParameterOptimizer

The `ParameterOptimizer` class provides automated parameter testing with statistical validation and overfitting mitigation.

#### Features:

- **Grid Search Optimization**: Exhaustive search across parameter space
- **Walk-Forward Optimization**: Rolling window validation
- **Statistical Significance Testing**: T-tests and Monte Carlo permutation tests
- **Overfitting Mitigation**: Cross-validation and stability metrics
- **Parallel Processing**: Multi-core support for faster optimization
- **Results Database**: SQLite storage for optimization results

#### Usage Example:

```python
from backend.executable_workflow.optimization import ParameterOptimizer, ParameterSpace

# Initialize optimizer
optimizer = ParameterOptimizer(
    db_path="optimization_results.db",
    min_sample_trades=30,
    significance_level=0.05,
    n_jobs=-1  # Use all CPUs
)

# Define parameter search space
param_space = ParameterSpace(
    tenkan_periods=[7, 8, 9, 10, 11, 12],
    kijun_periods=[22, 23, 24, 25, 26],
    senkou_b_periods=[48, 50, 52, 54, 56],
    signal_combinations=[
        ['cloud_breakout'],
        ['cloud_breakout', 'tk_cross'],
        ['cloud_breakout', 'tk_cross', 'price_momentum']
    ],
    stop_loss_percent=[0.02, 0.025, 0.03],
    take_profit_percent=[0.03, 0.04, 0.05]
)

# Run grid search
results = optimizer.grid_search_parameters(
    symbol="BTC-USD",
    start_date="2023-01-01",
    end_date="2024-01-01",
    parameter_space=param_space,
    parallel=True
)

# Select best parameters
best_params = optimizer.select_best_parameters(results)
```

#### Walk-Forward Optimization:

```python
# Run walk-forward optimization
wf_results = optimizer.walk_forward_optimization(
    symbol="BTC-USD",
    start_date="2022-01-01",
    end_date="2024-01-01",
    window_size_days=252,  # 1 year training
    test_size_days=63,     # 3 months testing
    step_days=21,          # 1 month step
    parameter_space=param_space
)
```

#### Statistical Validation:

```python
# Validate parameters on out-of-sample data
validation_results = optimizer.validate_optimization_results(
    symbol="BTC-USD",
    parameters=best_params['parameters'],
    test_start_date="2024-01-01",
    test_end_date="2024-06-01",
    n_monte_carlo=1000
)
```

### 2. LLMOptimizer

The `LLMOptimizer` class provides intelligent parameter optimization using Large Language Models.

#### Features:

- **LLM Integration**: Support for Google Gemini and OpenAI APIs
- **Intelligent Analysis**: LLM-guided parameter suggestions
- **Multi-Method Optimization**: Bayesian, genetic algorithm, and grid search
- **Market Adaptation**: Different optimization profiles (conservative, aggressive, balanced)

#### Usage Example:

```python
from backend.executable_workflow.optimization import LLMOptimizer

# Initialize LLM optimizer
llm_optimizer = LLMOptimizer(
    llm_provider="gemini",  # or "openai"
    optimization_profile="balanced",
    verbose=True
)

# Analyze backtest results
analysis = llm_optimizer.analyze_results(backtest_results)

# Generate parameter suggestions
suggestions = llm_optimizer.suggest_parameters(
    current_params=current_params,
    backtest_results=backtest_results,
    market_data=market_data
)

# Run optimization
optimized_params = llm_optimizer.optimize(
    data=market_data,
    current_params=current_params,
    method="bayesian",
    n_iterations=50
)
```

## Parameter Space Configuration

The `ParameterSpace` class defines the search space for optimization:

```python
@dataclass
class ParameterSpace:
    tenkan_periods: List[int]       # Default: [7-12]
    kijun_periods: List[int]        # Default: [22-26]
    senkou_b_periods: List[int]     # Default: [48-56]
    signal_combinations: List[List[str]]
    stop_loss_percent: List[float]
    take_profit_percent: List[float]
```

## Optimization Results

Results are stored in an SQLite database with the following structure:

- `optimization_date`: Timestamp of optimization
- `symbol`: Trading symbol
- `parameters`: JSON-encoded parameter set
- `in_sample_sharpe/return/drawdown/win_rate`: In-sample metrics
- `out_sample_sharpe/return/drawdown/win_rate`: Out-of-sample metrics
- `stability_score`: Parameter stability metric (0-1)
- `is_robust`: Boolean indicating statistical robustness
- `statistical_tests`: JSON-encoded statistical test results

## Best Practices

1. **Minimum Sample Size**: Ensure at least 30 trades per parameter set for statistical validity
2. **Walk-Forward Validation**: Always validate parameters on out-of-sample data
3. **Multiple Metrics**: Consider Sharpe ratio, drawdown, and stability together
4. **Statistical Testing**: Use significance tests to avoid data mining bias
5. **Parameter Stability**: Prefer parameters that work consistently across different periods

## Running the Examples

```bash
# Run the parameter optimizer example
python backend/executable_workflow/optimization/parameter_optimizer_example.py

# Run the LLM optimizer example
python backend/executable_workflow/optimization/optimization_example.py
```

## Requirements

- numpy
- pandas
- scipy
- scikit-learn
- sqlite3
- matplotlib
- seaborn
- google-generativeai (for Gemini)
- openai (for OpenAI)

## Output

Optimization results are saved to:
- SQLite database: `optimization_results.db`
- CSV reports: `frontend/backtest_results/optimization_report.csv`
- Visualizations: `frontend/backtest_results/walk_forward/`
- HTML reports: `frontend/backtest_results/optimized/`