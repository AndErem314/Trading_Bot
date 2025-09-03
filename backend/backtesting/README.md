# Crypto Trading Backtesting System

This backtesting system is designed specifically for cryptocurrency trading strategies, supporting individual strategy optimization and combined strategy testing on the same trading pairs.

## Overview

The system provides:
- **Individual Strategy Backtesting**: Test and optimize single strategies
- **Combined Strategy Testing**: Combine multiple strategies on the same trading pair
- **Parameter Optimization**: Grid search, random search, and Bayesian optimization
- **AI-Powered Analysis**: Integration with Gemini and OpenAI for strategy insights
- **Crypto-Specific Metrics**: Tailored for 24/7 crypto markets

## Directory Structure

```
backend/backtesting/
├── core/                      # Core backtesting components
│   ├── engine.py             # Main backtesting engine
│   ├── metrics.py            # Performance metrics (crypto-optimized)
│   └── strategy_combiner.py  # Strategy combination logic
├── optimization/              # Parameter optimization
│   └── optimizer.py          # Grid search, Bayesian optimization
├── analysis/                  # Analysis tools
│   └── llm_analyzer.py       # AI analysis (Gemini/OpenAI)
├── scripts/                   # Executable scripts
│   ├── run_single_strategy.py     # Single strategy backtesting
│   └── run_combined_strategies.py # Combined strategy backtesting
├── utils/                     # Utility modules
│   ├── data_loader.py        # Load data from SQLite
│   └── strategy_loader.py    # Load and initialize strategies
├── config/                    # Configuration files
│   ├── backtest_config.yaml      # Main configuration
│   └── optimization_ranges.yaml  # Parameter optimization ranges
├── data/                      # Data storage
│   ├── historical/           # Historical price data
│   ├── results/              # Backtesting results
│   └── cache/                # Cached calculations
└── .env.example              # Example environment variables
```

## Setup

### 1. Install Dependencies

```bash
pip install pandas numpy pyyaml tqdm
pip install google-generativeai  # For Gemini (optional)
pip install openai               # For OpenAI (optional)
pip install scikit-optimize      # For Bayesian optimization (optional)
```

### 2. Configure API Keys (Optional)

For AI-powered analysis:

```bash
cd backend/backtesting
cp .env.example .env
# Edit .env and add your API keys:
# GEMINI_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
```

## Usage Examples

### Single Strategy Backtesting

#### Basic backtest with default parameters:
```bash
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands
```

#### Optimize strategy parameters:
```bash
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --optimize
```

#### Different optimization methods:
```bash
# Grid search (default)
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --optimize --method grid_search

# Random search
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --optimize --method random_search

# Bayesian optimization
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --optimize --method bayesian
```

#### With AI analysis:
```bash
# Auto-select available LLM
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --analyze

# Use specific LLM
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --analyze --llm gemini
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --analyze --llm openai
```

#### Custom parameters:
```bash
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --params '{"bb_length": 25, "bb_std": 2.5}'
```

### Combined Strategy Testing

#### Basic combination (equal weights):
```bash
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence macd_momentum
```

#### Different trading pairs:
```bash
# Bitcoin
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence --symbol BTC/USDT

# Ethereum
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence --symbol ETH/USDT

# Solana
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence --symbol SOL/USDT
```

#### Combination methods:
```bash
# Majority vote (at least half strategies must agree)
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence macd_momentum --method majority_vote

# Weighted average (default)
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence macd_momentum --method weighted_average

# Unanimous (all must agree)
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence macd_momentum --method unanimous

# Any signal (trade when any strategy signals)
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence macd_momentum --method any_signal

# Score based (composite scoring)
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence macd_momentum --method score_based
```

#### Compare all combination methods:
```bash
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence macd_momentum --compare
```

#### Optimize combination weights:
```bash
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence --optimize-weights
```

#### Custom weights:
```bash
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence --weights '{"bollinger_bands": 0.7, "rsi_divergence": 0.3}'
```

#### Use default parameters (no optimization):
```bash
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence --use-defaults
```

## Available Strategies

1. **bollinger_bands** - Bollinger Bands Mean Reversion
2. **rsi_divergence** - RSI Momentum Divergence
3. **macd_momentum** - MACD Momentum Crossover
4. **volatility_breakout_short** - Volatility Breakout Short
5. **ichimoku_cloud** - Ichimoku Cloud Breakout
6. **parabolic_sar** - Parabolic SAR Trend Following
7. **fibonacci_retracement** - Fibonacci Retracement Support/Resistance
8. **gaussian_channel** - Gaussian Channel Breakout/Mean Reversion

## Performance Metrics (Crypto-Optimized)

### Returns & Performance
- **Total Return**: Absolute and percentage returns
- **Buy & Hold Return**: Comparison with simple holding
- **Strategy vs B&H**: Outperformance over buy-and-hold
- **Time in Market**: Percentage of time with active positions
- **Average Return per Trade**: When positions are active

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns (adjusted for 24/7 markets)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return to maximum drawdown ratio
- **Volatility**: Annualized volatility
- **Risk/Reward Ratio**: Average winning vs losing return
- **VaR (95%)**: Value at Risk at 95% confidence
- **CVaR (95%)**: Conditional Value at Risk
- **Tail Ratio**: Extreme gains vs extreme losses

### Drawdown Analysis
- **Max Drawdown**: Largest peak-to-trough decline
- **Average Drawdown**: Average of all drawdowns
- **Max Drawdown Duration**: Longest drawdown period
- **Recovery Factor**: Total return divided by max drawdown
- **Ulcer Index**: Root mean square of drawdowns

### Trade Statistics
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Expectancy**: Expected profit per trade
- **Average Win/Loss**: Average winning and losing trade amounts
- **Payoff Ratio**: Average win size / average loss size
- **Max Consecutive Wins/Losses**: Longest winning/losing streaks
- **Trade Duration**: Min/max/average trade duration in hours
- **Trades per Week**: Trading frequency

## Configuration Files

### backtest_config.yaml
Main configuration for backtesting parameters:
- Initial capital
- Commission and slippage
- Data settings (symbols, timeframes, date ranges)
- Risk management rules
- Performance criteria
- Output settings

### optimization_ranges.yaml
Parameter optimization ranges for each strategy:
- Minimum and maximum values
- Step sizes for grid search
- Parameter types (int, float, bool, list)

## Output Files

Results are saved in `backend/backtesting/data/results/`:

### Single Strategy Results
```
strategy_name/
├── strategy_name_metrics.json        # Performance metrics
├── strategy_name_trades.csv          # Trade history
├── strategy_name_equity_curve.csv    # Equity curve data
├── strategy_name_summary.txt         # Human-readable summary
├── strategy_name_best_params.json    # Optimized parameters
├── strategy_name_optimization_history.json  # All tested parameters
└── strategy_name_llm_analysis.md    # AI analysis report
```

### Combined Strategy Results
```
combined_strategy1_strategy2_method/
├── metrics.json                      # Combined performance metrics
├── trades.csv                        # All trades from combination
├── equity_curve.csv                  # Combined equity curve
└── summary.txt                       # Performance summary with agreement stats
```

## Tips for Best Results

1. **Data Requirements**: Ensure you have sufficient historical data (at least 1 year recommended)

2. **Optimization**: 
   - Start with grid search for initial exploration
   - Use Bayesian optimization for fine-tuning
   - Always validate on out-of-sample data

3. **Strategy Combination**:
   - Combine strategies with low correlation
   - Test different combination methods
   - Use `--compare` to find the best method

4. **Risk Management**:
   - Set appropriate position sizes in config
   - Monitor maximum drawdown limits
   - Consider market conditions

5. **AI Analysis**:
   - Use LLM analysis to understand parameter sensitivities
   - Review market condition recommendations
   - Validate AI suggestions with backtesting

## Example Workflow

1. **Optimize individual strategies**:
```bash
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --optimize --analyze
python backend/backtesting/scripts/run_single_strategy.py rsi_divergence --optimize --analyze
python backend/backtesting/scripts/run_single_strategy.py macd_momentum --optimize --analyze
```

2. **Test combinations**:
```bash
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence macd_momentum --compare --symbol BTC/USDT
```

3. **Optimize combination weights**:
```bash
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence macd_momentum --optimize-weights
```

4. **Run final backtest with optimized settings**:
```bash
python backend/backtesting/scripts/run_combined_strategies.py bollinger_bands rsi_divergence macd_momentum --method weighted_average --weights '{"bollinger_bands": 0.5, "rsi_divergence": 0.3, "macd_momentum": 0.2}'
```

## Troubleshooting

### No data found
- Check that SQLite database files exist in the `data/` directory
- Verify symbol format matches database (e.g., BTCUSDT.db)
- Ensure date range contains data

### Import errors
- Install missing dependencies
- Check Python path includes the project root

### API key errors
- Set environment variables in `.env` file
- Verify API keys are valid and have credits

### Memory issues with large datasets
- Reduce date range
- Use fewer parameter combinations
- Process strategies sequentially instead of parallel

## Future Enhancements

- Visualization module for charts and reports
- Real-time paper trading integration
- Multi-timeframe analysis
- More sophisticated position sizing algorithms
- Machine learning-based parameter optimization
- Web interface for results visualization
