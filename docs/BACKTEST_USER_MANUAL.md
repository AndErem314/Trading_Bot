# Trading Bot Backtesting User Manual

## Table of Contents
1. [Running a Single Strategy](#running-single-strategy)
2. [Understanding the Results](#understanding-results)  
3. [Implementing Optimizations](#implementing-optimizations)
4. [Best Workflow Approach](#best-workflow)
5. [Complete Workflow Example](#complete-workflow)

## 1. Running a Single Strategy {#running-single-strategy}

### Available Strategies
- `bollinger_bands` - Bollinger Bands Mean Reversion
- `rsi_divergence` - RSI Momentum Divergence  
- `macd_momentum` - MACD Momentum Crossover
- `volatility_breakout_short` - Volatility Breakout Short
- `ichimoku_cloud` - Ichimoku Cloud Breakout
- `parabolic_sar` - Parabolic SAR Trend Following
- `fibonacci_retracement` - Fibonacci Retracement Support/Resistance
- `gaussian_channel` - Gaussian Channel Breakout/Mean Reversion

### Basic Command
```bash
python3 backend/backtesting/scripts/run_single_strategy.py rsi_divergence --analyze --llm gemini
```

### With Different Timeframes
The default configuration uses BTC/USDT on 4h timeframe. Available timeframes are: `1h`, `4h`, `1d`

```bash
# For 1 day timeframe
python3 backend/backtesting/scripts/run_single_strategy.py bollinger_bands \
  --params '{"timeframe": "1d"}' \
  --analyze --llm gemini

# For 1 hour timeframe  
python3 backend/backtesting/scripts/run_single_strategy.py macd_momentum \
  --params '{"timeframe": "1h"}' \
  --analyze --llm gemini
```

## 2. Understanding the Results {#understanding-results}

### Files Automatically Generated
After running the command, you'll find these files in `backend/backtesting/data/results/{strategy_name}/`:

1. **`{strategy_name}_backtest_results.json`** - Complete backtest metrics
2. **`{strategy_name}_trades.csv`** - All trades with entry/exit details  
3. **`{strategy_name}_equity_curve.png`** - Visual chart of strategy performance
4. **`{strategy_name}_llm_analysis.md`** - AI recommendations for optimization

### Reading the AI Analysis
To view the optimization recommendations:
```bash
cat backend/backtesting/data/results/rsi_divergence/rsi_divergence_llm_analysis.md
```

Look for the **"Suggested Parameter Adjustments"** section - these are the optimized values recommended by the AI.

## 3. Implementing Optimizations {#implementing-optimizations}

### Method 1: Update optimized_strategies.yaml (Recommended)

After getting AI recommendations, update the central configuration:

```bash
# Open the optimized strategies configuration
nano backend/backtesting/config/optimized_strategies.yaml
```

Update your strategy section with the recommended parameters:
```yaml
rsi_divergence:
  description: "RSI Momentum Divergence - Optimized for BTC/USDT 4h"
  optimized_parameters:
    default:
      rsi_length: 16  # AI recommended: increased from 14
      rsi_sma_fast: 5
      rsi_sma_slow: 12
      rsi_oversold: 25  # AI recommended: lowered from 30
      rsi_overbought: 75  # AI recommended: raised from 70
      momentum_lookback: 7
      divergence_lookback: 25
```

### Method 2: Direct Parameter Input

For quick testing without updating the configuration:

```bash
python3 backend/backtesting/scripts/run_single_strategy.py rsi_divergence \
  --params '{"rsi_length": 16, "rsi_oversold": 25, "rsi_overbought": 75}' \
  --analyze
```

## 4. Best Workflow Approach {#best-workflow}

### Recommended Sequence:

#### 1. Initial Baseline Test
First, run the strategy with current parameters from `optimized_strategies.yaml`:
```bash
python3 backend/backtesting/scripts/run_single_strategy.py bollinger_bands --analyze --llm gemini
```
This gives you a baseline performance and AI recommendations.

#### 2. Review and Update Parameters
- Review the AI recommendations in the report
- The report shows both "Current Parameters Used" and "Suggested Parameter Adjustments"
- Manually update `optimized_strategies.yaml` with the suggested parameters

#### 3. Test Updated Parameters
Run again with the updated parameters:
```bash
python3 backend/backtesting/scripts/run_single_strategy.py bollinger_bands --analyze --llm gemini
```

#### 4. Optional: Run Optimization
If you want to explore the parameter space more thoroughly:
```bash
# Grid search (tests all combinations)
python3 backend/backtesting/scripts/run_single_strategy.py bollinger_bands --optimize --method grid_search

# Or Bayesian optimization (smarter search)
python3 backend/backtesting/scripts/run_single_strategy.py bollinger_bands --optimize --method bayesian
```

### Key Points:
1. **Without `--optimize`**: Uses parameters from `optimized_strategies.yaml`
2. **With `--optimize`**: Searches for new optimal parameters
3. **AI Analysis**: Always provides suggestions based on the results

### Viewing Optimization Results
When using `--optimize`, the results are saved in:
- `{strategy}_best_params.json` - Best parameters found
- `{strategy}_optimization_history.json` - All tested combinations (for grid search)
- `{strategy}_metrics.json` - Performance metrics of the final run

## 5. Complete Workflow Example {#complete-workflow}

Here's a full example from start to finish:

### Step 1: Run Initial Backtest
```bash
python3 backend/backtesting/scripts/run_single_strategy.py rsi_divergence --analyze --llm gemini
```

### Step 2: Check Results
```bash
# View the AI analysis
cat backend/backtesting/data/results/rsi_divergence/rsi_divergence_llm_analysis.md

# Check the CSV report
open backend/backtesting/data/results/rsi_divergence/rsi_divergence_trades.csv

# View the performance chart
open backend/backtesting/data/results/rsi_divergence/rsi_divergence_equity_curve.png
```

### Step 3: Update Optimized Parameters Configuration

After receiving parameter recommendations from the LLM analysis:

1. **Update the optimized strategies file**:
```bash
# Open the optimized strategies configuration
nano backend/backtesting/config/optimized_strategies.yaml
```

2. **Add your optimized parameters** under the appropriate strategy section:
```yaml
gaussian_channel:
  description: "Gaussian Channel - Optimized based on LLM analysis"
  optimized_parameters:
    default:
      period: 20
      std_dev: 2.0
      adaptive: true
```

3. **Verify the optimization** by running the same strategy again:
```bash
python3 backend/backtesting/scripts/run_single_strategy.py gaussian_channel --analyze
```

The script will automatically use the updated parameters from `optimized_strategies.yaml`.

**Note**: For live trading, manually update `/backend/config/strategy_config.json` with the tested parameters.

### Step 3: Create Optimized Config
```bash
# Create config file with AI suggestions
nano configs/strategies/rsi_divergence_optimized.yaml
```

### Step 4: Run with Optimized Config
```bash
python3 backend/backtesting/scripts/run_single_strategy.py rsi_divergence \
  --config configs/strategies/rsi_divergence_optimized.yaml \
  --analyze
```

### Alternative: Run Grid Search
```bash
# Let the system find optimal parameters
python3 backend/backtesting/scripts/run_single_strategy.py rsi_divergence \
  --optimize --method grid_search \
  --analyze --llm gemini
```

### Step 5: Compare Results
The new results will be saved with a timestamp. You can compare:
- Original: `backend/backtesting/data/results/rsi_divergence/`
- Optimized: Check the newest files in the same directory

## File Locations Summary

**All results are automatically saved to:**
```
backend/backtesting/data/results/{strategy_name}/
├── {strategy_name}_backtest_results.json    # Full metrics
├── {strategy_name}_trades.csv               # Trade details
├── {strategy_name}_equity_curve.png         # Performance chart
├── {strategy_name}_llm_analysis.md          # AI recommendations
└── {strategy_name}_best_params.json         # Best parameters (if optimized)
```

**Configuration files location:**
```
configs/strategies/
├── {strategy_name}_optimized.yaml           # Your custom configs
```

**No additional reporting steps needed** - everything is generated automatically!

## Summary

### Workflow Overview
1. **Run strategy** with current parameters from `optimized_strategies.yaml`
2. **Review AI analysis** showing current vs suggested parameters
3. **Update** `optimized_strategies.yaml` with recommendations
4. **Re-test** to verify improvements
5. **Optional**: Run full optimization for deeper parameter exploration

### Important Files
- **Configuration**: `backend/backtesting/config/optimized_strategies.yaml`
- **Results**: `backend/backtesting/data/results/{strategy_name}/`
- **Live Trading**: `backend/config/strategy_config.json` (update manually)

---

*Remember: Always test parameter changes thoroughly before deploying to live trading!*
