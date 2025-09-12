# Trading Bot Backtesting User Manual

## Table of Contents
1. [Running a Single Strategy](#running-single-strategy)
2. [Understanding the Results](#understanding-results)  
3. [Implementing Optimizations](#implementing-optimizations)
4. [Running with Optimized Parameters](#running-optimized)
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

### Method 1: Create an Optimized Configuration File

#### Step 1: Create a New Config File
```bash
nano configs/strategies/rsi_divergence_optimized.yaml
```

#### Step 2: Copy Suggested Parameters from LLM Analysis
Example content based on LLM suggestions:
```yaml
# Optimized RSI Divergence Strategy Configuration
# Based on LLM Analysis

strategy:
  name: rsi_divergence
  version: "1.1"
  description: "Optimized parameters from Gemini analysis"

parameters:
  # RSI Settings - Adjusted for better signal quality
  rsi_length: 16  # Increased from 14 for smoother signals
  rsi_sma_fast: 5
  rsi_sma_slow: 12  # Increased from 10 for better trend detection
  
  # Thresholds - Tightened for higher win rate
  rsi_oversold: 25  # Lowered from 30
  rsi_overbought: 75  # Raised from 70
  
  # Divergence Detection
  momentum_lookback: 7  # Increased from 5
  divergence_lookback: 25  # Increased from 20

# Risk Management - Added based on recommendations
risk_management:
  position_size: 0.02  # 2% per trade (reduced from default)
  stop_loss_pct: 0.03  # 3% stop loss
  take_profit_pct: 0.09  # 9% take profit (3:1 reward/risk)

# Backtesting Configuration
backtest:
  symbol: "BTC/USDT"
  timeframe: "4h"
  start_date: "2020-01-01"
  end_date: "2025-09-01"
```

#### Step 3: Run with Config File
```bash
python3 backend/backtesting/scripts/run_single_strategy.py rsi_divergence \
  --config configs/strategies/rsi_divergence_optimized.yaml \
  --analyze --llm gemini
```

### Method 2: Direct Parameter Input

Run the strategy with suggested parameters directly:

```bash
python3 backend/backtesting/scripts/run_single_strategy.py rsi_divergence \
  --params '{"rsi_length": 16, "rsi_oversold": 25, "rsi_overbought": 75}' \
  --analyze
```

## 4. Running with Optimized Parameters {#running-optimized}

### Option 1: Test Specific Parameters (Recommended)
Use the parameters from the AI analysis:
```bash
python3 backend/backtesting/scripts/run_single_strategy.py bollinger_bands \
  --params '{"bb_length": 25, "bb_std": 2.5}' \
  --analyze --llm gemini
```

### Option 2: Running Grid Search Optimization
Let the system test all parameter combinations automatically:
```bash
python3 backend/backtesting/scripts/run_single_strategy.py rsi_divergence \
  --optimize --method grid_search \
  --analyze --llm gemini
```

This will:
1. Test multiple parameter combinations systematically
2. Find the best performing parameter set
3. Generate AI analysis for the best results
4. Save all results with optimization history

### Option 3: Other Optimization Methods
```bash
# Random search (faster for large parameter spaces)
python3 backend/backtesting/scripts/run_single_strategy.py bollinger_bands \
  --optimize --method random_search \
  --analyze --llm gemini

# Bayesian optimization (most efficient)
python3 backend/backtesting/scripts/run_single_strategy.py macd_momentum \
  --optimize --method bayesian \
  --analyze --llm gemini
```

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

### Step 2a: Update Optimized Parameters Configuration

**Important**: After receiving parameter recommendations from the LLM analysis, you need to manually update the optimized parameters configuration:

1. **Update the optimized strategies file**:
```bash
# Open the optimized strategies configuration
nano backend/backtesting/config/optimized_strategies.yaml
```

2. **Add your optimized parameters** under the appropriate strategy section. For example, if you got recommendations for `gaussian_channel`:
```yaml
gaussian_channel:
  description: "Gaussian Channel - Optimized based on LLM analysis"
  optimized_parameters:
    default:
      period: 20
      std_dev: 2.0
      adaptive: true
      expected_return: "TBD - Run backtest to verify"
```

3. **Note**: These optimized parameters will **NOT** automatically update `strategy_config.json` for live trading. You must:
   - Either manually update `/backend/config/strategy_config.json` with the tested parameters
   - Or use the `run_optimized_strategy.py` script which reads from `optimized_strategies.yaml`

4. **Verify the optimization** by running:
```bash
python3 backend/backtesting/scripts/run_optimized_strategy.py gaussian_channel
```

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

## Parameter Optimization Workflow Summary

### From Backtesting to Live Trading

1. **Run backtest with LLM analysis** → Get parameter recommendations
2. **Manually update** `backend/backtesting/config/optimized_strategies.yaml` with the recommended parameters
3. **Test the optimized parameters** using `run_optimized_strategy.py`
4. **For live trading**, manually update `backend/config/strategy_config.json` with the tested parameters

**Key Points:**
- LLM parameter recommendations are NOT automatically applied
- `optimized_strategies.yaml` stores optimization results for backtesting
- `strategy_config.json` controls live trading parameters
- Always test optimized parameters before using in live trading

---

*That's it! Run strategy → Read AI analysis → Apply suggestions → Compare results → Update configs → Deploy to live trading*
