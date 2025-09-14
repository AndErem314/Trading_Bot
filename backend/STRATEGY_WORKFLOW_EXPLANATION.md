# Strategy Workflow and Data Flow Explanation

## Overview

The Trading Bot employs a dual-strategy system with two distinct but complementary approaches:

1. **SQL-based Strategy Descriptors** (Original)
2. **Executable Strategy Implementations** (New)

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        External Data Sources                      │
│  (Exchange APIs, WebSocket feeds, Historical data providers)     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Raw Data Collection                           │
│              (Handled by existing data pipeline)                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SQLite Database                                │
│                (data/trading_data_BTC.db)                         │
│  Tables: ohlcv_data, indicators, signals, trades, etc.           │
└──────────┬──────────────────────────────────┬────────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│  SQL-based Descriptors   │      │   Executable Strategies      │
│  (Historical Analysis)   │      │   (Real-time Trading)        │
└──────────────────────────┘      └──────────────────────────────┘
```

## 1. SQL-based Strategy Descriptors (Historical Analysis)

### Purpose
- Historical signal analysis
- Backtesting with pre-calculated indicators
- Documentation and strategy definition
- SQL-based pattern matching

### Data Source
- **Primary**: SQLite database with pre-calculated indicators
- **Tables Used**:
  - `ohlcv_data`: Raw price data
  - `indicators`: Pre-calculated technical indicators
  - Various indicator-specific tables (e.g., `rsi_data`, `macd_data`, `bollinger_bands_data`)

### Workflow
1. Raw OHLCV data is collected and stored in the database
2. Separate processes calculate indicators and store them in indicator tables
3. Strategy descriptors contain SQL queries that:
   - JOIN multiple tables (ohlcv + indicators)
   - Apply strategy-specific conditions
   - Generate historical signals
4. Results are used for:
   - Backtesting
   - Performance analysis
   - Strategy optimization

### Streamlined Backtesting Workflow

#### Single Configuration Source
All strategy parameters are now managed in `/backend/backtesting/config/optimized_strategies.yaml`:

```yaml
strategies:
  bollinger_bands:
    description: "Bollinger Bands Mean Reversion - Optimized for BTC/USDT 4h"
    optimized_parameters:
      default:
        bb_length: 25
        bb_std: 2.5
        rsi_length: 18
        rsi_oversold: 40
        rsi_overbought: 80
```

#### Running Backtests
```bash
# Run with default optimized parameters
python3 backend/backtesting/scripts/run_single_strategy.py bollinger_bands --analyze

# Run with parameter optimization (grid search or bayesian)
python3 backend/backtesting/scripts/run_single_strategy.py bollinger_bands --optimize --method grid_search

# Run with custom parameters
python3 backend/backtesting/scripts/run_single_strategy.py bollinger_bands --params '{"bb_length": 20, "bb_std": 2.0}'
```

#### Analysis Reports
All backtests automatically generate comprehensive reports including:
- **Current Parameters Used**: Shows exact parameters for the run
- **Performance Metrics**: Returns, Sharpe ratio, drawdown, etc.
- **Suggested Parameters**: AI-powered recommendations via Gemini
- **Risk Assessment**: Detailed risk analysis
- **Market Conditions**: Optimal conditions for the strategy

### Example SQL Query (RSI Momentum Strategy)
```sql
WITH signal_data AS (
    SELECT 
        o.timestamp,
        o.close,
        r.rsi,
        r.rsi_ma,
        LAG(r.rsi, 1) OVER (ORDER BY o.timestamp) as prev_rsi,
        -- Additional calculations...
    FROM ohlcv_data o
    JOIN rsi_data r ON o.timestamp = r.timestamp
    WHERE o.timestamp >= datetime('now', '-30 days')
)
SELECT * FROM signal_data
WHERE (conditions for entry/exit signals)
ORDER BY timestamp DESC
LIMIT 100;
```

## 2. Executable Strategy Implementations (Live Trading)

### Purpose
- Real-time signal generation
- Live trading decisions
- Dynamic market adaptation
- No dependency on pre-calculated indicators

### Data Source
- **Primary**: Live OHLCV data (pandas DataFrame)
- **Input Format**: DataFrame with columns: `open`, `high`, `low`, `close`, `volume`
- **Index**: Timestamp (datetime)

### Workflow
1. **Data Reception**:
   - Receives OHLCV DataFrame (typically last 200-500 candles)
   - Data can come from:
     - Direct API calls to exchanges
     - WebSocket streams
     - Database queries (for testing)
     - Any source that provides OHLCV format

2. **Indicator Calculation**:
   - Each strategy calculates its own indicators on-the-fly
   - Uses libraries like pandas_ta, talib, or custom implementations
   - No database dependency for calculations

3. **Signal Generation**:
   - Applies strategy logic to freshly calculated indicators
   - Returns signal dictionary with:
     - `signal`: 'BUY', 'SELL', or 'HOLD'
     - `confidence`: 0.0 to 1.0
     - `reason`: Explanation of the signal
     - Additional metadata

4. **Market Regime Adaptation**:
   - Checks current market conditions
   - Adjusts strategy behavior accordingly
   - Can disable strategy in unsuitable conditions

### Example Code Flow
```python
class RSIMomentumDivergence(TradingStrategy):
    def calculate_signal(self) -> Dict[str, float]:
        # 1. Calculate indicators on live data
        self.data['RSI'] = ta.rsi(self.data['close'], length=14)
        self.data['RSI_MA'] = self.data['RSI'].rolling(window=9).mean()
        
        # 2. Detect patterns
        bullish_divergence = self._detect_bullish_divergence()
        bearish_divergence = self._detect_bearish_divergence()
        
        # 3. Generate signal
        if bullish_divergence and self.data['RSI'].iloc[-1] < 30:
            return {
                'signal': 'BUY',
                'confidence': 0.8,
                'reason': 'Bullish divergence with oversold RSI'
            }
        # ... more logic
```

## Data Source Comparison

### SQL-based Descriptors
- **Pros**:
  - Fast historical queries
  - Complex multi-table joins
  - Backtesting entire datasets efficiently
  - Consistent historical analysis
  
- **Cons**:
  - Requires pre-calculated indicators
  - Database maintenance overhead
  - Delayed signals (calculation lag)
  - Less flexible for real-time adaptation

### Executable Strategies
- **Pros**:
  - Real-time calculations
  - No database dependency for trading
  - Flexible parameter adjustment
  - Immediate signal generation
  - Can adapt to market conditions dynamically
  
- **Cons**:
  - Computational overhead for each calculation
  - Requires sufficient historical data in memory
  - More complex testing setup

## Hybrid Approach with Strategy Bridge

The Strategy Bridge enables both systems to work together:

```python
# For historical analysis
bridge = StrategyBridge('RSI_Momentum_Divergence')
historical_signals = bridge.get_historical_signals()  # Uses SQL

# For live trading
live_data = get_live_ohlcv_data()  # From exchange API
bridge.initialize_executable(live_data)
current_signal = bridge.get_live_signal()  # Uses executable
```

## Data Requirements by Strategy

### Minimum Data Points Required:
- **RSI Momentum**: 50+ candles (RSI period + divergence detection)
- **Bollinger Bands**: 40+ candles (20 SMA + 2 std dev)
- **MACD**: 50+ candles (26 EMA + signal line)
- **SMA Golden Cross**: 200+ candles (for 200 SMA)
- **Ichimoku Cloud**: 120+ candles (52 period lookback)
- **Parabolic SAR**: 30+ candles
- **Fibonacci Retracement**: 100+ candles (for swing detection)
- **Gaussian Channel**: 50+ candles

## Production Data Flow

### Current Setup
1. **Data Collection**: External scripts collect data from exchanges
2. **Database Storage**: Raw data stored in SQLite
3. **Indicator Calculation**: Batch jobs calculate and store indicators
4. **Signal Generation**: 
   - Historical: SQL queries on stored data
   - Live: Executable strategies on recent data

### Recommended Future Setup
1. **Real-time Data Pipeline**:
   ```
   Exchange API → Message Queue → Strategy Engine → Trading Engine
                                         ↓
                                  Database (logging only)
   ```

2. **Dual-Purpose Database**:
   - Store raw OHLCV for historical analysis
   - Log executed trades and signals
   - No dependency for real-time decisions

3. **Memory-First Architecture**:
   - Keep recent data (e.g., last 500 candles) in memory
   - Calculate indicators on-demand
   - Use database for historical queries only

## Streamlined Workflow Benefits

### Single Source of Truth
- All strategy parameters in `optimized_strategies.yaml`
- No separate workflow for optimized vs non-optimized runs
- Easy to track and version control parameter changes

### Iterative Optimization Process
1. **Initial Run**: Test strategy with current parameters
2. **AI Analysis**: Gemini provides parameter recommendations
3. **Manual Update**: Update parameters in `optimized_strategies.yaml`
4. **Re-test**: Run again with new parameters
5. **Compare**: Analyze improvement in performance

### Optimization Methods
- **Grid Search**: Exhaustive search through parameter combinations
- **Bayesian Optimization**: Intelligent search using probabilistic model
- **No Random Search**: Removed for consistency and reproducibility

### Example Workflow
```bash
# Step 1: Test current parameters
python3 backend/backtesting/scripts/run_single_strategy.py gaussian_channel --analyze

# Step 2: Review AI recommendations in the report
# Report shows current parameters and suggested improvements

# Step 3: Update optimized_strategies.yaml with new parameters
# Edit the file manually based on recommendations

# Step 4: Run again with updated parameters
python3 backend/backtesting/scripts/run_single_strategy.py gaussian_channel --analyze

# Step 5: Compare performance improvements
```

## Summary

The executable strategies are designed to work with **live OHLCV data** from any source, calculating indicators on-the-fly without database dependency. The SQL-based descriptors continue to serve their purpose for historical analysis using the existing database infrastructure. This dual approach provides both real-time trading capability and comprehensive historical analysis tools.
