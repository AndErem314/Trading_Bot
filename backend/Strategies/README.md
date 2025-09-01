# Trading Strategies Documentation

> **Note**: This directory contains strategy descriptors for documentation and SQL-based analysis. For real-time signal generation, see the new executable strategies in `/backend/strategies_executable/`.

## RSI Momentum Divergence Swing Strategy

### Overview
The RSI Momentum Divergence Swing Strategy is a sophisticated trading system that combines multiple RSI-based indicators to identify high-probability swing trading opportunities in cryptocurrency markets. It operates 24/7 and is designed to capture medium-term price movements.

### Quick Start

#### 1. Check Current Signals
```bash
# Quick check for current trading signals
python3 backend/check_signals.py
```

#### 2. View Strategy Information
```bash
# Display detailed strategy information
python3 backend/strategy_runner.py --mode info
```

#### 3. Monitor Real-Time Signals
```bash
# Start monitoring for new signals (checks every 60 seconds)
python3 backend/strategy_runner.py --mode monitor

# Monitor with custom interval (e.g., every 30 seconds)
python3 backend/strategy_runner.py --mode monitor --interval 30
```

#### 4. Run Backtest
```bash
# Run comprehensive backtest with default parameters
python3 backend/backtest_strategy.py

# Run backtest with specific date range
python3 backend/backtest_strategy.py --start 2024-01-01 --end 2024-12-31

# Run backtest with custom parameters
python3 backend/backtest_strategy.py --capital 50000 --size 0.03 --fee 0.0005

# Export backtest results to CSV
python3 backend/backtest_strategy.py --export
```

### Strategy Details

#### Entry Signals
- **BUY Signal**: Generated when RSI crosses above 30 with bullish momentum
- **SELL Signal**: Generated when RSI crosses below 70 with bearish momentum

#### Exit Signals
- **EXIT_LONG**: When RSI reaches 70, bearish crossover, or bearish divergence
- **EXIT_SHORT**: When RSI reaches 30, bullish crossover, or bullish divergence

### Usage Examples

#### Example 1: Quick Signal Check
```bash
$ python3 backend/check_signals.py

================================================================================
RSI MOMENTUM DIVERGENCE STRATEGY - SIGNAL CHECK
Timestamp: 2025-08-26 18:30:00
================================================================================

✅ Found 5 signals!

📊 RECENT SIGNALS (Most Recent First):
--------------------------------------------------------------------------------

🔴 Signal #1:
   Time: 2025-07-14 04:00:00
   Type: SELL
   Price: $122,736.04
   RSI: 68.08
   Trend: bullish
   Divergence: none
```

#### Example 2: Real-Time Monitoring
```bash
$ python3 backend/strategy_runner.py --mode monitor --interval 60

Starting signal monitor for RSI Momentum Divergence Swing Strategy
Check interval: 60 seconds
Press Ctrl+C to stop monitoring
--------------------------------------------------------------------------------

🔔 NEW SIGNAL DETECTED!
Time: 2025-08-26 18:35:00
Signal: BUY
Price: $95,234.50
RSI: 32.45
Trend: bearish
Divergence: bullish

📊 Risk Management:
- Risk per trade: 2% of account
- Initial stop loss: 3% from entry price
- Trailing stop: Activate at 5% profit, trail by 2%
```

### Risk Management

The strategy includes comprehensive risk management rules:

1. **Position Sizing**
   - Risk 2% of account per trade
   - Maximum 3 concurrent positions
   - Scale into positions at extreme RSI levels (<25 or >75)

2. **Stop Loss**
   - Initial: 3% from entry price
   - Trailing: Activate at 5% profit, trail by 2%
   - Time-based: Exit after 168 hours (1 week)

3. **Filters**
   - Avoid entries when RSI changes >20 points in 1 hour
   - No new positions 1 hour before major news
   - Check correlation with market index

### Integration with Trading Systems

To integrate this strategy with your trading system:

1. **Import the strategy**:
```python
from backend.Strategies import RSIMomentumDivergenceSwingStrategy

strategy = RSIMomentumDivergenceSwingStrategy()
```

2. **Get SQL query for signals**:
```python
sql_query = strategy.get_sql_query()
# Execute this query against your database
```

3. **Access strategy rules**:
```python
entry_conditions = strategy.get_entry_conditions()
exit_conditions = strategy.get_exit_conditions()
risk_rules = strategy.get_risk_management_rules()
```

### Database Requirements

The strategy requires a SQLite database with the following tables:
- `ohlcv_data`: Price data (open, high, low, close, volume)
- `rsi_indicator`: RSI calculations and derived metrics

### Customization

You can customize the strategy by modifying:
- RSI thresholds (default: 30/70)
- Holding periods (default: 4-168 hours)
- Risk parameters (default: 2% risk per trade)

Edit the strategy file: `backend/Strategies/RSI_Momentum_Divergence_Swing_Strategy.py`

### Troubleshooting

1. **No signals found**:
   - Market conditions may not meet criteria
   - Check if RSI is in neutral territory
   - Ensure database has recent data

2. **Database errors**:
   - Verify database path: `data/trading_data_BTC.db`
   - Check table structure matches requirements

3. **Import errors**:
   - Run from project root directory
   - Ensure Python path includes backend directory

### Performance Notes

- The strategy works best in ranging and trending markets
- Avoid using during extreme volatility events
- Monitor performance and adjust parameters as needed
- Consider paper trading before live deployment

### Support

For issues or questions:
1. Check the strategy source code for detailed documentation
2. Review the SQL queries for signal logic
3. Examine the backtest results for historical performance

---

## New Executable Strategies

### Overview

In addition to the SQL-based strategy descriptors above, we now have executable strategy implementations that can generate signals in real-time from raw OHLCV data.

### Key Features

- **Real-time Signal Generation**: Calculate signals on-demand without database dependencies
- **Standardized Interface**: All strategies implement the `TradingStrategy` abstract base class
- **Market Regime Filtering**: Strategies can be enabled/disabled based on market conditions
- **On-demand Indicators**: Uses pandas_ta to calculate indicators from OHLCV data
- **Easy Testing**: Standardized interface makes unit testing straightforward

### Available Executable Strategies

1. **BollingerBandsMeanReversion** (`/backend/strategies_executable/bollinger_bands_strategy.py`)
2. **RSIMomentumDivergence** (`/backend/strategies_executable/rsi_momentum_strategy.py`)
3. **MACDMomentumCrossover** (`/backend/strategies_executable/macd_momentum_strategy.py`)
4. **SMAGoldenCross** (`/backend/strategies_executable/sma_golden_cross_strategy.py`)

### Usage with Strategy Bridge

The `StrategyBridge` class provides a unified interface to work with both old descriptors and new executable strategies:

```python
from backend.strategy_bridge import UnifiedStrategyFactory
import pandas as pd

# Load your OHLCV data
data = pd.read_csv('ohlcv_data.csv', index_col='timestamp', parse_dates=True)

# Create a strategy bridge
bridge = UnifiedStrategyFactory.create_strategy('RSI_Momentum_Divergence', data)

# Get historical signals (SQL-based)
historical_signals = bridge.get_historical_signals(limit=10)

# Get live signal (executable-based)
live_signal = bridge.get_live_signal()
print(f"Signal: {live_signal['signal']}, Confidence: {live_signal['confidence']}")

# Check if strategy is suitable for current market
is_allowed = bridge.is_strategy_allowed('Bullish')
```

### Using Executable Strategies Directly

```python
from backend.strategies_executable import RSIMomentumDivergence
import pandas as pd

# Load OHLCV data
data = pd.read_csv('ohlcv_data.csv', index_col='timestamp', parse_dates=True)

# Create strategy instance
strategy = RSIMomentumDivergence(data, config={'rsi_oversold': 25})

# Check if we have enough data
if strategy.has_sufficient_data():
    # Get signal
    signal = strategy.calculate_signal()
    print(f"Signal: {signal['signal']}")
    print(f"Reason: {signal['reason']}")
    print(f"RSI: {signal['rsi']}")
```

### Integration with Enhanced Orchestrator

The new `EnhancedMetaStrategyOrchestrator` automatically uses executable strategies:

```python
from backend.enhanced_meta_strategy_orchestrator import EnhancedMetaStrategyOrchestrator

# Initialize orchestrator
orchestrator = EnhancedMetaStrategyOrchestrator(
    db_connection_string="sqlite:///trading_data.db",
    symbols=['BTC/USDT', 'ETH/USDT'],
    config_path="backend/config/strategy_config.json"
)

# Setup and run
orchestrator.setup(primary_timeframe='H1')
results = orchestrator.run()

# Get composite signals
for symbol, signal_data in results['composite_signals'].items():
    print(f"{symbol}: {signal_data['signal']} (confidence: {signal_data['confidence']})")
```

### Configuration

Strategy parameters can be configured via JSON:

```json
{
    "strategies": {
        "rsi_divergence": {
            "class": "RSIMomentumDivergence",
            "enabled": true,
            "parameters": {
                "rsi_length": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70
            }
        }
    }
}
```

### Migration Path

1. **Phase 1**: Use StrategyBridge to run both systems in parallel
2. **Phase 2**: Validate executable strategies match expected behavior
3. **Phase 3**: Gradually migrate live trading to executable strategies
4. **Phase 4**: Keep descriptors for documentation and SQL analysis only
