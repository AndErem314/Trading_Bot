# Trading Bot Backend Structure

This directory contains the backend components of the Trading Bot, organized into two distinct workflows plus core shared components.

## 📁 Directory Structure

```
backend/
├── sql_workflow/               # SQL-based workflow for historical analysis
│   ├── strategies/            # SQL query descriptors for strategies
│   ├── data_collection/       # Data collection and storage tools
│   └── analysis/              # SQL-based analysis and backtesting
│
├── executable_workflow/        # Real-time executable workflow
│   ├── strategies/            # Live strategy implementations
│   ├── orchestration/         # Strategy orchestration and regime detection
│   └── interfaces/            # Common interfaces (TradingStrategy)
│
├── core/                      # Core components used by both workflows
│   ├── indicators/            # Technical indicator calculations
│   └── bridge/               # Bridge between SQL and executable workflows
│
└── main.py                    # Main entry point for data collection
```

## 🔄 SQL Workflow

The SQL workflow is designed for historical analysis, backtesting, and research. It uses pre-calculated indicators stored in SQLite databases.

### Components:
- **strategies/**: Contains strategy descriptors with SQL queries
- **data_collection/**: Tools for fetching and storing market data
  - `data_fetcher.py`: Fetches data from exchanges
  - `data_manager.py`: Manages database operations
  - `collect_historical_data.py`: Historical data collection
- **analysis/**: Analysis and execution tools
  - `check_signals.py`: Quick signal checking
  - `backtest_strategy.py`: Comprehensive backtesting
  - `strategy_runner.py`: Strategy monitoring and execution

### Usage:
```bash
# Check current signals
python backend/sql_workflow/analysis/check_signals.py

# Run backtest
python backend/sql_workflow/analysis/backtest_strategy.py --start 2024-01-01

# Monitor strategies
python backend/sql_workflow/analysis/strategy_runner.py --mode monitor
```

## ⚡ Executable Workflow

The executable workflow is designed for real-time trading. Strategies calculate indicators on-the-fly from raw OHLCV data.

### Components:
- **strategies/**: Live strategy implementations
  - Each strategy implements the `TradingStrategy` interface
  - Calculates indicators in real-time
  - Returns standardized signal format
- **orchestration/**: Strategy coordination
  - `refined_meta_strategy_orchestrator.py`: Manages multiple strategies
  - `enhanced_market_regime_detector.py`: ADX-based regime detection
- **interfaces/**: Common interfaces
  - `trading_strategy_interface.py`: Abstract base class for strategies

### Usage:
```python
from backend.executable_workflow.strategies import RSIMomentumDivergence
import pandas as pd

# Load OHLCV data
data = pd.read_csv('ohlcv_data.csv', index_col='timestamp', parse_dates=True)

# Create and use strategy
strategy = RSIMomentumDivergence(data)
signal = strategy.calculate_signal()
```

## 🔧 Core Components

Shared components used by both workflows.

### Components:
- **indicators/**: Technical indicator calculators
  - Bollinger Bands, RSI, MACD, etc.
  - Used by SQL workflow for batch calculations
  - Can be used by executable strategies if needed
- **bridge/**: Workflow integration
  - `strategy_bridge.py`: Unified interface for both workflows
  - `backtester.py`: Backtesting framework

### Usage:
```python
from backend.core.bridge.strategy_bridge import UnifiedStrategyFactory

# Create bridge for any strategy
bridge = UnifiedStrategyFactory.create_strategy('RSI_Momentum_Divergence', data)

# Use SQL workflow
historical_signals = bridge.get_historical_signals()

# Use executable workflow
live_signal = bridge.get_live_signal()
```

## 🚀 Quick Start

1. **For Historical Analysis (SQL Workflow)**:
   ```bash
   # Collect data
   python backend/main.py --mode collect --symbols BTC/USDT ETH/USDT
   
   # Calculate indicators
   python backend/main.py --mode all_indicators
   
   # Run analysis
   python backend/sql_workflow/analysis/check_signals.py
   ```

2. **For Live Trading (Executable Workflow)**:
   ```python
   from backend.executable_workflow.orchestration.refined_meta_strategy_orchestrator import RefinedMetaStrategyOrchestrator
   
   orchestrator = RefinedMetaStrategyOrchestrator(
       db_connection_string="sqlite:///data/trading_data.db",
       symbols=['BTC/USDT', 'ETH/USDT']
   )
   
   orchestrator.setup()
   results = orchestrator.run()
   ```

## 📊 Workflow Comparison

| Feature | SQL Workflow | Executable Workflow |
|---------|--------------|-------------------|
| **Purpose** | Historical analysis | Real-time trading |
| **Data Source** | Pre-calculated in DB | Live OHLCV data |
| **Speed** | Fast queries | Real-time calculation |
| **Flexibility** | Limited to SQL | Full Python flexibility |
| **Best For** | Backtesting, research | Live trading |

## 🔗 Integration

The `strategy_bridge.py` in the core module allows you to use both workflows together:

```python
# During development: Compare signals
bridge = UnifiedStrategyFactory.create_strategy('Bollinger_Bands', data)
comparison = bridge.compare_signals(data)

# In production: Use executable for trading, SQL for analysis
live_signal = bridge.get_live_signal()  # For trading
historical = bridge.get_historical_signals()  # For research
```

## 📝 Notes

- The SQL workflow requires a populated SQLite database
- The executable workflow needs at least 200-500 bars of OHLCV data
- Both workflows can run simultaneously without conflicts
- Use the bridge component for gradual migration from SQL to executable
