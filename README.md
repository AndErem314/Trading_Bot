# Trading Bot

A comprehensive Python trading bot that collects cryptocurrency market data, calculates different indicators, and provides flexible data visualization capabilities. The bot supports multiple trading pairs including BTC pairs (SOL/BTC, ETH/BTC) and USDT pairs, with dynamic chart generation and interactive user interfaces.

## Project Structure

```
Trading_Bot/
├── backend/                    # Backend data processing modules
│   ├── __init__.py
│   ├── main.py                # Main runner script that coordinates processes
│   ├── data_fetcher.py        # Collects raw OHLCV data from exchanges
│   ├── gaussian_channel.py    # Calculates Gaussian Channel indicators
│   └── collect_historical_data.py # Historical data collection script
├── frontend/                   # Frontend data visualization modules
│   ├── __init__.py
│   ├── data_visualizer.py     # Data visualization and charting
│   └── charts/                # Generated chart images
├── data/                      # SQLite database files
├── run_trading_bot.py         # Main entry point script
├── collect_data.py           # Data collection entry point
├── visualize_data.py         # Visualization entry point
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
└── README.md                # Project documentation
```

## Features

### Data Collection
- Fetches raw OHLCV data from cryptocurrency exchanges (using ccxt)
- Supports **5 trading pairs**: BTC/USDT, ETH/USDT, SOL/USDT, **SOL/BTC**, **ETH/BTC** (newly added)
- Historical data from August 1st, 2020 to present
- Multiple timeframes: 4-hour and daily intervals
- Automatic duplicate data prevention
- Robust error handling and retry mechanisms

### Data Analysis
- Calculates Gaussian Channel indicators (upper, middle, lower bands)
- SQLite database with optimized schema for raw data and indicators
- Supports batch processing of multiple symbols and timeframes
- Command-line interface for flexible operation

### Data Visualization
- **Interactive mode**: User-friendly interface for custom chart generation
- **Batch mode**: Generate charts for all trading pairs automatically
- **Flexible time periods**: Specify any number of days or view all available data
- **Multiple chart types**:
  - Candlestick charts with volume
  - Line charts with price trends
  - OHLC summary charts
  - Volume analysis charts
- High-quality chart exports (PNG format, 300 DPI)
- Automatic chart organization and naming

## Installation

### Prerequisites
- Python 3.8+ 
- pip or uv package manager

### Install Dependencies

**Option 1: Using pip (recommended)**
```bash
# Clone or download the repository
# Navigate to the project directory
cd Trading_Bot

# Install dependencies
pip3 install -r requirements.txt
```

**Option 2: Using uv**
```bash
uv sync
# or
uv add ccxt pandas numpy matplotlib
```

### Verify Installation
```bash
# Test that dependencies are installed correctly
python3 -c "import ccxt, pandas, matplotlib; print('All dependencies installed successfully!')"
```

## Usage

### Main Trading Bot (Data Collection + Analysis)
```bash
# Run complete pipeline (collect data and calculate indicators)
python run_trading_bot.py --mode both

# Collect raw data only
python run_trading_bot.py --mode collect

# Calculate Gaussian Channels only  
python run_trading_bot.py --mode calculate

# Custom symbols and timeframes
python run_trading_bot.py --symbols BTC/USDT ETH/USDT --timeframes 1h 4h --start-date 2022-01-01
```

### Data Collection
```bash
# Collect historical data for all symbols
python collect_data.py
```

### Data Visualization

The visualizer now offers three modes for maximum flexibility:

```bash
# Launch interactive visualizer
python visualize_data.py
```

**Mode 1: Interactive Mode (Default)**
- Select trading pairs from available data
- Choose timeframes (4h, 1d)
- Specify any number of days or view all data
- Pick chart type (candlestick, line, OHLC, volume)
- Optional chart saving with custom filenames

**Mode 2: Batch Mode**
- Generate charts for all trading pairs automatically
- Customize time period and chart type
- Perfect for bulk chart generation

**Mode 3: Quick Demo**
- Generates sample charts for all trading pairs
- Uses default 90-day candlestick charts
- Great for testing and demonstrations

#### Example Interactive Session:
```
Select mode:
1. Interactive mode (custom charts with user input)
2. Batch mode (generate charts for all symbols) 
3. Quick demo (legacy mode)

Enter your choice (1-3) [default: 1]: 1

Available trading pairs:
1. BTC/USDT
2. ETH/BTC  
3. ETH/USDT
4. SOL/BTC
5. SOL/USDT

Select a trading pair (1-5): 4
Select timeframe (1-2): 1
Enter number of days to visualize (or 'all'): 30
Select chart type (1-4) [default: candlestick]: 1
Save chart to file? (y/n): y
```

### Advanced Usage - Run Individual Modules
```bash
# Run backend modules directly
cd backend
python main.py --mode both
python data_fetcher.py
python gaussian_channel.py

# Run frontend modules directly
cd frontend  
python data_visualizer.py
```

## Database Schema

### Raw Data Table
- `id` - Primary key
- `symbol` - Trading pair (e.g., BTC/USDT)
- `timeframe` - Time interval (e.g., 4h, 1d)
- `timestamp` - Data timestamp
- `open, high, low, close, volume` - OHLCV data
- `created_at` - Record creation timestamp

### Gaussian Channel Data Table
- All fields from raw data table plus:
- `gc_upper` - Upper Gaussian Channel band
- `gc_middle` - Middle Gaussian Channel band (moving average)
- `gc_lower` - Lower Gaussian Channel band

## Data Summary

The bot currently maintains **63,546+ records** across **5 trading pairs**:

| Trading Pair | 4-Hour Records | Daily Records | Date Range |
|-------------|----------------|---------------|------------|
| BTC/USDT    | 10,905         | 1,818         | Aug 2020 - Present |
| ETH/USDT    | 10,905         | 1,818         | Aug 2020 - Present |
| SOL/USDT    | 10,844         | 1,808         | Aug 2020 - Present |
| **SOL/BTC** | **10,906**     | **1,818**     | **Aug 2020 - Present** |
| **ETH/BTC** | **10,906**     | **1,818**     | **Aug 2020 - Present** |

*Note: SOL/BTC and ETH/BTC pairs were recently added with full historical data.*

## Configuration

### Default Settings
- **Symbols**: BTC/USDT, ETH/USDT, SOL/USDT, SOL/BTC, ETH/BTC
- **Timeframes**: 4h (4-hour), 1d (daily)
- **Historical start**: August 1st, 2020
- **Database**: `data/market_data.db` (SQLite)
- **Exchange**: Binance
- **Chart output**: `charts/` directory

### Customization
All settings can be customized via command-line arguments:
```bash
# Custom symbols and timeframes
python backend/main.py --symbols SOL/BTC ETH/BTC --timeframes 1d --start-date 2023-01-01

# Custom visualization period
python visualize_data.py
# Then select custom days in interactive mode
```

## Recent Updates

### ✅ **Version 2.0 - Enhanced Visualization & New Trading Pairs**
- **Added new trading pairs**: SOL/BTC and ETH/BTC with full historical data
- **Interactive visualizer**: Dynamic user input for days, chart types, and trading pairs
- **Batch chart generation**: Create charts for all pairs automatically
- **Flexible time periods**: No longer limited to 90 days - specify any period
- **Multiple chart types**: Candlestick, Line, OHLC, and Volume analysis
- **Improved user experience**: Menu-driven interface with validation
- **High-quality exports**: 300 DPI PNG charts with organized naming

### 🔧 **Technical Improvements**
- Fixed requirements.txt (removed non-existent sqlite3 dependency)
- Enhanced error handling and data validation
- Optimized database operations for better performance
- Updated all modules to support new trading pairs consistently
