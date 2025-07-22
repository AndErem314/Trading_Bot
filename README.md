# Trading Bot

A Python trading bot that collects cryptocurrency market data and calculates Gaussian Channel indicators.

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

- Fetches raw OHLCV data from cryptocurrency exchanges (using ccxt)
- Stores data in SQLite database with separate tables for raw data and indicators
- Calculates Gaussian Channel indicators (upper, middle, lower bands)
- Supports multiple symbols and timeframes
- Handles duplicate data prevention
- Command-line interface for flexible operation

## Installation

1. Install dependencies using uv:
```bash
uv sync
# or
uv add ccxt pandas numpy
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
```bash
# Generate charts and visualizations
python visualize_data.py
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

## Configuration

- Default symbols: BTC/USDT, ETH/USDT, SOL/USDT
- Default timeframes: 4h, 1d
- Default start date: 2021-01-01
- Database location: `data/market_data.db`
- Exchange: Binance (configurable in code)
