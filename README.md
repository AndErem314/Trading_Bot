# Trading Bot

A Python trading bot that collects cryptocurrency market data and calculates Gaussian Channel indicators.

## Project Structure

- `data_fetcher.py` - Collects raw OHLCV data from exchanges and saves to database
- `gaussian_channel.py` - Calculates Gaussian Channel indicators from raw data
- `main.py` - Main runner script that coordinates both processes
- `requirements.txt` - Python dependencies
- `data/` - Directory for SQLite database files

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

### Collect Raw Data Only
```bash
python main.py --mode collect
```

### Calculate Gaussian Channels Only
```bash
python main.py --mode calculate
```

### Both (Default)
```bash
python main.py --mode both
```

### Custom Symbols and Timeframes
```bash
python main.py --symbols BTC/USDT ETH/USDT --timeframes 1h 4h --start-date 2022-01-01
```

### Run Individual Scripts
```bash
# Collect raw data
python data_fetcher.py

# Calculate Gaussian Channel indicators
python gaussian_channel.py
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
