# Ichimoku Cloud Trading Bot

A Python-based cryptocurrency trading bot that implements Ichimoku Cloud technical analysis strategies with backtesting capabilities and LLM-powered optimization.

## Overview

This trading bot focuses on the Ichimoku Cloud indicator system to generate trading signals for cryptocurrency pairs (BTC/USDT, ETH/USDT, SOL/USDT). It features:

- Historical data collection from cryptocurrency exchanges
- Ichimoku Cloud indicator calculations and signal generation
- Comprehensive backtesting framework
- Multiple pre-configured trading strategies
- LLM-powered strategy optimization (OpenAI and Google Gemini)
- Detailed reporting with visualizations

## Project Structure

```
Trading_Bot/
│
├── app.py                 # Main entry point - CLI interface
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (API keys)
├── .gitignore           # Git ignore file
│
├── config/              # Strategy configurations
│   ├── strategies.json  # Trading strategy definitions
│   └── strategies.yaml  # Alternative format
│
├── data/                # SQLite databases and schema
│   ├── symbol_schema.sql    # Database schema definition
│   ├── trading_data_BTC.db  # Bitcoin historical data
│   ├── trading_data_ETH.db  # Ethereum historical data
│   └── trading_data_SOL.db  # Solana historical data
│
├── data_fetching/       # Data collection modules
│   ├── __init__.py
│   ├── collect_historical_data.py  # Main data collection script
│   ├── data_fetcher.py            # CCXT integration
│   ├── data_manager.py            # Database operations
│   └── database_init.py           # Database initialization
│
├── strategy/            # Trading strategy implementation
│   ├── compute_ichimoku_to_sql.py  # Ichimoku calculations
│   └── ichimoku_strategy.py        # Strategy logic
│
├── backtesting/         # Backtesting engine
│   └── ichimoku_backtester.py      # Main backtester
│
├── llm_analysis/        # LLM integration for optimization
│   ├── __init__.py
│   ├── env_loader.py          # Load LLM credentials
│   ├── llm_client.py          # LLM API client
│   ├── payload_builder.py     # Build LLM payloads
│   ├── prompt_builder.py      # Generate prompts
│   ├── report_text_renderer.py # Render LLM responses
│   └── pdf_writer.py          # PDF report generation
│
├── reporting/           # Report generation
│   ├── __init__.py
│   └── report_generator.py    # Generate backtest reports
│
└── reports/            # Generated reports directory
```

## Database Schema

The project uses SQLite databases with a per-symbol architecture. Each cryptocurrency has its own database file containing:

### Tables

1. **ohlcv_data** - Historical price and volume data
   - `id`: Primary key
   - `timestamp`: Date/time of the candle
   - `open`, `high`, `low`, `close`: Price data
   - `volume`: Trading volume
   - `timeframe`: 1h, 4h, or 1d

2. **ichimoku_data** - Calculated Ichimoku indicators
   - `id`: Primary key
   - `ohlcv_id`: Foreign key to ohlcv_data
   - `tenkan_sen`: Conversion line (9-period)
   - `kijun_sen`: Base line (26-period)
   - `senkou_span_a`: Leading span A
   - `senkou_span_b`: Leading span B (52-period)
   - `chikou_span`: Lagging span
   - `cloud_color`: Bullish (green) or bearish (red)
   - `price_position`: Above, in, or below cloud
   - `trend_strength`: Strong bullish to strong bearish
   - `tk_cross`: Tenkan/Kijun cross signals

3. **metadata** - Database information
   - Key-value pairs for database metadata

### Views

- `ohlcv_ichimoku_view`: Combined OHLCV and Ichimoku data
- `latest_data_view`: Summary of available data per timeframe
- `ichimoku_signals_view`: Trading signals based on Ichimoku

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Trading_Bot.git
cd Trading_Bot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with your API keys:
```
# Exchange API (if needed for live data)
EXCHANGE_API_KEY=your_exchange_api_key
EXCHANGE_API_SECRET=your_exchange_api_secret

# LLM APIs for optimization
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

### Running the Application

Start the CLI interface:
```bash
python app.py
```

The main menu provides three options:

1. **Collect Historical Data**: Downloads OHLCV data for all configured symbols
2. **Compute Ichimoku**: Calculates Ichimoku indicators for selected symbols
3. **Backtest Ichimoku Strategy**: Run backtests with various strategies

### Workflow Example

1. **Initial Setup**:
   - Run option 1 to collect historical data
   - Run option 2 to compute Ichimoku indicators

2. **Backtesting**:
   - Select option 3
   - Choose a strategy from the available list
   - Select symbol (BTC, ETH, or SOL)
   - Select timeframe (1h, 4h, or 1d)
   - Optionally set a start date
   - Review generated reports in the `reports/` directory

3. **LLM Optimization** (Optional):
   - After backtesting, choose to generate LLM optimization report
   - Select prompt variant (analyst or risk-focused)
   - Choose LLM provider (OpenAI or Gemini)
   - Review the generated PDF report

## Trading Strategies

The bot includes several pre-configured Ichimoku-based strategies:

1. **Cloud-TK-SpanA Base TK Exit**: Entry on price above cloud + TK cross + bullish cloud
2. **Cloud Breakout with Trend**: Strong trend confirmation strategies
3. **Conservative Cloud Entry**: More restrictive entry conditions
4. **Aggressive Momentum**: Quick entry on momentum signals

Strategies are defined in `config/strategies.json` and can be customized.

## Requirements

- Python 3.8+
- SQLite3
- Internet connection for data fetching and LLM APIs
- Sufficient disk space for historical data storage

## Notes

- Historical data starts from August 1, 2020
- Backtesting uses fixed position sizing (100% of equity)
- Commission: 0.1%, Slippage: 0.03%
- The `archive/` folder contains old code and is excluded from git

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]