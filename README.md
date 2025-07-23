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
│   ├── bollinger_bands.py     # Calculates Bollinger Bands indicators
│   ├── simple_moving_average.py # Calculates SMA (50/200) indicators
│   ├── ichimoku_cloud.py      # Calculates Ichimoku Cloud indicators
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

The bot implements **4 comprehensive technical indicators** with advanced analysis capabilities:

#### **1. Gaussian Channel Indicator**
- Upper, middle, and lower channel bands based on moving averages
- Volatility-based trend analysis
- Breakout and reversal signal detection

#### **2. Bollinger Bands Indicator**
- Standard Bollinger Bands (20-period SMA ± 2 standard deviations)
- **Band Width**: Measures volatility expansion/contraction
- **%B Position**: Shows price position relative to bands (0-1 scale)
- **Squeeze Detection**: Identifies low volatility periods for breakout trading
- **Trading Signals**: Overbought/oversold conditions with buy/sell alerts

#### **3. Simple Moving Average (50/200)**
- **SMA 50**: Short-term trend identification (50-period average)
- **SMA 200**: Long-term trend identification (200-period average)
- **Golden Cross**: SMA 50 crosses above SMA 200 (major bullish signal)
- **Death Cross**: SMA 50 crosses below SMA 200 (major bearish signal)
- **Trend Strength**: Quantified analysis (-100 to +100 scale)
- **Position Analysis**: Price vs SMA positioning with detailed descriptions

#### **4. Ichimoku Cloud (Ichimoku Kinko Hyo)**
- **Tenkan-sen**: 9-period conversion line (short-term trend)
- **Kijun-sen**: 26-period base line (medium-term trend)
- **Senkou Span A**: Leading span A projected 26 periods forward
- **Senkou Span B**: Leading span B projected 26 periods forward
- **Chikou Span**: Lagging span (current close shifted 26 periods back)
- **Cloud Analysis**: Dynamic support/resistance with color-coded strength
- **Comprehensive Signals**: Multi-component confirmation system

#### **Technical Features**
- SQLite database with optimized schema for raw data and all indicators
- Supports batch processing of multiple symbols and timeframes
- Real-time pattern analysis and signal generation
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

### Individual Indicator Calculations
```bash
# Calculate specific indicators for all trading pairs
python backend/bollinger_bands.py     # Bollinger Bands analysis
python backend/simple_moving_average.py # SMA with Golden/Death Cross detection
python backend/ichimoku_cloud.py      # Ichimoku Cloud comprehensive analysis
python backend/gaussian_channel.py    # Gaussian Channel indicators
```

### Advanced Usage - Run Individual Modules
```bash
# Run backend modules directly
cd backend
python main.py --mode both
python data_fetcher.py
python gaussian_channel.py
python bollinger_bands.py
python simple_moving_average.py
python ichimoku_cloud.py

# Run frontend modules directly
cd frontend  
python data_visualizer.py
```

## Database Schema

The database contains **5 optimized tables** storing raw data and calculated indicators:

### Raw Data Table (`raw_data`)
- `id` - Primary key
- `symbol` - Trading pair (e.g., BTC/USDT)
- `timeframe` - Time interval (e.g., 4h, 1d)
- `timestamp` - Data timestamp
- `open, high, low, close, volume` - OHLCV data
- `created_at` - Record creation timestamp

### Gaussian Channel Data Table (`gaussian_channel_data`)
- All fields from raw data table plus:
- `gc_upper` - Upper Gaussian Channel band
- `gc_middle` - Middle Gaussian Channel band (moving average)
- `gc_lower` - Lower Gaussian Channel band

### Bollinger Bands Data Table (`bollinger_bands_data`)
- All fields from raw data table plus:
- `bb_upper` - Upper Bollinger Band (SMA + 2σ)
- `bb_middle` - Middle Bollinger Band (20-period SMA)
- `bb_lower` - Lower Bollinger Band (SMA - 2σ)
- `bb_width` - Band width (volatility measure)
- `bb_percent` - %B position indicator (0-1 scale)

### Simple Moving Average Data Table (`sma_data`)
- All fields from raw data table plus:
- `sma_50` - 50-period simple moving average
- `sma_200` - 200-period simple moving average
- `sma_ratio` - SMA 50/200 ratio (trend strength)
- `price_vs_sma50` - Price position vs SMA 50 (%)
- `price_vs_sma200` - Price position vs SMA 200 (%)
- `trend_strength` - Quantified trend analysis (-100 to +100)
- `sma_signal` - Trading signal (strong_buy, buy, hold, sell, strong_sell)
- `cross_signal` - Crossover detection (golden_cross, death_cross, none)

### Ichimoku Cloud Data Table (`ichimoku_data`)
- All fields from raw data table plus:
- `tenkan_sen` - Tenkan-sen (9-period conversion line)
- `kijun_sen` - Kijun-sen (26-period base line)
- `senkou_span_a` - Senkou Span A (leading span A, projected 26 periods forward)
- `senkou_span_b` - Senkou Span B (leading span B, projected 26 periods forward)
- `chikou_span` - Chikou Span (lagging span, shifted 26 periods backward)
- `cloud_color` - Cloud color indicator (green/red)
- `ichimoku_signal` - Overall signal (bullish/bearish/neutral)

## Data Summary

The bot currently maintains **317,730+ records** across **5 indicator tables** and **5 trading pairs**:

### Trading Pairs Data
| Trading Pair | 4-Hour Records | Daily Records | Date Range |
|-------------|----------------|---------------|------------|
| BTC/USDT    | 10,905         | 1,818         | Aug 2020 - Present |
| ETH/USDT    | 10,905         | 1,818         | Aug 2020 - Present |
| SOL/USDT    | 10,844         | 1,808         | Aug 2020 - Present |
| **SOL/BTC** | **10,906**     | **1,818**     | **Aug 2020 - Present** |
| **ETH/BTC** | **10,906**     | **1,818**     | **Aug 2020 - Present** |

### Technical Indicators Database
| Indicator | Records | Description |
|-----------|---------|-------------|
| **Raw Data** | 63,546 | Original OHLCV market data |
| **Gaussian Channel** | 63,546 | Volatility-based channel indicators |
| **Bollinger Bands** | 63,546 | Volatility bands with %B and squeeze detection |
| **SMA (50/200)** | 63,546 | Moving averages with Golden/Death Cross signals |
| **Ichimoku Cloud** | 63,546 | Complete Ichimoku system with all 5 components |
| **TOTAL** | **317,730** | **Comprehensive technical analysis dataset** |

*Note: All indicators calculated for the same time periods with consistent data coverage.*

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

### ✅ **Version 4.0 - Complete Technical Analysis Suite**
- **NEW: Ichimoku Cloud Indicator** with comprehensive 5-component analysis system
- **NEW: Bollinger Bands Indicator** with volatility analysis and squeeze detection
- **NEW: Simple Moving Average (50/200)** with Golden/Death Cross signals
- **Added new trading pairs**: SOL/BTC and ETH/BTC with full historical data
- **Interactive visualizer**: Dynamic user input for days, chart types, and trading pairs
- **Batch chart generation**: Create charts for all pairs automatically
- **Flexible time periods**: No longer limited to 90 days - specify any period
- **Multiple chart types**: Candlestick, Line, OHLC, and Volume analysis
- **Improved user experience**: Menu-driven interface with validation
- **High-quality exports**: 300 DPI PNG charts with organized naming

### 📊 **Advanced Technical Analysis Features**
- **Ichimoku Cloud**: Complete system with Tenkan-sen, Kijun-sen, Senkou spans, and Chikou span
- **Bollinger Bands**: %B position tracking, band width analysis, squeeze detection
- **SMA Crossovers**: Automated Golden Cross and Death Cross detection
- **Trend Strength**: Quantified trend analysis with -100 to +100 scoring
- **Advanced Signals**: Multiple signal types (strong_buy, buy, hold, sell, strong_sell)
- **Pattern Recognition**: Real-time analysis of market conditions and positioning
- **Multi-timeframe Analysis**: Comprehensive signals across 4h and daily intervals

### 🔧 **Technical Improvements**
- **Database expansion**: 317,730+ records across 5 indicator tables
- **Complete indicator suite**: 4 professional-grade technical indicators
- Fixed requirements.txt (removed non-existent sqlite3 dependency)
- Enhanced error handling and data validation
- Optimized database operations for better performance
- Updated all modules to support new trading pairs consistently
