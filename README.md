# Trading Bot

A comprehensive Python trading bot that collects cryptocurrency market data, calculates technical indicators, and provides flexible data visualization capabilities. The bot stores BTC/USDT OHLCV data and indicators in a per-symbol SQLite database (data/trading_data_BTC.db), with automatic duplicate prevention, incremental updates, and data integrity validation.

## Project Structure

```
Trading_Bot/
├── backend/                          # Backend data processing modules
│   ├── __init__.py                  # Package initialization
│   ├── main.py                      # Main coordinator script for all processes
│   ├── data_manager.py              # Core database operations
│   ├── data_fetcher.py              # Data collection and management
│   ├── collect_historical_data.py   # Historical data collection utility
│   ├── Indicators/                  # Technical indicator calculators
│   │   ├── __init__.py              # Package initialization
│   │   ├── simple_moving_average.py # SMA (50/200) indicator calculator
│   │   ├── bollinger_bands.py       # Bollinger Bands indicator calculator
│   │   ├── ichimoku_cloud.py        # Ichimoku Cloud indicator calculator
│   │   ├── macd.py                  # MACD (12,26,9) indicator calculator
│   │   ├── rsi.py                   # RSI (14-period) indicator calculator
│   │   ├── parabolic_sar.py         # Parabolic SAR indicator calculator
│   │   ├── fibonacci_retracement.py # Fibonacci Retracement calculator
│   │   └── gaussian_channel.py      # Gaussian Channel indicator calculator
├── frontend/                         # Frontend data visualization modules
│   ├── __init__.py                  # Package initialization
│   ├── data_visualizer.py           # Advanced charting and visualization
│   └── charts/                      # Generated chart images (PNG exports)
├── data/                            # Per-symbol SQLite databases
│   └── trading_data_BTC.db          # BTC/USDT OHLCV data and indicators
├── run_trading_bot.py               # Main application entry point
├── visualize_data.py                # Data visualization entry point script
├── requirements.txt                 # Python package dependencies
├── pyproject.toml                   # Project configuration and metadata
├── .gitignore                       # Git ignore patterns
├── .python-version                  # Python version specification
└── README.md                        # Comprehensive project documentation
```

## Features

### Data Management
- **Normalized Database Schema (per-symbol)**: All OHLCV data stored in a single, normalized SQLite database per trading pair
- **Automatic Duplicate Prevention**: Built-in UNIQUE constraints prevent data duplication
- **Incremental Updates**: Smart fetching that only retrieves new data since last update
- **Data Integrity Validation**: Automatic validation of OHLCV data quality
- **Gap Detection & Filling**: Identifies and fills missing data gaps
- **Batch Processing**: Efficient bulk operations with transaction management
- **Comprehensive Logging**: Detailed logging for all operations

### Data Collection
- Fetches raw OHLCV data from cryptocurrency exchanges (using ccxt)
- Supports BTC/USDT pair only
- Historical data from January 1st, 2020 to present
- Timeframes: 1-hour, 4-hour, and daily intervals (1h, 4h, 1d)
- Exchange: Binance (easily extendable)
- Robust error handling and retry mechanisms with exponential backoff

### Technical Indicators

The bot implements **9 comprehensive technical indicators** with advanced analysis capabilities, all integrated with the per-symbol database:

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

#### **5. MACD (Moving Average Convergence Divergence)**
- **MACD Line**: 12-period EMA - 26-period EMA (momentum indicator)
- **Signal Line**: 9-period EMA of MACD line (crossover signals)
- **Histogram**: MACD line - Signal line (momentum visualization)
- **Crossover Signals**: Bullish (MACD above signal) and bearish (MACD below signal)
- **Zero Line Analysis**: Trend direction confirmation (above/below zero)
- **Divergence Detection**: Price vs MACD divergence for reversal signals
- **Momentum Analysis**: Increasing/decreasing momentum tracking

#### **6. RSI (Relative Strength Index) - 14 Period**
- **RSI Value**: Momentum oscillator (0-100 scale) using 14-period calculation
- **Overbought/Oversold**: Classic levels at 70/30 with signal generation
- **Trend Strength Analysis**: Multi-level categorization (strong_bullish, bullish, neutral, bearish, strong_bearish)
- **Divergence Detection**: Bullish/bearish divergences between RSI and price movement
- **Support/Resistance**: Dynamic RSI level identification (30, 40, 50, 60, 70)
- **Momentum Shift**: Detection of significant RSI changes (>5 points)
- **RSI Smoothing**: 5 and 10-period SMAs of RSI for trend confirmation
- **Wilder's Smoothing**: Authentic RSI calculation using Wilder's exponential averaging

#### **7. Parabolic SAR (Stop and Reverse)**
- **SAR Value**: Dynamic stop-loss and trend reversal indicator
- **Trend Detection**: Real-time identification of uptrend/downtrend periods
- **Reversal Signals**: Automatic detection of trend change points with boolean flags
- **Signal Strength**: Percentage-based distance measurement between price and SAR
- **Acceleration Factor**: Progressive AF from 0.02 to 0.20 for trend acceleration
- **Pattern Analysis**: Current trend assessment with persistence measurement
- **Reversal History**: Complete tracking of trend changes with timestamps
- **Signal Classification**: Weak/moderate/strong categories based on price-SAR distance

#### **8. Fibonacci Retracement**
- **Retracement Levels**: Industry-standard Fibonacci ratios (23.6%, 38.2%, 50.0%, 61.8%, 76.4%)
- **Range-Based Calculation**: Consistent levels based on overall high/low range
- **Support/Resistance Identification**: Key price levels for technical analysis
- **Simple Implementation**: Clean, focused design suitable for combination with other indicators
- **Price Level Analysis**: Determine current price position relative to Fibonacci levels
- **Market Structure**: Identify potential reversal zones and continuation levels
- **Clean Data Structure**: Dedicated storage for each Fibonacci level
- **Historical Consistency**: Same retracement levels maintained across all time periods

#### **Technical Features**
- Per-symbol SQLite database with normalized, optimized schema
- Supports batch processing of multiple symbols and timeframes
- Real-time pattern analysis and signal generation
- Command-line interface for flexible operation
- Foreign key relationships for data integrity
- Optimized indexes for common query patterns
- Connection pooling and efficient database management

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

### Main Trading Bot
```bash
# Run complete pipeline (collect data and calculate all indicators)
python3 run_trading_bot.py --mode both

# Collect raw data only
  python3 run_trading_bot.py --mode collect --start-date 2020-01-01

# Calculate all technical indicators
python3 run_trading_bot.py --mode all_indicators

# Calculate simple moving averages only
python3 run_trading_bot.py --mode calculate

# Interactive indicator selection menu
python3 run_trading_bot.py --mode interactive

# Custom symbols and timeframes
python3 run_trading_bot.py --symbols BTC/USDT ETH/USDT --timeframes 1h 4h 1d --start-date 2022-01-01
```

### 🎯 Interactive Indicator Selection (NEW!)

The Trading Bot now features an interactive menu system for selecting specific technical indicators:

```bash
# Launch interactive mode
python3 run_trading_bot.py --mode interactive
# OR
python3 run_trading_bot.py --interactive
```

**Interactive Menu Options:**
- `1` - 🔢 Simple Moving Averages (SMA 50/200)
- `2` - 📈 Bollinger Bands
- `3` - ☁️ Ichimoku Cloud
- `4` - 📊 MACD (12,26,9)
- `5` - 📉 RSI (14-period)
- `6` - 🔄 Parabolic SAR
- `7` - 🌀 Fibonacci Retracement
- `8` - 📡 Gaussian Channel
- `9` - 🚀 ALL Indicators
- `0` - ❌ Exit

**Example Interactive Session:**
```
🎯 TECHNICAL INDICATORS MENU
==================================================
💱 Symbols: BTC/USDT
⏰ Timeframes: 1h, 4h, 1d

Select indicator to calculate:
  1. 🔢 Simple Moving Averages (SMA 50/200)
  2. 📈 Bollinger Bands
  3. ☁️ Ichimoku Cloud
  4. 📊 MACD (12,26,9)
  5. 📉 RSI (14-period)
  6. 🔄 Parabolic SAR
  7. 🌀 Fibonacci Retracement
  8. 📡 Gaussian Channel
  9. 🚀 ALL Indicators
  0. ❌ Exit

Enter your choice [1-9, 0 to exit]: 9

🚀 Running ALL Technical Indicators...
✅ Completed 8/8 indicators successfully!
```

### Data Management
```bash
# Using the data collector and data manager directly
from backend.data_fetcher import DataCollector
from backend.data_manager import DataManager

# Initialize the collector
collector = DataCollector()

# Update all data
symbols = ['BTC/USDT', 'ETH/USDT']
timeframes = ['1h', '4h', '1d']
collector.update_all_data(symbols, timeframes)

# Get database summary
data_manager = DataManager()
summary = data_manager.get_data_summary()
```


### Data Visualization

The visualizer now offers three modes for maximum flexibility:

```bash
# Launch interactive visualizer
python visualize_data.py
```

**Mode 1: Interactive Mode (Default)**
- Select trading pairs from available data
- Choose timeframes (1h, 4h, 1d)
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

Select a trading pair (1-5): 4
Select timeframe (1-2): 1
Enter number of days to visualize (or 'all'): 30
Select chart type (1-4) [default: candlestick]: 1
Save chart to file? (y/n): y
```

### Individual Indicator Calculations

**🎯 Interactive Method (Recommended):**
```bash
# Use the interactive menu for easy indicator selection
python3 run_trading_bot.py --interactive

# Interactive mode with custom symbols
python3 run_trading_bot.py --interactive --symbols BTC/USDT SOL/USDT
```

**🔧 Direct Backend Access (Advanced):**
```bash
# Calculate specific indicators directly (uses per-symbol database)
python3 backend/main.py --mode all_indicators  # All indicators via backend

# Note: Individual standalone scripts use the Indicators/ subfolder:
# All indicator calculators are now in backend/Indicators/
```

### Advanced Usage - Run Individual Modules

**🚀 Root Directory (Recommended):**
```bash
# Complete pipeline with all options
python3 run_trading_bot.py --mode both              # Complete pipeline
python3 run_trading_bot.py --interactive            # Interactive indicator menu
python3 run_trading_bot.py --status                 # Database status

# Data visualization
python3 visualize_data.py                           # Advanced visualization
```

**🔧 Backend Directory (Advanced):**
```bash
# Run backend modules directly
cd backend
python3 main.py --mode both              # Complete pipeline
python3 main.py --mode collect           # Data collection only

# Individual indicator calculators (in Indicators/ subfolder)
# Note: Use interactive mode instead for easier access
python3 -c "from Indicators import SimpleMovingAverageCalculator; print('SMA Calculator Available')"

# Run frontend modules directly
cd ../frontend  
python3 data_visualizer.py               # Advanced visualization
```

## Database Architecture

The project uses a per-symbol SQLite database with a normalized schema. For BTC/USDT, data is stored in:

### Database Structure (BTC/USDT)
```
data/
└── trading_data_BTC.db          # Per-symbol database
    ├── symbols                  # Trading pair lookup table
    ├── timeframes               # Timeframe lookup table
    ├── ohlcv_data               # Raw OHLCV market data
    ├── sma_indicator            # Simple Moving Average indicators
    ├── bollinger_bands_indicator # Bollinger Bands indicators
    ├── ichimoku_indicator       # Ichimoku Cloud indicators
    ├── macd_indicator           # MACD indicators
    ├── rsi_indicator            # RSI indicators
    ├── parabolic_sar_indicator  # Parabolic SAR indicators
    ├── fibonacci_retracement_indicator # Fibonacci Retracement indicators
    └── gaussian_channel_indicator # Gaussian Channel indicators
```

### Database Schema

#### **Core Tables**

**Table: `symbols`** (Lookup table)
- `id` - Primary key
- `symbol` - Trading pair (e.g., BTC/USDT)
- `created_at` - Record creation timestamp

**Table: `timeframes`** (Lookup table)
- `id` - Primary key
- `timeframe` - Time interval (e.g., 4h, 1d)
- `created_at` - Record creation timestamp

**Table: `ohlcv_data`** (Main OHLCV data)
- `id` - Primary key
- `symbol_id` - Foreign key to symbols table
- `timeframe_id` - Foreign key to timeframes table
- `timestamp` - Data timestamp
- `open, high, low, close, volume` - OHLCV data
- `created_at` - Record creation timestamp
- **UNIQUE constraint**: (symbol_id, timeframe_id, timestamp) prevents duplicates

#### **Technical Indicator Tables**

**Table: `sma_indicator`**
- Foreign keys to ohlcv_data, symbol, and timeframe tables
- `sma_50` - 50-period simple moving average
- `sma_200` - 200-period simple moving average
- `sma_ratio` - SMA 50/200 ratio (trend strength)
- `price_vs_sma50` - Price position vs SMA 50 (%)
- `price_vs_sma200` - Price position vs SMA 200 (%)
- `trend_strength` - Quantified trend analysis (-100 to +100)
- `sma_signal` - Trading signal (strong_buy, buy, hold, sell, strong_sell)
- `cross_signal` - Crossover detection (golden_cross, death_cross, none)

**Table: `bollinger_bands_indicator`**
- Foreign keys to ohlcv_data, symbol, and timeframe tables
- `bb_upper` - Upper Bollinger Band (SMA + 2σ)
- `bb_middle` - Middle Bollinger Band (20-period SMA)
- `bb_lower` - Lower Bollinger Band (SMA - 2σ)
- `bb_width` - Band width (volatility measure)
- `bb_percent` - %B position indicator (0-1 scale)

**Table: `ichimoku_indicator`**
- Foreign keys to ohlcv_data, symbol, and timeframe tables
- `tenkan_sen` - Tenkan-sen (9-period conversion line)
- `kijun_sen` - Kijun-sen (26-period base line)
- `senkou_span_a` - Senkou Span A (leading span A, projected 26 periods forward)
- `senkou_span_b` - Senkou Span B (leading span B, projected 26 periods forward)
- `chikou_span` - Chikou Span (lagging span, shifted 26 periods backward)
- `cloud_color` - Cloud color indicator (green/red)
- `ichimoku_signal` - Overall signal (bullish/bearish/neutral)

**Table: `macd_indicator`**
- Foreign keys to ohlcv_data, symbol, and timeframe tables
- `ema_12` - 12-period exponential moving average
- `ema_26` - 26-period exponential moving average
- `macd_line` - MACD line (EMA12 - EMA26)
- `signal_line` - Signal line (9-period EMA of MACD line)
- `histogram` - MACD histogram (MACD line - Signal line)
- `macd_signal` - MACD signal (bullish, bearish, strong_bullish, strong_bearish, neutral)

**Table: `rsi_indicator`**
- Foreign keys to ohlcv_data, symbol, and timeframe tables
- `rsi` - RSI value (0-100 scale, 14-period Wilder's smoothing)
- `rsi_sma_5` - 5-period simple moving average of RSI
- `rsi_sma_10` - 10-period simple moving average of RSI
- `overbought` - Overbought signal (RSI > 70)
- `oversold` - Oversold signal (RSI < 30)
- `trend_strength` - Trend categorization (strong_bullish, bullish, neutral, bearish, strong_bearish)
- `divergence_signal` - Divergence detection (bullish, bearish, none)
- `momentum_shift` - Significant RSI change detection (>5 points)
- `support_resistance` - Dynamic RSI support/resistance levels

**Table: `parabolic_sar_indicator`**
- Foreign keys to ohlcv_data, symbol, and timeframe tables
- `parabolic_sar` - Parabolic SAR value (dynamic stop-loss level)
- `trend` - Current trend direction (up/down)
- `reversal_signal` - Trend reversal detection (boolean flag)
- `signal_strength` - Signal strength based on price-SAR distance (percentage)

**Table: `fibonacci_retracement_indicator`**
- Foreign keys to ohlcv_data, symbol, and timeframe tables
- `level_23_6` - 23.6% Fibonacci retracement level
- `level_38_2` - 38.2% Fibonacci retracement level
- `level_50_0` - 50.0% Fibonacci retracement level (midpoint)
- `level_61_8` - 61.8% Fibonacci retracement level (golden ratio)
- `level_76_4` - 76.4% Fibonacci retracement level

**Table: `gaussian_channel_indicator`**
- Foreign keys to ohlcv_data, symbol, and timeframe tables
- `gc_upper` - Upper Gaussian Channel band
- `gc_middle` - Middle Gaussian Channel band (moving average)
- `gc_lower` - Lower Gaussian Channel band

### Database Architecture Benefits
- **Data Integrity**: Foreign key relationships ensure consistent data
- **Performance**: Optimized indexes and normalized schema for faster queries
- **No Duplicates**: Built-in unique constraints prevent data duplication
- **Atomic Operations**: Transaction-based operations ensure data consistency
- **Scalability**: Easy to add new indicators and extend the schema
- **Maintenance**: Single database file for simplified backup and management
- **Development**: Consistent data access patterns across all indicators
- **Memory Efficiency**: Normalized schema reduces storage requirements

#### Data Quality Notes
- Rare historical gaps from the exchange may exist on the 1h timeframe. To ensure strict continuity for analysis and indicators, missing 1h bars may be synthesized as follows:
  - Use the previous available 1h close if present
  - Otherwise, use the containing 4h bar’s close
  - Set open=high=low=close to that value and volume=0
- This synthesis is applied only to the exact missing hours and does not affect existing real data.

## Data Summary

Note: The following historical summary reflects an earlier unified database setup (archival). Current releases use per-symbol databases. For live counts, run:

python3 run_trading_bot.py --status

### Trading Pairs Data
| Trading Pair | 4-Hour Records | Daily Records | Date Range |
|-------------|----------------|---------------|-----------|
| BTC/USDT    | 10,905         | 1,818         | 2020-01-01 - Present |
| ETH/USDT    | 10,905         | 1,818         | 2020-01-01 - Present |
| SOL/USDT    | 10,844         | 1,808         | 2020-01-01 - Present |
| **SOL/BTC** | **10,906**     | **1,818**     | **2020-01-01 - Present** |
| **ETH/BTC** | **10,906**     | **1,818**     | **2020-01-01 - Present** |

### Historical Database Summary (archival)
| Component | Records/Tables | Size | Description |
|-----------|----------------|------|-------------|
| **Core OHLCV Data** | 64,499 records | ~59MB | Raw market data in unified normalized schema |
| **Technical Indicators** | 8 indicator tables | Included | All indicators linked via foreign keys |
| **Data Integrity** | 100% validated | N/A | No duplicate or invalid records |
| **Schema Design** | Normalized | Optimized | Foreign key relationships, unique constraints |
| **Total Database** | **1 unified file** | **~59MB** | **Complete trading system in single database** |

### Historical comparison (archival)
| Feature | Legacy System | Unified System |
|---------|--------------|----------------|
| Database Files | 9 separate files (~97MB) | 1 unified file (~59MB) |
| Data Duplicates | Possible | Prevented by design |
| Data Integrity | Manual validation | Automatic with foreign keys |
| Performance | Basic file-based | Optimized with indexes |
| Scalability | Limited | High with normalized schema |
| Maintenance | Complex (9 files) | Simple (1 file) |

*Note: All indicators calculated with consistent data coverage and automatic duplicate prevention.*

## Configuration

### Default Settings
- **Symbols**: BTC/USDT, ETH/USDT, SOL/USDT, SOL/BTC, ETH/BTC
- **Timeframes**: 1h (hourly), 4h (4-hour), 1d (daily)
- **Historical start**: January 1st, 2020
- **Database Architecture**: Per-symbol normalized database
  - **Single Database (per-symbol)**: `data/trading_data_BTC.db` - OHLCV data and indicators for BTC/USDT
  - **Normalized Schema**: Foreign key relationships with unique constraints
  - **Automatic Duplicate Prevention**: Built-in data integrity
- **Exchange**: Binance (easily extendable to other exchanges)
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

### Workflow Examples
```bash
# Daily update routine (incremental data collection)
python backend/main.py --mode collect

# Weekly indicator recalculation
python backend/main.py --mode all_indicators

# Data integrity validation
# (Automatic with normalized schema - no manual validation needed)
```

## Recent Updates

### ✅ **Version 7.1 - Interactive Indicator Selection - August 2025**
- **🎯 NEW: Interactive Indicator Menu** - User-friendly interface for selecting specific indicators
- **📊 Individual Indicator Control** - Choose from 8 technical indicators or run all at once
- **🎨 Professional UI** - Beautiful menu with emojis, progress tracking, and status feedback
- **🔄 Continuous Operation** - Menu stays active until user chooses to exit
- **⚡ Enhanced User Experience** - Real-time progress updates and error handling
- **🚀 Multiple Access Methods** - `--mode interactive` or `--interactive` flag
- **🎛️ Custom Configuration** - Works with custom symbols and timeframes
- **✅ Production Ready** - Fully integrated with the current database-backed workflow

### ✅ **Version 7.0 - Database Architecture Update (historical, Aug 2025)**
- **🚀 MAJOR: Unified Database Architecture** - Complete migration to single normalized database
- **🔧 Database Consolidation**: From 9 separate files (~97MB) to 1 unified file (~59MB)
- **⚡ Performance Optimization**: Normalized schema with foreign key relationships
- **🛡️ Data Integrity**: Built-in duplicate prevention and automatic validation
- **🔄 Incremental Updates**: Smart fetching that only retrieves new data since last update
- **📊 Complete System Overhaul**: All indicator calculators updated for unified system
- **🧹 Legacy Cleanup**: Removed deprecated scripts and databases
- **✅ Production Ready**: Fully tested and validated unified workflow

### ✅ **Project Name Standardization - August 2025**
- **Project Name Correction**: Standardized project name from "Traiding_Bot" to "Trading_Bot"
- **Consistent Naming**: Updated all references to use the correct spelling throughout the project
- **Repository Structure**: Maintained consistent "Trading_Bot" naming convention
- **Documentation Update**: Ensured all documentation reflects the correct project name

### ✅ **Version 6.3 - Fibonacci Retracement Indicator Integration**
- **NEW: Fibonacci Retracement Indicator** - Classic retracement levels for support/resistance analysis
- **Standard Fibonacci Ratios**: Industry-standard levels (23.6%, 38.2%, 50.0%, 61.8%, 76.4%)
- **Range-Based Calculation**: Consistent levels based on overall high/low range
- **Clean Implementation**: Simple, focused design suitable for combination with other indicators
- **Dedicated Database**: `fibonacci_retracement_data.db` (~9.2MB) with 63,546 records
- **Perfect for Integration**: Foundation for combining with other technical analysis tools
- **Demo Results**: BTC/USDT range $9,825-$123,218 with current price at 4.2% retracement

### ✅ **Version 6.2 - Parabolic SAR Indicator Integration**
- **NEW: Parabolic SAR Indicator** - Stop and Reverse trend following indicator
- **Advanced Trend Detection**: Real-time uptrend/downtrend identification with reversal signals
- **Signal Analysis**: 7,640+ trend reversals detected with signal strength measurement
- **Pattern Recognition**: Trend persistence analysis and reversal history tracking
- **Dedicated Database**: `parabolic_sar_data.db` (~10.4MB) with comprehensive SAR data
- **Complete Integration**: Seamlessly integrated with existing indicator architecture
- **Professional Features**: Acceleration factor progression, signal classification, pattern analysis

### ✅ **Version 6.1 - Database Architecture Optimization**
- **NEW: Dedicated Database Structure** - Split monolithic database into 7 specialized files
- **Performance Enhancement**: Faster access to individual indicators with dedicated databases
- **Improved Organization**: Each indicator has its own optimized database file
- **Better Scalability**: Easier maintenance, backup, and development workflows
- **Database Files**: `raw_market_data.db` + 6 dedicated indicator databases (~77.7MB total)
- **Architecture Benefits**: Isolated indicator development and individual optimization

### ✅ **Version 6.0 - Complete RSI Integration**
- **NEW: RSI Indicator (14-period)** with overbought/oversold levels and momentum analysis
- **Advanced RSI Features**: Wilder's smoothing, divergence detection, support/resistance levels
- **Complete Technical Suite**: Now 6 professional-grade indicators with momentum focus
- **Enhanced Signal Analysis**: RSI trend strength categorization and momentum shift detection
- **Comprehensive Database**: 444,822+ records across 7 indicator tables
- **RSI Pattern Recognition**: Real-time analysis with 5/10-period RSI smoothing

### ✅ **Version 5.0 - Professional MACD Integration**
- **NEW: MACD Indicator (12,26,9)** with momentum analysis and divergence detection
- **Complete Technical Suite**: 5 professional-grade indicators
- **Enhanced Signal Analysis**: Multi-component confirmation across all indicators
- **Advanced Momentum Tracking**: MACD crossovers, zero-line analysis, and trend strength
- **Pattern Recognition**: Real-time analysis with divergence risk assessment

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

### 📈 **Advanced Technical Analysis Features**
- **Fibonacci Retracement Analysis**: Classic retracement levels for support/resistance identification
- **Parabolic SAR Analysis**: Trend following with automatic reversal detection and signal strength measurement
- **RSI Analysis**: 14-period momentum oscillator with overbought/oversold signals and divergence detection
- **MACD Analysis**: Complete momentum system with signal crossovers and divergence detection
- **Ichimoku Cloud**: Complete system with Tenkan-sen, Kijun-sen, Senkou spans, and Chikou span
- **Bollinger Bands**: %B position tracking, band width analysis, squeeze detection
- **SMA Crossovers**: Automated Golden Cross and Death Cross detection
- **Trend Strength**: Quantified trend analysis with -100 to +100 scoring
- **Advanced Signals**: Multiple signal types (strong_buy, buy, hold, sell, strong_sell)
- **Pattern Recognition**: Real-time analysis of market conditions and positioning
- **Multi-timeframe Analysis**: Comprehensive signals across 4h and daily intervals

### 🔧 **Technical Improvements**
- **Interactive Interface**: User-friendly menu system for individual indicator selection
- **Normalized schema**: SQLite schema optimized for OHLCV and indicators
- **Complete Indicator Suite**: 8 professional-grade technical indicators with foreign key relationships
- **Enhanced User Experience**: Real-time progress tracking and professional UI
- **Data Integrity**: 100% validated data with automatic duplicate prevention
- **Performance Optimization**: Normalized schema with optimized indexes
- **Memory Efficiency**: ~40% reduction in storage requirements vs legacy system
- **Atomic Operations**: Transaction-based operations ensure data consistency
- **Incremental Updates**: Smart data fetching reduces API calls and processing time
- **Enhanced Error Handling**: Comprehensive error handling with retry logic
- **Connection Pooling**: Efficient database connection management
- **Flexible Operation Modes**: Batch processing and interactive selection
- **Production Ready**: Fully tested unified workflow with comprehensive validation
