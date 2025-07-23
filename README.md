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
│   ├── macd.py                # Calculates MACD (12,26,9) indicators
│   ├── rsi.py                 # Calculates RSI (14-period) indicators
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

The bot implements **8 comprehensive technical indicators** with advanced analysis capabilities:

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
python backend/macd.py                # MACD (12,26,9) momentum analysis
python backend/rsi.py                 # RSI (14-period) momentum oscillator
python backend/parabolic_sar.py       # Parabolic SAR (Stop and Reverse) trend indicator
python backend/fibonacci_retracement.py # Fibonacci Retracement levels
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
python macd.py
python rsi.py
python parabolic_sar.py
python fibonacci_retracement.py

# Run frontend modules directly
cd frontend  
python data_visualizer.py
```

## Database Architecture

The project uses a **dedicated database structure** with separate SQLite database files for optimal performance and organization:

### Database Files Structure
```
data/
├── raw_market_data.db          # Raw OHLCV market data
├── gaussian_channel_data.db    # Gaussian Channel indicators
├── bollinger_bands_data.db     # Bollinger Bands indicators
├── sma_data.db                 # Simple Moving Average indicators
├── ichimoku_data.db            # Ichimoku Cloud indicators
├── macd_data.db                # MACD indicators
├── rsi_data.db                 # RSI indicators
├── parabolic_sar_data.db       # Parabolic SAR indicators
└── fibonacci_retracement_data.db # Fibonacci Retracement indicators
```

### Database Schema

#### **Raw Market Data** (`raw_market_data.db`)
**Table: `raw_data`**
- `id` - Primary key
- `symbol` - Trading pair (e.g., BTC/USDT)
- `timeframe` - Time interval (e.g., 4h, 1d)
- `timestamp` - Data timestamp
- `open, high, low, close, volume` - OHLCV data
- `created_at` - Record creation timestamp

#### **Gaussian Channel Database** (`gaussian_channel_data.db`)
**Table: `gaussian_channel_data`**
- All OHLCV fields plus:
- `gc_upper` - Upper Gaussian Channel band
- `gc_middle` - Middle Gaussian Channel band (moving average)
- `gc_lower` - Lower Gaussian Channel band

#### **Bollinger Bands Database** (`bollinger_bands_data.db`)
**Table: `bollinger_bands_data`**
- All OHLCV fields plus:
- `bb_upper` - Upper Bollinger Band (SMA + 2σ)
- `bb_middle` - Middle Bollinger Band (20-period SMA)
- `bb_lower` - Lower Bollinger Band (SMA - 2σ)
- `bb_width` - Band width (volatility measure)
- `bb_percent` - %B position indicator (0-1 scale)

#### **Simple Moving Average Database** (`sma_data.db`)
**Table: `sma_data`**
- All OHLCV fields plus:
- `sma_50` - 50-period simple moving average
- `sma_200` - 200-period simple moving average
- `sma_ratio` - SMA 50/200 ratio (trend strength)
- `price_vs_sma50` - Price position vs SMA 50 (%)
- `price_vs_sma200` - Price position vs SMA 200 (%)
- `trend_strength` - Quantified trend analysis (-100 to +100)
- `sma_signal` - Trading signal (strong_buy, buy, hold, sell, strong_sell)
- `cross_signal` - Crossover detection (golden_cross, death_cross, none)

#### **Ichimoku Cloud Database** (`ichimoku_data.db`)
**Table: `ichimoku_data`**
- All OHLCV fields plus:
- `tenkan_sen` - Tenkan-sen (9-period conversion line)
- `kijun_sen` - Kijun-sen (26-period base line)
- `senkou_span_a` - Senkou Span A (leading span A, projected 26 periods forward)
- `senkou_span_b` - Senkou Span B (leading span B, projected 26 periods forward)
- `chikou_span` - Chikou Span (lagging span, shifted 26 periods backward)
- `cloud_color` - Cloud color indicator (green/red)
- `ichimoku_signal` - Overall signal (bullish/bearish/neutral)

#### **MACD Database** (`macd_data.db`)
**Table: `macd_data`**
- All OHLCV fields plus:
- `ema_12` - 12-period exponential moving average
- `ema_26` - 26-period exponential moving average
- `macd_line` - MACD line (EMA12 - EMA26)
- `signal_line` - Signal line (9-period EMA of MACD line)
- `histogram` - MACD histogram (MACD line - Signal line)
- `macd_signal` - MACD signal (bullish, bearish, strong_bullish, strong_bearish, neutral)

#### **RSI Database** (`rsi_data.db`)
**Table: `rsi_data`**
- All OHLCV fields plus:
- `rsi` - RSI value (0-100 scale, 14-period Wilder's smoothing)
- `rsi_sma_5` - 5-period simple moving average of RSI
- `rsi_sma_10` - 10-period simple moving average of RSI
- `overbought` - Overbought signal (RSI > 70)
- `oversold` - Oversold signal (RSI < 30)
- `trend_strength` - Trend categorization (strong_bullish, bullish, neutral, bearish, strong_bearish)
- `divergence_signal` - Divergence detection (bullish, bearish, none)
- `momentum_shift` - Significant RSI change detection (>5 points)
- `support_resistance` - Dynamic RSI support/resistance levels

#### **Parabolic SAR Database** (`parabolic_sar_data.db`)
**Table: `parabolic_sar_data`**
- All OHLCV fields plus:
- `parabolic_sar` - Parabolic SAR value (dynamic stop-loss level)
- `trend` - Current trend direction (up/down)
- `reversal_signal` - Trend reversal detection (boolean flag)
- `signal_strength` - Signal strength based on price-SAR distance (percentage)

#### **Fibonacci Retracement Database** (`fibonacci_retracement_data.db`)
**Table: `fibonacci_retracement_data`**
- Standard fields: symbol, timeframe, timestamp
- `level_23_6` - 23.6% Fibonacci retracement level
- `level_38_2` - 38.2% Fibonacci retracement level
- `level_50_0` - 50.0% Fibonacci retracement level (midpoint)
- `level_61_8` - 61.8% Fibonacci retracement level (golden ratio)
- `level_76_4` - 76.4% Fibonacci retracement level

### Database Architecture Benefits
- **Performance**: Faster access to specific indicators
- **Organization**: Clean separation of concerns
- **Scalability**: Easy to add new indicators
- **Maintenance**: Individual backup and optimization
- **Development**: Isolated indicator development

## Data Summary

The bot currently maintains **571,914+ records** across **9 indicator tables** and **5 trading pairs**:

### Trading Pairs Data
| Trading Pair | 4-Hour Records | Daily Records | Date Range |
|-------------|----------------|---------------|------------|
| BTC/USDT    | 10,905         | 1,818         | Aug 2020 - Present |
| ETH/USDT    | 10,905         | 1,818         | Aug 2020 - Present |
| SOL/USDT    | 10,844         | 1,808         | Aug 2020 - Present |
| **SOL/BTC** | **10,906**     | **1,818**     | **Aug 2020 - Present** |
| **ETH/BTC** | **10,906**     | **1,818**     | **Aug 2020 - Present** |

### Database Files Summary
| Database File | Records | Size | Description |
|---------------|---------|------|-------------|
| **raw_market_data.db** | 63,546 | ~9.5MB | Original OHLCV market data for all trading pairs |
| **gaussian_channel_data.db** | 63,546 | ~11.3MB | Volatility-based channel indicators |
| **bollinger_bands_data.db** | 63,546 | ~12.5MB | Volatility bands with %B and squeeze detection |
| **sma_data.db** | 63,546 | ~13.7MB | Moving averages with Golden/Death Cross signals |
| **ichimoku_data.db** | 63,546 | ~13.3MB | Complete Ichimoku system with all 5 components |
| **macd_data.db** | 63,546 | ~13.3MB | Moving Average Convergence Divergence with momentum analysis |
| **rsi_data.db** | 63,546 | ~4.1MB | Relative Strength Index with overbought/oversold and divergence analysis |
| **parabolic_sar_data.db** | 63,546 | ~10.4MB | Parabolic SAR with trend reversal detection and signal strength |
| **fibonacci_retracement_data.db** | 63,546 | ~9.2MB | Fibonacci retracement levels with standard ratios |
| **TOTAL** | **571,914** | **~97.3MB** | **Complete technical analysis dataset across 9 dedicated databases** |

*Note: All indicators calculated for the same time periods with consistent data coverage.*

## Configuration

### Default Settings
- **Symbols**: BTC/USDT, ETH/USDT, SOL/USDT, SOL/BTC, ETH/BTC
- **Timeframes**: 4h (4-hour), 1d (daily)
- **Historical start**: August 1st, 2020
- **Database Architecture**: Dedicated database files for optimal performance
  - **Raw Data**: `data/raw_market_data.db` - OHLCV market data
  - **Indicators**: 8 separate database files for each technical indicator
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
- **Database expansion**: 571,914+ records across 9 indicator tables
- **Complete indicator suite**: 8 professional-grade technical indicators
- **Fibonacci Retracement Integration**: Classic retracement levels with clean implementation
- **Parabolic SAR Integration**: Full trend reversal system with signal strength analysis
- **RSI Integration**: Full momentum oscillator with Wilder's smoothing and trend analysis
- **MACD Integration**: Full momentum analysis with EMA calculations and histogram tracking
- **Dedicated Database Architecture**: 9 optimized database files for maximum performance
- Fixed requirements.txt (removed non-existent sqlite3 dependency)
- Enhanced error handling and data validation
- Optimized database operations for better performance
- Updated all modules to support new trading pairs consistently
