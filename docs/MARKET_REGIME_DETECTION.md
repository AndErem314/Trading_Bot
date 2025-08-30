# Market Regime Detection Documentation

## Overview

The Market Regime Detection feature provides real-time analysis of market conditions for Bitcoin (BTC), Ethereum (ETH), and Solana (SOL). It uses the sophisticated `CryptoMarketRegimeDetector` class to classify market states and help traders adapt their strategies accordingly.

## Market Regimes

The system identifies five distinct market regimes:

1. **Bull Trend** 🟢
   - Strong, sustained upward momentum
   - ADX > 30 with positive directional movement
   - Ideal for trend-following strategies

2. **Bear Trend** 🔴
   - Strong, sustained downward momentum
   - ADX > 30 with negative directional movement
   - Consider defensive strategies or short positions

3. **Ranging / Accumulation** 🟡
   - Low volatility, sideways price action
   - ADX < 20 with normal volatility
   - Best for mean reversion strategies

4. **High Volatility / Breakout** 🟠
   - Rapidly expanding volatility
   - High ATR or Volume Z-scores
   - Reduce position sizes, wider stops needed

5. **Crypto Crash / Panic** 🔴🔴
   - Extreme downward momentum with panic selling
   - Price < 200 SMA, RSI < 25, ROC < -20%
   - Maximum caution advised

## Features

### 1. Real-Time Analysis
- Automatic refresh every 5 minutes
- Manual refresh available
- Configurable timeframes (1H, 4H, 1D)

### 2. Key Metrics Displayed
- **Current Price**: Live price updates
- **24h Change**: Percentage change over last 24 hours
- **ADX**: Average Directional Index (trend strength)
- **RSI**: Relative Strength Index (momentum)
- **Volatility Regime**: High or Normal
- **ATR Z-Score**: Volatility relative to historical average

### 3. Relative Strength Analysis
- ETH and SOL are compared against BTC
- Shows "Outperforming" or "Underperforming" status
- Helps identify the strongest assets

### 4. Historical Regime Visualization
- Click "View History" on any asset card
- Color-coded price chart showing regime changes
- Statistical breakdown of regime frequencies and durations

## How to Use

### Accessing the Feature

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Navigate to `http://localhost:5000`

3. Click on "Market Regime Detection" from the main menu

### Understanding the Display

Each asset (BTC, ETH, SOL) has its own card showing:
- Current regime with color-coded badge
- Price and 24-hour change
- Key technical indicators
- Relative strength (for ETH and SOL vs BTC)

### Customizing Analysis

1. **Timeframe Selection**:
   - 1 Hour: Short-term regime detection
   - 4 Hours: Medium-term (default)
   - 1 Day: Long-term regime detection

2. **Analysis Periods**:
   - 200 Periods: Recent analysis
   - 500 Periods: Standard analysis (default)
   - 1000 Periods: Extended historical analysis

### Strategy Recommendations by Regime

#### Bull Trend
- Use: SMA Golden Cross, Ichimoku Cloud, Parabolic SAR, MACD strategies
- Position sizing: 120% of base risk
- Focus on trend continuation patterns

#### Bear Trend
- Use: Parabolic SAR, MACD, RSI Divergence strategies
- Position sizing: 70% of base risk
- Consider defensive positions or shorts

#### Ranging / Accumulation
- Use: Bollinger Bands, RSI Divergence, Fibonacci Retracement strategies
- Position sizing: 100% of base risk
- Trade the range boundaries

#### High Volatility / Breakout
- Use: Gaussian Channel, Bollinger Bands strategies
- Position sizing: 50% of base risk
- Wait for volatility to settle

#### Crypto Crash
- Use: RSI oversold bounces, Fibonacci support levels
- Position sizing: 30% of base risk
- Extreme caution, consider staying out

## API Endpoints

### Get Current Regime
```
GET /api/market_regime?timeframe=4h&periods=500
```

Returns current regime and metrics for all three assets.

### Get Regime History
```
GET /api/market_regime_history/<asset>?timeframe=4h&periods=1000
```

Returns historical regime data for charting.

## Technical Details

The `CryptoMarketRegimeDetector` uses:
- ADX for trend strength measurement
- RSI for momentum and oversold/overbought conditions
- ATR Z-Score for volatility regime detection
- Volume Z-Score for unusual activity detection
- Rate of Change (ROC) for crash detection
- SMA 200 for long-term trend assessment

Asset-specific parameters are tuned for BTC, ETH, and general altcoins.

## Integration with Trading Strategies

See `backend/example_regime_strategy_integration.py` for examples of how to:
- Map strategies to different regimes
- Adjust risk parameters dynamically
- Create regime-aware trading systems

## Troubleshooting

1. **No Data Available Error**:
   - Ensure databases exist: `data/trading_data_BTC.db`, `data/trading_data_ETH.db`, `data/trading_data_SOL.db`
   - Run data collection first: `/update_data`

2. **Slow Loading**:
   - Reduce analysis periods
   - Use smaller timeframes
   - Check database performance

3. **Incorrect Regime Detection**:
   - Ensure sufficient historical data (minimum 200 periods)
   - Check for data gaps
   - Verify indicator calculations
