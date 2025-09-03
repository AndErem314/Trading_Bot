#!/bin/bash
# Examples to run all available strategies separately with run_single_strategy.py

# 1. Bollinger Bands Mean Reversion Strategy
echo "Running Bollinger Bands Strategy..."
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --params '{"symbol": "BTC/USDT", "bb_length": 20, "bb_std": 2.0, "rsi_length": 14, "rsi_oversold": 35, "rsi_overbought": 65}'

# 2. RSI Momentum Divergence Strategy (your example)
echo "Running RSI Divergence Strategy..."
python backend/backtesting/scripts/run_single_strategy.py rsi_divergence --params '{"symbol": "BTC/USDT", "rsi_length": 14, "rsi_sma_fast": 5, "rsi_sma_slow": 10, "rsi_oversold": 30, "rsi_overbought": 70, "momentum_lookback": 5, "divergence_lookback": 20}'

# 3. MACD Momentum Crossover Strategy
echo "Running MACD Momentum Strategy..."
python backend/backtesting/scripts/run_single_strategy.py macd_momentum --params '{"symbol": "BTC/USDT", "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "momentum_period": 14, "atr_period": 14, "volume_threshold": 1.5}'

# 4. Volatility Breakout Short Strategy
echo "Running Volatility Breakout Short Strategy..."
python backend/backtesting/scripts/run_single_strategy.py volatility_breakout_short --params '{"symbol": "BTC/USDT", "atr_period": 14, "lookback_period": 20, "volume_multiplier": 2.0, "rsi_period": 14, "rsi_extreme": 20, "atr_stop_multiplier": 2.0, "atr_trail_multiplier": 1.5}'

# 5. Ichimoku Cloud Breakout Strategy
echo "Running Ichimoku Cloud Strategy..."
python backend/backtesting/scripts/run_single_strategy.py ichimoku_cloud --params '{"symbol": "BTC/USDT", "tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52, "displacement": 26}'

# 6. Parabolic SAR Trend Following Strategy
echo "Running Parabolic SAR Strategy..."
python backend/backtesting/scripts/run_single_strategy.py parabolic_sar --params '{"symbol": "BTC/USDT", "start": 0.02, "increment": 0.02, "maximum": 0.2}'

# 7. Fibonacci Retracement Support/Resistance Strategy
echo "Running Fibonacci Retracement Strategy..."
python backend/backtesting/scripts/run_single_strategy.py fibonacci_retracement --params '{"symbol": "BTC/USDT", "lookback_period": 50, "fib_levels": [0.236, 0.382, 0.5, 0.618, 0.786]}'

# 8. Gaussian Channel Breakout Mean Reversion Strategy
echo "Running Gaussian Channel Strategy..."
python backend/backtesting/scripts/run_single_strategy.py gaussian_channel --params '{"symbol": "BTC/USDT", "period": 20, "std_dev": 2.0, "adaptive": true}'

# Additional Examples with different options:

# Run with optimization (finds best parameters automatically)
echo "Running with optimization..."
python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --optimize

# Run with LLM analysis
echo "Running with LLM analysis..."
python backend/backtesting/scripts/run_single_strategy.py rsi_divergence --params '{"symbol": "BTC/USDT"}' --analyze

# Run with optimization and LLM analysis
echo "Running with optimization and analysis..."
python backend/backtesting/scripts/run_single_strategy.py macd_momentum --optimize --analyze

# Run with different optimization methods
echo "Running with Bayesian optimization..."
python backend/backtesting/scripts/run_single_strategy.py ichimoku_cloud --optimize --method bayesian

# Run with specific LLM provider
echo "Running with Gemini LLM analysis..."
python backend/backtesting/scripts/run_single_strategy.py gaussian_channel --params '{"symbol": "BTC/USDT"}' --analyze --llm gemini
