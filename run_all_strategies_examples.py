#!/usr/bin/env python3
"""
Examples to run all available strategies separately
Each strategy can be run individually from the command line
"""

import subprocess
import json

# Define all available strategies with their parameters
STRATEGIES = {
    "bollinger_bands": {
        "description": "Bollinger Bands Mean Reversion Strategy",
        "params": {
            "symbol": "BTC/USDT",
            "bb_length": 20,
            "bb_std": 2.0,
            "rsi_length": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65
        }
    },
    "rsi_divergence": {
        "description": "RSI Momentum Divergence Strategy",
        "params": {
            "symbol": "BTC/USDT",
            "rsi_length": 14,
            "rsi_sma_fast": 5,
            "rsi_sma_slow": 10,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "momentum_lookback": 5,
            "divergence_lookback": 20
        }
    },
    "macd_momentum": {
        "description": "MACD Momentum Crossover Strategy",
        "params": {
            "symbol": "BTC/USDT",
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "momentum_period": 14,
            "atr_period": 14,
            "volume_threshold": 1.5
        }
    },
    "volatility_breakout_short": {
        "description": "Volatility Breakout Short Strategy",
        "params": {
            "symbol": "BTC/USDT",
            "atr_period": 14,
            "lookback_period": 20,
            "volume_multiplier": 2.0,
            "rsi_period": 14,
            "rsi_extreme": 20,
            "atr_stop_multiplier": 2.0,
            "atr_trail_multiplier": 1.5
        }
    },
    "ichimoku_cloud": {
        "description": "Ichimoku Cloud Breakout Strategy",
        "params": {
            "symbol": "BTC/USDT",
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
            "displacement": 26
        }
    },
    "parabolic_sar": {
        "description": "Parabolic SAR Trend Following Strategy",
        "params": {
            "symbol": "BTC/USDT",
            "start": 0.02,
            "increment": 0.02,
            "maximum": 0.2
        }
    },
    "fibonacci_retracement": {
        "description": "Fibonacci Retracement Support/Resistance Strategy",
        "params": {
            "symbol": "BTC/USDT",
            "lookback_period": 50,
            "fib_levels": [0.236, 0.382, 0.5, 0.618, 0.786]
        }
    },
    "gaussian_channel": {
        "description": "Gaussian Channel Breakout Mean Reversion Strategy",
        "params": {
            "symbol": "BTC/USDT",
            "period": 20,
            "std_dev": 2.0,
            "adaptive": True
        }
    }
}


def run_strategy(strategy_name, params=None, optimize=False, analyze=False, 
                 method='grid_search', llm='auto'):
    """
    Run a single strategy with specified parameters
    
    Args:
        strategy_name: Name of the strategy to run
        params: Dictionary of parameters (if None, uses defaults)
        optimize: Whether to run parameter optimization
        analyze: Whether to run LLM analysis
        method: Optimization method (grid_search, random_search, bayesian)
        llm: LLM provider (auto, gemini, openai)
    """
    cmd = ['python', 'backend/backtesting/scripts/run_single_strategy.py', strategy_name]
    
    if optimize:
        cmd.append('--optimize')
        if method != 'grid_search':
            cmd.extend(['--method', method])
    elif params:
        cmd.extend(['--params', json.dumps(params)])
    
    if analyze:
        cmd.append('--analyze')
        if llm != 'auto':
            cmd.extend(['--llm', llm])
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


def print_examples():
    """Print all command line examples"""
    print("=" * 80)
    print("TRADING BOT - ALL STRATEGY EXAMPLES")
    print("=" * 80)
    print()
    
    # Basic examples for each strategy
    print("BASIC EXAMPLES (with default parameters):")
    print("-" * 40)
    for strategy_name, info in STRATEGIES.items():
        params_json = json.dumps(info['params'])
        print(f"\n# {info['description']}")
        print(f"python backend/backtesting/scripts/run_single_strategy.py {strategy_name} --params '{params_json}'")
    
    print("\n" + "=" * 80)
    print("ADVANCED EXAMPLES:")
    print("-" * 40)
    
    # Minimal example (symbol only)
    print("\n# Run with minimal parameters (symbol only)")
    print("python backend/backtesting/scripts/run_single_strategy.py rsi_divergence --params '{\"symbol\": \"BTC/USDT\"}'")
    
    # Optimization examples
    print("\n# Run with parameter optimization")
    print("python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --optimize")
    
    print("\n# Run with Bayesian optimization")
    print("python backend/backtesting/scripts/run_single_strategy.py macd_momentum --optimize --method bayesian")
    
    print("\n# Run with Random search optimization")
    print("python backend/backtesting/scripts/run_single_strategy.py ichimoku_cloud --optimize --method random_search")
    
    # Analysis examples
    print("\n# Run with LLM analysis")
    print("python backend/backtesting/scripts/run_single_strategy.py rsi_divergence --params '{\"symbol\": \"BTC/USDT\"}' --analyze")
    
    print("\n# Run with specific LLM provider (Gemini)")
    print("python backend/backtesting/scripts/run_single_strategy.py gaussian_channel --params '{\"symbol\": \"BTC/USDT\"}' --analyze --llm gemini")
    
    print("\n# Run with specific LLM provider (OpenAI)")
    print("python backend/backtesting/scripts/run_single_strategy.py fibonacci_retracement --params '{\"symbol\": \"BTC/USDT\"}' --analyze --llm openai")
    
    # Combined examples
    print("\n# Run with optimization AND analysis")
    print("python backend/backtesting/scripts/run_single_strategy.py volatility_breakout_short --optimize --analyze")
    
    # Different symbols
    print("\n# Run with different trading pairs")
    print("python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --params '{\"symbol\": \"ETH/USDT\"}'")
    print("python backend/backtesting/scripts/run_single_strategy.py macd_momentum --params '{\"symbol\": \"SOL/USDT\"}'")
    
    # Custom configuration file
    print("\n# Run with custom configuration file")
    print("python backend/backtesting/scripts/run_single_strategy.py parabolic_sar --config path/to/custom_config.yaml")
    
    print("\n" + "=" * 80)


def run_all_strategies():
    """Run all strategies sequentially"""
    print("Running all strategies...")
    for strategy_name, info in STRATEGIES.items():
        print(f"\n{'='*60}")
        print(f"Running {info['description']}")
        print(f"{'='*60}")
        run_strategy(strategy_name, params=info['params'])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "run_all":
            run_all_strategies()
        elif sys.argv[1] == "examples":
            print_examples()
        elif sys.argv[1] in STRATEGIES:
            # Run specific strategy
            strategy = sys.argv[1]
            run_strategy(strategy, params=STRATEGIES[strategy]['params'])
        else:
            print(f"Unknown command or strategy: {sys.argv[1]}")
            print(f"Available strategies: {', '.join(STRATEGIES.keys())}")
            print("Commands: run_all, examples")
    else:
        # Just print examples by default
        print_examples()
