"""
Example Usage of the Enhanced Trading System

This script demonstrates how to use the enhanced meta strategy orchestrator
with executable strategies for live trading signal generation.

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

# Import the enhanced orchestrator
from enhanced_meta_strategy_orchestrator import EnhancedMetaStrategyOrchestrator
from backtester import Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data(symbol: str, periods: int = 500) -> pd.DataFrame:
    """
    Load or generate sample OHLCV data for testing.
    
    In production, this would load from your database.
    """
    # Generate sample data for demonstration
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
    
    # Simulate realistic price movements
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.0002, 0.01, periods)
    price = 50000 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    data = pd.DataFrame({
        'open': price * (1 + np.random.uniform(-0.001, 0.001, periods)),
        'high': price * (1 + np.abs(np.random.normal(0, 0.003, periods))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.003, periods))),
        'close': price,
        'volume': np.random.uniform(100, 1000, periods) * 1e6
    }, index=dates)
    
    return data


def example_live_trading():
    """
    Example of using the orchestrator for live trading signals.
    """
    logger.info("=== Live Trading Signal Generation Example ===")
    
    # Configuration
    db_connection_string = "sqlite:///trading_data.db"  # Use your actual DB
    symbols = ['BTC/USDT', 'ETH/USDT']
    config_path = "backend/config/strategy_config.json"
    
    # Initialize orchestrator
    orchestrator = EnhancedMetaStrategyOrchestrator(
        db_connection_string=db_connection_string,
        symbols=symbols,
        config_path=config_path
    )
    
    # For demonstration, inject sample data
    # In production, the orchestrator would load from your database
    for symbol in symbols:
        orchestrator.data_cache[symbol] = {
            'H1': load_sample_data(symbol, 500),
            'H4': load_sample_data(symbol, 500),
            'D1': load_sample_data(symbol, 500)
        }
    
    # Setup the orchestrator
    orchestrator.setup(primary_timeframe='H1')
    
    # Run the orchestrator to get signals
    results = orchestrator.run()
    
    # Display results
    print("\n📊 MARKET ANALYSIS RESULTS")
    print("=" * 60)
    
    # Overall market bias
    print(f"\n🌍 Overall Market Bias: {results['overall_bias']}")
    
    # Market regimes by symbol and timeframe
    print("\n📈 Market Regimes:")
    for symbol, regimes in results['market_regimes'].items():
        print(f"\n  {symbol}:")
        for timeframe, regime in regimes.items():
            print(f"    {timeframe}: {regime['bias']} (strength: {regime['strength']:.2f})")
    
    # Trading signals
    print("\n🚦 TRADING SIGNALS:")
    print("-" * 60)
    
    for symbol, signal_data in results['composite_signals'].items():
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        strategies = signal_data.get('contributing_strategies', [])
        
        # Determine action
        if signal > 0.6:
            action = "STRONG BUY 🟢"
            emoji = "📈"
        elif signal > 0.3:
            action = "BUY 🟢"
            emoji = "📈"
        elif signal < -0.6:
            action = "STRONG SELL 🔴"
            emoji = "📉"
        elif signal < -0.3:
            action = "SELL 🔴"
            emoji = "📉"
        else:
            action = "NEUTRAL ⚪"
            emoji = "➡️"
        
        print(f"\n{emoji} {symbol}:")
        print(f"  Signal: {signal:.3f} ({action})")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Contributing Strategies: {', '.join(strategies)}")
        
        # Show individual strategy signals
        if symbol in results['raw_signals']:
            print(f"  Strategy Breakdown:")
            for strategy_name, strategy_signal in results['raw_signals'][symbol].items():
                print(f"    - {strategy_name}: {strategy_signal['signal']:.3f} "
                      f"(weight: {strategy_signal['weight']:.2f})")
    
    # Performance metrics
    print("\n📊 Performance Metrics:")
    metrics = orchestrator.get_performance_metrics()
    if metrics:
        print(f"  Total Signals Generated: {metrics['total_signals']}")
        print(f"  Buy Signals: {metrics['buy_signals']}")
        print(f"  Sell Signals: {metrics['sell_signals']}")
        print(f"  Neutral Signals: {metrics['neutral_signals']}")
    
    return results


def example_backtesting():
    """
    Example of backtesting the orchestrator.
    """
    logger.info("\n=== Backtesting Example ===")
    
    # Configuration
    db_connection_string = "sqlite:///trading_data.db"
    symbols = ['BTC/USDT']
    
    # Initialize backtester
    backtester = Backtester(
        orchestrator_class=EnhancedMetaStrategyOrchestrator,
        db_connection_string=db_connection_string
    )
    
    # Prepare historical data
    # In production, load from your database
    historical_data = {}
    for symbol in symbols:
        historical_data[symbol] = load_sample_data(symbol, 1000)
    
    # Run backtest
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    results = backtester.run_backtest(
        full_data=historical_data,
        symbols=symbols,
        timeframe='H1',
        start_date=start_date,
        end_date=end_date,
        lookback_days=100
    )
    
    # Display backtest results
    print("\n📊 BACKTEST RESULTS")
    print("=" * 60)
    
    summary = results['summary']
    print(f"\nPerformance Summary:")
    print(f"  Initial Capital: ${summary['initial_capital']:,.2f}")
    print(f"  Final Value: ${summary['final_value']:,.2f}")
    print(f"  Total Return: {summary['total_return']:.2%}")
    print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {summary['max_drawdown']:.2%}")
    
    print(f"\nTrading Statistics:")
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Win Rate: {summary['win_rate']:.2%}")
    print(f"  Average Win: ${summary['avg_win']:.2f}")
    print(f"  Average Loss: ${summary['avg_loss']:.2f}")
    print(f"  Profit Factor: {summary['profit_factor']:.2f}")
    
    # Show recent trades
    if not results['trades'].empty:
        print("\n📝 Recent Trades:")
        print(results['trades'].tail(5).to_string())
    
    return results


def example_strategy_testing():
    """
    Example of testing individual strategies.
    """
    logger.info("\n=== Individual Strategy Testing ===")
    
    from strategies_executable import (
        BollingerBandsMeanReversion,
        RSIMomentumDivergence,
        MACDMomentumCrossover,
        SMAGoldenCross
    )
    
    # Load sample data
    data = load_sample_data('BTC/USDT', 500)
    
    # Test each strategy
    strategies = [
        BollingerBandsMeanReversion(data),
        RSIMomentumDivergence(data),
        MACDMomentumCrossover(data),
        SMAGoldenCross(data)
    ]
    
    print("\n🔬 INDIVIDUAL STRATEGY SIGNALS")
    print("=" * 60)
    
    for strategy in strategies:
        if strategy.has_sufficient_data():
            signal_data = strategy.calculate_signal()
            
            print(f"\n{strategy.name}:")
            print(f"  Signal: {signal_data['signal']:.3f}")
            print(f"  Confidence: {signal_data.get('confidence', 'N/A')}")
            print(f"  Reason: {signal_data.get('reason', 'N/A')}")
            
            # Show additional metadata
            for key, value in signal_data.items():
                if key not in ['signal', 'confidence', 'reason', 'error']:
                    print(f"  {key}: {value}")
        else:
            print(f"\n{strategy.name}: Insufficient data")


def example_configuration():
    """
    Example of working with strategy configuration.
    """
    logger.info("\n=== Configuration Management Example ===")
    
    # Load configuration
    config_path = Path("backend/config/strategy_config.json")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("\n⚙️ STRATEGY CONFIGURATION")
        print("=" * 60)
        
        print("\nEnabled Strategies:")
        for name, settings in config['strategies'].items():
            if settings.get('enabled', False):
                print(f"  ✓ {name}")
                print(f"    Class: {settings['class']}")
                print(f"    Parameters: {json.dumps(settings['parameters'], indent=6)}")
        
        print("\nRisk Management Settings:")
        for key, value in config['risk_management'].items():
            print(f"  {key}: {value}")
        
        print("\nSignal Filters:")
        for key, value in config['signal_filters'].items():
            print(f"  {key}: {value}")
    else:
        print("Configuration file not found!")


def main():
    """
    Run all examples.
    """
    print("\n🚀 ENHANCED TRADING SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Show configuration
        example_configuration()
        
        # Test individual strategies
        example_strategy_testing()
        
        # Generate live trading signals
        live_results = example_live_trading()
        
        # Run backtest (commented out for speed)
        # backtest_results = example_backtesting()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
