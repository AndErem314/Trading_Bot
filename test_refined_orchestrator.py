"""
Test Script for Refined Trading System

This script demonstrates the enhanced trading system with:
- ADX-based regime detection
- Optimized strategy weights
- Corrected mean reversion logic
- Volatility breakout short strategy for crashes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import the refined components
from backend.enhanced_market_regime_detector import EnhancedMarketRegimeDetector
from backend.refined_meta_strategy_orchestrator import RefinedMetaStrategyOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(trend='neutral', volatility='normal', periods=500):
    """
    Generate sample OHLCV data for testing different market conditions.
    
    Args:
        trend: 'bullish', 'bearish', 'neutral', 'crash'
        volatility: 'low', 'normal', 'high'
        periods: Number of periods to generate
    """
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
    
    # Base price
    base_price = 50000
    
    # Trend parameters
    if trend == 'bullish':
        drift = 0.0005
        trend_strength = 0.7
    elif trend == 'bearish':
        drift = -0.0003
        trend_strength = 0.6
    elif trend == 'crash':
        drift = -0.002
        trend_strength = 0.9
    else:  # neutral
        drift = 0.0001
        trend_strength = 0.2
    
    # Volatility parameters
    if volatility == 'high':
        vol = 0.03
    elif volatility == 'low':
        vol = 0.005
    else:  # normal
        vol = 0.015
    
    # Generate prices
    prices = [base_price]
    for i in range(1, periods):
        # Add trend component
        trend_move = drift * (1 + trend_strength * np.random.normal(0, 0.3))
        
        # Add random component
        random_move = np.random.normal(0, vol)
        
        # Special handling for crash
        if trend == 'crash' and i > periods * 0.7:
            # Intensify crash in later periods
            random_move = np.random.normal(-vol, vol * 1.5)
            trend_move *= 2
        
        new_price = prices[-1] * (1 + trend_move + random_move)
        prices.append(max(new_price, base_price * 0.5))  # Floor at 50% of base
    
    # Generate OHLCV
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate intraday movement
        daily_vol = vol * (1.5 if trend == 'crash' and i > periods * 0.7 else 1)
        high = close * (1 + abs(np.random.normal(0, daily_vol * 0.5)))
        low = close * (1 - abs(np.random.normal(0, daily_vol * 0.5)))
        open_price = prices[i-1] if i > 0 else close
        
        # Volume (higher in trends and crashes)
        base_volume = 1000000
        if trend == 'crash':
            volume = base_volume * np.random.uniform(2, 5)
        elif trend in ['bullish', 'bearish']:
            volume = base_volume * np.random.uniform(1.2, 2)
        else:
            volume = base_volume * np.random.uniform(0.8, 1.2)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


def test_regime_detection():
    """Test the enhanced regime detection on different market conditions."""
    logger.info("Testing Enhanced Regime Detection")
    logger.info("=" * 60)
    
    test_scenarios = [
        ('bullish', 'normal', 'STRONG_BULLISH'),
        ('bearish', 'normal', 'STRONG_BEARISH'),
        ('neutral', 'low', 'NEUTRAL_RANGING'),
        ('crash', 'high', 'CRASH_PANIC'),
    ]
    
    for trend, volatility, expected_regime in test_scenarios:
        logger.info(f"\nTesting {trend} market with {volatility} volatility")
        
        # Generate test data
        data = generate_sample_data(trend=trend, volatility=volatility)
        
        # Create detector
        detector = EnhancedMarketRegimeDetector(data)
        
        # Detect regime
        regime, metrics = detector.detect_market_regime()
        
        logger.info(f"Detected regime: {regime}")
        logger.info(f"Expected regime: {expected_regime}")
        logger.info(f"Metrics: {metrics}")
        logger.info(f"Match: {'✓' if regime == expected_regime else '✗'}")


def test_strategy_weights():
    """Test strategy weight assignments for different regimes."""
    logger.info("\n\nTesting Strategy Weight Assignments")
    logger.info("=" * 60)
    
    orchestrator = RefinedMetaStrategyOrchestrator(
        db_connection_string="sqlite:///test.db",
        symbols=['BTC/USDT']
    )
    
    regimes = ['STRONG_BULLISH', 'STRONG_BEARISH', 'NEUTRAL_RANGING', 'CRASH_PANIC']
    
    for regime in regimes:
        weights = orchestrator.STRATEGY_WEIGHTS[regime]
        logger.info(f"\n{regime} Regime Weights:")
        
        # Sort by weight
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        for strategy, weight in sorted_weights:
            if weight > 0:
                logger.info(f"  {strategy}: {weight}")


def test_mean_reversion_logic():
    """Test the corrected Gaussian Channel mean reversion logic."""
    logger.info("\n\nTesting Corrected Mean Reversion Logic")
    logger.info("=" * 60)
    
    # Import the corrected strategy
    from backend.strategies_executable.gaussian_channel_strategy import GaussianChannelBreakoutMeanReversion
    
    # Generate ranging market data
    data = generate_sample_data(trend='neutral', volatility='normal')
    
    # Create strategy instance
    strategy = GaussianChannelBreakoutMeanReversion(data)
    
    # Get signal
    signal = strategy.calculate_signal()
    
    logger.info(f"Signal: {signal['signal']}")
    logger.info(f"Reason: {signal['reason']}")
    logger.info(f"Channel Position: {signal.get('channel_position', 'N/A')}")
    
    # Verify mean reversion logic
    if 'lower band' in signal['reason'].lower() and signal['signal'] > 0:
        logger.info("✓ Correct: Buy signal at lower band (oversold)")
    elif 'upper band' in signal['reason'].lower() and signal['signal'] < 0:
        logger.info("✓ Correct: Sell signal at upper band (overbought)")
    else:
        logger.info("ℹ️  No extreme band touch - waiting for mean reversion opportunity")


def test_crash_strategy():
    """Test the new Volatility Breakout Short strategy in crash conditions."""
    logger.info("\n\nTesting Volatility Breakout Short Strategy")
    logger.info("=" * 60)
    
    # Import the crash strategy
    from backend.strategies_executable.volatility_breakout_short_strategy import VolatilityBreakoutShort
    
    # Generate crash market data
    data = generate_sample_data(trend='crash', volatility='high')
    
    # Create strategy instance
    strategy = VolatilityBreakoutShort(data)
    
    # Test regime allowance
    logger.info(f"Allowed in CRASH_PANIC: {strategy.is_strategy_allowed('CRASH_PANIC')}")
    logger.info(f"Allowed in NEUTRAL_RANGING: {strategy.is_strategy_allowed('NEUTRAL_RANGING')}")
    
    # Get signal
    signal = strategy.calculate_signal()
    
    logger.info(f"\nSignal: {signal['signal']}")
    logger.info(f"Confidence: {signal['confidence']}")
    logger.info(f"Reason: {signal['reason']}")
    logger.info(f"ATR Percentage: {signal.get('atr_percentage', 'N/A')}%")
    logger.info(f"Stop Loss Distance: {signal.get('stop_loss_distance_pct', 'N/A')}%")
    
    # Get risk parameters
    risk_params = strategy.get_risk_parameters()
    logger.info(f"\nRisk Parameters:")
    logger.info(f"  Position Size Multiplier: {risk_params['position_size_multiplier']}")
    logger.info(f"  Max Risk Per Trade: {risk_params['max_risk_per_trade']}")
    logger.info(f"  Time Stop: {risk_params['time_stop_hours']} hours")


def test_full_orchestration():
    """Test the complete orchestration system."""
    logger.info("\n\nTesting Full Orchestration System")
    logger.info("=" * 60)
    
    # Create sample data for different market conditions
    scenarios = {
        'BTC/USDT': ('crash', 'high'),  # Crash scenario
        'ETH/USDT': ('neutral', 'normal'),  # Ranging scenario
    }
    
    # Create orchestrator (with mock database)
    orchestrator = RefinedMetaStrategyOrchestrator(
        db_connection_string="sqlite:///test.db",
        symbols=list(scenarios.keys())
    )
    
    # Override data loading to use our test data
    for symbol, (trend, vol) in scenarios.items():
        orchestrator.data_cache[symbol] = generate_sample_data(trend=trend, volatility=vol)
    
    # Initialize regime detectors manually
    for symbol in orchestrator.symbols:
        if symbol in orchestrator.data_cache:
            orchestrator.regime_detectors[symbol] = EnhancedMarketRegimeDetector(
                orchestrator.data_cache[symbol]
            )
    
    # Run regime detection
    logger.info("\nRegime Detection Results:")
    for symbol in orchestrator.symbols:
        regime, metrics = orchestrator.detect_market_regime(symbol)
        orchestrator.current_regimes[symbol] = regime
        logger.info(f"{symbol}: {regime} (ADX: {metrics.get('adx', 'N/A')})")
    
    # Get regime summary
    summary = orchestrator.get_regime_summary()
    
    logger.info("\nActive Strategy Weights:")
    for symbol, weights in summary['strategy_weights'].items():
        logger.info(f"\n{symbol} ({summary['current_regimes'][symbol]}):")
        active_strategies = [(s, w) for s, w in weights.items() if w > 0]
        for strategy, weight in sorted(active_strategies, key=lambda x: x[1], reverse=True):
            logger.info(f"  {strategy}: {weight}")


def main():
    """Run all tests."""
    logger.info("REFINED TRADING SYSTEM TEST SUITE")
    logger.info("=" * 80)
    
    test_regime_detection()
    test_strategy_weights()
    test_mean_reversion_logic()
    test_crash_strategy()
    test_full_orchestration()
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUITE COMPLETE")


if __name__ == "__main__":
    main()
