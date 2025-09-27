"""
Multi-Timeframe Ichimoku Strategy Example

This module demonstrates how to use the enhanced IchimokuBacktester with
multi-timeframe analysis capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

from backend.executable_workflow.backtesting import (
    IchimokuBacktester,
    TimeframeConfig,
    MultiTimeframeSignal
)
from backend.executable_workflow.data_fetching import OHLCVDataFetcher, DataPreprocessor
from backend.executable_workflow.indicators import IchimokuCalculator, IchimokuSignalDetector
from backend.executable_workflow.config import StrategyConfigManager

logger = logging.getLogger(__name__)


class MultiTimeframeIchimokuStrategy:
    """
    Multi-timeframe Ichimoku trading strategy.
    
    This strategy combines signals from multiple timeframes to generate
    more reliable trading signals with proper conflict resolution.
    """
    
    def __init__(self):
        self.fetcher = OHLCVDataFetcher()
        self.preprocessor = DataPreprocessor()
        self.ichimoku_calc = IchimokuCalculator()
        self.signal_detector = IchimokuSignalDetector()
        self.backtester = IchimokuBacktester()
        
    def prepare_multi_timeframe_data(self, symbol: str, exchange: str = 'binance',
                                   timeframes: List[str] = ['1h', '4h'],
                                   days_back: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Fetch and prepare data for multiple timeframes.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            exchange: Exchange name
            timeframes: List of timeframes to fetch
            days_back: Number of days of historical data
            
        Returns:
            Dictionary mapping timeframe to prepared DataFrame
        """
        mtf_data = {}
        
        for timeframe in timeframes:
            logger.info(f"Fetching {timeframe} data for {symbol}")
            
            # Calculate limit based on timeframe
            tf_minutes = {'15m': 15, '1h': 60, '4h': 240, '1d': 1440}
            minutes_per_bar = tf_minutes.get(timeframe, 60)
            limit = int((days_back * 24 * 60) / minutes_per_bar)
            
            # Fetch data
            raw_data = self.fetcher.fetch_ohlcv(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                limit=min(limit, 1000)  # Most exchanges limit to 1000
            )
            
            if raw_data is None or raw_data.empty:
                logger.error(f"Failed to fetch {timeframe} data")
                continue
            
            # Preprocess data
            clean_data = self.preprocessor.process(raw_data)
            
            # Calculate Ichimoku indicators
            ichimoku_data = self.ichimoku_calc.calculate(clean_data)
            
            # Detect signals
            signals = self.signal_detector.detect_signals(ichimoku_data)
            
            # Combine all data
            final_data = pd.concat([ichimoku_data, signals], axis=1)
            
            mtf_data[timeframe] = final_data
            
            logger.info(f"Prepared {timeframe} data: {len(final_data)} bars")
        
        return mtf_data
    
    def run_mtf_backtest_example(self, symbol: str = 'BTC/USDT'):
        """
        Run a complete multi-timeframe backtest example.
        """
        logger.info("Starting Multi-Timeframe Ichimoku Backtest Example")
        
        # 1. Configure timeframes with different weights and priorities
        timeframe_configs = [
            TimeframeConfig(
                timeframe='1h',
                weight=1.0,
                priority=2,
                confirmation_required=False,
                min_bars_for_signal=100
            ),
            TimeframeConfig(
                timeframe='4h',
                weight=2.0,  # Higher weight for higher timeframe
                priority=1,   # Higher priority
                confirmation_required=True,
                min_bars_for_signal=50
            )
        ]
        
        self.backtester.configure_timeframes(timeframe_configs)
        
        # 2. Fetch and prepare multi-timeframe data
        mtf_data = self.prepare_multi_timeframe_data(
            symbol=symbol,
            timeframes=['1h', '4h'],
            days_back=30
        )
        
        # 3. Synchronize data across timeframes
        synced_data = self.backtester.synchronize_multi_timeframe_data(mtf_data)
        
        logger.info(f"Synchronized data timeframes: {list(synced_data.keys())}")
        
        # 4. Create a sample strategy configuration
        from backend.executable_workflow.config.models import (
            StrategyConfig, SignalCombination, IchimokuParameters,
            RiskManagement, PositionSizing
        )
        
        strategy_config = StrategyConfig(
            name="MTF Ichimoku Strategy",
            strategy_id="mtf_ichimoku_001",
            symbols=[symbol],
            timeframe="1h",  # Base timeframe
            ichimoku_params=IchimokuParameters(
                tenkan_period=9,
                kijun_period=26,
                senkou_span_b_period=52,
                displacement=26,
                chikou_span_displacement=26
            ),
            buy_signals=SignalCombination(
                conditions=["PriceAboveCloud", "TenkanAboveKijun"],
                combination_type="AND",
                min_strength=0.6
            ),
            sell_signals=SignalCombination(
                conditions=["PriceBelowCloud", "TenkanBelowKijun"],
                combination_type="AND",
                min_strength=0.6
            ),
            risk_management=RiskManagement(
                stop_loss_type="atr",
                stop_loss_value=2.0,
                take_profit_type="risk_reward",
                take_profit_value=2.0,
                trailing_stop_enabled=False
            ),
            position_sizing=PositionSizing(
                method="risk_based",
                risk_per_trade_pct=2.0,
                max_position_size_pct=10.0,
                min_position_size=0.001,
                fixed_size=1.0
            )
        )
        
        # 5. Configure different signal resolution methods and run backtests
        resolution_methods = ['weighted_average', 'priority', 'majority_vote', 'all_agree']
        results = {}
        
        for method in resolution_methods:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing resolution method: {method}")
            logger.info(f"{'='*60}")
            
            # Reset backtester
            self.backtester = IchimokuBacktester(
                initial_capital=10000,
                commission_rate=0.001,
                slippage_rate=0.0005
            )
            self.backtester.configure_timeframes(timeframe_configs)
            self.backtester.signal_resolution_method = method
            
            # Run backtest
            result = self.backtester.run_backtest(
                strategy_config=strategy_config,
                data=synced_data,
                timeframes=['1h', '4h']
            )
            
            results[method] = result
            
            # Print summary
            self._print_results_summary(method, result)
        
        # 6. Compare results across different methods
        self._compare_resolution_methods(results)
        
        return results
    
    def demonstrate_signal_conflict_resolution(self):
        """
        Demonstrate how signal conflicts are resolved.
        """
        logger.info("\n" + "="*60)
        logger.info("Demonstrating Signal Conflict Resolution")
        logger.info("="*60)
        
        # Create sample conflicting signals
        from backend.executable_workflow.backtesting import SignalEvent
        
        timestamp = datetime.now()
        
        # 1h timeframe says BUY
        signal_1h = SignalEvent(
            timestamp=timestamp,
            priority=1,
            symbol='BTC/USDT',
            signal_type='entry',
            signal_strength=0.8,
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            signal_data={'symbol': 'BTC/USDT', 'signal': 1, 'reason': '1h_buy'}
        )
        
        # 4h timeframe says SELL (conflict!)
        signal_4h = SignalEvent(
            timestamp=timestamp,
            priority=1,
            symbol='BTC/USDT',
            signal_type='entry',
            signal_strength=0.6,
            entry_price=50000,
            stop_loss=51000,
            take_profit=48000,
            signal_data={'symbol': 'BTC/USDT', 'signal': -1, 'reason': '4h_sell'}
        )
        
        signals_by_timeframe = {'1h': signal_1h, '4h': signal_4h}
        
        # Test different resolution methods
        methods = ['weighted_average', 'priority', 'majority_vote', 'all_agree']
        
        for method in methods:
            self.backtester.signal_resolution_method = method
            mtf_signal = self.backtester.resolve_signal_conflicts(
                signals_by_timeframe, timestamp
            )
            
            logger.info(f"\nResolution Method: {method}")
            logger.info(f"Conflicts Detected: {mtf_signal.conflicts}")
            logger.info(f"Combined Signal: {mtf_signal.combined_signal:.2f}")
            logger.info(f"Final Decision: {mtf_signal.final_decision}")
            logger.info(f"Signal Strength: {mtf_signal.combined_strength:.2f}")
    
    def demonstrate_htf_confirmation(self):
        """
        Demonstrate higher timeframe confirmation requirement.
        """
        logger.info("\n" + "="*60)
        logger.info("Demonstrating Higher Timeframe Confirmation")
        logger.info("="*60)
        
        # Enable HTF confirmation
        self.backtester.require_htf_confirmation = True
        
        from backend.executable_workflow.backtesting import SignalEvent
        timestamp = datetime.now()
        
        # Scenario 1: Both timeframes agree (BUY)
        signals_agree = {
            '1h': SignalEvent(
                timestamp=timestamp, priority=1, symbol='BTC/USDT',
                signal_type='entry', signal_strength=0.8,
                entry_price=50000, stop_loss=49000, take_profit=52000,
                signal_data={'symbol': 'BTC/USDT', 'signal': 1}
            ),
            '4h': SignalEvent(
                timestamp=timestamp, priority=1, symbol='BTC/USDT',
                signal_type='entry', signal_strength=0.7,
                entry_price=50000, stop_loss=49000, take_profit=52000,
                signal_data={'symbol': 'BTC/USDT', 'signal': 1}
            )
        }
        
        mtf_signal = self.backtester.resolve_signal_conflicts(signals_agree, timestamp)
        logger.info("\nScenario 1: Both timeframes agree")
        logger.info(f"HTF Confirmation: PASSED")
        logger.info(f"Final Decision: {mtf_signal.final_decision}")
        
        # Scenario 2: HTF disagrees
        signals_disagree = {
            '1h': SignalEvent(
                timestamp=timestamp, priority=1, symbol='BTC/USDT',
                signal_type='entry', signal_strength=0.8,
                entry_price=50000, stop_loss=49000, take_profit=52000,
                signal_data={'symbol': 'BTC/USDT', 'signal': 1}
            ),
            '4h': SignalEvent(
                timestamp=timestamp, priority=1, symbol='BTC/USDT',
                signal_type='entry', signal_strength=0.7,
                entry_price=50000, stop_loss=51000, take_profit=48000,
                signal_data={'symbol': 'BTC/USDT', 'signal': -1}
            )
        }
        
        mtf_signal = self.backtester.resolve_signal_conflicts(signals_disagree, timestamp)
        logger.info("\nScenario 2: HTF disagrees")
        logger.info(f"HTF Confirmation: FAILED")
        logger.info(f"Final Decision: {mtf_signal.final_decision}")
    
    def _print_results_summary(self, method: str, results: Dict):
        """Print backtest results summary."""
        metrics = results.get('metrics', {})
        
        logger.info(f"\nResults for {method}:")
        logger.info(f"  Total Return: {metrics.get('total_return', 0):.2f}%")
        logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        logger.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    
    def _compare_resolution_methods(self, results: Dict[str, Dict]):
        """Compare performance across different resolution methods."""
        logger.info("\n" + "="*60)
        logger.info("Comparison of Signal Resolution Methods")
        logger.info("="*60)
        
        comparison = pd.DataFrame()
        
        for method, result in results.items():
            metrics = result.get('metrics', {})
            comparison[method] = {
                'Total Return (%)': metrics.get('total_return', 0),
                'Total Trades': metrics.get('total_trades', 0),
                'Win Rate (%)': metrics.get('win_rate', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown (%)': metrics.get('max_drawdown', 0),
                'Profit Factor': metrics.get('profit_factor', 0)
            }
        
        logger.info("\n" + comparison.T.to_string())
        
        # Find best method by Sharpe ratio
        best_method = comparison.loc['Sharpe Ratio'].idxmax()
        logger.info(f"\nBest performing method (by Sharpe Ratio): {best_method}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create strategy instance
    mtf_strategy = MultiTimeframeIchimokuStrategy()
    
    # Run demonstrations
    print("\n1. Demonstrating signal conflict resolution:")
    mtf_strategy.demonstrate_signal_conflict_resolution()
    
    print("\n2. Demonstrating HTF confirmation:")
    mtf_strategy.demonstrate_htf_confirmation()
    
    print("\n3. Running full MTF backtest (this may take a moment):")
    # Uncomment to run full backtest
    # results = mtf_strategy.run_mtf_backtest_example()
    
    print("\nMulti-Timeframe Ichimoku Strategy Example Complete!")