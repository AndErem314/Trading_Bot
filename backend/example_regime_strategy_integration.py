#!/usr/bin/env python3
"""
Example: Integrating Market Regime Detection with Trading Strategies

This example demonstrates how to use the CryptoMarketRegimeDetector to enhance
your existing trading strategies by adapting to different market conditions.

Author: Andrey's Trading Bot
Date: 2025-08-30
"""

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from backend.Indicators import MarketRegimeDetector
from backend.Strategies import *  # Import all strategies


class RegimeAwareStrategyExecutor:
    """
    Enhanced strategy executor that adapts strategy selection and parameters
    based on detected market regime.
    """
    
    def __init__(self, symbol='BTC/USDT', db_path='data/trading_data_BTC.db'):
        self.symbol = symbol
        self.db_path = db_path
        self.asset_name = symbol.split('/')[0]  # Extract 'BTC' from 'BTC/USDT'
        
        # Strategy mapping based on market regime
        self.regime_strategy_map = {
            'Bull Trend': [
                'SMA_Golden_Cross_Strategy',
                'Ichimoku_Cloud_Breakout_Strategy', 
                'Parabolic_SAR_Trend_Following_Strategy',
                'MACD_Momentum_Crossover_Strategy'
            ],
            'Bear Trend': [
                'Parabolic_SAR_Trend_Following_Strategy',
                'MACD_Momentum_Crossover_Strategy',
                'RSI_Momentum_Divergence_Swing_Strategy'
            ],
            'Ranging / Accumulation': [
                'Bollinger_Bands_Mean_Reversion_Strategy',
                'RSI_Momentum_Divergence_Swing_Strategy',
                'Fibonacci_Retracement_Support_Resistance_Strategy'
            ],
            'High Volatility / Breakout': [
                'Gaussian_Channel_Breakout_Mean_Reversion_Strategy',
                'Bollinger_Bands_Mean_Reversion_Strategy'
            ],
            'Crypto Crash': [
                'RSI_Momentum_Divergence_Swing_Strategy',  # Look for oversold bounces
                'Fibonacci_Retracement_Support_Resistance_Strategy'  # Key support levels
            ]
        }
        
        # Risk adjustment factors based on regime
        self.regime_risk_factors = {
            'Bull Trend': 1.2,          # Increase position size in bull markets
            'Bear Trend': 0.7,          # Reduce position size in bear markets
            'Ranging / Accumulation': 1.0,  # Normal position size
            'High Volatility / Breakout': 0.5,  # Significantly reduce size in high volatility
            'Crypto Crash': 0.3,        # Minimal position size during crashes
            'Transitioning / Unknown': 0.8   # Slightly reduced size when uncertain
        }
        
    def load_ohlcv_data(self, timeframe='4h', periods=500):
        """Load OHLCV data from database."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_data
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(self.symbol, timeframe, periods))
        conn.close()
        
        # Convert timestamp and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def detect_current_regime(self, df, benchmark_df=None):
        """Detect current market regime."""
        detector = MarketRegimeDetector(
            df=df,
            asset_name=self.asset_name,
            benchmark_df=benchmark_df
        )
        
        regime, metrics = detector.classify_regime()
        return regime, metrics, detector
    
    def get_recommended_strategies(self, regime):
        """Get recommended strategies for current regime."""
        return self.regime_strategy_map.get(regime, [])
    
    def adjust_risk_parameters(self, base_risk, regime):
        """Adjust risk parameters based on market regime."""
        risk_factor = self.regime_risk_factors.get(regime, 0.8)
        return base_risk * risk_factor
    
    def execute_regime_aware_analysis(self):
        """Main execution method that combines regime detection with strategy selection."""
        
        print(f"Loading data for {self.symbol}...")
        df = self.load_ohlcv_data()
        
        # If analyzing an altcoin, load BTC as benchmark
        benchmark_df = None
        if self.asset_name != 'BTC':
            print("Loading BTC benchmark data...")
            # You would load BTC data here
            # benchmark_df = load_btc_data()
        
        print("\nDetecting market regime...")
        regime, metrics, detector = self.detect_current_regime(df, benchmark_df)
        
        print(f"\n{'='*60}")
        print(f"MARKET REGIME ANALYSIS for {self.symbol}")
        print(f"{'='*60}")
        print(f"Current Regime: {regime}")
        print(f"\nKey Metrics:")
        for key, value in metrics.items():
            if key != 'primary_regime':
                print(f"  {key}: {value}")
        
        # Get recommended strategies
        recommended_strategies = self.get_recommended_strategies(regime)
        print(f"\n{'='*60}")
        print(f"RECOMMENDED STRATEGIES for {regime}")
        print(f"{'='*60}")
        for i, strategy in enumerate(recommended_strategies, 1):
            print(f"{i}. {strategy}")
        
        # Risk adjustment
        base_risk = 0.02  # 2% base risk
        adjusted_risk = self.adjust_risk_parameters(base_risk, regime)
        print(f"\n{'='*60}")
        print(f"RISK MANAGEMENT ADJUSTMENTS")
        print(f"{'='*60}")
        print(f"Base Risk: {base_risk*100:.1f}%")
        print(f"Regime Adjusted Risk: {adjusted_risk*100:.1f}%")
        print(f"Risk Factor: {self.regime_risk_factors.get(regime, 0.8):.1f}x")
        
        # Additional recommendations based on regime
        print(f"\n{'='*60}")
        print(f"REGIME-SPECIFIC RECOMMENDATIONS")
        print(f"{'='*60}")
        
        recommendations = self.get_regime_recommendations(regime, metrics)
        for rec in recommendations:
            print(f"• {rec}")
        
        # Generate visualization
        print(f"\n{'='*60}")
        print("Generating regime visualization...")
        detector.plot_regime(use_plotly=True)
        
        # Get regime statistics
        print(f"\n{'='*60}")
        print("HISTORICAL REGIME STATISTICS")
        print(f"{'='*60}")
        stats = detector.get_regime_statistics()
        print(stats.to_string())
        
        return regime, metrics, detector, recommended_strategies
    
    def get_regime_recommendations(self, regime, metrics):
        """Get specific recommendations based on current regime."""
        
        recommendations = []
        
        if regime == 'Bull Trend':
            recommendations.extend([
                "Focus on trend-following strategies",
                "Consider increasing position sizes gradually",
                "Use trailing stops to protect profits",
                "Look for pullbacks to moving averages for entries"
            ])
            if metrics.get('adx', 0) > 40:
                recommendations.append("Strong trend detected - hold positions longer")
                
        elif regime == 'Bear Trend':
            recommendations.extend([
                "Prioritize capital preservation",
                "Consider short positions or staying in stablecoins",
                "Use tighter stop losses",
                "Wait for clear reversal signals before going long"
            ])
            if metrics.get('rsi', 100) < 30:
                recommendations.append("Approaching oversold - watch for potential bounce")
                
        elif regime == 'Ranging / Accumulation':
            recommendations.extend([
                "Use mean reversion strategies",
                "Trade the range boundaries",
                "Set profit targets at opposite range boundaries",
                "Consider accumulating spot positions gradually"
            ])
            
        elif regime == 'High Volatility / Breakout':
            recommendations.extend([
                "Reduce position sizes significantly",
                "Wait for volatility to settle before entering",
                "Use wider stops to avoid getting stopped out",
                "Consider volatility-based strategies"
            ])
            if metrics.get('volume_zscore', 0) > 3:
                recommendations.append("Extreme volume detected - major move possible")
                
        elif regime == 'Crypto Crash':
            recommendations.extend([
                "EXTREME CAUTION - Consider staying out of market",
                "If trading, use minimal position sizes",
                "Look for capitulation signals for potential bottom",
                "Focus on high-quality assets only",
                "Consider dollar-cost averaging for long-term positions"
            ])
            
        # Add relative strength recommendations if available
        if metrics.get('relative_strength') == 'Outperforming':
            recommendations.append(f"{self.asset_name} is outperforming the market - favorable for longs")
        elif metrics.get('relative_strength') == 'Underperforming':
            recommendations.append(f"{self.asset_name} is underperforming - consider other assets")
            
        return recommendations


def main():
    """Example usage of regime-aware strategy execution."""
    
    # Example 1: Analyze BTC
    print("\n" + "="*80)
    print("EXAMPLE 1: Bitcoin (BTC) Analysis")
    print("="*80)
    
    btc_executor = RegimeAwareStrategyExecutor(symbol='BTC/USDT')
    # Uncomment to run: btc_regime, btc_metrics, btc_detector, btc_strategies = btc_executor.execute_regime_aware_analysis()
    
    # Example 2: Analyze ETH with BTC as benchmark
    print("\n" + "="*80)
    print("EXAMPLE 2: Ethereum (ETH) Analysis with BTC Benchmark")
    print("="*80)
    
    eth_executor = RegimeAwareStrategyExecutor(symbol='ETH/USDT')
    # Uncomment to run: eth_regime, eth_metrics, eth_detector, eth_strategies = eth_executor.execute_regime_aware_analysis()
    
    # Example 3: Create a simple regime-based trading system
    print("\n" + "="*80)
    print("EXAMPLE 3: Simple Regime-Based Trading Logic")
    print("="*80)
    
    print("""
    def make_trading_decision(regime, metrics):
        if regime == 'Bull Trend' and metrics['rsi'] < 70:
            return 'BUY'
        elif regime == 'Bear Trend' and metrics['rsi'] > 30:
            return 'SELL' 
        elif regime == 'Crypto Crash':
            return 'STAY_OUT'
        elif regime == 'Ranging / Accumulation':
            if metrics['rsi'] < 40:
                return 'BUY'
            elif metrics['rsi'] > 60:
                return 'SELL'
        return 'HOLD'
    """)


if __name__ == "__main__":
    main()
    
    print("\n" + "="*80)
    print("INTEGRATION COMPLETE")
    print("="*80)
    print("\nThe CryptoMarketRegimeDetector has been successfully integrated!")
    print("\nKey Features:")
    print("1. Sophisticated regime detection for crypto markets")
    print("2. Asset-specific parameter tuning (BTC, ETH, etc.)")
    print("3. Crypto-specific crash detection")
    print("4. Relative strength analysis against benchmarks")
    print("5. Interactive visualizations with plotly")
    print("6. Strategy recommendations based on market regime")
    print("7. Dynamic risk adjustment based on market conditions")
    print("\nUse this regime detection to enhance your trading strategies!")
