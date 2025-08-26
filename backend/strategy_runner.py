"""
Strategy Runner for Trading Bot

This script provides functionality to run trading strategies and monitor signals.
For backtesting, use the separate backtest_strategy.py script.

Usage:
    python strategy_runner.py --strategy RSI_Momentum_Divergence --mode monitor
    python strategy_runner.py --strategy RSI_Momentum_Divergence --mode info
"""

import sqlite3
import argparse
import time
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.Strategies import RSIMomentumDivergenceSwingStrategy


class StrategyRunner:
    """Main class for running trading strategies"""
    
    def __init__(self, strategy_name, db_path='data/trading_data_BTC.db'):
        self.db_path = db_path
        self.strategy_name = strategy_name
        self.strategy = self._load_strategy()
        self.active_positions = []
        
    def _load_strategy(self):
        """Load the specified strategy"""
        if self.strategy_name == "RSI_Momentum_Divergence":
            return RSIMomentumDivergenceSwingStrategy()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
    
    def get_latest_signals(self, limit=10):
        """Get the latest trading signals from the strategy"""
        query = self.strategy.get_sql_query()
        # Modify query to get only recent signals
        query = query.replace("LIMIT 100", f"LIMIT {limit}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            signals = []
            for row in cursor.fetchall():
                signal_dict = dict(zip(columns, row))
                signals.append(signal_dict)
            return signals
    
    def monitor_signals(self, check_interval=60):
        """Monitor for new trading signals in real-time"""
        print(f"Starting signal monitor for {self.strategy.name}")
        print(f"Check interval: {check_interval} seconds")
        print("Press Ctrl+C to stop monitoring")
        print("-" * 80)
        
        last_signal_time = None
        
        try:
            while True:
                signals = self.get_latest_signals(limit=5)
                
                if signals:
                    # Check for new signals
                    for signal in signals:
                        signal_time = datetime.strptime(signal['timestamp'], "%Y-%m-%d %H:%M:%S")
                        
                        # If this is a new signal (not seen before)
                        if last_signal_time is None or signal_time > last_signal_time:
                            self._process_signal(signal)
                            
                    # Update last signal time
                    if signals:
                        last_signal_time = datetime.strptime(signals[0]['timestamp'], "%Y-%m-%d %H:%M:%S")
                
                # Wait before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
    
    def _process_signal(self, signal):
        """Process a trading signal"""
        timestamp = signal['timestamp']
        price = signal['price']
        rsi = signal['rsi']
        signal_name = signal['signal_name']
        
        print(f"\n🔔 NEW SIGNAL DETECTED!")
        print(f"Time: {timestamp}")
        print(f"Signal: {signal_name}")
        print(f"Price: ${price:,.2f}")
        print(f"RSI: {rsi:.2f}")
        print(f"Trend: {signal.get('trend_strength', 'N/A')}")
        print(f"Divergence: {signal.get('divergence_signal', 'N/A')}")
        
        # Add risk management info
        risk_rules = self.strategy.get_risk_management_rules()
        print(f"\n📊 Risk Management:")
        print(f"- Risk per trade: {risk_rules['position_sizing']['risk_per_trade']}")
        print(f"- Initial stop loss: {risk_rules['stop_loss']['initial']}")
        print(f"- Trailing stop: {risk_rules['stop_loss']['trailing']}")
        print("-" * 80)
    
    
    def display_strategy_info(self):
        """Display detailed strategy information"""
        print(f"\n{'='*80}")
        print(f"STRATEGY INFORMATION: {self.strategy.name}")
        print(f"{'='*80}\n")
        
        # Get strategy description
        desc = self.strategy.get_strategy_description()
        print("📋 Strategy Overview:")
        for key, value in desc.items():
            if key != 'description':
                print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📝 Description:{desc['description']}")
        
        # Entry conditions
        print(f"\n🎯 Entry Conditions:")
        entries = self.strategy.get_entry_conditions()
        for signal_type, conditions in entries.items():
            print(f"\n  {signal_type}:")
            print(f"    {conditions['description']}")
            print(f"    Primary Conditions:")
            for cond in conditions['primary_conditions']:
                print(f"      ✓ {cond}")
        
        # Exit conditions
        print(f"\n🚪 Exit Conditions:")
        exits = self.strategy.get_exit_conditions()
        for exit_type, conditions in exits.items():
            print(f"\n  {exit_type}:")
            print(f"    {conditions['description']}")
        
        # Risk management
        print(f"\n⚠️  Risk Management Rules:")
        risk_rules = self.strategy.get_risk_management_rules()
        print(f"  • Risk per trade: {risk_rules['position_sizing']['risk_per_trade']}")
        print(f"  • Max concurrent positions: {risk_rules['position_sizing']['max_concurrent_positions']}")
        print(f"  • Initial stop loss: {risk_rules['stop_loss']['initial']}")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run trading strategies')
    parser.add_argument('--strategy', type=str, default='RSI_Momentum_Divergence',
                        help='Strategy to run (default: RSI_Momentum_Divergence)')
    parser.add_argument('--mode', type=str, choices=['monitor', 'info'], 
                        default='info',
                        help='Mode to run in (default: info)')
    parser.add_argument('--interval', type=int, default=60,
                        help='Check interval in seconds for monitor mode (default: 60)')
    parser.add_argument('--db', type=str, default='data/trading_data_BTC.db',
                        help='Path to database (default: data/trading_data_BTC.db)')
    
    args = parser.parse_args()
    
    # Initialize strategy runner
    runner = StrategyRunner(args.strategy, args.db)
    
    if args.mode == 'info':
        runner.display_strategy_info()
    elif args.mode == 'monitor':
        runner.display_strategy_info()
        runner.monitor_signals(check_interval=args.interval)
    

if __name__ == "__main__":
    main()
