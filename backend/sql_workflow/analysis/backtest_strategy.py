#!/usr/bin/env python3
"""
Strategy Backtesting Module

This module provides comprehensive backtesting functionality for trading strategies.
It simulates trades based on historical data and calculates performance metrics.

Usage:
    python backtest_strategy.py --strategy RSI_Momentum_Divergence
    python backtest_strategy.py --strategy RSI_Momentum_Divergence --start 2024-01-01 --end 2024-12-31
    python backtest_strategy.py --strategy RSI_Momentum_Divergence --capital 10000 --fee 0.001
"""

import sqlite3
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from backend.sql_workflow.strategies import RSIMomentumDivergenceSwingStrategy


class StrategyBacktester:
    """Comprehensive backtesting system for trading strategies"""
    
    def __init__(self, strategy_name, db_path='data/trading_data_BTC.db'):
        self.db_path = db_path
        self.strategy_name = strategy_name
        self.strategy = self._load_strategy()
        
        # Backtesting parameters
        self.initial_capital = 10000
        self.position_size_pct = 0.02  # 2% risk per trade
        self.trading_fee = 0.001  # 0.1% trading fee
        self.slippage = 0.0005  # 0.05% slippage
        
        # Trade tracking
        self.trades = []
        self.positions = []
        self.equity_curve = []
        
    def _load_strategy(self):
        """Load the specified strategy"""
        if self.strategy_name == "RSI_Momentum_Divergence":
            return RSIMomentumDivergenceSwingStrategy()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
    
    def run_backtest(self, start_date=None, end_date=None, initial_capital=10000, 
                     position_size_pct=0.02, trading_fee=0.001):
        """
        Run a comprehensive backtest of the strategy
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Starting capital
            position_size_pct: Position size as percentage of capital
            trading_fee: Trading fee as percentage
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.trading_fee = trading_fee
        
        print(f"\n{'='*80}")
        print(f"BACKTESTING: {self.strategy.name}")
        print(f"{'='*80}\n")
        
        # Get all signals
        signals_df = self._get_historical_signals(start_date, end_date)
        
        if signals_df.empty:
            print("❌ No signals found in the specified period")
            return
        
        print(f"📊 Backtest Parameters:")
        print(f"  • Period: {signals_df['timestamp'].min()} to {signals_df['timestamp'].max()}")
        print(f"  • Initial Capital: ${initial_capital:,.2f}")
        print(f"  • Position Size: {position_size_pct*100}% of capital")
        print(f"  • Trading Fee: {trading_fee*100}%")
        print(f"  • Total Signals: {len(signals_df)}")
        
        # Simulate trades
        self._simulate_trades(signals_df)
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Generate report
        self._generate_backtest_report()
    
    def _get_historical_signals(self, start_date=None, end_date=None):
        """Get historical signals from the database"""
        # Modify the strategy query for backtesting
        query = self.strategy.get_sql_query()
        
        # Remove the limit
        query = query.replace("LIMIT 100;", "")
        query = query.replace("LIMIT 100", "")
        
        # Build date filter
        if start_date and end_date:
            date_filter = f"datetime('{start_date}') AND o.timestamp <= datetime('{end_date}')"
        elif start_date:
            date_filter = f"datetime('{start_date}')"
        else:
            date_filter = "datetime('2020-01-01')"
        
        # Replace the date filter
        query = query.replace("datetime('now', '-6 months')  -- Analyze last 6 months", date_filter)
        query = query.replace("datetime('now', '-6 months')", date_filter)
        
        # Execute query and return as DataFrame
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
            # Handle different timestamp formats
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
            except:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
            df = df.sort_values('timestamp')
            
        return df
    
    def _simulate_trades(self, signals_df):
        """Simulate trades based on signals"""
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = [(signals_df.iloc[0]['timestamp'], capital)]
        
        for idx, row in signals_df.iterrows():
            timestamp = row['timestamp']
            price = float(row['price'])
            signal = row['signal']
            signal_name = row['signal_name']
            
            # Handle entry signals
            if signal == 1 and position is None:  # BUY signal
                # Calculate position size
                position_value = capital * self.position_size_pct
                shares = position_value / price
                cost = position_value * (1 + self.trading_fee + self.slippage)
                
                position = {
                    'type': 'LONG',
                    'entry_time': timestamp,
                    'entry_price': price * (1 + self.slippage),
                    'shares': shares,
                    'cost': cost
                }
                capital -= cost
                
            elif signal == -1 and position is None:  # SELL signal
                # Calculate position size for short
                position_value = capital * self.position_size_pct
                shares = position_value / price
                cost = position_value * self.trading_fee
                
                position = {
                    'type': 'SHORT',
                    'entry_time': timestamp,
                    'entry_price': price * (1 - self.slippage),
                    'shares': shares,
                    'cost': cost,
                    'initial_value': position_value  # Track initial position value for shorts
                }
                # Don't subtract cost from capital for shorts, just track fee
                capital -= cost
                
            # Handle exit signals
            elif signal == -2 and position and position['type'] == 'LONG':  # EXIT LONG
                exit_value = position['shares'] * price * (1 - self.slippage)
                exit_cost = exit_value * self.trading_fee
                profit = exit_value - position['cost'] - exit_cost
                
                trades.append({
                    'type': 'LONG',
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': position['entry_price'],
                    'exit_price': price * (1 - self.slippage),
                    'shares': position['shares'],
                    'profit': profit,
                    'profit_pct': (profit / position['cost']) * 100,
                    'duration': (timestamp - position['entry_time']).total_seconds() / 3600
                })
                
                capital += exit_value - exit_cost
                position = None
                
            elif signal == 2 and position and position['type'] == 'SHORT':  # EXIT SHORT
                # Calculate profit/loss for short position
                entry_value = position['shares'] * position['entry_price']
                exit_value = position['shares'] * price * (1 + self.slippage)
                exit_cost = exit_value * self.trading_fee
                profit = entry_value - exit_value - position['cost'] - exit_cost
                
                trades.append({
                    'type': 'SHORT',
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': position['entry_price'],
                    'exit_price': price * (1 + self.slippage),
                    'shares': position['shares'],
                    'profit': profit,
                    'profit_pct': (profit / position.get('initial_value', position_value)) * 100,
                    'duration': (timestamp - position['entry_time']).total_seconds() / 3600
                })
                
                # For shorts: receive back the entry value minus exit costs and profit/loss
                capital += entry_value - exit_value - exit_cost
                position = None
            
            # Track equity curve
            current_value = capital
            if position:
                if position['type'] == 'LONG':
                    current_value += position['shares'] * price
                else:  # SHORT
                    # For short positions: profit = entry_price - current_price
                    current_value += position['shares'] * position['entry_price'] - position['shares'] * price
            
            equity_curve.append((timestamp, current_value))
        
        # Close any open positions at the end
        if position:
            final_price = signals_df.iloc[-1]['price']
            if position['type'] == 'LONG':
                exit_value = position['shares'] * final_price * (1 - self.slippage)
                exit_cost = exit_value * self.trading_fee
                profit = exit_value - position['cost'] - exit_cost
                capital += exit_value - exit_cost
            else:  # SHORT
                entry_value = position['shares'] * position['entry_price']
                exit_value = position['shares'] * final_price * (1 + self.slippage)
                exit_cost = exit_value * self.trading_fee
                profit = entry_value - exit_value - position['cost'] - exit_cost
                capital += entry_value - exit_value - exit_cost
            
            trades.append({
                'type': position['type'],
                'entry_time': position['entry_time'],
                'exit_time': signals_df.iloc[-1]['timestamp'],
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'shares': position['shares'],
                'profit': profit,
                'profit_pct': (profit / position['cost']) * 100,
                'duration': (signals_df.iloc[-1]['timestamp'] - position['entry_time']).total_seconds() / 3600
            })
        
        self.trades = pd.DataFrame(trades)
        self.equity_curve = pd.DataFrame(equity_curve, columns=['timestamp', 'equity'])
        self.final_capital = capital
        
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if self.trades.empty:
            self.metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'avg_trade_duration': 0
            }
            return
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len(self.trades[self.trades['profit'] > 0])
        losing_trades = len(self.trades[self.trades['profit'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit metrics
        total_profit = self.trades['profit'].sum()
        gross_profit = self.trades[self.trades['profit'] > 0]['profit'].sum()
        gross_loss = abs(self.trades[self.trades['profit'] < 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns
        total_return = ((self.final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Risk metrics
        returns = self.equity_curve['equity'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown
        self.equity_curve['peak'] = self.equity_curve['equity'].cummax()
        self.equity_curve['drawdown'] = (self.equity_curve['equity'] - self.equity_curve['peak']) / self.equity_curve['peak']
        max_drawdown = self.equity_curve['drawdown'].min() * 100
        
        # Trade duration
        avg_trade_duration = self.trades['duration'].mean() if not self.trades.empty else 0
        
        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'total_profit': total_profit,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_trade_duration': avg_trade_duration,
            'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0,
            'best_trade': self.trades['profit'].max() if not self.trades.empty else 0,
            'worst_trade': self.trades['profit'].min() if not self.trades.empty else 0,
            'avg_winning_trade': self.trades[self.trades['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0,
            'avg_losing_trade': self.trades[self.trades['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0
        }
    
    def _generate_backtest_report(self):
        """Generate and display comprehensive backtest report"""
        print(f"\n{'='*80}")
        print("📈 BACKTEST RESULTS")
        print(f"{'='*80}\n")
        
        # Performance Summary
        print("💰 PERFORMANCE SUMMARY:")
        print(f"  • Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  • Final Capital: ${self.final_capital:,.2f}")
        print(f"  • Total Return: {self.metrics['total_return']:.2f}%")
        print(f"  • Total Profit: ${self.metrics['total_profit']:,.2f}")
        
        print("\n📊 TRADE STATISTICS:")
        print(f"  • Total Trades: {self.metrics['total_trades']}")
        print(f"  • Winning Trades: {self.metrics['winning_trades']}")
        print(f"  • Losing Trades: {self.metrics['losing_trades']}")
        print(f"  • Win Rate: {self.metrics['win_rate']:.2f}%")
        print(f"  • Profit Factor: {self.metrics['profit_factor']:.2f}")
        print(f"  • Average Trade Duration: {self.metrics['avg_trade_duration']:.1f} hours")
        
        print("\n💵 PROFIT ANALYSIS:")
        print(f"  • Gross Profit: ${self.metrics['gross_profit']:,.2f}")
        print(f"  • Gross Loss: ${self.metrics['gross_loss']:,.2f}")
        print(f"  • Average Profit per Trade: ${self.metrics['avg_profit_per_trade']:,.2f}")
        print(f"  • Best Trade: ${self.metrics['best_trade']:,.2f}")
        print(f"  • Worst Trade: ${self.metrics['worst_trade']:,.2f}")
        print(f"  • Avg Winning Trade: ${self.metrics['avg_winning_trade']:,.2f}")
        print(f"  • Avg Losing Trade: ${self.metrics['avg_losing_trade']:,.2f}")
        
        print("\n📉 RISK METRICS:")
        print(f"  • Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"  • Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
        
        # Recent trades
        if not self.trades.empty:
            print("\n🔄 LAST 10 TRADES:")
            print("-" * 80)
            recent_trades = self.trades.tail(10)
            for idx, trade in recent_trades.iterrows():
                print(f"  {trade['type']} | Entry: {trade['entry_time'].strftime('%Y-%m-%d')} @ ${trade['entry_price']:,.2f} | "
                      f"Exit: {trade['exit_time'].strftime('%Y-%m-%d')} @ ${trade['exit_price']:,.2f} | "
                      f"Profit: ${trade['profit']:,.2f} ({trade['profit_pct']:.2f}%)")
        
        print(f"\n{'='*80}\n")
    
    def export_results(self, filename=None):
        """Export backtest results to CSV files"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{self.strategy_name}_{timestamp}"
        
        # Export trades
        if not self.trades.empty:
            trades_file = f"{filename}_trades.csv"
            self.trades.to_csv(trades_file, index=False)
            print(f"✅ Trades exported to: {trades_file}")
        
        # Export equity curve
        if not self.equity_curve.empty:
            equity_file = f"{filename}_equity.csv"
            self.equity_curve.to_csv(equity_file, index=False)
            print(f"✅ Equity curve exported to: {equity_file}")
        
        # Export metrics
        metrics_file = f"{filename}_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"Backtest Results for {self.strategy.name}\n")
            f.write("="*50 + "\n\n")
            for key, value in self.metrics.items():
                f.write(f"{key}: {value}\n")
        print(f"✅ Metrics exported to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--strategy', type=str, default='RSI_Momentum_Divergence',
                        help='Strategy to backtest')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--size', type=float, default=0.02,
                        help='Position size as percentage (default: 0.02)')
    parser.add_argument('--fee', type=float, default=0.001,
                        help='Trading fee as percentage (default: 0.001)')
    parser.add_argument('--export', action='store_true',
                        help='Export results to CSV files')
    parser.add_argument('--db', type=str, default='data/trading_data_BTC.db',
                        help='Path to database')
    
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = StrategyBacktester(args.strategy, args.db)
    
    # Run backtest
    backtester.run_backtest(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        position_size_pct=args.size,
        trading_fee=args.fee
    )
    
    # Export results if requested
    if args.export:
        backtester.export_results()


if __name__ == "__main__":
    main()
