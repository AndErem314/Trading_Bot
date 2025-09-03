"""
Performance Metrics Module for Backtesting

This module provides comprehensive performance metrics calculation
for analyzing backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for backtesting results
    """
    
    @staticmethod
    def calculate_returns_metrics(equity_curve: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate return-based metrics for crypto trading
        
        Args:
            equity_curve: DataFrame with 'timestamp' and 'equity' columns
            
        Returns:
            Dictionary of return metrics
        """
        if len(equity_curve) < 2:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'avg_return_per_trade': 0.0,
                'best_period_return': 0.0,
                'worst_period_return': 0.0,
                'time_in_market': 0.0,
                'buy_and_hold_return': 0.0,
                'strategy_vs_buy_hold': 0.0
            }
        
        # Calculate returns
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        
        # Total return (absolute and percentage)
        initial_capital = equity_curve['equity'].iloc[0]
        final_capital = equity_curve['equity'].iloc[-1]
        total_return = final_capital - initial_capital
        total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
        
        # Buy and hold comparison (if price data available)
        buy_hold_return = 0.0
        if 'price' in equity_curve.columns:
            initial_price = equity_curve['price'].iloc[0]
            final_price = equity_curve['price'].iloc[-1]
            buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        
        strategy_vs_buy_hold = total_return_pct - buy_hold_return
        
        # Period returns (4-hour periods for crypto)
        period_returns = equity_curve['returns'].dropna()
        
        # Best and worst period returns
        best_period = period_returns.max() * 100 if len(period_returns) > 0 else 0
        worst_period = period_returns.min() * 100 if len(period_returns) > 0 else 0
        
        # Average return per period (when position is active)
        avg_return = period_returns[period_returns != 0].mean() * 100 if len(period_returns[period_returns != 0]) > 0 else 0
        
        # Time in market (percentage of time with active positions)
        periods_with_position = (period_returns != 0).sum()
        total_periods = len(period_returns)
        time_in_market = (periods_with_position / total_periods * 100) if total_periods > 0 else 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'avg_return_per_trade': avg_return,
            'best_period_return': best_period,
            'worst_period_return': worst_period,
            'time_in_market': time_in_market,
            'buy_and_hold_return': buy_hold_return,
            'strategy_vs_buy_hold': strategy_vs_buy_hold
        }
    
    @staticmethod
    def calculate_risk_metrics(
        equity_curve: pd.DataFrame,
        risk_free_rate: float = 0.0  # Set to 0 for crypto (no risk-free rate)
    ) -> Dict[str, float]:
        """
        Calculate risk-based metrics for crypto trading
        
        Args:
            equity_curve: DataFrame with equity values
            risk_free_rate: Risk-free rate (usually 0 for crypto)
            
        Returns:
            Dictionary of risk metrics
        """
        if len(equity_curve) < 2:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'volatility': 0.0,
                'downside_deviation': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'risk_reward_ratio': 0.0,
                'tail_ratio': 0.0
            }
        
        # Calculate returns if not present
        if 'returns' not in equity_curve.columns:
            equity_curve['returns'] = equity_curve['equity'].pct_change()
        
        returns = equity_curve['returns'].dropna()
        
        # Volatility (for crypto, use hourly periods - 24*365 hours per year)
        periods_per_year = 24 * 365  # Hourly periods in a year for 24/7 crypto markets
        volatility = returns.std() * np.sqrt(periods_per_year) * 100
        
        # Sharpe Ratio (adjusted for crypto markets)
        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino Ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        sortino_ratio = (np.sqrt(periods_per_year) * excess_returns.mean() / 
                        downside_deviation) if downside_deviation > 0 else 0
        
        # Calmar Ratio (return to max drawdown ratio)
        total_return = ((equity_curve['equity'].iloc[-1] - equity_curve['equity'].iloc[0]) / 
                       equity_curve['equity'].iloc[0]) * 100
        max_dd = PerformanceMetrics.calculate_drawdown_metrics(equity_curve)['max_drawdown']
        calmar_ratio = (total_return / max_dd) if max_dd > 0 else 0
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100 if len(returns) > 0 else 0
        
        # Risk-Reward Ratio
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
        risk_reward_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0
        
        # Tail Ratio (95th percentile gain / 5th percentile loss)
        if len(returns) > 20:
            tail_gain = np.percentile(returns, 95)
            tail_loss = abs(np.percentile(returns, 5))
            tail_ratio = tail_gain / tail_loss if tail_loss > 0 else 0
        else:
            tail_ratio = 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
            'downside_deviation': downside_deviation * 100,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'risk_reward_ratio': risk_reward_ratio,
            'tail_ratio': tail_ratio
        }
    
    @staticmethod
    def calculate_drawdown_metrics(equity_curve: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate drawdown-based metrics
        
        Args:
            equity_curve: DataFrame with equity values
            
        Returns:
            Dictionary of drawdown metrics
        """
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'recovery_factor': 0.0,
                'ulcer_index': 0.0
            }
        
        # Calculate running maximum
        equity = equity_curve['equity'].values
        running_max = np.maximum.accumulate(equity)
        
        # Drawdown series
        drawdown = (running_max - equity) / running_max * 100
        
        # Maximum drawdown
        max_drawdown = drawdown.max()
        
        # Average drawdown
        avg_drawdown = drawdown[drawdown > 0].mean() if len(drawdown[drawdown > 0]) > 0 else 0
        
        # Drawdown duration
        in_drawdown = drawdown > 0
        drawdown_starts = (~in_drawdown[:-1]) & in_drawdown[1:]
        drawdown_ends = in_drawdown[:-1] & (~in_drawdown[1:])
        
        if drawdown_starts.any() and drawdown_ends.any():
            # Find longest drawdown period
            max_duration = 0
            current_duration = 0
            
            for i in range(len(drawdown)):
                if drawdown[i] > 0:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
        else:
            max_duration = 0
        
        # Recovery factor (total return / max drawdown)
        total_return = ((equity[-1] - equity[0]) / equity[0]) * 100
        recovery_factor = (total_return / max_drawdown) if max_drawdown > 0 else 0
        
        # Ulcer Index (root mean square of drawdowns)
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_duration,
            'recovery_factor': recovery_factor,
            'ulcer_index': ulcer_index
        }
    
    @staticmethod
    def calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trade-based metrics
        
        Args:
            trades_df: DataFrame with trade records
            
        Returns:
            Dictionary of trade metrics
        """
        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_loss_ratio': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'avg_trade_duration': 0.0,
                'trades_per_day': 0.0
            }
        
        # Winning and losing trades
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        # Basic metrics
        total_trades = len(trades_df)
        win_rate = len(winning_trades) / total_trades * 100
        
        # Average win/loss
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        
        # Win/loss ratio
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        
        # Consecutive wins/losses
        is_win = (trades_df['pnl'] > 0).astype(int)
        win_streaks = []
        loss_streaks = []
        current_streak = 0
        
        for i in range(len(is_win)):
            if i == 0:
                current_streak = 1
            elif is_win.iloc[i] == is_win.iloc[i-1]:
                current_streak += 1
            else:
                if is_win.iloc[i-1]:
                    win_streaks.append(current_streak)
                else:
                    loss_streaks.append(current_streak)
                current_streak = 1
        
        # Add final streak
        if len(is_win) > 0:
            if is_win.iloc[-1]:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
        
        max_consecutive_wins = max(win_streaks) if win_streaks else 0
        max_consecutive_losses = max(loss_streaks) if loss_streaks else 0
        
        # Trade duration
        if 'entry_timestamp' in trades_df.columns and 'exit_timestamp' in trades_df.columns:
            durations = pd.to_datetime(trades_df['exit_timestamp']) - pd.to_datetime(trades_df['entry_timestamp'])
            avg_duration_hours = durations.dt.total_seconds().mean() / 3600 if len(durations) > 0 else 0
            max_duration_hours = durations.dt.total_seconds().max() / 3600 if len(durations) > 0 else 0
            min_duration_hours = durations.dt.total_seconds().min() / 3600 if len(durations) > 0 else 0
        else:
            avg_duration_hours = 0
            max_duration_hours = 0
            min_duration_hours = 0
        
        # Trading frequency (per week for crypto)
        if len(trades_df) > 1:
            first_trade = pd.to_datetime(trades_df['entry_timestamp'].iloc[0])
            last_trade = pd.to_datetime(trades_df['exit_timestamp'].iloc[-1])
            total_hours = (last_trade - first_trade).total_seconds() / 3600
            total_weeks = total_hours / (24 * 7)
            trades_per_week = total_trades / total_weeks if total_weeks > 0 else 0
        else:
            trades_per_week = 0
        
        # Payoff Ratio (average win / average loss)
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'payoff_ratio': payoff_ratio,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_trade_duration': avg_duration_hours,
            'max_trade_duration': max_duration_hours,
            'min_trade_duration': min_duration_hours,
            'trades_per_week': trades_per_week
        }
    
    @staticmethod
    def calculate_monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly returns table
        
        Args:
            equity_curve: DataFrame with timestamp and equity columns
            
        Returns:
            DataFrame with monthly returns by year
        """
        if len(equity_curve) == 0:
            return pd.DataFrame()
        
        # Set timestamp as index
        df = equity_curve.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate monthly returns
        monthly = df['equity'].resample('M').last()
        monthly_returns = monthly.pct_change() * 100
        
        # Create pivot table
        monthly_returns_df = pd.DataFrame(monthly_returns)
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month
        
        # Pivot to create year x month table
        pivot_table = monthly_returns_df.pivot_table(
            values='equity',
            index='Year',
            columns='Month',
            aggfunc='first'
        )
        
        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
        
        # Add yearly total
        pivot_table['Year'] = pivot_table.sum(axis=1)
        
        return pivot_table
    
    @staticmethod
    def generate_complete_metrics(
        equity_curve: pd.DataFrame,
        trades_df: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> Dict[str, any]:
        """
        Generate all performance metrics
        
        Args:
            equity_curve: DataFrame with equity values
            trades_df: DataFrame with trade records
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary containing all metrics
        """
        # Calculate all metric groups
        returns_metrics = PerformanceMetrics.calculate_returns_metrics(equity_curve)
        risk_metrics = PerformanceMetrics.calculate_risk_metrics(equity_curve, risk_free_rate)
        drawdown_metrics = PerformanceMetrics.calculate_drawdown_metrics(equity_curve)
        trade_metrics = PerformanceMetrics.calculate_trade_metrics(trades_df)
        monthly_returns = PerformanceMetrics.calculate_monthly_returns(equity_curve)
        
        # Combine all metrics
        all_metrics = {
            **returns_metrics,
            **risk_metrics,
            **drawdown_metrics,
            **trade_metrics,
            'monthly_returns': monthly_returns
        }
        
        return all_metrics
    
    @staticmethod
    def create_performance_summary(metrics: Dict[str, float]) -> str:
        """
        Create a formatted performance summary
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Formatted string summary
        """
        summary = f"""
═══════════════════════════════════════════════════════════════
                    PERFORMANCE SUMMARY
═══════════════════════════════════════════════════════════════

RETURNS & PERFORMANCE
--------------------
Total Return:          ${metrics.get('total_return', 0):.2f} ({metrics.get('total_return_pct', 0):.2f}%)
Buy & Hold Return:     {metrics.get('buy_and_hold_return', 0):.2f}%
Strategy vs B&H:       {metrics.get('strategy_vs_buy_hold', 0):+.2f}%
Time in Market:        {metrics.get('time_in_market', 0):.1f}%
Avg Return/Trade:      {metrics.get('avg_return_per_trade', 0):.3f}%

RISK METRICS
------------
Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.2f}
Sortino Ratio:         {metrics.get('sortino_ratio', 0):.2f}
Calmar Ratio:          {metrics.get('calmar_ratio', 0):.2f}
Volatility:            {metrics.get('volatility', 0):.2f}%
Risk/Reward Ratio:     {metrics.get('risk_reward_ratio', 0):.2f}
VaR (95%):             {metrics.get('var_95', 0):.2f}%
CVaR (95%):            {metrics.get('cvar_95', 0):.2f}%
Tail Ratio:            {metrics.get('tail_ratio', 0):.2f}

DRAWDOWN
--------
Max Drawdown:          {metrics.get('max_drawdown', 0):.2f}%
Avg Drawdown:          {metrics.get('avg_drawdown', 0):.2f}%
Max DD Duration:       {metrics.get('max_drawdown_duration', 0)} days
Recovery Factor:       {metrics.get('recovery_factor', 0):.2f}
Ulcer Index:           {metrics.get('ulcer_index', 0):.2f}

TRADE STATISTICS
----------------
Total Trades:          {metrics.get('total_trades', 0)}
Win Rate:              {metrics.get('win_rate', 0):.2f}%
Profit Factor:         {metrics.get('profit_factor', 0):.2f}
Expectancy:            ${metrics.get('expectancy', 0):.2f}
Avg Win:               ${metrics.get('avg_win', 0):.2f}
Avg Loss:              ${metrics.get('avg_loss', 0):.2f}
Win/Loss Ratio:        {metrics.get('win_loss_ratio', 0):.2f}
Payoff Ratio:          {metrics.get('payoff_ratio', 0):.2f}
Max Consec. Wins:      {metrics.get('max_consecutive_wins', 0)}
Max Consec. Losses:    {metrics.get('max_consecutive_losses', 0)}
Avg Trade Duration:    {metrics.get('avg_trade_duration', 0):.1f} hours
Max Trade Duration:    {metrics.get('max_trade_duration', 0):.1f} hours
Trades Per Week:       {metrics.get('trades_per_week', 0):.2f}

═══════════════════════════════════════════════════════════════
        """
        return summary
