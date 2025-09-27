"""
Performance Analytics Module

This module provides comprehensive performance analysis for trading strategies,
including calculation of various risk-adjusted returns, drawdown analysis,
and detailed trade statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""
    # Return metrics
    total_return: float
    cumulative_return: float
    annualized_return: float
    monthly_returns: pd.Series
    
    # Risk metrics
    volatility: float
    annualized_volatility: float
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    underwater_periods: List[Dict[str, Any]]
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    
    # Trade performance
    gross_profit: float
    gross_loss: float
    net_profit: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    
    # Streaks
    win_streak_current: int
    win_streak_max: int
    loss_streak_current: int
    loss_streak_max: int
    
    # Additional metrics
    recovery_factor: float
    payoff_ratio: float
    profit_per_day: float
    trades_per_day: float
    
    # Statistical metrics
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional Value at Risk
    
    # Time-based analysis
    best_month: Tuple[str, float]
    worst_month: Tuple[str, float]
    positive_months_pct: float
    
    # Rolling metrics
    rolling_sharpe: Optional[pd.Series] = None
    rolling_volatility: Optional[pd.Series] = None
    rolling_win_rate: Optional[pd.Series] = None


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    
    This class calculates various performance metrics from trade history
    and equity curves, supporting both aggregate and rolling calculations.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.metrics: Optional[PerformanceMetrics] = None
        self.trades_df: Optional[pd.DataFrame] = None
        self.equity_curve: Optional[pd.DataFrame] = None
        
    def calculate_all_metrics(self, trades: pd.DataFrame, 
                            equity_curve: pd.DataFrame) -> PerformanceMetrics:
        """
        Calculate all performance metrics.
        
        Args:
            trades: DataFrame with trade history
            equity_curve: DataFrame with equity values over time
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        logger.info("Calculating comprehensive performance metrics...")
        
        self.trades_df = trades.copy()
        self.equity_curve = equity_curve.copy()
        
        # Ensure proper datetime index
        if not isinstance(self.equity_curve.index, pd.DatetimeIndex):
            self.equity_curve.index = pd.to_datetime(self.equity_curve.index)
        
        # Calculate returns if not present
        if 'returns' not in self.equity_curve.columns:
            self.equity_curve['returns'] = self.equity_curve['total_value'].pct_change()
        
        # Return metrics
        total_return = self._calculate_total_return()
        cumulative_return = self._calculate_cumulative_return()
        annualized_return = self._calculate_annualized_return()
        monthly_returns = self._calculate_monthly_returns()
        
        # Risk metrics
        volatility, annualized_vol = self._calculate_volatility()
        drawdown_metrics = self._calculate_drawdown_metrics()
        underwater_periods = self._analyze_underwater_periods()
        
        # Risk-adjusted metrics
        sharpe = self._calculate_sharpe_ratio(annualized_return, annualized_vol)
        sortino = self._calculate_sortino_ratio()
        calmar = self._calculate_calmar_ratio(annualized_return, drawdown_metrics['max_dd_pct'])
        info_ratio = self._calculate_information_ratio()
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics()
        
        # Streaks
        streak_stats = self._calculate_streak_statistics()
        
        # Additional metrics
        recovery_factor = self._calculate_recovery_factor(total_return, drawdown_metrics['max_dd'])
        payoff_ratio = self._calculate_payoff_ratio(trade_stats)
        daily_stats = self._calculate_daily_statistics()
        
        # Statistical metrics
        stat_metrics = self._calculate_statistical_metrics()
        
        # Time-based analysis
        time_analysis = self._analyze_time_based_performance(monthly_returns)
        
        # Rolling metrics
        rolling_metrics = self._calculate_rolling_metrics()
        
        # Create metrics object
        self.metrics = PerformanceMetrics(
            # Returns
            total_return=total_return,
            cumulative_return=cumulative_return,
            annualized_return=annualized_return,
            monthly_returns=monthly_returns,
            
            # Risk
            volatility=volatility,
            annualized_volatility=annualized_vol,
            max_drawdown=drawdown_metrics['max_dd'],
            max_drawdown_pct=drawdown_metrics['max_dd_pct'],
            max_drawdown_duration=drawdown_metrics['max_dd_duration'],
            underwater_periods=underwater_periods,
            
            # Risk-adjusted
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            
            # Trade stats
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            expectancy=trade_stats['expectancy'],
            
            # Trade performance
            gross_profit=trade_stats['gross_profit'],
            gross_loss=trade_stats['gross_loss'],
            net_profit=trade_stats['net_profit'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            largest_win=trade_stats['largest_win'],
            largest_loss=trade_stats['largest_loss'],
            avg_trade_duration=trade_stats['avg_trade_duration'],
            
            # Streaks
            win_streak_current=streak_stats['win_streak_current'],
            win_streak_max=streak_stats['win_streak_max'],
            loss_streak_current=streak_stats['loss_streak_current'],
            loss_streak_max=streak_stats['loss_streak_max'],
            
            # Additional
            recovery_factor=recovery_factor,
            payoff_ratio=payoff_ratio,
            profit_per_day=daily_stats['profit_per_day'],
            trades_per_day=daily_stats['trades_per_day'],
            
            # Statistical
            skewness=stat_metrics['skewness'],
            kurtosis=stat_metrics['kurtosis'],
            var_95=stat_metrics['var_95'],
            cvar_95=stat_metrics['cvar_95'],
            
            # Time-based
            best_month=time_analysis['best_month'],
            worst_month=time_analysis['worst_month'],
            positive_months_pct=time_analysis['positive_months_pct'],
            
            # Rolling
            rolling_sharpe=rolling_metrics['rolling_sharpe'],
            rolling_volatility=rolling_metrics['rolling_volatility'],
            rolling_win_rate=rolling_metrics['rolling_win_rate']
        )
        
        logger.info("Performance metrics calculation complete")
        return self.metrics
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Formatted performance report as string
        """
        if not self.metrics:
            raise ValueError("No metrics calculated. Run calculate_all_metrics first.")
        
        report = []
        report.append("=" * 80)
        report.append("TRADING STRATEGY PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Return Metrics
        report.append("RETURN METRICS")
        report.append("-" * 40)
        report.append(f"Total Return:         {self.metrics.total_return:>10.2f}%")
        report.append(f"Cumulative Return:    {self.metrics.cumulative_return:>10.2f}%")
        report.append(f"Annualized Return:    {self.metrics.annualized_return:>10.2f}%")
        report.append(f"Monthly Avg Return:   {self.metrics.monthly_returns.mean():>10.2f}%")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-" * 40)
        report.append(f"Volatility (Daily):   {self.metrics.volatility:>10.4f}")
        report.append(f"Volatility (Annual):  {self.metrics.annualized_volatility:>10.2f}%")
        report.append(f"Max Drawdown:         ${self.metrics.max_drawdown:>10.2f}")
        report.append(f"Max Drawdown %:       {self.metrics.max_drawdown_pct:>10.2f}%")
        report.append(f"Max DD Duration:      {self.metrics.max_drawdown_duration:>10} days")
        report.append(f"Value at Risk (95%):  {self.metrics.var_95:>10.2f}%")
        report.append(f"Conditional VaR:      {self.metrics.cvar_95:>10.2f}%")
        report.append("")
        
        # Risk-Adjusted Returns
        report.append("RISK-ADJUSTED RETURNS")
        report.append("-" * 40)
        report.append(f"Sharpe Ratio:         {self.metrics.sharpe_ratio:>10.2f}")
        report.append(f"Sortino Ratio:        {self.metrics.sortino_ratio:>10.2f}")
        report.append(f"Calmar Ratio:         {self.metrics.calmar_ratio:>10.2f}")
        report.append(f"Information Ratio:    {self.metrics.information_ratio:>10.2f}")
        report.append("")
        
        # Trading Statistics
        report.append("TRADING STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Trades:         {self.metrics.total_trades:>10}")
        report.append(f"Winning Trades:       {self.metrics.winning_trades:>10}")
        report.append(f"Losing Trades:        {self.metrics.losing_trades:>10}")
        report.append(f"Win Rate:             {self.metrics.win_rate:>10.2f}%")
        report.append(f"Profit Factor:        {self.metrics.profit_factor:>10.2f}")
        report.append(f"Expectancy:           ${self.metrics.expectancy:>10.2f}")
        report.append(f"Payoff Ratio:         {self.metrics.payoff_ratio:>10.2f}")
        report.append("")
        
        # Trade Performance
        report.append("TRADE PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Gross Profit:         ${self.metrics.gross_profit:>10.2f}")
        report.append(f"Gross Loss:           ${self.metrics.gross_loss:>10.2f}")
        report.append(f"Net Profit:           ${self.metrics.net_profit:>10.2f}")
        report.append(f"Average Win:          ${self.metrics.avg_win:>10.2f}")
        report.append(f"Average Loss:         ${self.metrics.avg_loss:>10.2f}")
        report.append(f"Largest Win:          ${self.metrics.largest_win:>10.2f}")
        report.append(f"Largest Loss:         ${self.metrics.largest_loss:>10.2f}")
        report.append(f"Avg Trade Duration:   {self.metrics.avg_trade_duration:>10.1f} bars")
        report.append("")
        
        # Streak Analysis
        report.append("STREAK ANALYSIS")
        report.append("-" * 40)
        report.append(f"Current Win Streak:   {self.metrics.win_streak_current:>10}")
        report.append(f"Max Win Streak:       {self.metrics.win_streak_max:>10}")
        report.append(f"Current Loss Streak:  {self.metrics.loss_streak_current:>10}")
        report.append(f"Max Loss Streak:      {self.metrics.loss_streak_max:>10}")
        report.append("")
        
        # Time-Based Analysis
        report.append("TIME-BASED ANALYSIS")
        report.append("-" * 40)
        report.append(f"Best Month:           {self.metrics.best_month[0]} ({self.metrics.best_month[1]:.2f}%)")
        report.append(f"Worst Month:          {self.metrics.worst_month[0]} ({self.metrics.worst_month[1]:.2f}%)")
        report.append(f"Positive Months:      {self.metrics.positive_months_pct:>10.2f}%")
        report.append(f"Profit per Day:       ${self.metrics.profit_per_day:>10.2f}")
        report.append(f"Trades per Day:       {self.metrics.trades_per_day:>10.2f}")
        report.append("")
        
        # Statistical Properties
        report.append("STATISTICAL PROPERTIES")
        report.append("-" * 40)
        report.append(f"Returns Skewness:     {self.metrics.skewness:>10.2f}")
        report.append(f"Returns Kurtosis:     {self.metrics.kurtosis:>10.2f}")
        report.append(f"Recovery Factor:      {self.metrics.recovery_factor:>10.2f}")
        report.append("")
        
        # Underwater Periods
        report.append("UNDERWATER PERIODS (Top 5)")
        report.append("-" * 40)
        for i, period in enumerate(self.metrics.underwater_periods[:5]):
            report.append(f"{i+1}. Duration: {period['duration']} days, "
                         f"Depth: {period['max_depth']:.2f}%")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Performance report saved to {output_path}")
        
        return report_text
    
    def plot_equity_curve(self, save_path: Optional[str] = None, 
                         show_drawdowns: bool = True,
                         show_trades: bool = True) -> plt.Figure:
        """
        Plot the equity curve with optional drawdowns and trade markers.
        
        Args:
            save_path: Optional path to save the plot
            show_drawdowns: Whether to highlight drawdown periods
            show_trades: Whether to mark trade entry/exit points
            
        Returns:
            Matplotlib figure object
        """
        if self.equity_curve is None:
            raise ValueError("No equity curve data. Run calculate_all_metrics first.")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Equity curve
        ax1 = axes[0]
        ax1.plot(self.equity_curve.index, self.equity_curve['total_value'], 
                'b-', linewidth=2, label='Portfolio Value')
        
        if show_drawdowns:
            # Calculate and plot drawdowns
            running_max = self.equity_curve['total_value'].expanding().max()
            drawdown = (self.equity_curve['total_value'] - running_max) / running_max
            
            # Fill drawdown areas
            ax1.fill_between(self.equity_curve.index, 
                           self.equity_curve['total_value'],
                           running_max,
                           where=(drawdown < 0),
                           color='red', alpha=0.3, label='Drawdown')
        
        if show_trades and self.trades_df is not None:
            # Mark trade entries and exits
            for _, trade in self.trades_df.iterrows():
                if 'entry_time' in trade:
                    ax1.axvline(x=trade['entry_time'], color='g', alpha=0.3, 
                              linestyle='--', linewidth=1)
                if 'exit_time' in trade:
                    ax1.axvline(x=trade['exit_time'], color='r', alpha=0.3, 
                              linestyle='--', linewidth=1)
        
        ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns
        ax2 = axes[1]
        returns = self.equity_curve['returns'].fillna(0)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        ax2.bar(positive_returns.index, positive_returns * 100, 
               color='green', alpha=0.7, label='Positive')
        ax2.bar(negative_returns.index, negative_returns * 100, 
               color='red', alpha=0.7, label='Negative')
        
        ax2.set_title('Daily Returns', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Returns (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax3 = axes[2]
        cum_returns = (1 + returns).cumprod() - 1
        ax3.plot(cum_returns.index, cum_returns * 100, 'purple', 
                linewidth=2, label='Cumulative Return')
        ax3.fill_between(cum_returns.index, 0, cum_returns * 100, 
                        alpha=0.3, color='purple')
        
        ax3.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve plot saved to {save_path}")
        
        return fig
    
    def get_drawdown_analysis(self) -> pd.DataFrame:
        """
        Get detailed drawdown analysis.
        
        Returns:
            DataFrame with drawdown periods and statistics
        """
        if self.equity_curve is None:
            raise ValueError("No equity curve data. Run calculate_all_metrics first.")
        
        # Calculate drawdowns
        running_max = self.equity_curve['total_value'].expanding().max()
        drawdown = (self.equity_curve['total_value'] - running_max) / running_max
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = (~in_drawdown.shift(1).fillna(False)) & in_drawdown
        drawdown_ends = in_drawdown.shift(1).fillna(False) & (~in_drawdown)
        
        drawdown_periods = []
        current_start = None
        
        for idx, is_start in drawdown_starts.items():
            if is_start:
                current_start = idx
            
            if current_start and idx in drawdown_ends.index and drawdown_ends[idx]:
                # Calculate drawdown statistics
                dd_period = drawdown[current_start:idx]
                duration = (idx - current_start).days
                
                drawdown_periods.append({
                    'start': current_start,
                    'end': idx,
                    'duration_days': duration,
                    'max_depth': dd_period.min() * 100,
                    'recovery_time': duration,
                    'peak_value': running_max[current_start],
                    'trough_value': self.equity_curve['total_value'][dd_period.idxmin()]
                })
                
                current_start = None
        
        # Check for ongoing drawdown
        if current_start:
            dd_period = drawdown[current_start:]
            duration = (self.equity_curve.index[-1] - current_start).days
            
            drawdown_periods.append({
                'start': current_start,
                'end': 'Ongoing',
                'duration_days': duration,
                'max_depth': dd_period.min() * 100,
                'recovery_time': 'N/A',
                'peak_value': running_max[current_start],
                'trough_value': self.equity_curve['total_value'][dd_period.idxmin()]
            })
        
        # Convert to DataFrame and sort by depth
        dd_df = pd.DataFrame(drawdown_periods)
        if not dd_df.empty:
            dd_df = dd_df.sort_values('max_depth', ascending=True)
        
        return dd_df
    
    # Private calculation methods
    
    def _calculate_total_return(self) -> float:
        """Calculate total return percentage."""
        initial_value = self.equity_curve['total_value'].iloc[0]
        final_value = self.equity_curve['total_value'].iloc[-1]
        return ((final_value - initial_value) / initial_value) * 100
    
    def _calculate_cumulative_return(self) -> float:
        """Calculate cumulative return."""
        returns = self.equity_curve['returns'].fillna(0)
        return ((1 + returns).prod() - 1) * 100
    
    def _calculate_annualized_return(self) -> float:
        """Calculate annualized return."""
        total_days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if total_days == 0:
            return 0
        years = total_days / 365.25
        total_return = self._calculate_total_return() / 100
        return ((1 + total_return) ** (1 / years) - 1) * 100
    
    def _calculate_monthly_returns(self) -> pd.Series:
        """Calculate monthly returns."""
        monthly = self.equity_curve['total_value'].resample('M').last()
        return monthly.pct_change().dropna() * 100
    
    def _calculate_volatility(self) -> Tuple[float, float]:
        """Calculate daily and annualized volatility."""
        daily_vol = self.equity_curve['returns'].std()
        annual_vol = daily_vol * np.sqrt(252) * 100  # Assuming 252 trading days
        return daily_vol, annual_vol
    
    def _calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """Calculate drawdown metrics."""
        running_max = self.equity_curve['total_value'].expanding().max()
        drawdown = self.equity_curve['total_value'] - running_max
        drawdown_pct = drawdown / running_max * 100
        
        max_dd = drawdown.min()
        max_dd_pct = drawdown_pct.min()
        
        # Calculate drawdown duration
        in_drawdown = drawdown < 0
        dd_start = None
        max_duration = 0
        
        for idx, is_dd in in_drawdown.items():
            if is_dd and dd_start is None:
                dd_start = idx
            elif not is_dd and dd_start is not None:
                duration = (idx - dd_start).days
                max_duration = max(max_duration, duration)
                dd_start = None
        
        # Check for ongoing drawdown
        if dd_start is not None:
            duration = (self.equity_curve.index[-1] - dd_start).days
            max_duration = max(max_duration, duration)
        
        return {
            'max_dd': abs(max_dd),
            'max_dd_pct': abs(max_dd_pct),
            'max_dd_duration': max_duration
        }
    
    def _analyze_underwater_periods(self) -> List[Dict[str, Any]]:
        """Analyze periods when equity is below peak."""
        running_max = self.equity_curve['total_value'].expanding().max()
        underwater = (self.equity_curve['total_value'] - running_max) / running_max * 100
        
        periods = []
        in_underwater = underwater < 0
        start_idx = None
        
        for i, (idx, is_underwater) in enumerate(in_underwater.items()):
            if is_underwater and start_idx is None:
                start_idx = idx
            elif not is_underwater and start_idx is not None:
                duration = (idx - start_idx).days
                max_depth = underwater[start_idx:idx].min()
                periods.append({
                    'start': start_idx,
                    'end': idx,
                    'duration': duration,
                    'max_depth': max_depth
                })
                start_idx = None
        
        # Sort by duration
        periods.sort(key=lambda x: x['duration'], reverse=True)
        
        return periods
    
    def _calculate_sharpe_ratio(self, annual_return: float, annual_vol: float) -> float:
        """Calculate Sharpe ratio."""
        if annual_vol == 0:
            return 0
        return (annual_return - self.risk_free_rate * 100) / annual_vol
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (using downside deviation)."""
        returns = self.equity_curve['returns'].fillna(0)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = downside_returns.std() * np.sqrt(252)
        annual_return = self._calculate_annualized_return() / 100
        
        if downside_std == 0:
            return np.inf
        
        return (annual_return - self.risk_free_rate) / downside_std
    
    def _calculate_calmar_ratio(self, annual_return: float, max_dd_pct: float) -> float:
        """Calculate Calmar ratio."""
        if max_dd_pct == 0:
            return np.inf
        return annual_return / abs(max_dd_pct)
    
    def _calculate_information_ratio(self) -> float:
        """Calculate Information ratio (simplified - using returns vs 0)."""
        returns = self.equity_curve['returns'].fillna(0)
        if returns.std() == 0:
            return 0
        return (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    def _calculate_trade_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive trade statistics."""
        if self.trades_df is None or self.trades_df.empty:
            return self._empty_trade_stats()
        
        # Basic counts
        total_trades = len(self.trades_df)
        winning_trades = len(self.trades_df[self.trades_df['net_pnl'] > 0])
        losing_trades = len(self.trades_df[self.trades_df['net_pnl'] <= 0])
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit/Loss calculations
        gross_profit = self.trades_df[self.trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(self.trades_df[self.trades_df['net_pnl'] <= 0]['net_pnl'].sum())
        net_profit = gross_profit - gross_loss
        
        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Average win/loss
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        
        # Largest win/loss
        largest_win = self.trades_df['net_pnl'].max() if not self.trades_df.empty else 0
        largest_loss = self.trades_df['net_pnl'].min() if not self.trades_df.empty else 0
        
        # Expectancy
        expectancy = net_profit / total_trades if total_trades > 0 else 0
        
        # Average trade duration
        if 'bars_held' in self.trades_df.columns:
            avg_duration = self.trades_df['bars_held'].mean()
        else:
            avg_duration = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': net_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_trade_duration': avg_duration
        }
    
    def _calculate_streak_statistics(self) -> Dict[str, int]:
        """Calculate win/loss streak statistics."""
        if self.trades_df is None or self.trades_df.empty:
            return {
                'win_streak_current': 0,
                'win_streak_max': 0,
                'loss_streak_current': 0,
                'loss_streak_max': 0
            }
        
        # Calculate streaks
        is_win = self.trades_df['net_pnl'] > 0
        
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for win in is_win:
            if win:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return {
            'win_streak_current': current_win_streak,
            'win_streak_max': max_win_streak,
            'loss_streak_current': current_loss_streak,
            'loss_streak_max': max_loss_streak
        }
    
    def _calculate_recovery_factor(self, total_return: float, max_dd: float) -> float:
        """Calculate recovery factor."""
        if max_dd == 0:
            return np.inf
        return abs(total_return) / max_dd
    
    def _calculate_payoff_ratio(self, trade_stats: Dict[str, Any]) -> float:
        """Calculate payoff ratio (avg win / avg loss)."""
        if trade_stats['avg_loss'] == 0:
            return np.inf
        return trade_stats['avg_win'] / trade_stats['avg_loss']
    
    def _calculate_daily_statistics(self) -> Dict[str, float]:
        """Calculate daily performance statistics."""
        total_days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if total_days == 0:
            return {'profit_per_day': 0, 'trades_per_day': 0}
        
        net_profit = self.equity_curve['total_value'].iloc[-1] - self.equity_curve['total_value'].iloc[0]
        total_trades = len(self.trades_df) if self.trades_df is not None else 0
        
        return {
            'profit_per_day': net_profit / total_days,
            'trades_per_day': total_trades / total_days
        }
    
    def _calculate_statistical_metrics(self) -> Dict[str, float]:
        """Calculate statistical properties of returns."""
        returns = self.equity_curve['returns'].fillna(0)
        
        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * 100
        
        # Conditional Value at Risk
        returns_below_var = returns[returns <= np.percentile(returns, 5)]
        cvar_95 = returns_below_var.mean() * 100 if len(returns_below_var) > 0 else var_95
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _analyze_time_based_performance(self, monthly_returns: pd.Series) -> Dict[str, Any]:
        """Analyze time-based performance metrics."""
        if monthly_returns.empty:
            return {
                'best_month': ('N/A', 0),
                'worst_month': ('N/A', 0),
                'positive_months_pct': 0
            }
        
        # Best and worst months
        best_month_idx = monthly_returns.idxmax()
        worst_month_idx = monthly_returns.idxmin()
        
        best_month = (best_month_idx.strftime('%Y-%m'), monthly_returns[best_month_idx])
        worst_month = (worst_month_idx.strftime('%Y-%m'), monthly_returns[worst_month_idx])
        
        # Percentage of positive months
        positive_months_pct = (monthly_returns > 0).sum() / len(monthly_returns) * 100
        
        return {
            'best_month': best_month,
            'worst_month': worst_month,
            'positive_months_pct': positive_months_pct
        }
    
    def _calculate_rolling_metrics(self, window: int = 60) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics."""
        returns = self.equity_curve['returns'].fillna(0)
        
        # Rolling Sharpe ratio
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean - self.risk_free_rate) / rolling_std
        
        # Rolling volatility
        rolling_volatility = returns.rolling(window).std() * np.sqrt(252) * 100
        
        # Rolling win rate (if trades available)
        rolling_win_rate = None
        if self.trades_df is not None and not self.trades_df.empty:
            # This would require trade-by-trade calculation
            # Simplified for now
            rolling_win_rate = pd.Series(index=self.equity_curve.index)
        
        return {
            'rolling_sharpe': rolling_sharpe,
            'rolling_volatility': rolling_volatility,
            'rolling_win_rate': rolling_win_rate
        }
    
    def _empty_trade_stats(self) -> Dict[str, Any]:
        """Return empty trade statistics."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'net_profit': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_trade_duration': 0
        }


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Performance Analyzer Module")
    print("-" * 50)
    print("Available metrics:")
    print("- Total/Cumulative/Annualized Returns")
    print("- Volatility and Drawdown Analysis")
    print("- Sharpe, Sortino, Calmar Ratios")
    print("- Win Rate and Profit Factor")
    print("- Trade Statistics and Streaks")
    print("- Time-based Performance Analysis")
    print("- Statistical Properties (VaR, Skewness, etc.)")
    print("\nUse with backtest results for comprehensive analysis")