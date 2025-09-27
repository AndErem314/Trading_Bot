"""
Dynamic Strategy Builder

This module provides a dynamic strategy builder that generates trading strategies
from configuration files. It parses strategy configurations and creates executable
trading logic with proper entry/exit conditions, risk management, and position handling.

Features:
- Dynamic strategy generation from configurations
- Support for AND/OR logic in signal combinations
- Flexible exit conditions (signal-based, fixed targets, crosses)
- Position management with pyramiding support
- Risk parameter integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

# Import configuration and signal detection components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import StrategyConfig, SignalCondition
from data_fetching import IchimokuSignalDetector, SignalType

# Configure logging
logger = logging.getLogger(__name__)


class ExitType(Enum):
    """Types of exit conditions."""
    FIXED_TARGET = "fixed_target"
    SIGNAL_BASED = "signal_based"
    TENKAN_KIJUN_CROSS = "tenkan_kijun_cross"
    CLOUD_BASED_STOP = "cloud_based_stop"
    TIME_BASED = "time_based"
    TRAILING_STOP = "trailing_stop"


@dataclass
class Position:
    """Represents an open trading position."""
    entry_price: float
    entry_time: datetime
    position_size: float
    side: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    entry_signal: str
    pyramid_level: int = 0


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    position_size: float
    side: str
    pnl: float
    pnl_pct: float
    exit_reason: str
    holding_periods: int


class StrategyBuilder:
    """
    Dynamic strategy builder that creates trading strategies from configurations.
    
    This class takes strategy configurations and generates executable trading logic
    with proper signal handling, risk management, and position tracking.
    """
    
    def __init__(self, max_pyramiding: int = 1):
        """
        Initialize the strategy builder.
        
        Args:
            max_pyramiding: Maximum number of pyramid positions allowed
        """
        self.max_pyramiding = max_pyramiding
        self.signal_detector = IchimokuSignalDetector()
        self.current_positions: List[Position] = []
        self.completed_trades: List[Trade] = []
        self.strategy_config: Optional[StrategyConfig] = None
        
    def build_strategy_from_config(self, config: StrategyConfig) -> Callable:
        """
        Build a complete trading strategy from configuration.
        
        Args:
            config: Strategy configuration object
            
        Returns:
            Callable strategy function that can be executed on data
        """
        self.strategy_config = config
        
        def strategy_function(data: pd.DataFrame) -> Dict[str, Any]:
            """
            Execute the configured strategy on the provided data.
            
            Args:
                data: DataFrame with OHLCV and Ichimoku indicators
                
            Returns:
                Dictionary with signals, positions, and metrics
            """
            # Detect all signals
            signal_data = self.signal_detector.detect_all_signals(data)
            
            # Generate entry and exit signals
            entry_signals = self.generate_entry_conditions(
                signal_data,
                config.signal_conditions.buy_conditions,
                config.signal_conditions.buy_logic
            )
            
            exit_signals = self.generate_exit_conditions(
                signal_data,
                config.signal_conditions.sell_conditions,
                config.signal_conditions.sell_logic
            )
            
            # Apply risk parameters
            stop_levels, target_levels = self.calculate_risk_levels(
                signal_data,
                config.risk_management
            )
            
            # Execute trading logic
            results = self.execute_strategy(
                signal_data,
                entry_signals,
                exit_signals,
                stop_levels,
                target_levels,
                config
            )
            
            return results
        
        # Attach configuration to the function for reference
        strategy_function.config = config
        strategy_function.name = config.name
        
        return strategy_function
    
    def generate_entry_conditions(self, data: pd.DataFrame, 
                                buy_signals: List[str], 
                                logic: str = "AND") -> pd.Series:
        """
        Generate entry signals based on configured conditions.
        
        Args:
            data: DataFrame with signal columns
            buy_signals: List of signal conditions for entry
            logic: "AND" or "OR" logic for combining signals
            
        Returns:
            Boolean Series indicating entry points
        """
        if not buy_signals:
            return pd.Series(False, index=data.index)
        
        # Convert signal names to column names
        signal_columns = []
        for signal in buy_signals:
            # Convert from PascalCase to snake_case
            col_name = self._signal_to_column(signal)
            if col_name in data.columns:
                signal_columns.append(col_name)
            else:
                logger.warning(f"Signal column '{col_name}' not found for condition '{signal}'")
        
        if not signal_columns:
            return pd.Series(False, index=data.index)
        
        # Combine signals based on logic
        if logic == "AND":
            entry_signal = pd.Series(True, index=data.index)
            for col in signal_columns:
                entry_signal = entry_signal & data[col]
        else:  # OR logic
            entry_signal = pd.Series(False, index=data.index)
            for col in signal_columns:
                entry_signal = entry_signal | data[col]
        
        # Add additional filters
        entry_signal = self._apply_entry_filters(data, entry_signal)
        
        return entry_signal
    
    def generate_exit_conditions(self, data: pd.DataFrame,
                               sell_signals: List[str],
                               logic: str = "AND") -> pd.Series:
        """
        Generate exit signals based on configured conditions.
        
        Args:
            data: DataFrame with signal columns
            sell_signals: List of signal conditions for exit
            logic: "AND" or "OR" logic for combining signals
            
        Returns:
            Boolean Series indicating exit points
        """
        if not sell_signals:
            return pd.Series(False, index=data.index)
        
        # Convert signal names to column names
        signal_columns = []
        for signal in sell_signals:
            col_name = self._signal_to_column(signal)
            if col_name in data.columns:
                signal_columns.append(col_name)
        
        if not signal_columns:
            return pd.Series(False, index=data.index)
        
        # Combine signals based on logic
        if logic == "AND":
            exit_signal = pd.Series(True, index=data.index)
            for col in signal_columns:
                exit_signal = exit_signal & data[col]
        else:  # OR logic
            exit_signal = pd.Series(False, index=data.index)
            for col in signal_columns:
                exit_signal = exit_signal | data[col]
        
        # Add special exit conditions
        exit_signal = self._add_special_exits(data, exit_signal)
        
        return exit_signal
    
    def set_risk_parameters(self, stop_loss_pct: float, take_profit_pct: float,
                          use_cloud_stop: bool = False) -> Dict[str, Any]:
        """
        Set risk management parameters for the strategy.
        
        Args:
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            use_cloud_stop: Whether to use cloud-based stops
            
        Returns:
            Dictionary with risk parameters
        """
        return {
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'use_cloud_stop': use_cloud_stop,
            'trailing_stop': self.strategy_config.risk_management.trailing_stop if self.strategy_config else False,
            'trailing_stop_pct': self.strategy_config.risk_management.trailing_stop_pct if self.strategy_config else 0
        }
    
    def calculate_risk_levels(self, data: pd.DataFrame, 
                            risk_management: Any) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            data: DataFrame with price and indicator data
            risk_management: Risk management configuration
            
        Returns:
            Tuple of (stop_levels, target_levels) Series
        """
        # Calculate basic percentage-based levels
        stop_levels = data['close'] * (1 - risk_management.stop_loss_pct / 100)
        target_levels = data['close'] * (1 + risk_management.take_profit_pct / 100)
        
        # Optionally adjust stops based on cloud
        if hasattr(data, 'cloud_bottom') and hasattr(data, 'cloud_top'):
            # For long positions, use cloud bottom as dynamic stop
            cloud_stop_long = data['cloud_bottom'] * 0.98  # 2% below cloud
            stop_levels = pd.DataFrame({
                'fixed': stop_levels,
                'cloud': cloud_stop_long
            }).max(axis=1)
        
        return stop_levels, target_levels
    
    def execute_strategy(self, data: pd.DataFrame, entry_signals: pd.Series,
                        exit_signals: pd.Series, stop_levels: pd.Series,
                        target_levels: pd.Series, config: StrategyConfig) -> Dict[str, Any]:
        """
        Execute the complete strategy logic with position management.
        
        Args:
            data: Complete DataFrame with prices and signals
            entry_signals: Boolean Series for entries
            exit_signals: Boolean Series for exits
            stop_levels: Series with stop loss levels
            target_levels: Series with take profit levels
            config: Strategy configuration
            
        Returns:
            Dictionary with execution results
        """
        # Initialize results
        positions = []
        trades = []
        equity_curve = [10000]  # Starting capital
        signals_log = []
        
        # Track position state
        in_position = False
        current_position = None
        
        for i in range(1, len(data)):
            current_idx = data.index[i]
            current_price = data['close'].iloc[i]
            
            # Check for exit conditions first (if in position)
            if in_position and current_position:
                exit_triggered = False
                exit_reason = ""
                
                # Check stop loss
                if current_price <= current_position.stop_loss:
                    exit_triggered = True
                    exit_reason = "Stop Loss"
                
                # Check take profit
                elif current_price >= current_position.take_profit:
                    exit_triggered = True
                    exit_reason = "Take Profit"
                
                # Check signal-based exit
                elif exit_signals.iloc[i]:
                    exit_triggered = True
                    exit_reason = "Exit Signal"
                
                # Check Tenkan/Kijun cross exit
                elif self._check_tk_cross_exit(data, i, current_position.side):
                    exit_triggered = True
                    exit_reason = "TK Cross Exit"
                
                # Execute exit
                if exit_triggered:
                    trade = self._close_position(
                        current_position, current_price, 
                        current_idx, exit_reason, i - entry_bar_idx
                    )
                    trades.append(trade)
                    
                    # Update equity
                    equity_curve.append(equity_curve[-1] + trade.pnl)
                    
                    in_position = False
                    current_position = None
                    
                    signals_log.append({
                        'time': current_idx,
                        'action': 'exit',
                        'price': current_price,
                        'reason': exit_reason
                    })
                
                # Update trailing stop if enabled
                elif config.risk_management.trailing_stop:
                    current_position = self._update_trailing_stop(
                        current_position, current_price, 
                        config.risk_management.trailing_stop_pct
                    )
            
            # Check for entry conditions (if not in position)
            elif not in_position and entry_signals.iloc[i]:
                # Check pyramiding rules
                if len(self.current_positions) < self.max_pyramiding:
                    # Calculate position size
                    position_size = self._calculate_position_size(
                        equity_curve[-1], 
                        config.position_sizing,
                        current_price,
                        stop_levels.iloc[i]
                    )
                    
                    # Create new position
                    current_position = Position(
                        entry_price=current_price,
                        entry_time=current_idx,
                        position_size=position_size,
                        side='long',  # Assuming long-only for now
                        stop_loss=stop_levels.iloc[i],
                        take_profit=target_levels.iloc[i],
                        entry_signal=self._get_active_signals(data, i),
                        pyramid_level=len(self.current_positions)
                    )
                    
                    positions.append(current_position)
                    in_position = True
                    entry_bar_idx = i
                    
                    signals_log.append({
                        'time': current_idx,
                        'action': 'entry',
                        'price': current_price,
                        'size': position_size,
                        'stop_loss': current_position.stop_loss,
                        'take_profit': current_position.take_profit
                    })
            
            # Update equity curve even if no position
            if not in_position:
                equity_curve.append(equity_curve[-1])
        
        # Close any remaining positions
        if in_position and current_position:
            final_price = data['close'].iloc[-1]
            trade = self._close_position(
                current_position, final_price,
                data.index[-1], "End of Data", len(data) - entry_bar_idx
            )
            trades.append(trade)
            equity_curve.append(equity_curve[-1] + trade.pnl)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(trades, equity_curve)
        
        return {
            'trades': trades,
            'positions': positions,
            'equity_curve': equity_curve,
            'signals_log': signals_log,
            'metrics': metrics,
            'entry_signals': entry_signals,
            'exit_signals': exit_signals
        }
    
    def _signal_to_column(self, signal_name: str) -> str:
        """Convert PascalCase signal name to snake_case column name."""
        # Handle known conversions
        conversions = {
            'PriceAboveCloud': 'price_above_cloud',
            'PriceBelowCloud': 'price_below_cloud',
            'TenkanAboveKijun': 'tenkan_above_kijun',
            'TenkanBelowKijun': 'tenkan_below_kijun',
            'ChikouAbovePrice': 'chikou_above_price',
            'ChikouBelowPrice': 'chikou_below_price',
            'ChikouAboveCloud': 'chikou_above_cloud',
            'ChikouBelowCloud': 'chikou_below_cloud',
            'ChikouCrossAboveSenkouB': 'chikou_cross_above_senkou_b',
            'ChikouCrossBelowSenkouB': 'chikou_cross_below_senkou_b'
        }
        
        return conversions.get(signal_name, signal_name.lower())
    
    def _apply_entry_filters(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Apply additional filters to entry signals."""
        # Example: No entry during low volume
        if 'volume' in data.columns:
            avg_volume = data['volume'].rolling(window=20).mean()
            volume_filter = data['volume'] > avg_volume * 0.5
            signals = signals & volume_filter.fillna(True)
        
        return signals
    
    def _add_special_exits(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Add special exit conditions beyond configured signals."""
        # Can be extended with additional exit logic
        return signals
    
    def _check_tk_cross_exit(self, data: pd.DataFrame, idx: int, position_side: str) -> bool:
        """Check for Tenkan/Kijun cross exit."""
        if 'tenkan_sen' not in data.columns or 'kijun_sen' not in data.columns:
            return False
        
        if idx < 1:
            return False
        
        current_tenkan = data['tenkan_sen'].iloc[idx]
        current_kijun = data['kijun_sen'].iloc[idx]
        prev_tenkan = data['tenkan_sen'].iloc[idx-1]
        prev_kijun = data['kijun_sen'].iloc[idx-1]
        
        # For long position, exit on bearish cross
        if position_side == 'long':
            return prev_tenkan >= prev_kijun and current_tenkan < current_kijun
        
        # For short position, exit on bullish cross
        return prev_tenkan <= prev_kijun and current_tenkan > current_kijun
    
    def _update_trailing_stop(self, position: Position, current_price: float, 
                            trailing_pct: float) -> Position:
        """Update trailing stop for a position."""
        if position.side == 'long':
            # For long positions, trail stop upward
            new_stop = current_price * (1 - trailing_pct / 100)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
        else:
            # For short positions, trail stop downward
            new_stop = current_price * (1 + trailing_pct / 100)
            if new_stop < position.stop_loss:
                position.stop_loss = new_stop
        
        return position
    
    def _calculate_position_size(self, capital: float, position_sizing: Any,
                               entry_price: float, stop_price: float) -> float:
        """Calculate position size based on configuration."""
        if position_sizing.method == 'fixed':
            return position_sizing.fixed_size
        
        elif position_sizing.method == 'risk_based':
            # Calculate based on risk per trade
            risk_amount = capital * (position_sizing.risk_per_trade_pct / 100)
            price_risk = abs(entry_price - stop_price)
            
            if price_risk > 0:
                position_size = risk_amount / price_risk
            else:
                position_size = position_sizing.fixed_size
            
            # Apply limits
            position_size = max(position_sizing.min_position_size, 
                              min(position_size, position_sizing.max_position_size))
            
            return position_size
        
        else:  # volatility_based or other methods
            # Simplified volatility-based sizing
            return position_sizing.fixed_size
    
    def _close_position(self, position: Position, exit_price: float,
                       exit_time: datetime, exit_reason: str,
                       holding_periods: int) -> Trade:
        """Close a position and create a trade record."""
        # Calculate P&L
        if position.side == 'long':
            pnl = (exit_price - position.entry_price) * position.position_size
            pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl = (position.entry_price - exit_price) * position.position_size
            pnl_pct = ((position.entry_price - exit_price) / position.entry_price) * 100
        
        return Trade(
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            position_size=position.position_size,
            side=position.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            holding_periods=holding_periods
        )
    
    def _get_active_signals(self, data: pd.DataFrame, idx: int) -> str:
        """Get a description of active signals at a given index."""
        active_signals = []
        
        signal_columns = [
            'price_above_cloud', 'price_below_cloud',
            'tenkan_above_kijun', 'tenkan_below_kijun',
            'chikou_above_price', 'chikou_below_price'
        ]
        
        for col in signal_columns:
            if col in data.columns and data[col].iloc[idx]:
                active_signals.append(col)
        
        return ', '.join(active_signals)
    
    def _calculate_performance_metrics(self, trades: List[Trade], 
                                     equity_curve: List[float]) -> Dict[str, float]:
        """Calculate strategy performance metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
        
        # Basic metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns
        returns = pd.Series(equity_curve).pct_change().dropna()
        
        # Sharpe ratio (assuming daily returns)
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        # Maximum drawdown
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Total return
        total_return = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'avg_holding_periods': np.mean([t.holding_periods for t in trades])
        }


# Example usage
if __name__ == "__main__":
    import logging
    from datetime import datetime
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example: Create a strategy builder
    builder = StrategyBuilder(max_pyramiding=2)
    
    # Example configuration
    from config import (
        StrategyConfig, SignalConditions, IchimokuParameters,
        RiskManagement, PositionSizing
    )
    
    example_config = StrategyConfig(
        name="Example Dynamic Strategy",
        description="Dynamically built strategy from configuration",
        enabled=True,
        signal_conditions=SignalConditions(
            buy_conditions=["PriceAboveCloud", "TenkanAboveKijun", "ChikouAbovePrice"],
            sell_conditions=["PriceBelowCloud", "TenkanBelowKijun"],
            buy_logic="AND",
            sell_logic="OR"
        ),
        ichimoku_parameters=IchimokuParameters(),
        risk_management=RiskManagement(
            stop_loss_pct=2.0,
            take_profit_pct=6.0,
            trailing_stop=True,
            trailing_stop_pct=1.5
        ),
        position_sizing=PositionSizing(
            method="risk_based",
            risk_per_trade_pct=2.0
        ),
        timeframe="1h",
        symbols=["BTC/USDT"]
    )
    
    # Build strategy
    strategy_func = builder.build_strategy_from_config(example_config)
    
    print(f"Built strategy: {strategy_func.name}")
    print(f"Description: {strategy_func.config.description}")
    print(f"Buy conditions: {', '.join(strategy_func.config.signal_conditions.buy_conditions)}")
    print(f"Risk per trade: {strategy_func.config.risk_management.risk_per_trade_pct}%")
    
    # The strategy_func can now be used to process market data
    # Example:
    # results = strategy_func(market_data_df)
    # print(f"Total trades: {results['metrics']['total_trades']}")
    # print(f"Win rate: {results['metrics']['win_rate']:.1f}%")