"""
Core Backtesting Engine for Trading Strategies

This module provides the main backtesting functionality to simulate trading
based on strategy signals and track performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of orders that can be placed"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionSide(Enum):
    """Side of the position"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Represents a trading order"""
    timestamp: datetime
    symbol: str
    side: PositionSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "pending"


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_timestamp: datetime
    current_price: float
    unrealized_pnl: float
    position_id: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    exit_price: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    pnl: float
    pnl_percentage: float
    fees: float
    trade_id: str


class BacktestEngine:
    """
    Main backtesting engine that simulates trading based on strategy signals
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1% commission
        slippage: float = 0.0005,   # 0.05% slippage
        use_leverage: bool = False,
        max_leverage: float = 1.0
    ):
        """
        Initialize the backtesting engine
        
        Args:
            initial_capital: Starting capital for backtest
            commission: Trading commission as a fraction
            slippage: Price slippage as a fraction
            use_leverage: Whether to allow leverage
            max_leverage: Maximum leverage allowed
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.use_leverage = use_leverage
        self.max_leverage = max_leverage
        
        # Portfolio state
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # Performance tracking
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Trade counter
        self._trade_counter = 0
        self._position_counter = 0
        
    def run_backtest(
        self,
        strategy_signals: pd.DataFrame,
        price_data: pd.DataFrame,
        symbol: str = "BTC/USDT"
    ) -> Dict[str, Any]:
        """
        Run the backtest with given strategy signals
        
        Args:
            strategy_signals: DataFrame with columns ['timestamp', 'signal', 'strength', 'price']
                            signal: 1 for buy, -1 for sell, 0 for hold
            price_data: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            Dictionary containing backtesting results
        """
        logger.info(f"Starting backtest for {symbol}")
        
        # Reset state
        self._reset()
        
        # Ensure data is sorted by timestamp
        strategy_signals = strategy_signals.sort_values('timestamp')
        price_data = price_data.sort_values('timestamp')
        
        # Iterate through each signal
        for idx, signal_row in strategy_signals.iterrows():
            timestamp = signal_row['timestamp']
            signal = signal_row['signal']
            strength = signal_row.get('strength', 1.0)
            current_price = signal_row['price']
            
            # Update current positions with latest price
            self._update_positions(current_price, timestamp)
            
            # Process signal
            if signal == 1:  # Buy signal
                self._process_buy_signal(symbol, current_price, timestamp, strength)
            elif signal == -1:  # Sell signal
                self._process_sell_signal(symbol, current_price, timestamp, strength)
            
            # Record equity
            equity = self._calculate_total_equity(current_price)
            self._update_equity_curve(timestamp, equity, current_price)
        
        # Close all remaining positions at end
        if self.positions:
            last_price = price_data.iloc[-1]['close']
            last_timestamp = price_data.iloc[-1].name
            self._close_all_positions(last_price, last_timestamp)
        
        # Calculate final metrics
        results = self._calculate_results()
        logger.info(f"Backtest completed. Total return: {results['total_return']:.2%}")
        
        return results
    
    def _reset(self):
        """Reset engine state for new backtest"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.peak_equity = self.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self._trade_counter = 0
        self._position_counter = 0
    
    def _process_buy_signal(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        strength: float
    ):
        """Process a buy signal"""
        # Check if we already have a position
        if symbol in self.positions:
            return
        
        # Calculate position size based on signal strength
        position_size = self._calculate_position_size(price, strength)
        
        if position_size <= 0:
            return
        
        # Apply slippage
        entry_price = price * (1 + self.slippage)
        
        # Calculate commission
        commission = position_size * entry_price * self.commission
        
        # Check if we have enough capital
        total_cost = position_size * entry_price + commission
        if total_cost > self.current_capital:
            # Adjust position size if needed
            position_size = (self.current_capital - commission) / entry_price
            total_cost = position_size * entry_price + commission
        
        # Create position
        self._position_counter += 1
        position = Position(
            symbol=symbol,
            side=PositionSide.LONG,
            quantity=position_size,
            entry_price=entry_price,
            entry_timestamp=timestamp,
            current_price=entry_price,
            unrealized_pnl=0.0,
            position_id=f"POS_{self._position_counter}"
        )
        
        # Update capital and positions
        self.current_capital -= total_cost
        self.positions[symbol] = position
        
        logger.debug(f"Opened long position: {position_size:.4f} {symbol} @ {entry_price:.2f}")
    
    def _process_sell_signal(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        strength: float
    ):
        """Process a sell signal"""
        # Check if we have a position to close
        if symbol not in self.positions:
            # Could open short position here if enabled
            return
        
        position = self.positions[symbol]
        
        # Apply slippage
        exit_price = price * (1 - self.slippage)
        
        # Close position
        self._close_position(position, exit_price, timestamp)
    
    def _close_position(self, position: Position, exit_price: float, timestamp: datetime):
        """Close a position and record the trade"""
        # Calculate P&L
        if position.side == PositionSide.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        # Calculate commission
        commission = position.quantity * exit_price * self.commission
        net_pnl = pnl - commission
        
        # Calculate percentage return
        pnl_percentage = (net_pnl / (position.entry_price * position.quantity)) * 100
        
        # Create trade record
        self._trade_counter += 1
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=timestamp,
            pnl=net_pnl,
            pnl_percentage=pnl_percentage,
            fees=commission,
            trade_id=f"TRADE_{self._trade_counter}"
        )
        
        # Update records
        self.trades.append(trade)
        self.current_capital += position.quantity * exit_price - commission
        del self.positions[position.symbol]
        
        logger.debug(f"Closed position: {position.symbol} P&L: {net_pnl:.2f} ({pnl_percentage:.2f}%)")
    
    def _close_all_positions(self, price: float, timestamp: datetime):
        """Close all open positions"""
        positions_to_close = list(self.positions.values())
        for position in positions_to_close:
            self._close_position(position, price, timestamp)
    
    def _update_positions(self, current_price: float, timestamp: datetime):
        """Update unrealized P&L for open positions"""
        for position in self.positions.values():
            position.current_price = current_price
            if position.side == PositionSide.LONG:
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
    
    def _calculate_position_size(self, price: float, strength: float) -> float:
        """Calculate position size based on available capital and signal strength"""
        # Use a fraction of capital based on signal strength
        max_position_value = self.current_capital * 0.95  # Keep 5% as buffer
        position_value = max_position_value * min(strength, 1.0)
        position_size = position_value / price
        
        return position_size
    
    def _calculate_total_equity(self, current_price: float) -> float:
        """Calculate total equity including open positions"""
        equity = self.current_capital
        
        for position in self.positions.values():
            position_value = position.quantity * current_price
            equity += position_value
        
        return equity
    
    def _update_equity_curve(self, timestamp: datetime, equity: float, price: float):
        """Update equity curve and drawdown calculations"""
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'price': price,
            'drawdown': 0.0
        })
        
        # Update peak equity and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Update drawdown in equity curve
        self.equity_curve[-1]['drawdown'] = self.current_drawdown
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate final backtesting results and metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0,
                'trades': [],
                'equity_curve': pd.DataFrame(self.equity_curve)
            }
        
        # Convert trades to DataFrame for easier analysis
        trades_df = pd.DataFrame([
            {
                'symbol': t.symbol,
                'side': t.side.value,
                'quantity': t.quantity,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'entry_timestamp': t.entry_timestamp,
                'exit_timestamp': t.exit_timestamp,
                'pnl': t.pnl,
                'pnl_percentage': t.pnl_percentage,
                'fees': t.fees,
                'trade_id': t.trade_id
            }
            for t in self.trades
        ])
        
        # Calculate metrics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        total_return = ((self.equity_curve[-1]['equity'] - self.initial_capital) / 
                       self.initial_capital) * 100
        
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate Sharpe ratio
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 1:
            returns = equity_df['equity'].pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        results = {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'best_trade': trades_df['pnl'].max() if len(trades_df) > 0 else 0,
            'worst_trade': trades_df['pnl'].min() if len(trades_df) > 0 else 0,
            'total_fees': trades_df['fees'].sum(),
            'final_equity': self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital,
            'trades': trades_df,
            'equity_curve': equity_df
        }
        
        return results
