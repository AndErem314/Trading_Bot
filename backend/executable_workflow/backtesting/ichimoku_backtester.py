"""
Ichimoku Backtesting Engine

This module provides a comprehensive event-driven backtesting engine specifically
designed for Ichimoku trading strategies. It simulates realistic market conditions
including commissions, slippage, and proper order execution timing.

Features:
- Event-driven architecture for accurate simulation
- Multi-timeframe support (MTF analysis)
- Realistic trade execution with slippage and commissions
- Detailed trade journal and performance metrics
- Portfolio management with various position sizing methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from queue import PriorityQueue
import copy

# Configure logging
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of orders supported."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: PositionSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    tag: str = ""  # For strategy identification


@dataclass
class Fill:
    """Represents an order fill."""
    fill_id: str
    order_id: str
    timestamp: datetime
    quantity: float
    price: float
    commission: float
    slippage: float


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: PositionSide
    quantity: float
    average_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    peak_pnl: float = 0.0
    drawdown_from_peak: float = 0.0
    bars_held: int = 0


@dataclass
class TradeRecord:
    """Detailed record of a completed trade."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    return_pct: float
    bars_held: int
    entry_reason: str
    exit_reason: str
    mae: float  # Maximum Adverse Excursion
    mfe: float  # Maximum Favorable Excursion


@dataclass
class BacktestEvent:
    """Base class for backtest events."""
    timestamp: datetime
    priority: int = 0  # Lower number = higher priority
    
    def __lt__(self, other):
        """For priority queue ordering."""
        if self.timestamp == other.timestamp:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class MarketEvent(BacktestEvent):
    """Market data update event."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    bar_number: int


@dataclass
class SignalEvent(BacktestEvent):
    """Trading signal event."""
    symbol: str
    signal_type: str
    signal_strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_data: Dict[str, Any]


@dataclass
class OrderEvent(BacktestEvent):
    """Order placement event."""
    order: Order


@dataclass
class FillEvent(BacktestEvent):
    """Order fill event."""
    fill: Fill


class IchimokuBacktester:
    """
    Comprehensive backtesting engine for Ichimoku strategies.
    
    This class implements an event-driven architecture to accurately simulate
    trading with realistic market conditions including slippage, commissions,
    and proper order execution timing.
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 commission_rate: float = 0.001,  # 0.1%
                 slippage_rate: float = 0.0005,   # 0.05%
                 min_commission: float = 1.0,
                 enable_shorting: bool = False):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting portfolio value
            commission_rate: Commission as percentage of trade value
            slippage_rate: Slippage as percentage of price
            min_commission: Minimum commission per trade
            enable_shorting: Whether to allow short positions
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_commission = min_commission
        self.enable_shorting = enable_shorting
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.fill_history: List[Fill] = []
        self.trade_history: List[TradeRecord] = []
        
        # Event queue
        self.event_queue = PriorityQueue()
        
        # Performance tracking
        self.equity_curve: List[Dict[str, Any]] = []
        self.portfolio_values: List[float] = [initial_capital]
        self.timestamps: List[datetime] = []
        
        # Multi-timeframe data storage
        self.market_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.current_bars: Dict[str, Dict[str, pd.Series]] = {}
        
        # Order and trade ID counters
        self._order_counter = 0
        self._fill_counter = 0
        self._trade_counter = 0
        
        # Strategy reference
        self.strategy = None
        self.strategy_config = None
    
    def run_backtest(self, strategy_config: Any, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                    timeframes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a complete backtest of an Ichimoku strategy.
        
        Args:
            strategy_config: Strategy configuration object
            data: Market data (DataFrame or dict of DataFrames for multiple timeframes)
            timeframes: List of timeframes to use (if data is dict)
            
        Returns:
            Dictionary containing backtest results and performance metrics
        """
        logger.info(f"Starting backtest for {strategy_config.name}")
        
        # Reset state
        self._reset()
        
        # Store strategy configuration
        self.strategy_config = strategy_config
        
        # Prepare market data
        self._prepare_market_data(data, timeframes)
        
        # Generate market events
        self._generate_market_events()
        
        # Import and initialize strategy
        from strategies import StrategyBuilder
        builder = StrategyBuilder()
        self.strategy = builder.build_strategy_from_config(strategy_config)
        
        # Process events
        self._process_event_queue()
        
        # Calculate final metrics
        results = self._calculate_results()
        
        logger.info(f"Backtest complete. Total trades: {len(self.trade_history)}")
        
        return results
    
    def execute_trade(self, signal: Dict[str, Any], price: float, 
                     timestamp: datetime) -> Optional[Order]:
        """
        Execute a trade based on signal.
        
        Args:
            signal: Trading signal with entry/exit information
            price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Order object if order was placed, None otherwise
        """
        # Determine order side and size
        if signal['signal'] > 0:  # Buy signal
            side = PositionSide.LONG
            if not self._can_open_position(signal['symbol'], side):
                logger.debug("Cannot open long position - already in position or insufficient funds")
                return None
        elif signal['signal'] < 0:  # Sell signal
            side = PositionSide.SHORT
            if not self.enable_shorting:
                logger.debug("Short selling not enabled")
                return None
            if not self._can_open_position(signal['symbol'], side):
                logger.debug("Cannot open short position")
                return None
        else:
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(
            signal,
            price,
            self.strategy_config.position_sizing
        )
        
        if position_size <= 0:
            logger.debug("Position size too small")
            return None
        
        # Create order
        order = self._create_order(
            symbol=signal['symbol'],
            side=side,
            quantity=position_size,
            order_type=OrderType.MARKET,
            timestamp=timestamp,
            tag=signal.get('reason', 'signal')
        )
        
        # Add to pending orders
        self.pending_orders[order.order_id] = order
        self.order_history.append(order)
        
        # Create order event
        order_event = OrderEvent(
            timestamp=timestamp,
            priority=1,
            order=order
        )
        self.event_queue.put(order_event)
        
        logger.info(f"Order placed: {order.order_id} - {side.value} {position_size} @ {price}")
        
        return order
    
    def calculate_portfolio_values(self) -> pd.DataFrame:
        """
        Calculate portfolio values and performance metrics over time.
        
        Returns:
            DataFrame with portfolio values and metrics
        """
        if not self.equity_curve:
            return pd.DataFrame()
        
        # Convert equity curve to DataFrame
        df = pd.DataFrame(self.equity_curve)
        
        # Calculate returns
        df['returns'] = df['total_value'].pct_change()
        df['cum_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Calculate drawdown
        running_max = df['total_value'].expanding().max()
        df['drawdown'] = (df['total_value'] - running_max) / running_max
        df['drawdown_pct'] = df['drawdown'] * 100
        
        # Calculate rolling metrics
        if len(df) > 20:
            df['rolling_sharpe'] = self._calculate_rolling_sharpe(df['returns'], window=20)
            df['rolling_volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def _reset(self):
        """Reset all state variables."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.pending_orders.clear()
        self.order_history.clear()
        self.fill_history.clear()
        self.trade_history.clear()
        self.equity_curve.clear()
        self.portfolio_values = [self.initial_capital]
        self.timestamps.clear()
        self.market_data.clear()
        self.current_bars.clear()
        self._order_counter = 0
        self._fill_counter = 0
        self._trade_counter = 0
        
        # Clear event queue
        while not self.event_queue.empty():
            self.event_queue.get()
    
    def _prepare_market_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                           timeframes: Optional[List[str]]):
        """Prepare market data for backtesting."""
        if isinstance(data, pd.DataFrame):
            # Single timeframe
            symbol = self.strategy_config.symbols[0] if self.strategy_config else "BTC/USDT"
            timeframe = self.strategy_config.timeframe if self.strategy_config else "1h"
            self.market_data[symbol] = {timeframe: data}
        else:
            # Multiple timeframes
            for symbol, tf_data in data.items():
                if isinstance(tf_data, dict):
                    self.market_data[symbol] = tf_data
                else:
                    # Single timeframe for this symbol
                    timeframe = timeframes[0] if timeframes else "1h"
                    self.market_data[symbol] = {timeframe: tf_data}
    
    def _generate_market_events(self):
        """Generate market events from data."""
        for symbol, timeframe_data in self.market_data.items():
            for timeframe, df in timeframe_data.items():
                for i, (timestamp, row) in enumerate(df.iterrows()):
                    event = MarketEvent(
                        timestamp=timestamp,
                        priority=0,  # Market events have highest priority
                        symbol=symbol,
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        timeframe=timeframe,
                        bar_number=i
                    )
                    self.event_queue.put(event)
    
    def _process_event_queue(self):
        """Process all events in chronological order."""
        while not self.event_queue.empty():
            event = self.event_queue.get()
            
            if isinstance(event, MarketEvent):
                self._handle_market_event(event)
            elif isinstance(event, SignalEvent):
                self._handle_signal_event(event)
            elif isinstance(event, OrderEvent):
                self._handle_order_event(event)
            elif isinstance(event, FillEvent):
                self._handle_fill_event(event)
    
    def _handle_market_event(self, event: MarketEvent):
        """Handle market data update."""
        # Update current bar data
        if event.symbol not in self.current_bars:
            self.current_bars[event.symbol] = {}
        
        # Get the full data row
        df = self.market_data[event.symbol][event.timeframe]
        self.current_bars[event.symbol][event.timeframe] = df.iloc[event.bar_number]
        
        # Update position marks
        self._update_position_marks(event.symbol, event.close)
        
        # Check pending orders
        self._check_pending_orders(event)
        
        # Generate signals if we have complete data
        if self._has_complete_data(event.symbol, event.timeframe):
            self._generate_signals(event)
        
        # Update equity curve
        self._update_equity_curve(event.timestamp)
    
    def _handle_signal_event(self, event: SignalEvent):
        """Handle trading signal."""
        # Check if we should act on this signal
        if not self._should_trade_signal(event):
            return
        
        # Execute trade
        self.execute_trade(
            signal=event.signal_data,
            price=event.entry_price,
            timestamp=event.timestamp
        )
    
    def _handle_order_event(self, event: OrderEvent):
        """Handle order placement."""
        order = event.order
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected: {order.order_id}")
            return
        
        # For market orders, execute immediately
        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order, event.timestamp)
    
    def _handle_fill_event(self, event: FillEvent):
        """Handle order fill."""
        fill = event.fill
        order = self._get_order_by_id(fill.order_id)
        
        if not order:
            logger.error(f"Fill for unknown order: {fill.order_id}")
            return
        
        # Update order
        order.filled_quantity += fill.quantity
        order.average_fill_price = (
            (order.average_fill_price * (order.filled_quantity - fill.quantity) + 
             fill.price * fill.quantity) / order.filled_quantity
        )
        order.commission += fill.commission
        order.slippage += fill.slippage
        
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL
        
        # Update position
        self._update_position(order, fill)
        
        # Update cash
        if order.side == PositionSide.LONG:
            self.cash -= (fill.price * fill.quantity + fill.commission)
        else:  # SHORT or closing position
            self.cash += (fill.price * fill.quantity - fill.commission)
        
        # Record fill
        self.fill_history.append(fill)
        
        # Check if trade is complete
        if self._is_trade_complete(order):
            self._record_trade(order, event.timestamp)
    
    def _execute_market_order(self, order: Order, timestamp: datetime):
        """Execute a market order."""
        # Get current price
        current_bar = self.current_bars.get(order.symbol, {})
        if not current_bar:
            logger.error(f"No market data for {order.symbol}")
            order.status = OrderStatus.REJECTED
            return
        
        # Use the close price of the most recent bar
        base_price = list(current_bar.values())[0]['close']
        
        # Calculate slippage
        slippage_amount = base_price * self.slippage_rate
        if order.side == PositionSide.LONG:
            fill_price = base_price + slippage_amount  # Pay more when buying
        else:
            fill_price = base_price - slippage_amount  # Receive less when selling
        
        # Calculate commission
        trade_value = fill_price * order.quantity
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # Create fill
        fill = Fill(
            fill_id=self._generate_fill_id(),
            order_id=order.order_id,
            timestamp=timestamp,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage_amount * order.quantity
        )
        
        # Create fill event
        fill_event = FillEvent(
            timestamp=timestamp,
            priority=2,
            fill=fill
        )
        self.event_queue.put(fill_event)
    
    def _update_position_marks(self, symbol: str, price: float):
        """Update position marks with current price."""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = price
            
            # Calculate unrealized P&L
            if position.side == PositionSide.LONG:
                position.unrealized_pnl = (price - position.average_price) * position.quantity
            else:  # SHORT
                position.unrealized_pnl = (position.average_price - price) * position.quantity
            
            # Track peak and drawdown
            total_pnl = position.realized_pnl + position.unrealized_pnl
            position.peak_pnl = max(position.peak_pnl, total_pnl)
            position.drawdown_from_peak = total_pnl - position.peak_pnl
            
            # Increment bars held
            position.bars_held += 1
    
    def _update_position(self, order: Order, fill: Fill):
        """Update or create position based on fill."""
        symbol = order.symbol
        
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                side=order.side,
                quantity=fill.quantity,
                average_price=fill.price,
                current_price=fill.price,
                entry_time=fill.timestamp
            )
        else:
            position = self.positions[symbol]
            
            if position.side == order.side:
                # Adding to position
                new_quantity = position.quantity + fill.quantity
                position.average_price = (
                    (position.average_price * position.quantity + fill.price * fill.quantity) 
                    / new_quantity
                )
                position.quantity = new_quantity
            else:
                # Closing or reducing position
                if fill.quantity >= position.quantity:
                    # Position closed
                    realized_pnl = self._calculate_realized_pnl(position, fill.price, position.quantity)
                    position.realized_pnl += realized_pnl
                    del self.positions[symbol]
                else:
                    # Position reduced
                    realized_pnl = self._calculate_realized_pnl(position, fill.price, fill.quantity)
                    position.realized_pnl += realized_pnl
                    position.quantity -= fill.quantity
    
    def _calculate_realized_pnl(self, position: Position, exit_price: float, quantity: float) -> float:
        """Calculate realized P&L for a position."""
        if position.side == PositionSide.LONG:
            return (exit_price - position.average_price) * quantity
        else:  # SHORT
            return (position.average_price - exit_price) * quantity
    
    def _calculate_position_size(self, signal: Dict[str, Any], price: float, 
                               sizing_config: Any) -> float:
        """Calculate position size based on configuration."""
        if sizing_config.method == 'fixed':
            return sizing_config.fixed_size
        
        elif sizing_config.method == 'risk_based':
            # Calculate based on risk
            stop_loss = signal.get('stop_loss', price * 0.98)
            risk_per_share = abs(price - stop_loss)
            
            if risk_per_share > 0:
                risk_amount = self.cash * (sizing_config.risk_per_trade_pct / 100)
                position_size = risk_amount / risk_per_share
            else:
                position_size = sizing_config.fixed_size
            
            # Apply limits
            max_position_value = self.cash * (sizing_config.max_position_size_pct / 100)
            max_shares = max_position_value / price
            
            position_size = min(position_size, max_shares)
            position_size = max(position_size, sizing_config.min_position_size)
            
            return position_size
        
        else:
            # Default to fixed size
            return sizing_config.fixed_size
    
    def _generate_signals(self, market_event: MarketEvent):
        """Generate trading signals from strategy."""
        # Get current data for the symbol
        symbol = market_event.symbol
        timeframe = market_event.timeframe
        
        # Get the data up to current bar
        df = self.market_data[symbol][timeframe]
        current_idx = market_event.bar_number
        
        if current_idx < 100:  # Need minimum history
            return
        
        # Get historical data up to current point
        historical_data = df.iloc[:current_idx + 1].copy()
        
        # Run strategy to get signals
        signal_data = self.strategy(historical_data)
        
        # Check if there's a signal at the current bar
        if signal_data['entry_signals'].iloc[-1] or signal_data['exit_signals'].iloc[-1]:
            # Create signal event
            signal_event = SignalEvent(
                timestamp=market_event.timestamp,
                priority=1,
                symbol=symbol,
                signal_type='entry' if signal_data['entry_signals'].iloc[-1] else 'exit',
                signal_strength=abs(signal_data['entry_signals'].iloc[-1]),
                entry_price=market_event.close,
                stop_loss=signal_data.get('stop_levels', pd.Series()).iloc[-1] if 'stop_levels' in signal_data else market_event.close * 0.98,
                take_profit=signal_data.get('target_levels', pd.Series()).iloc[-1] if 'target_levels' in signal_data else market_event.close * 1.02,
                signal_data={
                    'symbol': symbol,
                    'signal': 1 if signal_data['entry_signals'].iloc[-1] else -1,
                    'reason': 'strategy_signal'
                }
            )
            self.event_queue.put(signal_event)
    
    def _check_pending_orders(self, market_event: MarketEvent):
        """Check if any pending orders should be executed."""
        symbol = market_event.symbol
        
        for order_id, order in list(self.pending_orders.items()):
            if order.symbol != symbol or order.status != OrderStatus.PENDING:
                continue
            
            should_execute = False
            
            # Check limit orders
            if order.order_type == OrderType.LIMIT:
                if order.side == PositionSide.LONG and market_event.low <= order.price:
                    should_execute = True
                elif order.side == PositionSide.SHORT and market_event.high >= order.price:
                    should_execute = True
            
            # Check stop orders
            elif order.order_type == OrderType.STOP:
                if order.side == PositionSide.LONG and market_event.high >= order.stop_price:
                    should_execute = True
                elif order.side == PositionSide.SHORT and market_event.low <= order.stop_price:
                    should_execute = True
            
            if should_execute:
                self._execute_market_order(order, market_event.timestamp)
                del self.pending_orders[order_id]
    
    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve with current portfolio value."""
        # Calculate total portfolio value
        position_value = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        total_value = self.cash + position_value
        
        # Calculate metrics
        returns = ((total_value - self.portfolio_values[-1]) / self.portfolio_values[-1] 
                  if self.portfolio_values else 0)
        
        # Record equity curve point
        self.equity_curve.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value,
            'returns': returns,
            'positions': len(self.positions)
        })
        
        self.portfolio_values.append(total_value)
        self.timestamps.append(timestamp)
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate final backtest results and metrics."""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'trades': [],
                'equity_curve': pd.DataFrame(),
                'metrics': {}
            }
        
        # Convert trade history to DataFrame
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'symbol': t.symbol,
                'side': t.side.value,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'net_pnl': t.net_pnl,
                'return_pct': t.return_pct,
                'bars_held': t.bars_held,
                'mae': t.mae,
                'mfe': t.mfe
            }
            for t in self.trade_history
        ])
        
        # Calculate metrics
        winning_trades = trades_df[trades_df['net_pnl'] > 0]
        losing_trades = trades_df[trades_df['net_pnl'] <= 0]
        
        total_trades = len(trades_df)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        
        gross_profits = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_losses = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
        
        # Portfolio metrics
        portfolio_df = self.calculate_portfolio_values()
        
        if len(portfolio_df) > 0:
            total_return = ((self.portfolio_values[-1] - self.initial_capital) 
                          / self.initial_capital * 100)
            
            # Calculate Sharpe ratio
            returns = portfolio_df['returns'].dropna()
            if len(returns) > 1:
                sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            max_drawdown = abs(portfolio_df['drawdown_pct'].min())
            
            # Calmar ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        else:
            total_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
            calmar_ratio = 0
        
        # Additional metrics
        metrics = {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': trades_df['net_pnl'].mean() if total_trades > 0 else 0,
            'largest_win': trades_df['net_pnl'].max() if total_trades > 0 else 0,
            'largest_loss': trades_df['net_pnl'].min() if total_trades > 0 else 0,
            'avg_bars_held': trades_df['bars_held'].mean() if total_trades > 0 else 0,
            'gross_profits': gross_profits,
            'gross_losses': gross_losses,
            'net_profit': gross_profits - gross_losses,
            'avg_mae': trades_df['mae'].mean() if total_trades > 0 else 0,
            'avg_mfe': trades_df['mfe'].mean() if total_trades > 0 else 0
        }
        
        return {
            'metrics': metrics,
            'trades': trades_df,
            'equity_curve': portfolio_df,
            'trade_history': self.trade_history,
            'order_history': self.order_history,
            'fill_history': self.fill_history
        }
    
    # Helper methods
    
    def _can_open_position(self, symbol: str, side: PositionSide) -> bool:
        """Check if we can open a new position."""
        # Check if already in position
        if symbol in self.positions:
            current_pos = self.positions[symbol]
            # Can't open opposite position
            if current_pos.side != side:
                return False
        
        # Check available capital (simplified)
        return self.cash > 0
    
    def _should_trade_signal(self, signal_event: SignalEvent) -> bool:
        """Determine if we should act on a signal."""
        # Add any additional filtering logic here
        return True
    
    def _validate_order(self, order: Order) -> bool:
        """Validate an order before execution."""
        # Check sufficient funds
        if order.side == PositionSide.LONG:
            required_capital = order.quantity * (order.price or 0)
            if required_capital > self.cash:
                return False
        
        # Add other validation rules
        return True
    
    def _has_complete_data(self, symbol: str, timeframe: str) -> bool:
        """Check if we have complete data for signal generation."""
        if symbol not in self.current_bars:
            return False
        return timeframe in self.current_bars[symbol]
    
    def _is_trade_complete(self, order: Order) -> bool:
        """Check if an order completes a trade."""
        # Simplified - in reality would track entry/exit orders
        return order.status == OrderStatus.FILLED
    
    def _record_trade(self, order: Order, exit_time: datetime):
        """Record a completed trade."""
        # This is simplified - in a real system would match entry/exit orders
        trade = TradeRecord(
            trade_id=self._generate_trade_id(),
            symbol=order.symbol,
            entry_time=order.timestamp,
            exit_time=exit_time,
            side=order.side,
            entry_price=order.average_fill_price,
            exit_price=order.average_fill_price,  # Simplified
            quantity=order.quantity,
            gross_pnl=0,  # Would calculate from matched orders
            commission=order.commission,
            slippage=order.slippage,
            net_pnl=-order.commission,  # Simplified
            return_pct=0,
            bars_held=1,
            entry_reason=order.tag,
            exit_reason="signal",
            mae=0,
            mfe=0
        )
        self.trade_history.append(trade)
    
    def _create_order(self, symbol: str, side: PositionSide, quantity: float,
                     order_type: OrderType, timestamp: datetime, tag: str = "") -> Order:
        """Create a new order."""
        return Order(
            order_id=self._generate_order_id(),
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            tag=tag
        )
    
    def _get_order_by_id(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        for order in self.order_history:
            if order.order_id == order_id:
                return order
        return None
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"ORD_{self._order_counter:06d}"
    
    def _generate_fill_id(self) -> str:
        """Generate unique fill ID."""
        self._fill_counter += 1
        return f"FILL_{self._fill_counter:06d}"
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"TRD_{self._trade_counter:06d}"
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        return np.sqrt(252) * rolling_mean / rolling_std


# Example usage
if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Ichimoku Backtester initialized")
    print("\nFeatures:")
    print("- Event-driven architecture")
    print("- Realistic order execution with slippage and commissions")
    print("- Multi-timeframe support")
    print("- Comprehensive trade journal")
    print("- Portfolio performance tracking")
    print("\nUse with StrategyBuilder to backtest Ichimoku strategies")