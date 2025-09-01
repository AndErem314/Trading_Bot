"""
Backtesting Framework for MetaStrategyOrchestrator

This module provides a backtesting framework that simulates running
the orchestrator over historical data in a walk-forward manner.

"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    stop_loss: float
    take_profit: Optional[float]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    market_bias: str
    strategy_weights: Dict[str, float]
    signal_strength: float
    status: str  # 'open', 'closed'
    exit_reason: Optional[str]  # 'stop_loss', 'take_profit', 'signal', 'time'


class Backtester:
    """
    Backtester for MetaStrategyOrchestrator.
    
    Simulates trading over historical data using a walk-forward approach.
    """
    
    def __init__(self, orchestrator_class, db_connection_string: str):
        """
        Initialize the backtester.
        
        Args:
            orchestrator_class: The MetaStrategyOrchestrator class
            db_connection_string: Database connection string
        """
        self.orchestrator_class = orchestrator_class
        self.db_connection_string = db_connection_string
        
        # Portfolio tracking
        self.initial_capital = 10000.0
        self.cash = self.initial_capital
        self.holdings: Dict[str, float] = {}
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        
        # Performance tracking
        self.portfolio_values = []
        self.equity_curve = pd.DataFrame()
        
        # Risk parameters
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.max_open_trades = 5
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.05  # 5% take profit
        
        # Commission and slippage
        self.commission_rate = 0.001  # 0.1%
        self.slippage_rate = 0.0005  # 0.05%
        
    def run_backtest(
        self,
        full_data: Dict[str, pd.DataFrame],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        lookback_days: int = 100
    ) -> Dict[str, Any]:
        """
        Run backtest over historical data.
        
        Args:
            full_data: Dictionary of DataFrames keyed by symbol
            symbols: List of symbols to trade
            timeframe: Timeframe to use (e.g., '1h', '4h', '1d')
            start_date: Backtest start date
            end_date: Backtest end date
            lookback_days: Days of history needed for indicators
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Reset portfolio
        self._reset_portfolio()
        
        # Get all unique timestamps across symbols
        all_timestamps = set()
        for symbol_data in full_data.values():
            mask = (symbol_data.index >= start_date) & (symbol_data.index <= end_date)
            all_timestamps.update(symbol_data[mask].index)
        
        timestamps = sorted(list(all_timestamps))
        
        # Progress tracking
        total_bars = len(timestamps)
        checkpoint_interval = max(1, total_bars // 20)  # 5% intervals
        
        # Walk forward through time
        for i, current_time in enumerate(timestamps):
            # Progress reporting
            if i % checkpoint_interval == 0:
                progress = (i / total_bars) * 100
                logger.info(f"Backtest progress: {progress:.1f}% ({i}/{total_bars} bars)")
            
            # Get lookback window end
            lookback_start = current_time - timedelta(days=lookback_days)
            
            # Prepare data for orchestrator
            orchestrator_data = {}
            for symbol in symbols:
                if symbol in full_data:
                    # Get data up to current time
                    symbol_data = full_data[symbol]
                    mask = (symbol_data.index >= lookback_start) & (symbol_data.index <= current_time)
                    orchestrator_data[symbol] = {
                        timeframe: symbol_data[mask].copy()
                    }
            
            # Skip if insufficient data
            if not orchestrator_data:
                continue
            
            # Update open positions with current prices
            self._update_open_positions(orchestrator_data, current_time)
            
            # Create new orchestrator instance with current data
            try:
                orchestrator = self.orchestrator_class(
                    db_connection_string=self.db_connection_string,
                    symbols=symbols,
                    lookback_period=lookback_days
                )
                
                # Override data loading to use our prepared data
                orchestrator.data_cache = orchestrator_data
                
                # Setup and run orchestrator
                orchestrator.setup()
                signals = orchestrator.run()
                
                # Process signals
                self._process_signals(signals, orchestrator_data, current_time, orchestrator)
                
            except Exception as e:
                logger.error(f"Error at {current_time}: {e}")
                continue
            
            # Record portfolio value
            self._record_portfolio_value(orchestrator_data, current_time)
        
        # Close any remaining open positions
        self._close_all_positions(orchestrator_data, timestamps[-1])
        
        # Generate results
        results = self._generate_results()
        
        logger.info("Backtest completed")
        return results
    
    def _reset_portfolio(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.holdings = {}
        self.trades = []
        self.open_trades = []
        self.portfolio_values = []
    
    def _update_open_positions(self, data: Dict, current_time: datetime) -> None:
        """
        Update open positions and check for stop loss/take profit.
        
        Args:
            data: Current market data
            current_time: Current timestamp
        """
        for trade in self.open_trades[:]:  # Copy list to allow modification
            symbol = trade.symbol
            if symbol not in data:
                continue
            
            # Get current price
            timeframe = list(data[symbol].keys())[0]
            current_data = data[symbol][timeframe]
            if current_data.empty:
                continue
                
            current_price = current_data['close'].iloc[-1]
            
            # Check stop loss
            if trade.direction == 'long':
                if current_price <= trade.stop_loss:
                    self._close_trade(trade, current_price, current_time, 'stop_loss')
                elif trade.take_profit and current_price >= trade.take_profit:
                    self._close_trade(trade, current_price, current_time, 'take_profit')
            else:  # short
                if current_price >= trade.stop_loss:
                    self._close_trade(trade, current_price, current_time, 'stop_loss')
                elif trade.take_profit and current_price <= trade.take_profit:
                    self._close_trade(trade, current_price, current_time, 'take_profit')
    
    def _process_signals(
        self,
        signals: Dict[str, Any],
        data: Dict,
        current_time: datetime,
        orchestrator: Any
    ) -> None:
        """
        Process trading signals from orchestrator.
        
        Args:
            signals: Signals from orchestrator
            data: Current market data
            current_time: Current timestamp
            orchestrator: Orchestrator instance for metadata
        """
        # Extract composite signal
        composite_signal = signals.get('composite_signal', {})
        
        for symbol, signal_data in composite_signal.items():
            signal_value = signal_data.get('signal', 0)
            confidence = signal_data.get('confidence', 0)
            
            # Skip weak signals
            if abs(signal_value) < 0.6:
                continue
            
            # Check if we already have a position
            existing_trade = self._get_open_trade(symbol)
            
            if existing_trade:
                # Check for exit signal
                if (existing_trade.direction == 'long' and signal_value < -0.5) or \
                   (existing_trade.direction == 'short' and signal_value > 0.5):
                    # Close position due to signal reversal
                    timeframe = list(data[symbol].keys())[0]
                    current_price = data[symbol][timeframe]['close'].iloc[-1]
                    self._close_trade(existing_trade, current_price, current_time, 'signal')
                    
                    # Open new position in opposite direction
                    if len(self.open_trades) < self.max_open_trades:
                        self._open_trade(
                            symbol, signal_value, confidence, 
                            data, current_time, orchestrator
                        )
            else:
                # No existing position, check if we can open new
                if len(self.open_trades) < self.max_open_trades:
                    self._open_trade(
                        symbol, signal_value, confidence,
                        data, current_time, orchestrator
                    )
    
    def _open_trade(
        self,
        symbol: str,
        signal: float,
        confidence: float,
        data: Dict,
        current_time: datetime,
        orchestrator: Any
    ) -> None:
        """Open a new trade."""
        # Get current price
        timeframe = list(data[symbol].keys())[0]
        current_data = data[symbol][timeframe]
        if current_data.empty:
            return
            
        entry_price = current_data['close'].iloc[-1]
        
        # Apply slippage
        if signal > 0:  # Long
            entry_price *= (1 + self.slippage_rate)
            direction = 'long'
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # Short
            entry_price *= (1 - self.slippage_rate)
            direction = 'short'
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        # Calculate position size
        position_value = self.cash * self.max_position_size * confidence
        commission = position_value * self.commission_rate
        quantity = (position_value - commission) / entry_price
        
        # Check if we have enough cash
        if position_value > self.cash:
            return
        
        # Create trade
        trade = Trade(
            entry_time=current_time,
            exit_time=None,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl=None,
            pnl_pct=None,
            market_bias=orchestrator.overall_bias,
            strategy_weights=orchestrator.weighted_signals.get(symbol, {}),
            signal_strength=abs(signal),
            status='open',
            exit_reason=None
        )
        
        # Update portfolio
        self.cash -= (position_value + commission)
        self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
        self.trades.append(trade)
        self.open_trades.append(trade)
        
        logger.debug(f"Opened {direction} trade on {symbol} at {entry_price:.2f}")
    
    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        current_time: datetime,
        exit_reason: str
    ) -> None:
        """Close an existing trade."""
        # Apply slippage
        if trade.direction == 'long':
            exit_price *= (1 - self.slippage_rate)
        else:
            exit_price *= (1 + self.slippage_rate)
        
        # Calculate PnL
        if trade.direction == 'long':
            pnl = (exit_price - trade.entry_price) * trade.quantity
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price
        
        # Commission on exit
        commission = abs(trade.quantity * exit_price * self.commission_rate)
        pnl -= commission
        
        # Update trade
        trade.exit_time = current_time
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.status = 'closed'
        trade.exit_reason = exit_reason
        
        # Update portfolio
        self.cash += (trade.quantity * exit_price - commission)
        self.holdings[trade.symbol] -= trade.quantity
        if abs(self.holdings[trade.symbol]) < 1e-8:
            del self.holdings[trade.symbol]
        
        # Remove from open trades
        self.open_trades.remove(trade)
        
        logger.debug(
            f"Closed {trade.direction} trade on {trade.symbol} at {exit_price:.2f}, "
            f"PnL: ${pnl:.2f} ({pnl_pct*100:.2f}%)"
        )
    
    def _get_open_trade(self, symbol: str) -> Optional[Trade]:
        """Get open trade for symbol."""
        for trade in self.open_trades:
            if trade.symbol == symbol:
                return trade
        return None
    
    def _close_all_positions(self, data: Dict, current_time: datetime) -> None:
        """Close all open positions."""
        for trade in self.open_trades[:]:
            if trade.symbol in data:
                timeframe = list(data[trade.symbol].keys())[0]
                current_price = data[trade.symbol][timeframe]['close'].iloc[-1]
                self._close_trade(trade, current_price, current_time, 'end_of_backtest')
    
    def _record_portfolio_value(self, data: Dict, current_time: datetime) -> None:
        """Record current portfolio value."""
        # Calculate holdings value
        holdings_value = 0.0
        for symbol, quantity in self.holdings.items():
            if symbol in data:
                timeframe = list(data[symbol].keys())[0]
                current_price = data[symbol][timeframe]['close'].iloc[-1]
                holdings_value += quantity * current_price
        
        total_value = self.cash + holdings_value
        
        self.portfolio_values.append({
            'timestamp': current_time,
            'cash': self.cash,
            'holdings_value': holdings_value,
            'total_value': total_value,
            'returns': (total_value - self.initial_capital) / self.initial_capital
        })
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate backtest results and statistics."""
        # Create equity curve
        self.equity_curve = pd.DataFrame(self.portfolio_values)
        if not self.equity_curve.empty:
            self.equity_curve.set_index('timestamp', inplace=True)
        
        # Calculate metrics
        total_trades = len([t for t in self.trades if t.status == 'closed'])
        winning_trades = len([t for t in self.trades if t.status == 'closed' and t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.status == 'closed' and t.pnl < 0])
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_win = np.mean([t.pnl for t in self.trades if t.status == 'closed' and t.pnl > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t.pnl for t in self.trades if t.status == 'closed' and t.pnl < 0]) if losing_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else np.inf
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Calculate returns
        if not self.equity_curve.empty:
            final_value = self.equity_curve['total_value'].iloc[-1]
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Calculate Sharpe ratio (assuming daily data)
            daily_returns = self.equity_curve['total_value'].pct_change().dropna()
            if len(daily_returns) > 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            final_value = self.initial_capital
            total_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Create trades DataFrame
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'market_bias': t.market_bias,
                'signal_strength': t.signal_strength,
                'exit_reason': t.exit_reason
            }
            for t in self.trades if t.status == 'closed'
        ])
        
        return {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'equity_curve': self.equity_curve,
            'trades': trades_df,
            'all_trades': self.trades
        }
