"""
Ichimoku Strategy Backtester

This module provides a streamlined backtesting engine specifically designed
for Ichimoku trading strategies with fixed position sizing and simplified execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
import json
import yaml
from pathlib import Path

# Local imports for streamlined workflow
from streamline_workflow.data_fetching.data_manager import DataManager
from strategy.ichimoku_strategy import (
    UnifiedIchimokuAnalyzer,
    IchimokuStrategyConfig,
    IchimokuParameters,
)
from streamline_workflow.reporting.report_generator import ReportGenerator

# Configure logging
logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    FLAT = "flat"


@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    commission: float
    slippage: float
    net_pnl: float
    return_pct: float
    bars_held: int
    entry_reason: str
    exit_reason: str


@dataclass
class BacktestResult:
    """Comprehensive backtest results."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_return_pct: float
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]


class IchimokuBacktester:
    """
    Simplified backtesting engine for Ichimoku strategies.

    Features:
    - Fixed position sizing (100% of equity)
    - Pyramiding = 1 (only one position at a time)
    - Commission = 0.1%
    - Slippage = 0.03%
    - Close on sell signal
    """

    def __init__(self,
                 commission_rate: float = 0.001,  # 0.1%
                 slippage_rate: float = 0.0003,  # 0.03%
                 pyramiding: int = 1):
        """
        Initialize the backtesting engine.

        Args:
            commission_rate: Commission as percentage of trade value
            slippage_rate: Slippage as percentage of price
            pyramiding: Maximum number of simultaneous positions (1 = no pyramiding)
        """
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.pyramiding = pyramiding

        # Trading state
        self.initial_capital = 0
        self.cash = 0
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

        # Order and trade ID counters
        self._trade_counter = 0

        # Strategy reference
        self.strategy_config = None

        logger.info("Ichimoku Backtester initialized with fixed position sizing")

    def run_backtest(self,
                     strategy_config: Dict,
                     data: pd.DataFrame,
                     initial_capital: float = 10000.0) -> BacktestResult:
        """
        Run a complete backtest of an Ichimoku strategy.

        Args:
            strategy_config: Strategy configuration dictionary
            data: Market data DataFrame with OHLCV and Ichimoku signals
            initial_capital: Starting portfolio value

        Returns:
            BacktestResult with comprehensive results
        """
        logger.info(f"Starting backtest for {strategy_config['name']}")

        # Reset state
        self._reset()
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.strategy_config = strategy_config

        # Validate data
        if not self._validate_data(data):
            raise ValueError("Data validation failed - missing required columns")

        # Run backtest
        self._execute_trading(data)

        # Calculate results
        results = self._calculate_results()

        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")

        return results

    def _reset(self):
        """Reset all state variables."""
        self.initial_capital = 0
        self.cash = 0
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self._trade_counter = 0

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Ensure price columns
        if not all(col in data.columns for col in required_columns):
            return False

        # If signal columns are missing, they will be computed upstream by helper utilities.
        return True

    def _execute_trading(self, data: pd.DataFrame):
        """Execute trading logic based on strategy signals."""
        in_position = False
        current_position = None

        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Update equity curve
            self._update_equity_curve(timestamp, row['close'], current_position)

            # Check for signals
            if not in_position:
                # Check for buy signal
                if self._check_buy_signal(row):
                    # Enter long position
                    entry_price = self._calculate_entry_price(row, True)
                    position_size = self._calculate_position_size(entry_price)

                    if position_size > 0:
                        current_position = self._enter_long(
                            symbol=self.strategy_config['symbols'][0],
                            timestamp=timestamp,
                            price=entry_price,
                            quantity=position_size,
                            reason="buy_signal"
                        )
                        in_position = True
                        logger.debug(f"Entered long position at {entry_price}")

            else:
                # Check for sell signal or stop loss
                exit_signal = False
                exit_reason = ""

                # Check sell signal
                if self._check_sell_signal(row):
                    exit_signal = True
                    exit_reason = "sell_signal"

                # Check stop loss
                elif self._check_stop_loss(row, current_position):
                    exit_signal = True
                    exit_reason = "stop_loss"

                if exit_signal:
                    # Exit position
                    exit_price = self._calculate_entry_price(row, False)
                    self._exit_position(
                        timestamp=timestamp,
                        price=exit_price,
                        reason=exit_reason
                    )
                    in_position = False
                    current_position = None
                    logger.debug(f"Exited position at {exit_price} ({exit_reason})")

    def _check_buy_signal(self, row: pd.Series) -> bool:
        """Check if buy conditions are met."""
        conditions = self.strategy_config['signal_conditions']['buy_conditions']
        logic = self.strategy_config['signal_conditions'].get('buy_logic', 'AND')

        signal_mapping = {
            'PriceAboveCloud': 'price_above_cloud',
            'PriceBelowCloud': 'price_below_cloud',
            'TenkanAboveKijun': 'tenkan_above_kijun',
            'TenkanBelowKijun': 'tenkan_below_kijun',
            'SpanAaboveSpanB': 'SpanAaboveSpanB',
            'SpanAbelowSpanB': 'SpanAbelowSpanB'
        }

        conditions_met = []

        for condition in conditions:
            column_name = signal_mapping.get(condition)
            if column_name and column_name in row:
                if row[column_name]:
                    conditions_met.append(True)
                else:
                    conditions_met.append(False)

        if logic.upper() == "AND":
            return all(conditions_met)
        elif logic.upper() == "OR":
            return any(conditions_met)
        else:
            return all(conditions_met)

    def _check_sell_signal(self, row: pd.Series) -> bool:
        """Check if sell conditions are met."""
        conditions = self.strategy_config['signal_conditions']['sell_conditions']
        logic = self.strategy_config['signal_conditions'].get('sell_logic', 'AND')

        signal_mapping = {
            'PriceAboveCloud': 'price_above_cloud',
            'PriceBelowCloud': 'price_below_cloud',
            'TenkanAboveKijun': 'tenkan_above_kijun',
            'TenkanBelowKijun': 'tenkan_below_kijun',
            'SpanAaboveSpanB': 'SpanAaboveSpanB',
            'SpanAbelowSpanB': 'SpanAbelowSpanB'
        }

        conditions_met = []

        for condition in conditions:
            column_name = signal_mapping.get(condition)
            if column_name and column_name in row:
                if row[column_name]:
                    conditions_met.append(True)
                else:
                    conditions_met.append(False)

        if logic.upper() == "AND":
            return all(conditions_met)
        elif logic.upper() == "OR":
            return any(conditions_met)
        else:
            return all(conditions_met)

    def _check_stop_loss(self, row: pd.Series, position: Dict) -> bool:
        """Check if stop loss condition is met."""
        if not position:
            return False

        stop_loss_pct = self.strategy_config['risk_management'].get('stop_loss_pct')
        if stop_loss_pct is None:
            return False

        current_price = row['close']
        entry_price = position['entry_price']

        # Calculate stop loss price
        stop_price = entry_price * (1 - stop_loss_pct / 100)

        return current_price <= stop_price

    def _calculate_entry_price(self, row: pd.Series, is_buy: bool) -> float:
        """Calculate entry/exit price with slippage."""
        base_price = row['close']
        slippage = base_price * self.slippage_rate

        if is_buy:
            # Buy at ask price (higher)
            return base_price + slippage
        else:
            # Sell at bid price (lower)
            return base_price - slippage

    def _calculate_position_size(self, entry_price: float) -> float:
        """Calculate position size (100% of equity)."""
        if self.cash <= 0:
            return 0

        # Fixed position sizing - use 100% of available cash
        position_value = self.cash
        quantity = position_value / entry_price

        return quantity

    def _enter_long(self, symbol: str, timestamp: datetime, price: float,
                    quantity: float, reason: str) -> Dict:
        """Enter a long position."""
        # Calculate commission
        trade_value = price * quantity
        commission = trade_value * self.commission_rate

        # Update cash
        self.cash -= (trade_value + commission)

        # Create position
        position = {
            'symbol': symbol,
            'entry_time': timestamp,
            'entry_price': price,
            'quantity': quantity,
            'commission_paid': commission,
            'entry_reason': reason
        }

        self.positions[symbol] = position

        return position

    def _exit_position(self, timestamp: datetime, price: float, reason: str):
        """Exit current position."""
        if not self.positions:
            return

        symbol = list(self.positions.keys())[0]
        position = self.positions[symbol]

        # Calculate exit values
        exit_value = price * position['quantity']
        commission = exit_value * self.commission_rate
        net_proceeds = exit_value - commission

        # Update cash
        self.cash += net_proceeds

        # Calculate P&L
        entry_value = position['entry_price'] * position['quantity']
        gross_pnl = exit_value - entry_value
        total_commission = position['commission_paid'] + commission
        net_pnl = gross_pnl - total_commission
        return_pct = (net_pnl / entry_value) * 100

        # Calculate slippage
        base_price = (position['entry_price'] + price) / 2
        slippage_amount = base_price * self.slippage_rate * position['quantity']

        # Calculate bars held
        bars_held = len(self.equity_curve) - self._find_entry_bar_index(position['entry_time'])

        # Record trade
        trade = Trade(
            trade_id=self._generate_trade_id(),
            symbol=symbol,
            entry_time=position['entry_time'],
            exit_time=timestamp,
            side=PositionSide.LONG,
            entry_price=position['entry_price'],
            exit_price=price,
            quantity=position['quantity'],
            commission=total_commission,
            slippage=slippage_amount,
            net_pnl=net_pnl,
            return_pct=return_pct,
            bars_held=bars_held,
            entry_reason=position['entry_reason'],
            exit_reason=reason
        )

        self.trades.append(trade)

        # Remove position
        self.positions.clear()

        logger.info(
            f"Trade closed: {symbol} | PnL: ${net_pnl:.2f} ({return_pct:.2f}%) | "
            f"Reason: {reason}"
        )

    def _find_entry_bar_index(self, entry_time: datetime) -> int:
        """Find the equity curve index for entry time."""
        for i, point in enumerate(self.equity_curve):
            if point['timestamp'] == entry_time:
                return i
        return 0

    def _update_equity_curve(self, timestamp: datetime, price: float, position: Optional[Dict]):
        """Update equity curve with current portfolio value."""
        # Calculate position value if any
        position_value = 0
        if position:
            position_value = price * position['quantity']

        total_value = self.cash + position_value

        self.equity_curve.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value,
            'price': price
        })

    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        if not self.trades:
            return self._empty_results()

        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'symbol': t.symbol,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'net_pnl': t.net_pnl,
            'return_pct': t.return_pct,
            'bars_held': t.bars_held,
            'commission': t.commission,
            'slippage': t.slippage
        } for t in self.trades])

        # Calculate basic metrics
        total_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate profit factor
        gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate total return
        final_equity = self.equity_curve[-1]['total_value'] if self.equity_curve else self.initial_capital
        total_return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        # Calculate max drawdown
        equity_values = [point['total_value'] for point in self.equity_curve]
        max_drawdown_pct = self._calculate_max_drawdown(equity_values)

        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(trades_df)

        # Additional metrics
        avg_trade = trades_df['net_pnl'].mean()
        avg_winning_trade = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean()
        avg_losing_trade = trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean()
        largest_win = trades_df['net_pnl'].max()
        largest_loss = trades_df['net_pnl'].min()
        avg_bars_held = trades_df['bars_held'].mean()

        metrics = {
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_pnl': avg_trade,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_bars_held': avg_bars_held,
            'total_commission': trades_df['commission'].sum(),
            'total_slippage': trades_df['slippage'].sum(),
            'net_profit': trades_df['net_pnl'].sum(),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            # Set timestamp as index for time-series operations
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df.set_index('timestamp', inplace=True)
            
            # Add 'equity' column for compatibility with ReportGenerator
            equity_df['equity'] = equity_df['total_value']
            
            equity_df['returns'] = equity_df['total_value'].pct_change()
            equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1

            # Calculate running max for drawdown
            equity_df['running_max'] = equity_df['total_value'].expanding().max()
            equity_df['drawdown'] = (equity_df['total_value'] - equity_df['running_max']) / equity_df['running_max']

        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_return_pct=total_return_pct,
            win_rate=win_rate * 100,
            profit_factor=profit_factor,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            trades=self.trades,
            equity_curve=equity_df,
            metrics=metrics
        )

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd * 100  # Convert to percentage

    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio from trade returns."""
        if len(trades_df) < 2:
            return 0.0

        # Use trade returns to calculate Sharpe
        returns = trades_df['return_pct'] / 100  # Convert to decimal

        if returns.std() == 0:
            return 0.0

        # Annualize assuming 252 trading days
        # Rough approximation: average trades per day * 252
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

        return sharpe

    def _empty_results(self) -> BacktestResult:
        """Return empty results when no trades occurred."""
        final_equity = self.equity_curve[-1]['total_value'] if self.equity_curve else self.initial_capital
        total_return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        return BacktestResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_return_pct=total_return_pct,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            trades=[],
            equity_curve=pd.DataFrame(),
            metrics={
                'total_return_pct': total_return_pct,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'net_profit': 0.0
            }
        )

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"TRD_{self._trade_counter:06d}"


# Strategy JSON Integration and utilities
class StrategyBacktestRunner:
    """Helper class to run backtests from strategy JSON configurations."""

    def __init__(self, backtester: IchimokuBacktester):
        self.backtester = backtester

    def run_strategy_backtest(self,
                              strategy_config: Dict,
                              data: pd.DataFrame,
                              initial_capital: float = 10000.0) -> BacktestResult:
        """
        Run backtest for a strategy from JSON configuration.

        Args:
            strategy_config: Strategy configuration from JSON
            data: Market data with Ichimoku signals
            initial_capital: Initial capital for backtest

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Running backtest for: {strategy_config['name']}")

        # Validate strategy configuration
        if not self._validate_strategy_config(strategy_config):
            raise ValueError("Invalid strategy configuration")

        # Run backtest
        result = self.backtester.run_backtest(
            strategy_config=strategy_config,
            data=data,
            initial_capital=initial_capital
        )

        return result

    def _validate_strategy_config(self, config: Dict) -> bool:
        """Validate strategy configuration."""
        required_fields = [
            'name', 'symbols', 'signal_conditions',
            'ichimoku_parameters', 'risk_management', 'position_sizing'
        ]

        if not all(field in config for field in required_fields):
            logger.error("Missing required fields in strategy configuration")
            return False

        # Validate signal conditions
        signal_conditions = config['signal_conditions']
        if 'buy_conditions' not in signal_conditions:
            logger.error("Missing buy_conditions in signal_conditions")
            return False

        # Validate position sizing
        position_sizing = config['position_sizing']
        if position_sizing.get('method') != 'fixed':
            logger.warning("Only fixed position sizing is supported")

        return True

    # ---------- Convenience high-level helpers ----------
    def load_strategy_from_json(self, strategy_key: str,
                                json_path: Optional[Union[str, Path]] = None) -> Dict:
        """Load a single strategy configuration by key from strategies.json.
        Tries both streamline_workflow/config and strategy/config locations.
        """
        candidates: List[Path] = []
        if json_path:
            candidates.append(Path(json_path))
        # Preferred path as per user instruction
        base = Path(__file__).resolve().parents[1]
        candidates.extend([
            base / 'config' / 'strategies.json',
            base / 'config' / 'strategies.yaml',
            base / 'strategy' / 'config' / 'strategies.json',
            base / 'strategy' / 'config' / 'strategies.yaml',
        ])

        data: Dict[str, Any] = {}
        file_found = None
        for p in candidates:
            if p.exists():
                try:
                    with open(p, 'r') as f:
                        if p.suffix == '.json':
                            data = json.load(f)
                        else:
                            data = yaml.safe_load(f)
                    file_found = p
                    break
                except Exception:
                    continue
        if not data:
            raise FileNotFoundError("No valid strategies file found in expected locations (json or yaml)")

        strategies = data.get('strategies', {})
        if strategy_key not in strategies:
            raise KeyError(f"Strategy key '{strategy_key}' not found in {file_found}")
        return strategies[strategy_key]

    def fetch_sql_data_with_signals(self, symbol_short: str, timeframe: str,
                                    start: Optional[str] = None,
                                    end: Optional[str] = None,
                                    ichimoku_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Fetch OHLCV+Ichimoku from per-symbol DB and add boolean signals columns."""
        dm = DataManager(symbol=symbol_short)
        start_dt = pd.to_datetime(start) if start else None
        end_dt = pd.to_datetime(end) if end else None
        # Prefer combined view with Ichimoku if present
        try:
            df = dm.get_ichimoku_data(timeframe=timeframe, start_date=start_dt, end_date=end_dt)
        except Exception:
            df = dm.get_ohlcv_data(timeframe=timeframe, start_date=start_dt, end_date=end_dt)
        dm.close_connection()
        if df.empty:
            return df

        # If Ichimoku components are not present, compute them from price data using analyzer
        analyzer = UnifiedIchimokuAnalyzer()
        params = IchimokuStrategyConfig.create_parameters(**(ichimoku_params or {}))
        if not set(['tenkan_sen','kijun_sen','senkou_span_a','senkou_span_b','chikou_span']).issubset(df.columns):
            df = analyzer.calculate_ichimoku_components(df, params)
        # Add boolean signals
        df = analyzer.detect_boolean_signals(df, params)
        return df

    def run_from_json(self, strategy_key: str, symbol_short: str, timeframe: str,
                       start: Optional[str] = None, end: Optional[str] = None,
                       initial_capital: float = 10000.0,
                       report_formats: str = 'all',
                       output_dir: str = 'results') -> Dict[str, Any]:
        """Load strategy by key, fetch data, run backtest, and generate report."""
        strategy_config = self.load_strategy_from_json(strategy_key)
        data = self.fetch_sql_data_with_signals(symbol_short, timeframe,
                                                start, end,
                                                strategy_config.get('ichimoku_parameters'))
        if data.empty:
            raise ValueError("No data available for backtest")

        result = self.run_strategy_backtest(strategy_config, data, initial_capital)

        # Prepare structures for reporting
        trades_df = pd.DataFrame([t.__dict__ for t in result.trades]) if result.trades else pd.DataFrame()
        # Rename net_pnl to pnl for compatibility with ReportGenerator
        if not trades_df.empty and 'net_pnl' in trades_df.columns:
            trades_df['pnl'] = trades_df['net_pnl']
        equity_df = result.equity_curve.copy() if isinstance(result.equity_curve, pd.DataFrame) else pd.DataFrame(result.equity_curve)
        
        report_payload = {
            'data': data,
            'trades': trades_df,
            'equity_curve': equity_df,
            'metrics': {
                'performance_metrics': {
                    'total_return': result.metrics.get('total_return_pct', 0)/100.0,
                    'annualized_return': None,
                    'cumulative_return': equity_df['cumulative_returns'].iloc[-1] if not equity_df.empty and 'cumulative_returns' in equity_df.columns else None
                }
            },
            'strategy_config': {
                'name': strategy_config.get('name'),
                'symbol': f"{symbol_short}/USDT",
                'timeframe': timeframe
            }
        }
        rg = ReportGenerator(output_dir=output_dir)
        reports = rg.generate_backtest_report(report_payload, format=report_formats, filename_prefix=f"{symbol_short}_{timeframe}")

        return {
            'result': result,
            'reports': reports,
            'strategy_config': strategy_config
        }


# Example usage
if __name__ == "__main__":
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example strategy configuration (from your JSON)
    strategy_config = {
        "name": "Cloud-TK-SpanA Base TK Exit",
        "description": "Price above cloud + TK cross + SpanA>SpanB, exit on TK cross down",
        "enabled": True,
        "timeframes": ["1h", "4h"],
        "symbols": ["BTC/USDT"],
        "signal_conditions": {
            "buy_conditions": ["PriceAboveCloud", "TenkanAboveKijun", "SpanAaboveSpanB"],
            "sell_conditions": ["TenkanBelowKijun"],
            "buy_logic": "AND",
            "sell_logic": "AND"
        },
        "ichimoku_parameters": {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
            "chikou_offset": 26,
            "senkou_offset": 26
        },
        "risk_management": {
            "stop_loss_pct": 5.0,
            "take_profit_pct": None,
            "close_on_sell_signal": True,
            "trailing_stop": False,
            "max_position_size_pct": 100.0,
            "risk_per_trade_pct": 5.0
        },
        "position_sizing": {
            "method": "fixed",
            "fixed_size": 1000
        }
    }

    # Initialize backtester with your specified parameters
    backtester = IchimokuBacktester(
        commission_rate=0.001,  # 0.1%
        slippage_rate=0.0003,  # 0.03%
        pyramiding=1
    )

    # Initialize strategy runner
    runner = StrategyBacktestRunner(backtester)

    print("Ichimoku Strategy Backtester ready")
    print("Features:")
    print("- Fixed position sizing (100% of equity)")
    print("- Pyramiding = 1 (single position)")
    print("- Commission = 0.1%")
    print("- Slippage = 0.03%")
    print("- Close on sell signal")
    print("- Stop loss support")