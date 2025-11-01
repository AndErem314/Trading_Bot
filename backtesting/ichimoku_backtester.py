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
from data_fetching.data_manager import DataManager
from strategy.ichimoku_strategy import (
    UnifiedIchimokuAnalyzer,
    IchimokuStrategyConfig,
    IchimokuParameters,
)
from reporting.report_generator import ReportGenerator

# Configure logging
logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
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
                # Check for LONG entry
                if self._check_entry_signal(row, PositionSide.LONG):
                    entry_price = self._calculate_fill_price(PositionSide.LONG, is_entry=True, row=row)
                    position_size = self._calculate_position_size(entry_price)
                    if position_size > 0:
                        current_position = self._enter_long(
                            symbol=self.strategy_config['symbols'][0],
                            timestamp=timestamp,
                            price=entry_price,
                            quantity=position_size,
                            reason="long_entry"
                        )
                        in_position = True
                        logger.debug(f"Entered LONG at {entry_price}")
                # If not LONG, check SHORT entry
                elif self._check_entry_signal(row, PositionSide.SHORT):
                    entry_price = self._calculate_fill_price(PositionSide.SHORT, is_entry=True, row=row)
                    position_size = self._calculate_position_size(entry_price)
                    if position_size > 0:
                        current_position = self._enter_short(
                            symbol=self.strategy_config['symbols'][0],
                            timestamp=timestamp,
                            price=entry_price,
                            quantity=position_size,
                            reason="short_entry"
                        )
                        in_position = True
                        logger.debug(f"Entered SHORT at {entry_price}")

            else:
                # Check for exit signal or stop loss based on side
                exit_signal = False
                exit_reason = ""

                # Exit on configured signal
                if self._check_exit_signal(row, PositionSide(current_position['side'])):
                    exit_signal = True
                    exit_reason = "signal_exit"
                # Check stop loss
                elif self._check_stop_loss(row, current_position):
                    exit_signal = True
                    exit_reason = "stop_loss"

                if exit_signal:
                    # Exit position
                    side = PositionSide(current_position['side'])
                    exit_price = self._calculate_fill_price(side, is_entry=False, row=row)
                    self._exit_position(
                        timestamp=timestamp,
                        price=exit_price,
                        reason=exit_reason
                    )
                    in_position = False
                    current_position = None
                    logger.debug(f"Exited position at {exit_price} ({exit_reason})")

    def _get_signal_mapping(self) -> Dict[str, str]:
        """Map config condition names to DataFrame columns."""
        return {
            'PriceAboveCloud': 'price_above_cloud',
            'PriceBelowCloud': 'price_below_cloud',
            'TenkanAboveKijun': 'tenkan_above_kijun',
            'TenkanBelowKijun': 'tenkan_below_kijun',
            'SpanAaboveSpanB': 'SpanAaboveSpanB',
            'SpanAbelowSpanB': 'SpanAbelowSpanB',
            'ChikouAbovePrice': 'chikou_above_price',
            'ChikouBelowPrice': 'chikou_below_price',
            'ChikouAboveCloud': 'chikou_above_cloud',
            'ChikouBelowCloud': 'chikou_below_cloud',
            'PSARUptrend': 'psar_uptrend',
            'PSARDowntrend': 'psar_downtrend',
        }

    def _check_conditions(self, row: pd.Series, conditions: List[str], logic: str) -> bool:
        """Generic condition checker with AND/OR logic."""
        if not conditions:
            return False
        mapping = self._get_signal_mapping()
        flags: List[bool] = []
        for cond in conditions:
            col = mapping.get(cond)
            if col and col in row:
                flags.append(bool(row[col]))
        if not flags:
            return False
        if (logic or 'AND').upper() == 'OR':
            return any(flags)
        return all(flags)

    def _check_entry_signal(self, row: pd.Series, side: PositionSide) -> bool:
        sc = self.strategy_config.get('signal_conditions', {})
        if side == PositionSide.LONG:
            conditions = sc.get('long_entry_conditions') or sc.get('buy_conditions') or []
            logic = sc.get('long_entry_logic') or sc.get('buy_logic', 'AND')
        else:
            conditions = sc.get('short_entry_conditions')
            if not conditions:
                base = sc.get('long_entry_conditions') or sc.get('buy_conditions') or []
                conditions = self._mirror_conditions(base)
            logic = sc.get('short_entry_logic') or sc.get('long_entry_logic') or sc.get('buy_logic', 'AND')
        base_ok = self._check_conditions(row, conditions, logic)
        # PSAR confirmation (no look-ahead; assumed precomputed on closed bar)
        if side == PositionSide.LONG:
            psar_ok = (('psar_uptrend' in row and bool(row['psar_uptrend'])) or ('psar_trend' in row and row['psar_trend'] == 1))
        else:
            psar_ok = (('psar_downtrend' in row and bool(row['psar_downtrend'])) or ('psar_trend' in row and row['psar_trend'] == -1))
        # If PSAR columns are not present, don't block entries
        if (('psar_uptrend' in row) or ('psar_trend' in row)):
            return base_ok and psar_ok
        return base_ok

    def _check_exit_signal(self, row: pd.Series, side: PositionSide) -> bool:
        sc = self.strategy_config.get('signal_conditions', {})
        if side == PositionSide.LONG:
            conditions = sc.get('long_exit_conditions') or sc.get('sell_conditions') or []
            logic = sc.get('long_exit_logic') or sc.get('sell_logic', 'AND')
        else:
            conditions = sc.get('short_exit_conditions')
            if not conditions:
                base = sc.get('long_exit_conditions') or sc.get('sell_conditions') or []
                conditions = self._mirror_conditions(base)
            logic = sc.get('short_exit_logic') or sc.get('long_exit_logic') or sc.get('sell_logic', 'AND')
        return self._check_conditions(row, conditions, logic)

    def _mirror_condition(self, cond: str) -> Optional[str]:
        """Return the opposite condition name for mirroring LONG<->SHORT."""
        mirror_map = {
            'PriceAboveCloud': 'PriceBelowCloud',
            'PriceBelowCloud': 'PriceAboveCloud',
            'TenkanAboveKijun': 'TenkanBelowKijun',
            'TenkanBelowKijun': 'TenkanAboveKijun',
            'SpanAaboveSpanB': 'SpanAbelowSpanB',
            'SpanAbelowSpanB': 'SpanAaboveSpanB',
            'ChikouAbovePrice': 'ChikouBelowPrice',
            'ChikouBelowPrice': 'ChikouAbovePrice',
            'ChikouAboveCloud': 'ChikouBelowCloud',
            'ChikouBelowCloud': 'ChikouAboveCloud',
        }
        return mirror_map.get(cond)

    def _mirror_conditions(self, conditions: List[str]) -> List[str]:
        return [self._mirror_condition(c) for c in conditions if self._mirror_condition(c)]

    def _check_stop_loss(self, row: pd.Series, position: Dict) -> bool:
        """Check if stop loss condition is met (based on closed bars)."""
        if not position:
            return False

        stop_loss_pct = self.strategy_config['risk_management'].get('stop_loss_pct')
        if stop_loss_pct is None:
            return False

        current_price = row['close']
        entry_price = position['entry_price']
        side = PositionSide(position.get('side', PositionSide.LONG.value))

        if side == PositionSide.LONG:
            stop_price = entry_price * (1 - stop_loss_pct / 100)
            return current_price <= stop_price
        else:
            stop_price = entry_price * (1 + stop_loss_pct / 100)
            return current_price >= stop_price

    def _calculate_fill_price(self, side: PositionSide, is_entry: bool, row: pd.Series) -> float:
        """Calculate trade fill price with slippage for LONG/SHORT entries and exits."""
        base_price = row['close']
        slippage = base_price * self.slippage_rate
        if side == PositionSide.LONG:
            # Long entry buys (ask), exit sells (bid)
            return base_price + slippage if is_entry else base_price - slippage
        elif side == PositionSide.SHORT:
            # Short entry sells (bid), exit buys (ask)
            return base_price - slippage if is_entry else base_price + slippage
        else:
            return base_price

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
        trade_value = price * quantity
        commission = trade_value * self.commission_rate
        self.cash -= (trade_value + commission)
        position = {
            'symbol': symbol,
            'side': PositionSide.LONG.value,
            'entry_time': timestamp,
            'entry_price': price,
            'quantity': quantity,
            'commission_paid': commission,
            'entry_reason': reason
        }
        self.positions[symbol] = position
        return position

    def _enter_short(self, symbol: str, timestamp: datetime, price: float,
                     quantity: float, reason: str) -> Dict:
        """Enter a short position."""
        trade_value = price * quantity
        commission = trade_value * self.commission_rate
        # Receive proceeds from short sale minus commission
        self.cash += (trade_value - commission)
        position = {
            'symbol': symbol,
            'side': PositionSide.SHORT.value,
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
        side = PositionSide(position.get('side', PositionSide.LONG.value))

        # Calculate exit values
        exit_value = price * position['quantity']
        commission_exit = exit_value * self.commission_rate

        # Update cash and compute P&L based on side
        entry_value = position['entry_price'] * position['quantity']
        commission_entry = position['commission_paid']
        if side == PositionSide.LONG:
            net_proceeds = exit_value - commission_exit
            self.cash += net_proceeds
            gross_pnl = exit_value - entry_value
        else:  # SHORT
            # Buy-to-cover reduces cash
            self.cash -= (exit_value + commission_exit)
            gross_pnl = entry_value - exit_value

        total_commission = commission_entry + commission_exit
        net_pnl = gross_pnl - total_commission
        return_pct = (net_pnl / entry_value) * 100 if entry_value != 0 else 0.0

        # Calculate slippage (approximate, symmetric)
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
            side=side,
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
        position_value = 0.0
        if position:
            qty = position['quantity']
            side = PositionSide(position.get('side', PositionSide.LONG.value))
            if side == PositionSide.LONG:
                position_value = price * qty
            else:  # SHORT
                position_value = -price * qty

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

    def generate_llm_optimization_report(self,
                                         *,
                                         result: BacktestResult,
                                         data_df: pd.DataFrame,
                                         trades_df: pd.DataFrame,
                                         equity_df: pd.DataFrame,
                                         strategy_config: Dict[str, Any],
                                         output_dir: str,
                                         symbol_short: str,
                                         timeframe: str,
                                         analysis_start: Optional[str] = None,
                                         analysis_end: Optional[str] = None,
                                         llm_provider: Optional[str] = None,
                                         llm_model_override: Optional[str] = None,
                                         prompt_variant: str = 'analyst') -> Optional[str]:
        """Generate an LLM-only optimization PDF using existing backtest artifacts.

        Returns the path to the generated PDF or None if it failed.
        """
        try:
            from llm_analysis import (
                load_llm_config, LLMClient, build_llm_payload, build_prompt,
                parse_llm_output, build_final_text, write_llm_pdf
            )
            payload = build_llm_payload(
                result_metrics=result.metrics,
                trades_df=trades_df,
                equity_df=equity_df,
                strategy_config=strategy_config,
                analysis_start=analysis_start,
                analysis_end=analysis_end,
                budget='standard'
            )
            prompt = build_prompt(payload, variant=prompt_variant)
            cfg = load_llm_config()
            client = LLMClient(cfg)

            # Determine effective provider/model to use for token estimation
            effective_provider = (llm_provider or cfg.provider or 'openai').lower()
            if effective_provider not in ('openai', 'gemini'):
                effective_provider = 'openai' if cfg.openai_api_key else 'gemini'
            if effective_provider == 'openai':
                effective_model = llm_model_override or cfg.openai_model or 'gpt-4o-mini'
            else:
                effective_model = llm_model_override or cfg.gemini_model or 'gemini-2.5-pro'

            # Token counts for prompt and (later) output
            try:
                from llm_analysis.token_utils import count_tokens as _count_tokens
                prompt_tokens = _count_tokens(prompt, effective_provider, effective_model)
            except Exception:
                prompt_tokens = 0

            raw = client.generate(prompt, provider=llm_provider, model_override=llm_model_override)

            try:
                from llm_analysis.token_utils import count_tokens as _count_tokens
                output_tokens = _count_tokens(raw or '', effective_provider, effective_model)
            except Exception:
                output_tokens = 0

            json_obj, memo = parse_llm_output(raw)
            title = 'Strategy Settings Optimization — Executive Summary' if prompt_variant == 'analyst' else 'Risk-Focused Optimization'
            final_text = build_final_text(title, json_obj, memo)

            # Prepend usage header to the final text
            usage_header = (
                f"Provider: {effective_provider} | Model: {effective_model} | "
                f"Prompt tokens: {prompt_tokens} | Output tokens: {output_tokens} | Total: {prompt_tokens + output_tokens}"
            )
            final_text = usage_header + "\n\n" + final_text

            # Optionally write optimized YAML config derived from LLM JSON
            try:
                # Save YAML into project config/llm_strategy_config (project root = parents[1])
                proj_root = Path(__file__).resolve().parents[1]
                yaml_dir = proj_root / 'config' / 'llm_strategy_config'
                opt_yaml = self._write_llm_optimized_yaml(
                    base_strategy_config=strategy_config,
                    llm_json=json_obj,
                    symbol_short=symbol_short,
                    timeframe=timeframe,
                    output_dir=str(yaml_dir)
                )
            except Exception:
                opt_yaml = None

            pdf_path = write_llm_pdf(
                output_dir=output_dir,
                filename_prefix=f"{symbol_short}_{timeframe}",
                title=title,
                text_body=final_text
            )
            return pdf_path
        except Exception as e:
            logger.error(f"LLM optimization generation failed: {e}")
            return None

    def _write_llm_optimized_yaml(self, *, base_strategy_config: Dict[str, Any], llm_json: Dict[str, Any], symbol_short: str, timeframe: str, output_dir: Union[str, Path]) -> Optional[str]:
        """Build and write an optimized strategy YAML (single strategy) based on LLM JSON suggestions.

        The YAML structure matches strategies.yaml with a single key under 'strategies'.
        """
        try:
            import yaml  # lazy import
        except Exception:
            return None
        from pathlib import Path as _P
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        key_base = base_strategy_config.get('name', 'strategy').lower().replace(' ', '_')
        strat_key = f"{key_base}_llm_{symbol_short.lower()}_{timeframe}"

        # Start from the original strategy config to preserve fields
        sc = json.loads(json.dumps(base_strategy_config)) if isinstance(base_strategy_config, dict) else {}

        pc = (llm_json or {}).get('parameter_changes', {}) if isinstance(llm_json, dict) else {}
        # Ichimoku params
        ichi = pc.get('ichimoku', {}) or {}
        if 'ichimoku_parameters' not in sc:
            sc['ichimoku_parameters'] = {}
        for p in ('tenkan_period','kijun_period','senkou_b_period','chikou_offset','senkou_offset'):
            if p in ichi and isinstance(ichi[p], dict) and 'suggested' in ichi[p] and isinstance(ichi[p]['suggested'], (int, float)):
                sc['ichimoku_parameters'][p] = int(ichi[p]['suggested']) if isinstance(ichi[p]['suggested'], float) and p.endswith('period') else ichi[p]['suggested']

        # Signal logic and conditions
        sl = pc.get('signal_logic', {}) or {}
        if 'signal_conditions' not in sc:
            sc['signal_conditions'] = {'buy_conditions': [], 'sell_conditions': [], 'buy_logic': 'AND', 'sell_logic': 'AND'}
        if 'buy_logic' in sl and isinstance(sl['buy_logic'], dict):
            sc['signal_conditions']['buy_logic'] = sl['buy_logic'].get('suggested', sc['signal_conditions'].get('buy_logic', 'AND'))
        if 'sell_logic' in sl and isinstance(sl['sell_logic'], dict):
            sc['signal_conditions']['sell_logic'] = sl['sell_logic'].get('suggested', sc['signal_conditions'].get('sell_logic', 'AND'))
        add_conditions = sl.get('add_conditions', []) or []
        remove_conditions = set(sl.get('remove_conditions', []) or [])
        if isinstance(add_conditions, list):
            # Add to buy_conditions by default (keeps original semantics)
            bc = list(sc['signal_conditions'].get('buy_conditions', []))
            for c in add_conditions:
                if c not in bc:
                    bc.append(c)
            sc['signal_conditions']['buy_conditions'] = bc
        if remove_conditions:
            for list_name in ('buy_conditions','sell_conditions'):
                cur = [c for c in sc['signal_conditions'].get(list_name, []) if c not in remove_conditions]
                sc['signal_conditions'][list_name] = cur

        # Risk management
        rm = pc.get('risk_management', {}) or {}
        if 'risk_management' not in sc:
            sc['risk_management'] = {}
        for k in ('stop_loss_pct','take_profit_pct'):
            v = rm.get(k)
            if isinstance(v, dict) and 'suggested' in v and isinstance(v['suggested'], (int, float)):
                sc['risk_management'][k] = float(v['suggested'])
        # Position sizing suggestion
        ps = rm.get('position_sizing')
        if isinstance(ps, dict):
            suggested = ps.get('suggested')
            if suggested in ('fixed','volatility'):
                sc.setdefault('position_sizing', {}).update({'method': suggested})

        # Symbols and timeframe harmonization
        sc['symbols'] = sc.get('symbols') or [f"{symbol_short}/USDT"]
        if isinstance(sc.get('timeframes'), list):
            if timeframe not in sc['timeframes']:
                sc['timeframes'].append(timeframe)
        else:
            sc['timeframes'] = [timeframe]

        # Build final YAML mapping
        out = {'strategies': {strat_key: sc}}

        out_dir = _P(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{strat_key}_{ts}.yaml"
        with open(out_path, 'w') as f:
            yaml.safe_dump(out, f, sort_keys=False)
        return str(out_path)

    def run_strategy_backtest(self, strategy_config: Dict,
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

        # Validate signal conditions (support legacy buy/sell and new long/short)
        signal_conditions = config['signal_conditions']
        has_legacy = 'buy_conditions' in signal_conditions and 'sell_conditions' in signal_conditions
        has_directional = ('long_entry_conditions' in signal_conditions and 'long_exit_conditions' in signal_conditions)
        if not (has_legacy or has_directional):
            logger.error("Missing entry/exit conditions in signal_conditions")
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
            # Prefer YAML first
            base / 'config' / 'strategies.yaml',
            base / 'config' / 'strategies.json',
            base / 'strategy' / 'config' / 'strategies.yaml',
            base / 'strategy' / 'config' / 'strategies.json',
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
                                    ichimoku_params: Optional[Dict[str, Any]] = None,
                                    force_recompute: bool = False) -> pd.DataFrame:
        """Fetch OHLCV+Ichimoku and add boolean signals.

        If force_recompute=True, ignore any precomputed Ichimoku in SQL and recompute
        components using the provided ichimoku_params purely in-memory.
        """
        dm = DataManager(symbol=symbol_short)
        start_dt = pd.to_datetime(start) if start else None
        end_dt = pd.to_datetime(end) if end else None

        analyzer = UnifiedIchimokuAnalyzer()
        params = IchimokuStrategyConfig.create_parameters(**(ichimoku_params or {}))

        if force_recompute:
            # Load raw OHLCV only and recompute all components with strategy params
            df = dm.get_ohlcv_data(timeframe=timeframe, start_date=start_dt, end_date=end_dt)
            dm.close_connection()
            if df.empty:
                return df
            df = analyzer.calculate_ichimoku_components(df, params)
            df = analyzer.detect_boolean_signals(df, params)
            # Compute PSAR in-memory for confirmation
            try:
                from strategy.psar_indicator import compute_psar
                psar_df = compute_psar(df[['high','low','close']])
                df['psar'] = psar_df['psar']
                df['psar_trend'] = psar_df['psar_trend']
                df['psar_reversal'] = psar_df['psar_reversal']
            except Exception:
                pass
            # Derive boolean PSAR signals on closed bars
            if 'psar_trend' in df.columns:
                closed_mask = df.index.to_series().notna()
                if len(df) > 0:
                    closed_mask.iloc[-1] = False
                df['psar_uptrend'] = (df['psar_trend'] == 1) & closed_mask
                df['psar_downtrend'] = (df['psar_trend'] == -1) & closed_mask
            return df

        # Default path: use SQL ichimoku view if present, otherwise OHLCV and compute missing
        try:
            df = dm.get_ichimoku_data(timeframe=timeframe, start_date=start_dt, end_date=end_dt)
        except Exception:
            df = dm.get_ohlcv_data(timeframe=timeframe, start_date=start_dt, end_date=end_dt)
        dm.close_connection()
        if df.empty:
            return df

        # If Ichimoku components are not present, compute them from price data using analyzer
        if not set(['tenkan_sen','kijun_sen','senkou_span_a','senkou_span_b','chikou_span']).issubset(df.columns):
            df = analyzer.calculate_ichimoku_components(df, params)
        # Add boolean signals
        df = analyzer.detect_boolean_signals(df, params)

        # Ensure PSAR columns present; if not, compute in-memory
        psar_present = 'psar' in df.columns
        if not psar_present:
            try:
                from strategy.psar_indicator import compute_psar
                psar_df = compute_psar(df[['high','low','close']])
                df['psar'] = psar_df['psar']
                df['psar_trend'] = psar_df['psar_trend']
                df['psar_reversal'] = psar_df['psar_reversal']
            except Exception:
                pass
        # Derive boolean PSAR signals on closed bars
        if 'psar_trend' in df.columns:
            closed_mask = df.index.to_series().notna()
            if len(df) > 0:
                closed_mask.iloc[-1] = False
            df['psar_uptrend'] = (df['psar_trend'] == 1) & closed_mask
            df['psar_downtrend'] = (df['psar_trend'] == -1) & closed_mask
        return df

    def run_from_json(self, strategy_key: str, symbol_short: str, timeframe: str,
                       start: Optional[str] = None, end: Optional[str] = None,
                       initial_capital: float = 10000.0,
                       report_formats: str = 'pdf',
                       output_dir: str = 'results',
                       with_llm_optimization: bool = False,
                       llm_provider: Optional[str] = None,
                       analysis_start: Optional[str] = None,
                       analysis_end: Optional[str] = None,
                       llm_model_override: Optional[str] = None,
                       prompt_variant: str = 'analyst',
                       force_recompute_ichimoku: bool = False) -> Dict[str, Any]:
        """Load strategy by key, fetch data, run backtest, and generate report.

        If with_llm_optimization=True, an LLM-only optimization PDF is generated AFTER the standard report.
        """
        strategy_config = self.load_strategy_from_json(strategy_key)
        data = self.fetch_sql_data_with_signals(symbol_short, timeframe,
                                                start, end,
                                                strategy_config.get('ichimoku_parameters'),
                                                force_recompute=force_recompute_ichimoku)
        if data.empty:
            raise ValueError("No data available for backtest")

        result = self.run_strategy_backtest(strategy_config, data, initial_capital)

        # Prepare structures for reporting
        trades_df = pd.DataFrame([t.__dict__ for t in result.trades]) if result.trades else pd.DataFrame()
        # Normalize columns for reporting
        if not trades_df.empty:
            # pnl alias
            if 'net_pnl' in trades_df.columns:
                trades_df['pnl'] = trades_df['net_pnl']
            # direction string from side enum
            if 'side' in trades_df.columns:
                trades_df['direction'] = trades_df['side'].apply(lambda s: s.value if hasattr(s, 'value') else str(s).lower())
            # entry_signal from entry_reason
            if 'entry_reason' in trades_df.columns:
                trades_df['entry_signal'] = trades_df['entry_reason']
        equity_df = result.equity_curve.copy() if isinstance(result.equity_curve, pd.DataFrame) else pd.DataFrame(result.equity_curve)
        
        # Map backtester metrics into the format expected by ReportGenerator
        perf_metrics = {
            'total_return': (result.metrics.get('total_return_pct', 0) or 0) / 100.0,
            'annual_return': (result.metrics.get('annual_return', 0) or 0) / 100.0 if 'annual_return' in result.metrics else None,
            'cumulative_return': equity_df['cumulative_returns'].iloc[-1] if not equity_df.empty and 'cumulative_returns' in equity_df.columns else None,
            'sharpe_ratio': result.metrics.get('sharpe_ratio', 0.0) or 0.0,
            'max_drawdown': (result.metrics.get('max_drawdown_pct', 0) or 0) / 100.0,
            'win_rate': (result.metrics.get('win_rate_pct', 0) or 0) / 100.0,
            'profit_factor': result.metrics.get('profit_factor', 0.0) if np.isfinite(result.metrics.get('profit_factor', 0.0) or 0.0) else float('inf'),
            'total_trades': result.metrics.get('total_trades', len(result.trades) if isinstance(result.trades, list) else 0),
            # Optional fields used by risk/insights if available
            'avg_win': result.metrics.get('avg_winning_trade', 0),
            'avg_loss': result.metrics.get('avg_losing_trade', 0),
            'loss_std': result.metrics.get('loss_std', 0),
            'var_95': result.metrics.get('var_95', 0),
            'cvar_95': result.metrics.get('cvar_95', 0)
        }

        report_payload = {
            'data': data,
            'trades': trades_df,
            'equity_curve': equity_df,
            'metrics': {
                'performance_metrics': perf_metrics
            },
            'strategy_config': {
                'name': strategy_config.get('name'),
                'symbol': f"{symbol_short}/USDT",
                'timeframe': timeframe
            }
        }
        rg = ReportGenerator(output_dir=output_dir)
        reports = rg.generate_backtest_report(report_payload, format=report_formats, filename_prefix=f"{symbol_short}_{timeframe}")

        # Optional LLM optimization step AFTER standard reporting
        llm_pdf_path = None
        if with_llm_optimization:
            try:
                from llm_analysis import (
                    load_llm_config, LLMClient, build_llm_payload, build_prompt,
                    parse_llm_output, build_final_text, write_llm_pdf
                )
                # Build compact payload
                payload = build_llm_payload(
                    result_metrics=result.metrics,
                    trades_df=trades_df,
                    equity_df=equity_df,
                    strategy_config=strategy_config,
                    analysis_start=analysis_start,
                    analysis_end=analysis_end,
                    budget='standard'
                )
                # Compose prompt
                prompt = build_prompt(payload, variant=prompt_variant)
                # Generate with configured provider
                cfg = load_llm_config()
                client = LLMClient(cfg)

                # Determine effective provider/model to use for token estimation
                effective_provider = (llm_provider or cfg.provider or 'openai').lower()
                if effective_provider not in ('openai', 'gemini'):
                    effective_provider = 'openai' if cfg.openai_api_key else 'gemini'
                if effective_provider == 'openai':
                    effective_model = llm_model_override or cfg.openai_model or 'gpt-4o-mini'
                else:
                    effective_model = llm_model_override or cfg.gemini_model or 'gemini-2.5-pro'

                # Token counts for prompt and (later) output
                try:
                    from llm_analysis.token_utils import count_tokens as _count_tokens
                    prompt_tokens = _count_tokens(prompt, effective_provider, effective_model)
                except Exception:
                    prompt_tokens = 0

                raw = client.generate(prompt, provider=llm_provider, model_override=llm_model_override)

                try:
                    from llm_analysis.token_utils import count_tokens as _count_tokens
                    output_tokens = _count_tokens(raw or '', effective_provider, effective_model)
                except Exception:
                    output_tokens = 0

                json_obj, memo = parse_llm_output(raw)
                title = 'Strategy Settings Optimization — Executive Summary' if prompt_variant == 'analyst' else 'Risk-Focused Optimization'
                final_text = build_final_text(title, json_obj, memo)

                # Prepend usage header to the final text
                usage_header = (
                    f"Provider: {effective_provider} | Model: {effective_model} | "
                    f"Prompt tokens: {prompt_tokens} | Output tokens: {output_tokens} | Total: {prompt_tokens + output_tokens}"
                )
                final_text = usage_header + "\n\n" + final_text

                # Optionally write optimized YAML config derived from LLM JSON
                try:
                    # Save YAML into project config/llm_strategy_config (project root = parents[1])
                    proj_root = Path(__file__).resolve().parents[1]
                    yaml_dir = proj_root / 'config' / 'llm_strategy_config'
                    opt_yaml = self._write_llm_optimized_yaml(
                        base_strategy_config=strategy_config,
                        llm_json=json_obj,
                        symbol_short=symbol_short,
                        timeframe=timeframe,
                        output_dir=str(yaml_dir)
                    )
                except Exception:
                    opt_yaml = None

                llm_pdf_path = write_llm_pdf(
                    output_dir=output_dir,
                    filename_prefix=f"{symbol_short}_{timeframe}",
                    title=title,
                    text_body=final_text
                )
            except Exception as e:
                logger.error(f"LLM optimization step failed: {e}")

        return {
            'result': result,
            'reports': reports,
            'strategy_config': strategy_config,
            'data_df': data,
            'trades_df': trades_df,
            'equity_df': equity_df,
            'llm_pdf': llm_pdf_path
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