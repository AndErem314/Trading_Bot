"""
Main Workflow Controller for Ichimoku Cloud Trading Strategy

This module serves as the central orchestrator that connects all components
of the trading system into a seamless workflow. It manages the data pipeline,
strategy execution, backtesting, analysis, and reporting.

Features:
- Integrated data fetching and preprocessing
- Ichimoku indicator calculation and signal detection
- Strategy configuration and backtesting
- Performance analysis and optimization
- Report generation and visualization
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
import json
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import all required modules
from data_fetching import (
    OHLCVDataFetcher,
    DataPreprocessor,
    IchimokuCalculator,
    IchimokuSignalDetector
)
from config import StrategyConfigManager, StrategyConfig
from backtesting import IchimokuBacktester
from analytics import PerformanceAnalyzer
from optimization import ParameterOptimizer
from reporting import ReportGenerator
from visualization import ResultsVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for the workflow execution."""
    # Data configuration
    exchange: str = 'binance'
    symbol: str = 'BTC/USDT'
    timeframe: str = '1h'
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lookback_days: int = 365
    
    # Strategy configuration
    strategy_name: str = 'IchimokuCloud_Default'
    strategy_config_path: Optional[str] = None
    
    # Execution configuration (matching TradingView settings)
    initial_capital: float = 1000.0  # Fixed at 1000
    position_size_pct: float = 1.0   # 100% of equity
    pyramiding: int = 1              # Max 1 position
    commission: float = 0.001        # 0.1%
    slippage: float = 0.0003         # 3 ticks (approximate)
    slippage_ticks: int = 3          # Actual slippage in ticks
    
    # Optimization configuration
    enable_optimization: bool = False
    optimization_metric: str = 'sharpe_ratio'
    n_jobs: int = -1
    
    # Output configuration
    output_dir: str = './results'
    generate_report: bool = True
    report_formats: List[str] = field(default_factory=lambda: ['html', 'pdf'])
    show_plots: bool = True
    

@dataclass
class WorkflowResults:
    """Container for workflow execution results."""
    market_data: pd.DataFrame
    ichimoku_data: pd.DataFrame
    signals: pd.DataFrame
    backtest_results: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    optimization_results: Optional[Dict[str, Any]] = None
    report_paths: List[str] = field(default_factory=list)
    

class IchimokuWorkflowController:
    """
    Main controller that orchestrates the entire Ichimoku trading workflow.
    
    This class integrates all components of the trading system and provides
    a unified interface for executing backtests, optimizations, and analyses.
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        """
        Initialize the workflow controller.
        
        Args:
            config: Workflow configuration object
        """
        self.config = config or WorkflowConfig()
        self._validate_config()
        
        # Initialize components
        self.data_fetcher = None
        self.preprocessor = DataPreprocessor()
        self.ichimoku_calculator = IchimokuCalculator()
        self.signal_detector = IchimokuSignalDetector()
        self.strategy_config_manager = None
        self.backtester = None
        self.performance_analyzer = None
        self.optimizer = None
        self.report_generator = None
        self.visualizer = ResultsVisualizer()
        
        # Results storage
        self.results = None
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger.info(f"Workflow controller initialized for {self.config.symbol} on {self.config.timeframe}")
        
    def _validate_config(self):
        """Validate the workflow configuration."""
        if self.config.position_size_pct <= 0 or self.config.position_size_pct > 1:
            raise ValueError("Position size percentage must be between 0 and 1")
        
        if self.config.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
            
        # Set date range if not provided
        if not self.config.end_date:
            self.config.end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not self.config.start_date:
            end_dt = datetime.strptime(self.config.end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=self.config.lookback_days)
            self.config.start_date = start_dt.strftime('%Y-%m-%d')
    
    def execute_full_workflow(self) -> WorkflowResults:
        """
        Execute the complete workflow from data fetching to report generation.
        
        Returns:
            WorkflowResults object containing all results
        """
        logger.info("Starting full workflow execution")
        
        try:
            # Step 1: Fetch and prepare data
            logger.info("Step 1: Fetching and preparing data...")
            market_data = self._fetch_and_prepare_data()
            
            # Step 2: Calculate Ichimoku indicators
            logger.info("Step 2: Calculating Ichimoku indicators...")
            ichimoku_data = self._calculate_ichimoku(market_data)
            
            # Step 3: Detect signals
            logger.info("Step 3: Detecting trading signals...")
            signals = self._detect_signals(ichimoku_data)
            
            # Step 4: Load strategy configuration
            logger.info("Step 4: Loading strategy configuration...")
            strategy_config = self._load_strategy_config()
            
            # Step 5: Run backtest or optimization
            if self.config.enable_optimization:
                logger.info("Step 5: Running parameter optimization...")
                backtest_results, optimization_results = self._run_optimization(
                    ichimoku_data, signals, strategy_config
                )
            else:
                logger.info("Step 5: Running backtest...")
                backtest_results = self._run_backtest(ichimoku_data, signals, strategy_config)
                optimization_results = None
            
            # Step 6: Analyze performance
            logger.info("Step 6: Analyzing performance...")
            performance_metrics = self._analyze_performance(backtest_results)
            
            # Step 7: Generate reports and visualizations
            logger.info("Step 7: Generating reports and visualizations...")
            report_paths = self._generate_reports(
                backtest_results, performance_metrics, ichimoku_data, signals
            )
            
            # Store results
            self.results = WorkflowResults(
                market_data=market_data,
                ichimoku_data=ichimoku_data,
                signals=signals,
                backtest_results=backtest_results,
                performance_metrics=performance_metrics,
                optimization_results=optimization_results,
                report_paths=report_paths
            )
            
            logger.info("Workflow execution completed successfully!")
            self._print_summary()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise
    
    def _fetch_and_prepare_data(self) -> pd.DataFrame:
        """Fetch market data and preprocess it."""
        # Initialize data fetcher
        self.data_fetcher = OHLCVDataFetcher(
            exchange_id=self.config.exchange,
            cache_dir=os.path.join(self.config.output_dir, 'cache')
        )
        
        # Fetch data
        logger.info(f"Fetching {self.config.symbol} data from {self.config.start_date} to {self.config.end_date}")
        
        # Convert string dates to datetime objects
        start_dt = datetime.strptime(self.config.start_date, '%Y-%m-%d') if self.config.start_date else None
        end_dt = datetime.strptime(self.config.end_date, '%Y-%m-%d') if self.config.end_date else None
        
        # Calculate limit based on date range and timeframe
        if start_dt and end_dt:
            days_diff = (end_dt - start_dt).days
            timeframe_minutes = {'15m': 15, '1h': 60, '4h': 240, '1d': 1440}
            candles_per_day = 1440 / timeframe_minutes.get(self.config.timeframe, 60)
            limit = min(int(days_diff * candles_per_day), 1000)  # Cap at 1000
        else:
            limit = 500  # Default limit
        
        raw_data = self.data_fetcher.fetch_data(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            limit=limit,
            start_time=start_dt,
            end_time=end_dt
        )
        
        # Preprocess data
        logger.info("Preprocessing data...")
        clean_data = self.preprocessor.clean_data(
            raw_data,
            timeframe=self.config.timeframe
        )
        
        # Add additional features (if the method exists)
        processed_data = self._add_features(clean_data)
        
        logger.info(f"Data prepared: {len(processed_data)} candles")
        return processed_data
    
    def _add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add additional features to the data."""
        # Add simple returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Add ATR (Average True Range)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['atr'] = true_range.rolling(14).mean()
        
        # Add volume moving average
        data['volume_ma'] = data['volume'].rolling(20).mean()
        
        # Forward fill any NaN values
        data = data.fillna(method='ffill')
        
        return data
    
    def _calculate_ichimoku(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud indicators."""
        ichimoku_data = self.ichimoku_calculator.calculate_ichimoku(data)
        
        # Validate calculations
        required_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 
                        'senkou_span_b', 'chikou_span']
        missing_cols = [col for col in required_cols if col not in ichimoku_data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing Ichimoku indicators: {missing_cols}")
        
        logger.info("Ichimoku indicators calculated successfully")
        return ichimoku_data
    
    def _detect_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect Ichimoku trading signals."""
        signals = self.signal_detector.detect_all_signals(data)
        
        # Log signal summary
        signal_cols = [col for col in signals.columns if col.endswith('_signal')]
        total_signals = sum(signals[col].sum() for col in signal_cols)
        logger.info(f"Detected {total_signals} total signals across {len(signal_cols)} signal types")
        
        return signals
    
    def _load_strategy_config(self) -> StrategyConfig:
        """Load or create strategy configuration."""
        if self.config.strategy_config_path and os.path.exists(self.config.strategy_config_path):
            self.strategy_config_manager = StrategyConfigManager(self.config.strategy_config_path)
            strategy_config = self.strategy_config_manager.get_strategy(self.config.strategy_name)
            
            if not strategy_config:
                logger.warning(f"Strategy '{self.config.strategy_name}' not found in config file")
                strategy_config = self._create_default_config()
        else:
            logger.info("Using default Ichimoku strategy configuration")
            strategy_config = self._create_default_config()
        
        return strategy_config
    
    def _create_default_config(self) -> StrategyConfig:
        """Create default Ichimoku strategy configuration."""
        from config import (
            StrategyConfig, SignalConditions, IchimokuParameters,
            RiskManagement, PositionSizing
        )
        
        return StrategyConfig(
            name=self.config.strategy_name,
            description="Default Ichimoku Cloud strategy",
            enabled=True,
            
            signal_conditions=SignalConditions(
                buy_conditions=[
                    "PriceAboveCloud",
                    "TenkanAboveKijun",
                    "ChikouAbovePrice"
                ],
                buy_logic="AND",
                sell_conditions=[
                    "PriceBelowCloud",
                    "TenkanBelowKijun"
                ],
                sell_logic="OR"
            ),
            
            ichimoku_parameters=IchimokuParameters(
                tenkan_period=9,
                kijun_period=26,
                senkou_b_period=52,
                chikou_offset=26,
                senkou_offset=26
            ),
            
            risk_management=RiskManagement(
                stop_loss_pct=2.0,  # 2%
                take_profit_pct=6.0,  # 6%
                trailing_stop=False,
                trailing_stop_pct=1.5
            ),
            
            position_sizing=PositionSizing(
                method="fixed",
                fixed_size=self.config.position_size_pct,  # 1.0 = 100% of equity
                max_leverage=1.0,
                min_position_size=0.01,
                max_position_size=1.0  # Max 100% of equity
            )
        )
    
    def _run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame, 
                     strategy_config: StrategyConfig) -> Dict[str, Any]:
        """Run backtest with the configured strategy."""
        # Initialize backtester
        self.backtester = IchimokuBacktester(
            initial_capital=self.config.initial_capital,
            commission_rate=self.config.commission,
            slippage_rate=self.config.slippage,
            enable_shorting=False
        )
        
        # For now, create a simple backtest using the signals directly
        # This is a simplified approach that processes signals to generate trades
        trades = []
        positions = []
        equity_curve = []
        current_balance = self.config.initial_capital
        position = None
        open_positions = 0  # Track number of open positions for pyramiding
        
        # Combine signals based on strategy config
        buy_signals = self._combine_signals(signals, strategy_config.signal_conditions.buy_conditions, 
                                          strategy_config.signal_conditions.buy_logic)
        sell_signals = self._combine_signals(signals, strategy_config.signal_conditions.sell_conditions,
                                           strategy_config.signal_conditions.sell_logic)
        
        for i in range(len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            
            # Check for entry signal (respect pyramiding limit)
            if buy_signals.iloc[i] and open_positions < self.config.pyramiding:
                # Enter long position
                # Calculate position size based on current equity (100% of available capital)
                position_value = current_balance * strategy_config.position_sizing.fixed_size
                position_size = position_value / current_price
                
                # Apply commission on entry
                entry_commission = position_value * self.config.commission
                current_balance -= entry_commission
                
                position = {
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'size': position_size,
                    'side': 'long',
                    'entry_commission': entry_commission
                }
                positions.append(position)
                open_positions += 1
                
            # Check for exit signal or stop loss/take profit
            elif position is not None:
                exit_trade = False
                exit_reason = ''
                
                # Check sell signal
                if sell_signals.iloc[i]:
                    exit_trade = True
                    exit_reason = 'signal'
                    
                # Check stop loss
                elif position['side'] == 'long' and current_price <= position['entry_price'] * (1 - strategy_config.risk_management.stop_loss_pct):
                    exit_trade = True
                    exit_reason = 'stop_loss'
                    
                # Check take profit
                elif position['side'] == 'long' and current_price >= position['entry_price'] * (1 + strategy_config.risk_management.take_profit_pct):
                    exit_trade = True
                    exit_reason = 'take_profit'
                
                if exit_trade:
                    # Apply slippage on exit (adverse price movement)
                    slippage_amount = current_price * self.config.slippage
                    exit_price = current_price - slippage_amount  # Worse price for long exit
                    
                    # Calculate P&L
                    gross_pnl = (exit_price - position['entry_price']) * position['size']
                    
                    # Apply commission on exit
                    exit_commission = position['size'] * exit_price * self.config.commission
                    
                    # Total commission (entry + exit)
                    total_commission = position['entry_commission'] + exit_commission
                    
                    # Net P&L after commissions
                    net_pnl = gross_pnl - exit_commission
                    
                    # Update balance (add back position value + P&L)
                    current_balance += (position['size'] * exit_price) - exit_commission
                    
                    # Record trade
                    trades.append({
                        'entry_time': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'size': position['size'],
                        'side': position['side'],
                        'gross_profit': gross_pnl,
                        'commission': total_commission,
                        'profit': net_pnl,
                        'profit_pct': net_pnl / (position['entry_price'] * position['size']),
                        'exit_reason': exit_reason
                    })
                    
                    position = None
                    open_positions -= 1
            
            # Record equity
            equity_value = current_balance
            if position:
                # Add unrealized P&L
                unrealized_pnl = (current_price - position['entry_price']) * position['size']
                equity_value += unrealized_pnl
                
            equity_curve.append({
                'timestamp': current_time,
                'total_value': equity_value,
                'cash': current_balance
            })
        
        # Close any open position at end
        if position is not None:
            final_price = data['close'].iloc[-1]
            pnl = (final_price - position['entry_price']) * position['size']
            pnl_after_commission = pnl - (position['size'] * final_price * self.config.commission * 2)
            current_balance += pnl_after_commission
            
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': data.index[-1],
                'exit_price': final_price,
                'size': position['size'],
                'side': position['side'],
                'profit': pnl_after_commission,
                'profit_pct': pnl_after_commission / (position['entry_price'] * position['size']),
                'exit_reason': 'end_of_data'
            })
        
        results = {
            'trades': trades,
            'positions': positions,
            'equity_curve': equity_curve,
            'final_balance': current_balance,
            'initial_capital': self.config.initial_capital
        }
        
        logger.info(f"Backtest completed: {len(trades)} trades executed")
        return results
    
    def _combine_signals(self, signals: pd.DataFrame, conditions: List[str], logic: str) -> pd.Series:
        """Combine multiple signal conditions based on logic (AND/OR)."""
        if not conditions:
            return pd.Series(False, index=signals.index)
        
        # Map condition names to signal columns
        signal_mapping = {
            'PriceAboveCloud': 'price_above_cloud_signal',
            'PriceBelowCloud': 'price_below_cloud_signal',
            'TenkanAboveKijun': 'tenkan_above_kijun_signal',
            'TenkanBelowKijun': 'tenkan_below_kijun_signal',
            'ChikouAbovePrice': 'chikou_above_price_signal',
            'ChikouBelowPrice': 'chikou_below_price_signal',
            'ChikouAboveCloud': 'chikou_above_cloud_signal',
            'ChikouBelowCloud': 'chikou_below_cloud_signal'
        }
        
        # Get relevant signal columns
        signal_series = []
        for condition in conditions:
            if condition in signal_mapping and signal_mapping[condition] in signals.columns:
                signal_series.append(signals[signal_mapping[condition]])
        
        if not signal_series:
            return pd.Series(False, index=signals.index)
        
        # Combine based on logic
        if logic == 'AND':
            combined = signal_series[0]
            for series in signal_series[1:]:
                combined = combined & series
            return combined
        else:  # OR logic
            combined = signal_series[0]
            for series in signal_series[1:]:
                combined = combined | series
            return combined
    
    def _run_optimization(self, data: pd.DataFrame, signals: pd.DataFrame,
                         base_config: StrategyConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run parameter optimization."""
        # Initialize optimizer with correct parameters
        self.optimizer = ParameterOptimizer(
            db_path=os.path.join(self.config.output_dir, "optimization_results.db"),
            min_sample_trades=30,
            significance_level=0.05,
            n_jobs=self.config.n_jobs if hasattr(self.config, 'n_jobs') else -1
        )
        
        # Define parameter space using ParameterSpace class
        from optimization.parameter_optimizer import ParameterSpace
        param_space = ParameterSpace(
            tenkan_periods=[7, 9, 12],
            kijun_periods=[20, 26, 30],
            senkou_b_periods=[44, 52, 60],
            stop_loss_percent=[0.01, 0.02, 0.03],
            take_profit_percent=[0.04, 0.06, 0.08]
        )
        
        # Run optimization
        logger.info("Starting parameter optimization...")
        # Get date range from data
        start_date = data.index[0].strftime('%Y-%m-%d')
        end_date = data.index[-1].strftime('%Y-%m-%d')
        
        optimization_results = self.optimizer.grid_search_parameters(
            symbol=self.config.symbol,
            start_date=start_date,
            end_date=end_date,
            parameter_space=param_space,
            parallel=False  # Disable parallel to avoid import issues in multiprocessing
        )
        
        # Get best parameters (first result is best due to sorting)
        if optimization_results:
            best_result = optimization_results[0]
            best_params = best_result.parameters
            all_results = optimization_results
        else:
            logger.warning("No optimization results found")
            best_params = {}
            all_results = []
        
        # Run backtest with best parameters
        optimized_config = self._update_config_with_params(base_config, best_params)
        backtest_results = self._run_backtest(data, signals, optimized_config)
        
        optimization_results = {
            'best_params': best_params,
            'all_results': all_results,
            'optimized_config': optimized_config
        }
        
        return backtest_results, optimization_results
    
    def _update_config_with_params(self, config: StrategyConfig, params: Dict) -> StrategyConfig:
        """Update strategy config with optimized parameters."""
        # Create a copy of the config
        updated_config = StrategyConfig(**config.dict())
        
        # Update parameters
        if 'tenkan_period' in params:
            updated_config.ichimoku_parameters.tenkan_period = params['tenkan_period']
        if 'kijun_period' in params:
            updated_config.ichimoku_parameters.kijun_period = params['kijun_period']
        if 'senkou_b_period' in params:
            updated_config.ichimoku_parameters.senkou_b_period = params['senkou_b_period']
        if 'stop_loss_pct' in params:
            updated_config.risk_management.stop_loss_pct = params['stop_loss_pct']
        if 'take_profit_pct' in params:
            updated_config.risk_management.take_profit_pct = params['take_profit_pct']
        
        return updated_config
    
    def _analyze_performance(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze backtest performance."""
        # Initialize analyzer
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Calculate metrics
        trades_df = pd.DataFrame(backtest_results['trades'])
        equity_curve = pd.DataFrame(backtest_results['equity_curve'])
        
        metrics = self.performance_analyzer.calculate_all_metrics(trades_df, equity_curve)
        
        # Convert to dictionary for easy access
        metrics_dict = {
            'returns': {
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'cumulative_return': metrics.cumulative_return
            },
            'risk': {
                'volatility': metrics.annualized_volatility,
                'max_drawdown': metrics.max_drawdown_pct,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio
            },
            'trades': {
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'avg_win': metrics.avg_win,
                'avg_loss': metrics.avg_loss
            }
        }
        
        return metrics_dict
    
    def _generate_reports(self, backtest_results: Dict[str, Any],
                         performance_metrics: Dict[str, Any],
                         data: pd.DataFrame, signals: pd.DataFrame) -> List[str]:
        """Generate reports and visualizations."""
        if not self.config.generate_report:
            return []
        
        # Initialize report generator
        self.report_generator = ReportGenerator(
            output_dir=self.config.output_dir
        )
        
        # Create report filename
        report_name = f"{self.config.symbol.replace('/', '_')}_{self.config.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report_paths = []
        
        # Generate visualizations
        if self.config.show_plots:
            try:
                # Equity curve (matplotlib figure)
                equity_fig = self.performance_analyzer.plot_equity_curve()
                equity_path = os.path.join(self.config.output_dir, 'equity_curve.png')
                if equity_fig:
                    equity_fig.savefig(equity_path)
                    logger.info(f"Saved equity curve to {equity_path}")
                    # Close the figure to free memory
                    import matplotlib.pyplot as plt
                    plt.close(equity_fig)
            except Exception as e:
                logger.error(f"Failed to generate equity curve: {e}")
            
            try:
                # Ichimoku chart with signals (plotly figure)
                ichimoku_fig = self.visualizer.plot_trading_chart(
                    data=data,
                    trades=pd.DataFrame(backtest_results.get('trades', [])) if backtest_results.get('trades') else pd.DataFrame(),
                    ichimoku_data=data  # The data already contains ichimoku indicators
                )
                
                # Save plotly figure
                ichimoku_path = os.path.join(self.config.output_dir, 'ichimoku_chart.html')
                if ichimoku_fig:
                    ichimoku_fig.write_html(ichimoku_path)
                    logger.info(f"Saved Ichimoku chart to {ichimoku_path}")
                    
                    # Also save as static image if kaleido is installed
                    try:
                        ichimoku_png_path = os.path.join(self.config.output_dir, 'ichimoku_chart.png')
                        ichimoku_fig.write_image(ichimoku_png_path)
                        logger.info(f"Saved Ichimoku chart image to {ichimoku_png_path}")
                    except Exception:
                        logger.warning("Could not save static image. Install kaleido for static image export.")
            except Exception as e:
                logger.error(f"Failed to generate Ichimoku chart: {e}")
        
        # Prepare data for report generator
        # Convert equity curve to proper format
        equity_curve_list = backtest_results.get('equity_curve', [])
        if equity_curve_list:
            equity_df = pd.DataFrame(equity_curve_list)
            # Rename 'total_value' to 'equity' as expected by ReportGenerator
            if 'total_value' in equity_df.columns:
                equity_df['equity'] = equity_df['total_value']
            # Ensure timestamp is the index
            if 'timestamp' in equity_df.columns:
                equity_df.set_index('timestamp', inplace=True)
        else:
            equity_df = pd.DataFrame()
            
        report_data = {
            'data': data,
            'trades': pd.DataFrame(backtest_results['trades']) if backtest_results['trades'] else pd.DataFrame(),
            'equity_curve': equity_df,
            'metrics': {
                'performance_metrics': performance_metrics
            },
            'strategy_config': {
                'name': self.config.strategy_name,
                'symbol': self.config.symbol,
                'timeframe': self.config.timeframe
            }
        }
        
        # Generate reports using the correct method
        if self.config.report_formats:
            # Determine format string for generate_backtest_report
            if 'html' in self.config.report_formats and 'pdf' in self.config.report_formats:
                format_str = 'all'
            elif 'html' in self.config.report_formats:
                format_str = 'html'
            elif 'pdf' in self.config.report_formats:
                format_str = 'pdf'
            elif 'json' in self.config.report_formats:
                format_str = 'json'
            else:
                format_str = 'all'
            
            # Generate reports
            generated_reports = self.report_generator.generate_backtest_report(
                results=report_data,
                format=format_str,
                filename_prefix=report_name
            )
            
            # Collect report paths
            for fmt, path in generated_reports.items():
                if isinstance(path, list):  # CSV returns a list
                    report_paths.extend(path)
                else:
                    report_paths.append(path)
        
        logger.info(f"Generated {len(report_paths)} reports")
        return report_paths
    
    def _print_summary(self):
        """Print a summary of the workflow results."""
        if not self.results:
            return
        
        print("\n" + "="*60)
        print("WORKFLOW EXECUTION SUMMARY")
        print("="*60)
        print(f"Symbol: {self.config.symbol}")
        print(f"Timeframe: {self.config.timeframe}")
        print(f"Period: {self.config.start_date} to {self.config.end_date}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        
        if self.results.backtest_results:
            trades = self.results.backtest_results.get('trades', [])
            final_balance = self.results.backtest_results.get('final_balance', 0)
            
            print(f"\nBacktest Results:")
            print(f"  Total Trades: {len(trades)}")
            print(f"  Final Balance: ${final_balance:,.2f}")
            
        if self.results.performance_metrics:
            metrics = self.results.performance_metrics
            
            print(f"\nPerformance Metrics:")
            print(f"  Total Return: {metrics['returns']['total_return']:.2%}")
            print(f"  Sharpe Ratio: {metrics['risk']['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {metrics['risk']['max_drawdown']:.2%}")
            print(f"  Win Rate: {metrics['trades']['win_rate']:.2%}")
            
        if self.results.optimization_results:
            best_params = self.results.optimization_results.get('best_params', {})
            print(f"\nOptimization Results:")
            print(f"  Best Parameters: {best_params}")
            
        if self.results.report_paths:
            print(f"\nGenerated Reports:")
            for path in self.results.report_paths:
                print(f"  - {path}")
        
        print("="*60 + "\n")
    
    def save_session(self, session_name: str):
        """Save the current workflow session."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        session_path = os.path.join(self.config.output_dir, f"{session_name}_session.pkl")
        
        # Save using pickle or joblib
        import pickle
        with open(session_path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'results': self.results
            }, f)
        
        logger.info(f"Session saved to {session_path}")
    
    def load_session(self, session_path: str):
        """Load a previous workflow session."""
        import pickle
        
        with open(session_path, 'rb') as f:
            session_data = pickle.load(f)
        
        self.config = session_data['config']
        self.results = session_data['results']
        
        logger.info(f"Session loaded from {session_path}")


# Convenience functions
def create_workflow(config_dict: Dict[str, Any]) -> IchimokuWorkflowController:
    """Create a workflow controller from a configuration dictionary."""
    config = WorkflowConfig(**config_dict)
    return IchimokuWorkflowController(config)


def run_quick_backtest(symbol: str, timeframe: str = '1h', 
                      lookback_days: int = 365) -> WorkflowResults:
    """Run a quick backtest with default settings."""
    config = WorkflowConfig(
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
        generate_report=True,
        show_plots=True
    )
    
    controller = IchimokuWorkflowController(config)
    return controller.execute_full_workflow()


if __name__ == "__main__":
    # Example usage
    config = WorkflowConfig(
        symbol='BTC/USDT',
        timeframe='4h',
        lookback_days=180,
        enable_optimization=False,
        generate_report=True,
        show_plots=True
    )
    
    controller = IchimokuWorkflowController(config)
    results = controller.execute_full_workflow()