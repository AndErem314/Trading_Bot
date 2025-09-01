"""
Meta Strategy Orchestrator

A sophisticated orchestration system that integrates multiple trading strategies
with market regime detection across different timeframes and symbols.

This class serves as the central coordination point for:
- Multi-timeframe market regime detection
- Multiple trading strategy execution
- Data management and caching
- Risk management coordination

Author: Trading Bot Team
Date: 2024
"""

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from contextlib import contextmanager
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

# Import our market regime detector
from backend.Indicators import MarketRegimeDetector

# Import strategy classes (these are assumed to exist)
from backend.Strategies import (
    MACDMomentumCrossoverStrategy,
    RSIMomentumDivergenceSwingStrategy,
    BollingerBandsMeanReversionStrategy,
    IchimokuCloudBreakoutStrategy,
    ParabolicSARTrendFollowingStrategy,
    FibonacciRetracementSupportResistanceStrategy,
    GaussianChannelBreakoutMeanReversionStrategy,
    SMAGoldenCrossStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetaStrategyOrchestrator:
    """
    Meta Strategy Orchestrator for multi-strategy, multi-timeframe trading.
    
    This class coordinates multiple trading strategies across different symbols
    and timeframes, integrating market regime detection to optimize strategy
    selection and risk management.
    
    Attributes:
        db_connection_string (str): SQLAlchemy connection string
        symbols (List[str]): List of trading symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
        lookback_period (int): Number of periods to fetch for historical data
        regime_detectors (Dict): Nested dict of regime detectors by symbol and timeframe
        strategies (Dict): Dictionary of initialized strategy instances
        engine: SQLAlchemy engine for database connections
    """
    
    # Timeframe mappings
    TIMEFRAMES = ['D1', 'H4', 'H1']
    TIMEFRAME_MAP = {
        'D1': '1d',
        'H4': '4h', 
        'H1': '1h'
    }
    
    # Strategy configuration
    STRATEGY_CLASSES = {
        'macd_momentum': MACDMomentumCrossoverStrategy,
        'rsi_divergence': RSIMomentumDivergenceSwingStrategy,
        'bollinger_bands': BollingerBandsMeanReversionStrategy,
        'ichimoku_cloud': IchimokuCloudBreakoutStrategy,
        'parabolic_sar': ParabolicSARTrendFollowingStrategy,
        'fibonacci_retracement': FibonacciRetracementSupportResistanceStrategy,
        'gaussian_channel': GaussianChannelBreakoutMeanReversionStrategy,
        'sma_golden_cross': SMAGoldenCrossStrategy
    }
    
    # Strategy weighting matrix based on market bias
    STRATEGY_WEIGHTS = {
        'Strong Bullish': {
            'macd_momentum': 0.8,
            'ichimoku_cloud': 0.9,
            'parabolic_sar': 0.8,
            'sma_golden_cross': 0.7,
            'rsi_divergence': 0.3,
            'bollinger_bands': 0.0,  # No mean reversion in strong trends
            'fibonacci_retracement': 0.2,
            'gaussian_channel': 0.1
        },
        'Bullish': {
            'macd_momentum': 0.6,
            'ichimoku_cloud': 0.7,
            'parabolic_sar': 0.6,
            'sma_golden_cross': 0.5,
            'rsi_divergence': 0.4,
            'bollinger_bands': 0.2,
            'fibonacci_retracement': 0.3,
            'gaussian_channel': 0.2
        },
        'Neutral': {
            'macd_momentum': 0.1,
            'ichimoku_cloud': 0.1,
            'parabolic_sar': 0.1,
            'sma_golden_cross': 0.1,
            'rsi_divergence': 0.7,  # Good for ranging markets
            'bollinger_bands': 0.8,  # Excellent for ranges
            'fibonacci_retracement': 0.6,
            'gaussian_channel': 0.7
        },
        'Bearish': {
            'macd_momentum': 0.6,  # Can catch downtrends
            'ichimoku_cloud': 0.7,
            'parabolic_sar': 0.6,
            'sma_golden_cross': 0.5,
            'rsi_divergence': 0.4,
            'bollinger_bands': 0.2,
            'fibonacci_retracement': 0.3,
            'gaussian_channel': 0.2
        },
        'Strong Bearish': {
            'macd_momentum': 0.8,
            'ichimoku_cloud': 0.9,
            'parabolic_sar': 0.8,
            'sma_golden_cross': 0.7,
            'rsi_divergence': 0.3,
            'bollinger_bands': 0.0,  # No mean reversion in strong trends
            'fibonacci_retracement': 0.2,
            'gaussian_channel': 0.1
        },
        'Crash': {
            # Only short-biased or protective strategies
            'macd_momentum': 1.0,  # For short signals
            'ichimoku_cloud': 1.0,
            'parabolic_sar': 1.0,
            'sma_golden_cross': 0.0,  # Disable long-only strategies
            'rsi_divergence': 0.0,    # Too risky in crash
            'bollinger_bands': 0.0,
            'fibonacci_retracement': 0.0,
            'gaussian_channel': 0.0
        }
    }
    
    # Trading configuration
    SIGNAL_THRESHOLD = 0.6  # Minimum absolute weighted signal to trade
    MAX_RISK_PER_TRADE = 0.01  # 1% of portfolio per trade
    STOP_LOSS_ATR_MULTIPLIER = 2.0  # Stop loss at 2x ATR
    
    def __init__(
        self, 
        db_connection_string: str,
        symbols: List[str] = None,
        lookback_period: int = 500
    ):
        """
        Initialize the Meta Strategy Orchestrator.
        
        Args:
            db_connection_string: SQLAlchemy database connection string
            symbols: List of trading symbols (default: ['BTC/USDT', 'ETH/USDT'])
            lookback_period: Number of historical periods to fetch (default: 500)
        """
        # Store configuration
        self.db_connection_string = db_connection_string
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        self.lookback_period = lookback_period
        
        # Initialize containers
        self.regime_detectors: Dict[str, Dict[str, MarketRegimeDetector]] = {}
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Initialize regime detector structure
        for symbol in self.symbols:
            self.regime_detectors[symbol] = {}
            self.data_cache[symbol] = {}
        
        # Create database engine with connection pooling
        self.engine = self._create_engine()
        
        logger.info(
            f"MetaStrategyOrchestrator initialized with symbols: {self.symbols}, "
            f"lookback: {self.lookback_period} periods"
        )
    
    def _create_engine(self):
        """Create SQLAlchemy engine with connection pooling."""
        try:
            engine = create_engine(
                self.db_connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=3600
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database engine created successfully")
            return engine
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def fetch_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol and timeframe.
        
        This method connects to the database and retrieves historical price data,
        handling the mapping between our timeframe notation and database tables.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe code ('D1', 'H4', or 'H1')
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index, sorted ascending
            
        Raises:
            ValueError: If invalid symbol or timeframe
            SQLAlchemyError: If database query fails
        """
        # Validate inputs
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not in configured symbols")
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Invalid timeframe {timeframe}. Must be one of {self.TIMEFRAMES}")
        
        # Check cache first
        if symbol in self.data_cache and timeframe in self.data_cache[symbol]:
            logger.debug(f"Returning cached data for {symbol} {timeframe}")
            return self.data_cache[symbol][timeframe].copy()
        
        # Map to database format
        db_timeframe = self.TIMEFRAME_MAP[timeframe]
        asset_name = symbol.split('/')[0]  # Extract 'BTC' from 'BTC/USDT'
        
        logger.info(f"Fetching data for {symbol} {timeframe} (last {self.lookback_period} periods)")
        
        # Build query based on your database structure
        query = text("""
            SELECT 
                o.timestamp,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume
            FROM ohlcv_data o
            JOIN symbols s ON o.symbol_id = s.id
            JOIN timeframes t ON o.timeframe_id = t.id
            WHERE s.symbol = :symbol
            AND t.timeframe = :timeframe
            ORDER BY o.timestamp DESC
            LIMIT :limit
        """)
        
        try:
            with self.get_db_connection() as conn:
                df = pd.read_sql(
                    query,
                    conn,
                    params={
                        'symbol': symbol,
                        'timeframe': db_timeframe,
                        'limit': self.lookback_period
                    }
                )
                
            if df.empty:
                logger.warning(f"No data found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Process the dataframe
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            df.set_index('timestamp', inplace=True)
            df.sort_index(ascending=True, inplace=True)  # Sort ascending for indicators
            
            # Validate data quality
            self._validate_data(df, symbol, timeframe)
            
            # Cache the data
            self.data_cache[symbol][timeframe] = df.copy()
            
            logger.info(
                f"Successfully fetched {len(df)} rows for {symbol} {timeframe}. "
                f"Date range: {df.index[0]} to {df.index[-1]}"
            )
            
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"Database error fetching {symbol} {timeframe}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol} {timeframe}: {e}")
            raise
    
    def _validate_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Validate the quality of fetched data."""
        # Check for missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning(
                f"Found null values in {symbol} {timeframe} data: "
                f"{null_counts[null_counts > 0].to_dict()}"
            )
        
        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                logger.error(f"Found zero or negative prices in {col} for {symbol} {timeframe}")
                raise ValueError(f"Invalid price data in {col}")
        
        # Check high/low consistency
        if not (df['high'] >= df['low']).all():
            logger.error(f"High < Low detected in {symbol} {timeframe} data")
            raise ValueError("Invalid OHLC data: High < Low")
    
    def setup(self):
        """
        Initialize all components of the orchestrator.
        
        This method:
        1. Fetches data for all symbols and timeframes
        2. Initializes regime detectors for each symbol/timeframe combination
        3. Initializes trading strategies with appropriate data
        
        This should be called once after instantiation before running the main loop.
        """
        logger.info("Starting MetaStrategyOrchestrator setup...")
        start_time = datetime.now()
        
        # Step 1: Fetch all data
        logger.info("Step 1: Fetching data for all symbols and timeframes...")
        for symbol in self.symbols:
            for timeframe in self.TIMEFRAMES:
                try:
                    self.fetch_data(symbol, timeframe)
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol} {timeframe}: {e}")
                    # Continue with other symbols/timeframes
        
        # Step 2: Initialize regime detectors
        logger.info("Step 2: Initializing regime detectors...")
        self._initialize_regime_detectors()
        
        # Step 3: Initialize trading strategies
        logger.info("Step 3: Initializing trading strategies...")
        self._initialize_strategies()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Setup completed in {elapsed:.2f} seconds")
    
    def _initialize_regime_detectors(self):
        """Initialize regime detectors for each symbol and timeframe."""
        for symbol in self.symbols:
            asset_name = symbol.split('/')[0]
            
            for timeframe in self.TIMEFRAMES:
                # Skip if no data available
                if timeframe not in self.data_cache.get(symbol, {}):
                    logger.warning(f"No data for {symbol} {timeframe}, skipping regime detector")
                    continue
                
                df = self.data_cache[symbol][timeframe]
                
                try:
                    # Get benchmark data if not BTC
                    benchmark_df = None
                    if asset_name != 'BTC' and 'BTC/USDT' in self.symbols:
                        if timeframe in self.data_cache.get('BTC/USDT', {}):
                            benchmark_df = self.data_cache['BTC/USDT'][timeframe]
                    
                    # Initialize regime detector
                    detector = MarketRegimeDetector(
                        df=df,
                        asset_name=asset_name,
                        benchmark_df=benchmark_df
                    )
                    
                    self.regime_detectors[symbol][timeframe] = detector
                    
                    # Log current regime
                    regime, metrics = detector.classify_regime()
                    logger.info(
                        f"Initialized {symbol} {timeframe} regime detector. "
                        f"Current regime: {regime}"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to initialize regime detector for {symbol} {timeframe}: {e}")
    
    def _initialize_strategies(self):
        """Initialize all trading strategy instances."""
        # Use H4 timeframe as the primary timeframe for strategies
        primary_timeframe = 'H4'
        
        for symbol in self.symbols:
            self.strategies[symbol] = {}
            
            # Check if we have data for the primary timeframe
            if primary_timeframe not in self.data_cache.get(symbol, {}):
                logger.warning(f"No {primary_timeframe} data for {symbol}, skipping strategies")
                continue
            
            df = self.data_cache[symbol][primary_timeframe]
            
            # Initialize each strategy
            for strategy_name, strategy_class in self.STRATEGY_CLASSES.items():
                try:
                    # Create strategy instance
                    strategy = strategy_class(
                        data=df,
                        symbol=symbol,
                        timeframe=primary_timeframe
                    )
                    
                    self.strategies[symbol][strategy_name] = strategy
                    
                    # Generate initial signals
                    signals = strategy.generate_signals()
                    last_signal = signals.iloc[-1] if len(signals) > 0 else 'No signal'
                    
                    logger.info(
                        f"Initialized {strategy_name} for {symbol}. "
                        f"Last signal: {last_signal}"
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Failed to initialize {strategy_name} for {symbol}: {e}"
                    )
    
    def get_current_regimes(self, symbol: str) -> Dict[str, str]:
        """
        Get current market regimes for all timeframes of a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary mapping timeframe to current regime
        """
        regimes = {}
        
        for timeframe in self.TIMEFRAMES:
            if symbol in self.regime_detectors and timeframe in self.regime_detectors[symbol]:
                try:
                    regime, _ = self.regime_detectors[symbol][timeframe].classify_regime()
                    regimes[timeframe] = regime
                except Exception as e:
                    logger.error(f"Error getting regime for {symbol} {timeframe}: {e}")
                    regimes[timeframe] = "Error"
            else:
                regimes[timeframe] = "Not initialized"
        
        return regimes
    
    def get_strategy_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Get current signals from all strategies for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary mapping strategy name to current signal
        """
        signals = {}
        
        if symbol not in self.strategies:
            return signals
        
        for strategy_name, strategy in self.strategies[symbol].items():
            try:
                strategy_signals = strategy.generate_signals()
                if not strategy_signals.empty:
                    signals[strategy_name] = {
                        'signal': strategy_signals.iloc[-1],
                        'confidence': getattr(strategy, 'get_confidence', lambda: 0.5)()
                    }
            except Exception as e:
                logger.error(f"Error getting signal from {strategy_name} for {symbol}: {e}")
                signals[strategy_name] = {'signal': 'Error', 'confidence': 0}
        
        return signals
    
    def determine_overall_bias(self, symbol: str) -> tuple[str, Dict[str, str]]:
        """
        Determine overall market bias by synthesizing regime information from multiple timeframes.
        
        This method analyzes regimes from D1, H4, and H1 timeframes to produce a single
        actionable bias for trading decisions.
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            tuple: (overall_bias, bias_metadata)
                - overall_bias: One of ['Strong Bullish', 'Bullish', 'Neutral', 
                                       'Bearish', 'Strong Bearish', 'Crash']
                - bias_metadata: Dict with regime for each timeframe
                
        Bias Logic:
            - Crash: Any timeframe shows "Crypto Crash" (highest priority)
            - Strong Bullish: D1 Bull Trend AND H4 Bull Trend
            - Bullish: D1 Bull Trend AND H4 Ranging/High Volatility
            - Neutral: D1 Ranging
            - Bearish: D1 Bear Trend AND H4 Ranging/High Volatility
            - Strong Bearish: D1 Bear Trend AND H4 Bear Trend
        """
        # Initialize metadata
        bias_metadata = {
            'D1': 'Unknown',
            'H4': 'Unknown', 
            'H1': 'Unknown'
        }
        
        # Validate symbol
        if symbol not in self.symbols:
            logger.error(f"Symbol {symbol} not in configured symbols")
            return ('Unknown', bias_metadata)
        
        # Check if regime detectors are initialized
        if symbol not in self.regime_detectors:
            logger.error(f"No regime detectors initialized for {symbol}")
            return ('Unknown', bias_metadata)
        
        # Collect regimes from all timeframes
        for timeframe in self.TIMEFRAMES:
            try:
                if timeframe in self.regime_detectors[symbol]:
                    detector = self.regime_detectors[symbol][timeframe]
                    regime, _ = detector.classify_regime()
                    bias_metadata[timeframe] = regime
                else:
                    logger.warning(f"No regime detector for {symbol} {timeframe}")
                    bias_metadata[timeframe] = 'Not initialized'
            except Exception as e:
                logger.error(f"Error getting regime for {symbol} {timeframe}: {e}")
                bias_metadata[timeframe] = 'Error'
        
        # Extract individual regimes
        d1_regime = bias_metadata['D1']
        h4_regime = bias_metadata['H4']
        h1_regime = bias_metadata['H1']
        
        # PRIORITY 1: Check for crash in any timeframe
        if 'Crypto Crash / Panic' in [d1_regime, h4_regime, h1_regime]:
            overall_bias = 'Crash'
            logger.info(f"{symbol} Overall Bias: {overall_bias} (Crash detected!) | Regimes: {bias_metadata}")
            return (overall_bias, bias_metadata)
        
        # PRIORITY 2: Analyze based on D1 and H4 alignment
        # Define regime categories for cleaner logic
        ranging_or_volatile = ['Ranging / Accumulation', 'High Volatility / Breakout', 
                               'Transitioning / Unknown']
        
        # Determine bias based on rules
        if d1_regime == 'Bull Trend':
            if h4_regime == 'Bull Trend':
                overall_bias = 'Strong Bullish'
            elif h4_regime in ranging_or_volatile:
                overall_bias = 'Bullish'
            else:  # H4 is bearish
                overall_bias = 'Bullish'  # D1 takes precedence
                
        elif d1_regime == 'Bear Trend':
            if h4_regime == 'Bear Trend':
                overall_bias = 'Strong Bearish'
            elif h4_regime in ranging_or_volatile:
                overall_bias = 'Bearish'
            else:  # H4 is bullish
                overall_bias = 'Bearish'  # D1 takes precedence
                
        elif d1_regime == 'Ranging / Accumulation':
            overall_bias = 'Neutral'
            
        elif d1_regime in ['High Volatility / Breakout', 'Transitioning / Unknown']:
            # When D1 is uncertain, look at H4 for guidance
            if h4_regime == 'Bull Trend':
                overall_bias = 'Bullish'
            elif h4_regime == 'Bear Trend':
                overall_bias = 'Bearish'
            else:
                overall_bias = 'Neutral'
        else:
            # Fallback for any unhandled cases
            overall_bias = 'Neutral'
        
        # Log the result
        logger.info(
            f"{symbol} Overall Bias: {overall_bias} | "
            f"D1: {d1_regime}, H4: {h4_regime}, H1: {h1_regime}"
        )
        
        return (overall_bias, bias_metadata)
    
    def refresh_data(self, symbol: str = None, timeframe: str = None):
        """
        Refresh data for specific symbol/timeframe or all.
        
        Args:
            symbol: Specific symbol to refresh (None for all)
            timeframe: Specific timeframe to refresh (None for all)
        """
        symbols = [symbol] if symbol else self.symbols
        timeframes = [timeframe] if timeframe else self.TIMEFRAMES
        
        logger.info(f"Refreshing data for symbols: {symbols}, timeframes: {timeframes}")
        
        for sym in symbols:
            for tf in timeframes:
                # Clear cache
                if sym in self.data_cache and tf in self.data_cache[sym]:
                    del self.data_cache[sym][tf]
                
                # Fetch fresh data
                try:
                    self.fetch_data(sym, tf)
                except Exception as e:
                    logger.error(f"Failed to refresh {sym} {tf}: {e}")
    
    def _get_strategy_signal(self, strategy) -> float:
        """
        Get normalized signal from a strategy.
        
        Strategies should implement get_signal() returning -1 to +1.
        This is a wrapper to handle different strategy interfaces.
        
        Args:
            strategy: Strategy instance
            
        Returns:
            float: Signal value between -1 (strong short) and +1 (strong long)
        """
        try:
            # Try to get signal using standard method
            if hasattr(strategy, 'get_signal'):
                return strategy.get_signal()
            
            # Fallback: interpret generate_signals() output
            signals = strategy.generate_signals()
            if signals.empty:
                return 0.0
            
            last_signal = signals.iloc[-1]
            
            # Convert various signal formats to -1 to +1 range
            if isinstance(last_signal, (int, float)):
                return max(-1, min(1, last_signal))
            elif isinstance(last_signal, str):
                signal_map = {
                    'buy': 1.0, 'strong_buy': 1.0, 'long': 1.0,
                    'sell': -1.0, 'strong_sell': -1.0, 'short': -1.0,
                    'hold': 0.0, 'neutral': 0.0
                }
                return signal_map.get(last_signal.lower(), 0.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting signal from strategy: {e}")
            return 0.0
    
    def _calculate_position_size(
        self, 
        symbol: str, 
        signal_strength: float,
        portfolio_value: float = 100000.0
    ) -> Dict[str, float]:
        """
        Calculate position size based on ATR and risk management rules.
        
        Args:
            symbol: Trading symbol
            signal_strength: Absolute value of weighted signal (0 to 1)
            portfolio_value: Total portfolio value in USD
            
        Returns:
            Dict with position details: size, stop_loss_price, risk_amount
        """
        try:
            # Get current regime metrics for ATR
            if symbol not in self.regime_detectors or 'H4' not in self.regime_detectors[symbol]:
                logger.error(f"No regime detector for {symbol} H4")
                return {'size': 0, 'stop_loss_price': 0, 'risk_amount': 0}
            
            _, metrics = self.regime_detectors[symbol]['H4'].classify_regime()
            
            current_price = metrics.get('price', 0)
            atr = metrics.get('atr', 0)
            
            if current_price <= 0 or atr <= 0:
                logger.error(f"Invalid price or ATR for {symbol}")
                return {'size': 0, 'stop_loss_price': 0, 'risk_amount': 0}
            
            # Calculate risk amount (adjusted by signal strength)
            base_risk = portfolio_value * self.MAX_RISK_PER_TRADE
            risk_amount = base_risk * signal_strength
            
            # Calculate stop loss distance
            stop_loss_distance = atr * self.STOP_LOSS_ATR_MULTIPLIER
            
            # Calculate position size
            # Position size = Risk Amount / Stop Loss Distance
            position_size = risk_amount / stop_loss_distance
            
            # Convert to asset units
            asset_units = position_size / current_price
            
            return {
                'size': round(asset_units, 8),
                'stop_loss_distance': round(stop_loss_distance, 2),
                'stop_loss_price': round(current_price - stop_loss_distance, 2),
                'risk_amount': round(risk_amount, 2),
                'current_price': round(current_price, 2),
                'atr': round(atr, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {'size': 0, 'stop_loss_price': 0, 'risk_amount': 0}
    
    def run(self, symbol: str = 'BTC/USDT', portfolio_value: float = 100000.0):
        """
        Main execution method for one trading cycle.
        
        This method:
        1. Determines overall market bias
        2. Applies dynamic strategy weights based on bias
        3. Calculates consensus signal from all strategies
        4. Executes trade if signal exceeds threshold
        
        Args:
            symbol: Trading symbol to analyze (default: 'BTC/USDT')
            portfolio_value: Total portfolio value for position sizing
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TRADING CYCLE START - {symbol} - {datetime.now()}")
        logger.info(f"{'='*80}")
        
        # Step 1: Determine overall market bias
        logger.info("\nStep 1: Determining market bias...")
        overall_bias, bias_metadata = self.determine_overall_bias(symbol)
        
        logger.info(f"Market Bias: {overall_bias}")
        logger.info(f"Regime Details: D1={bias_metadata['D1']}, "
                   f"H4={bias_metadata['H4']}, H1={bias_metadata['H1']}")
        
        # Handle unknown bias
        if overall_bias == 'Unknown':
            logger.warning("Unable to determine market bias. Skipping trade execution.")
            return
        
        # Step 2: Get strategy weights for current bias
        logger.info("\nStep 2: Applying strategy weights...")
        strategy_weights = self.STRATEGY_WEIGHTS.get(overall_bias, self.STRATEGY_WEIGHTS['Neutral'])
        
        # Step 3: Collect and weight strategy signals
        logger.info("\nStep 3: Collecting strategy signals...")
        
        if symbol not in self.strategies:
            logger.error(f"No strategies initialized for {symbol}")
            return
        
        weighted_signals = []
        total_weight = 0
        signal_details = {}
        
        for strategy_name, strategy in self.strategies[symbol].items():
            weight = strategy_weights.get(strategy_name, 0)
            
            if weight > 0:
                signal = self._get_strategy_signal(strategy)
                weighted_signal = signal * weight
                weighted_signals.append(weighted_signal)
                total_weight += weight
                
                signal_details[strategy_name] = {
                    'signal': round(signal, 3),
                    'weight': weight,
                    'weighted': round(weighted_signal, 3)
                }
                
                logger.info(f"  {strategy_name}: signal={signal:.3f}, "
                          f"weight={weight:.1f}, weighted={weighted_signal:.3f}")
        
        # Step 4: Calculate consensus signal
        logger.info("\nStep 4: Calculating consensus signal...")
        
        if total_weight == 0:
            logger.warning("No strategies have positive weights. No trade possible.")
            return
        
        consensus_signal = sum(weighted_signals) / total_weight
        signal_strength = abs(consensus_signal)
        
        logger.info(f"Consensus Signal: {consensus_signal:.3f}")
        logger.info(f"Signal Strength: {signal_strength:.3f}")
        logger.info(f"Threshold: {self.SIGNAL_THRESHOLD}")
        
        # Step 5: Execute trade if signal exceeds threshold
        logger.info("\nStep 5: Trade execution decision...")
        
        if signal_strength < self.SIGNAL_THRESHOLD:
            logger.info(f"Signal too weak ({signal_strength:.3f} < {self.SIGNAL_THRESHOLD}). "
                       "No trade executed.")
            logger.info(f"\n{'='*80}")
            logger.info("TRADING CYCLE COMPLETE - NO ACTION TAKEN")
            logger.info(f"{'='*80}\n")
            return
        
        # Determine trade direction
        trade_direction = "LONG" if consensus_signal > 0 else "SHORT"
        
        # Step 6: Calculate position size with risk management
        logger.info("\nStep 6: Calculating position size...")
        position_details = self._calculate_position_size(symbol, signal_strength, portfolio_value)
        
        if position_details['size'] == 0:
            logger.error("Failed to calculate valid position size. No trade executed.")
            return
        
        # Step 7: Log trade execution (simulated)
        logger.info("\nStep 7: TRADE EXECUTION")
        logger.info(f"{'='*60}")
        logger.info(f"Direction: {trade_direction}")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Position Size: {position_details['size']:.8f} units")
        logger.info(f"Entry Price: ${position_details['current_price']:,.2f}")
        logger.info(f"Stop Loss: ${position_details['stop_loss_price']:,.2f}")
        logger.info(f"Stop Distance: ${position_details['stop_loss_distance']:.2f} ({self.STOP_LOSS_ATR_MULTIPLIER}x ATR)")
        logger.info(f"Risk Amount: ${position_details['risk_amount']:.2f}")
        logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"Risk Percentage: {(position_details['risk_amount']/portfolio_value)*100:.2f}%")
        logger.info(f"{'='*60}")
        
        # Log complete analysis for audit
        logger.info("\nCOMPLETE ANALYSIS SUMMARY:")
        logger.info(f"Market Bias: {overall_bias}")
        logger.info(f"Bias Metadata: {bias_metadata}")
        logger.info("Strategy Signals:")
        for strategy, details in signal_details.items():
            logger.info(f"  {strategy}: {details}")
        logger.info(f"Consensus Signal: {consensus_signal:.3f}")
        logger.info(f"Trade Direction: {trade_direction}")
        logger.info(f"ATR: {position_details['atr']}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRADING CYCLE COMPLETE - {trade_direction} ORDER EXECUTED")
        logger.info(f"{'='*80}\n")
    
    def __repr__(self):
        """String representation of the orchestrator."""
        return (
            f"MetaStrategyOrchestrator("
            f"symbols={self.symbols}, "
            f"strategies={list(self.STRATEGY_CLASSES.keys())}, "
            f"timeframes={self.TIMEFRAMES})"
        )


# Example usage and testing
if __name__ == "__main__":
    # Example connection string (adjust for your database)
    # For SQLite
    connection_string = "sqlite:///data/trading_data.db"
    
    # For PostgreSQL
    # connection_string = "postgresql://user:password@localhost:5432/trading_db"
    
    # Create orchestrator
    orchestrator = MetaStrategyOrchestrator(
        db_connection_string=connection_string,
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        lookback_period=500
    )
    
    # Initialize all components
    orchestrator.setup()
    
    # Get current regimes for BTC
    btc_regimes = orchestrator.get_current_regimes('BTC/USDT')
    print(f"BTC Regimes: {btc_regimes}")
    
    # Determine overall bias for all symbols
    print("\nOverall Market Bias:")
    for symbol in orchestrator.symbols:
        bias, metadata = orchestrator.determine_overall_bias(symbol)
        print(f"{symbol}: {bias} (D1: {metadata['D1']}, H4: {metadata['H4']}, H1: {metadata['H1']})")
    
    # Get strategy signals for ETH
    eth_signals = orchestrator.get_strategy_signals('ETH/USDT')
    print(f"\nETH Signals: {eth_signals}")
