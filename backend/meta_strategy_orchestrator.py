#!/usr/bin/env python3
"""
Meta Strategy Orchestrator

This module provides a sophisticated orchestration layer that integrates multiple trading strategies
with market regime detection across different timeframes and symbols. It manages the initialization,
data fetching, and coordination of various strategy components for a comprehensive trading system.

Author: Andrey's Trading Bot
Date: 2025-08-30
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

# Import the market regime detector
from backend.Indicators.market_regime_detector import CryptoMarketRegimeDetector

# Import all strategy classes
from backend.Strategies.Bollinger_Bands_Mean_Reversion_Strategy import BollingerBandsMeanReversionStrategy
from backend.Strategies.Fibonacci_Retracement_Support_Resistance_Strategy import FibonacciRetracementSupportResistanceStrategy
from backend.Strategies.Gaussian_Channel_Breakout_Mean_Reversion_Strategy import GaussianChannelBreakoutMeanReversionStrategy
from backend.Strategies.Ichimoku_Cloud_Breakout_Strategy import IchimokuCloudBreakoutStrategy
from backend.Strategies.MACD_Momentum_Crossover_Strategy import MACDMomentumCrossoverStrategy
from backend.Strategies.Parabolic_SAR_Trend_Following_Strategy import ParabolicSARTrendFollowingStrategy
from backend.Strategies.RSI_Momentum_Divergence_Swing_Strategy import RSIMomentumDivergenceSwingStrategy
from backend.Strategies.SMA_Golden_Cross_Strategy import SMAGoldenCrossStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meta_strategy_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MetaStrategyOrchestrator:
    """
    Orchestrates multiple trading strategies across different symbols and timeframes,
    integrating market regime detection for adaptive strategy selection and risk management.
    
    This class serves as the central coordinator for:
    - Data fetching and management across multiple symbols and timeframes
    - Market regime detection on multiple timeframes
    - Strategy initialization and management
    - Cross-timeframe and cross-asset analysis
    
    Attributes:
        db_connection_string (str): Database connection string
        symbols (List[str]): List of trading symbols to analyze
        lookback_period (int): Number of periods to fetch for historical analysis
        regime_detectors (Dict): Nested dictionary of regime detectors by symbol and timeframe
        strategies (Dict): Dictionary of initialized strategy instances
        data_cache (Dict): Cache for fetched OHLCV data
    """
    
    # Timeframe mappings for consistency
    TIMEFRAME_MAPPINGS = {
        'D1': ('1d', 'Daily'),
        'H4': ('4h', '4 Hour'),
        'H1': ('1h', '1 Hour')
    }
    
    # Strategy configurations
    STRATEGY_CONFIGS = {
        'trend_following': ['SMAGoldenCross', 'IchimokuCloud', 'ParabolicSAR', 'MACD'],
        'mean_reversion': ['BollingerBands', 'GaussianChannel', 'Fibonacci'],
        'momentum': ['RSI', 'MACD'],
        'support_resistance': ['Fibonacci']
    }
    
    def __init__(self, 
                 db_connection_string: str,
                 symbols: List[str] = None,
                 lookback_period: int = 500):
        """
        Initialize the Meta Strategy Orchestrator.
        
        Args:
            db_connection_string: Database connection string (can be SQLite path or SQLAlchemy URI)
            symbols: List of trading symbols to analyze (default: ['BTC/USDT', 'ETH/USDT'])
            lookback_period: Number of periods to fetch for historical analysis (default: 500)
        """
        self.db_connection_string = self._validate_connection_string(db_connection_string)
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        self.lookback_period = lookback_period
        
        # Initialize containers
        self.regime_detectors: Dict[str, Dict[str, CryptoMarketRegimeDetector]] = {}
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Initialize structure for regime detectors
        for symbol in self.symbols:
            self.regime_detectors[symbol] = {}
            self.data_cache[symbol] = {}
            
        # Initialize strategy container structure
        self.strategies = {
            'trend_following': {},
            'mean_reversion': {},
            'momentum': {},
            'support_resistance': {}
        }
        
        # SQLAlchemy engine for database operations
        self.engine: Optional[Engine] = None
        
        logger.info(f"MetaStrategyOrchestrator initialized for symbols: {self.symbols}")
        logger.info(f"Lookback period: {self.lookback_period} periods")
        
    def _validate_connection_string(self, connection_string: str) -> str:
        """
        Validate and format the database connection string.
        
        Args:
            connection_string: Raw connection string
            
        Returns:
            Formatted connection string for SQLAlchemy
        """
        # If it's a path to SQLite file, format it properly
        if not connection_string.startswith('sqlite://') and connection_string.endswith('.db'):
            return f'sqlite:///{connection_string}'
        return connection_string
    
    @contextmanager
    def _get_db_connection(self):
        """
        Context manager for database connections using SQLAlchemy.
        
        Yields:
            SQLAlchemy connection object
        """
        if self.engine is None:
            # Create engine with NullPool to avoid connection pooling issues with SQLite
            self.engine = create_engine(
                self.db_connection_string,
                poolclass=NullPool,
                echo=False
            )
        
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def _get_symbol_db_path(self, symbol: str) -> str:
        """
        Get the database path for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            
        Returns:
            Database file path
        """
        # Extract base currency from symbol
        base_currency = symbol.split('/')[0]
        return f'data/trading_data_{base_currency}.db'
    
    def fetch_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch OHLCV data for a specific symbol and timeframe from the SQL database.
        
        This method handles the database connection, query execution, and data formatting
        to return a properly indexed pandas DataFrame ready for analysis.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe code ('D1', 'H4', or 'H1')
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index, sorted ascending
            
        Raises:
            ValueError: If invalid timeframe is provided
            Exception: If database query fails
        """
        if timeframe not in self.TIMEFRAME_MAPPINGS:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(self.TIMEFRAME_MAPPINGS.keys())}")
        
        # Get the appropriate database path for the symbol
        db_path = self._get_symbol_db_path(symbol)
        db_string = f'sqlite:///{db_path}'
        
        # Map timeframe to database format
        db_timeframe, timeframe_label = self.TIMEFRAME_MAPPINGS[timeframe]
        
        logger.info(f"Fetching {timeframe_label} data for {symbol} from {db_path}")
        
        # SQL query to fetch OHLCV data with proper joins
        query = """
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
        """
        
        try:
            # Create a temporary engine for this specific database
            temp_engine = create_engine(db_string, poolclass=NullPool)
            
            with temp_engine.connect() as connection:
                # Execute query with parameters
                result = connection.execute(
                    text(query),
                    {
                        'symbol': symbol,
                        'timeframe': db_timeframe,
                        'limit': self.lookback_period
                    }
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall(), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                if df.empty:
                    logger.warning(f"No data found for {symbol} {timeframe}")
                    return pd.DataFrame()
                
                # Convert timestamp to datetime and set as index
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Ensure data types are correct
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Sort by timestamp ascending (oldest first)
                df.sort_index(inplace=True)
                
                # Remove any duplicate timestamps
                df = df[~df.index.duplicated(keep='last')]
                
                # Cache the data
                self.data_cache[symbol][timeframe] = df
                
                logger.info(f"Successfully fetched {len(df)} records for {symbol} {timeframe}")
                logger.debug(f"Date range: {df.index[0]} to {df.index[-1]}")
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {str(e)}")
            raise
    
    def _initialize_regime_detector(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[CryptoMarketRegimeDetector]:
        """
        Initialize a market regime detector for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe code
            df: OHLCV DataFrame
            
        Returns:
            Initialized CryptoMarketRegimeDetector or None if initialization fails
        """
        try:
            # Extract asset name from symbol
            asset_name = symbol.split('/')[0]
            
            # For altcoins, use BTC as benchmark if available
            benchmark_df = None
            if asset_name != 'BTC' and 'BTC/USDT' in self.data_cache:
                benchmark_df = self.data_cache.get('BTC/USDT', {}).get(timeframe)
            
            # Initialize detector
            detector = CryptoMarketRegimeDetector(
                df=df,
                asset_name=asset_name,
                benchmark_df=benchmark_df
            )
            
            # Test detector by getting current regime
            regime, metrics = detector.classify_regime()
            logger.info(f"Initialized regime detector for {symbol} {timeframe}: Current regime = {regime}")
            
            return detector
            
        except Exception as e:
            logger.error(f"Failed to initialize regime detector for {symbol} {timeframe}: {str(e)}")
            return None
    
    def _initialize_strategy(self, strategy_class, df: pd.DataFrame, symbol: str) -> Optional[Any]:
        """
        Initialize a trading strategy instance.
        
        Args:
            strategy_class: Strategy class to instantiate
            df: OHLCV DataFrame
            symbol: Trading symbol for context
            
        Returns:
            Initialized strategy instance or None if initialization fails
        """
        try:
            # Create strategy instance
            strategy = strategy_class()
            
            # Log strategy initialization
            logger.info(f"Initialized {strategy.name} for {symbol}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to initialize {strategy_class.__name__} for {symbol}: {str(e)}")
            return None
    
    def setup(self):
        """
        Set up all components of the orchestrator.
        
        This method performs the following initialization steps:
        1. Fetches data for all symbols and timeframes
        2. Initializes regime detectors for each symbol-timeframe combination
        3. Initializes all trading strategy instances
        4. Validates that all components are properly initialized
        
        The setup process includes comprehensive error handling and logging
        to ensure robust initialization in production environments.
        """
        logger.info("Starting MetaStrategyOrchestrator setup...")
        setup_start_time = datetime.now()
        
        # Step 1: Fetch data for all symbols and timeframes
        logger.info("Step 1: Fetching data for all symbols and timeframes...")
        
        # Fetch BTC data first (needed as benchmark for other assets)
        if 'BTC/USDT' in self.symbols:
            for timeframe in self.TIMEFRAME_MAPPINGS.keys():
                try:
                    df = self.fetch_data('BTC/USDT', timeframe)
                    if df.empty:
                        logger.warning(f"No data retrieved for BTC/USDT {timeframe}")
                except Exception as e:
                    logger.error(f"Failed to fetch BTC/USDT {timeframe}: {str(e)}")
        
        # Fetch data for other symbols
        for symbol in self.symbols:
            if symbol == 'BTC/USDT':
                continue  # Already fetched
                
            for timeframe in self.TIMEFRAME_MAPPINGS.keys():
                try:
                    df = self.fetch_data(symbol, timeframe)
                    if df.empty:
                        logger.warning(f"No data retrieved for {symbol} {timeframe}")
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol} {timeframe}: {str(e)}")
        
        # Step 2: Initialize regime detectors
        logger.info("Step 2: Initializing market regime detectors...")
        
        for symbol in self.symbols:
            for timeframe in self.TIMEFRAME_MAPPINGS.keys():
                if symbol in self.data_cache and timeframe in self.data_cache[symbol]:
                    df = self.data_cache[symbol][timeframe]
                    if not df.empty:
                        detector = self._initialize_regime_detector(symbol, timeframe, df)
                        if detector:
                            self.regime_detectors[symbol][timeframe] = detector
                        else:
                            logger.warning(f"Failed to initialize detector for {symbol} {timeframe}")
                    else:
                        logger.warning(f"No data available for {symbol} {timeframe} detector")
        
        # Step 3: Initialize trading strategies
        logger.info("Step 3: Initializing trading strategies...")
        
        # Map strategy names to classes
        strategy_classes = {
            'SMAGoldenCross': SMAGoldenCrossStrategy,
            'IchimokuCloud': IchimokuCloudBreakoutStrategy,
            'ParabolicSAR': ParabolicSARTrendFollowingStrategy,
            'MACD': MACDMomentumCrossoverStrategy,
            'BollingerBands': BollingerBandsMeanReversionStrategy,
            'GaussianChannel': GaussianChannelBreakoutMeanReversionStrategy,
            'Fibonacci': FibonacciRetracementSupportResistanceStrategy,
            'RSI': RSIMomentumDivergenceSwingStrategy
        }
        
        # Initialize strategies for each symbol
        for symbol in self.symbols:
            # Use H4 timeframe as default for strategies
            if symbol in self.data_cache and 'H4' in self.data_cache[symbol]:
                df_h4 = self.data_cache[symbol]['H4']
                
                if not df_h4.empty:
                    # Initialize each strategy type
                    for category, strategy_names in self.STRATEGY_CONFIGS.items():
                        for strategy_name in strategy_names:
                            if strategy_name in strategy_classes:
                                strategy_key = f"{symbol}_{strategy_name}"
                                strategy = self._initialize_strategy(
                                    strategy_classes[strategy_name],
                                    df_h4,
                                    symbol
                                )
                                if strategy:
                                    self.strategies[category][strategy_key] = strategy
                                else:
                                    logger.warning(f"Failed to initialize {strategy_name} for {symbol}")
        
        # Step 4: Validation and summary
        logger.info("Step 4: Validating setup...")
        
        # Count initialized components
        total_detectors = sum(len(detectors) for detectors in self.regime_detectors.values())
        total_strategies = sum(len(strategies) for strategies in self.strategies.values())
        
        # Log summary
        setup_duration = (datetime.now() - setup_start_time).total_seconds()
        logger.info(f"Setup completed in {setup_duration:.2f} seconds")
        logger.info(f"Initialized {total_detectors} regime detectors")
        logger.info(f"Initialized {total_strategies} trading strategies")
        
        # Log detailed breakdown
        logger.info("Regime Detectors Summary:")
        for symbol, detectors in self.regime_detectors.items():
            logger.info(f"  {symbol}: {list(detectors.keys())}")
        
        logger.info("Strategies Summary:")
        for category, strategies in self.strategies.items():
            if strategies:
                logger.info(f"  {category}: {len(strategies)} strategies")
                for strategy_key in list(strategies.keys())[:3]:  # Show first 3
                    logger.info(f"    - {strategy_key}")
        
        # Validation warnings
        if total_detectors == 0:
            logger.error("No regime detectors were initialized!")
        if total_strategies == 0:
            logger.error("No trading strategies were initialized!")
        
        logger.info("MetaStrategyOrchestrator setup complete!")
    
    def get_current_regimes(self) -> Dict[str, Dict[str, Tuple[str, Dict]]]:
        """
        Get current market regimes for all symbols and timeframes.
        
        Returns:
            Nested dictionary with structure:
            {symbol: {timeframe: (regime_string, metrics_dict)}}
        """
        current_regimes = {}
        
        for symbol, detectors in self.regime_detectors.items():
            current_regimes[symbol] = {}
            for timeframe, detector in detectors.items():
                try:
                    regime, metrics = detector.classify_regime()
                    current_regimes[symbol][timeframe] = (regime, metrics)
                except Exception as e:
                    logger.error(f"Error getting regime for {symbol} {timeframe}: {str(e)}")
                    current_regimes[symbol][timeframe] = ('Unknown', {})
        
        return current_regimes
    
    def get_strategy_recommendations(self, symbol: str) -> Dict[str, List[str]]:
        """
        Get recommended strategies based on current market regime.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary mapping regime to recommended strategy names
        """
        recommendations = {}
        
        # Get current regimes for the symbol
        if symbol in self.regime_detectors:
            for timeframe, detector in self.regime_detectors[symbol].items():
                try:
                    regime, _ = detector.classify_regime()
                    
                    # Map regimes to strategy categories
                    if regime == 'Bull Trend':
                        recommended = self.STRATEGY_CONFIGS['trend_following']
                    elif regime == 'Bear Trend':
                        recommended = self.STRATEGY_CONFIGS['trend_following'] + ['RSI']
                    elif regime == 'Ranging / Accumulation':
                        recommended = self.STRATEGY_CONFIGS['mean_reversion']
                    elif regime == 'High Volatility / Breakout':
                        recommended = ['GaussianChannel', 'BollingerBands']
                    elif regime == 'Crypto Crash':
                        recommended = ['RSI', 'Fibonacci']  # Look for oversold bounces
                    else:
                        recommended = []
                    
                    recommendations[f"{timeframe}_{regime}"] = recommended
                    
                except Exception as e:
                    logger.error(f"Error getting recommendations for {symbol}: {str(e)}")
        
        return recommendations
    
    def refresh_data(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """
        Refresh data for specific symbol/timeframe or all if not specified.
        
        Args:
            symbol: Optional specific symbol to refresh
            timeframe: Optional specific timeframe to refresh
        """
        symbols_to_refresh = [symbol] if symbol else self.symbols
        timeframes_to_refresh = [timeframe] if timeframe else list(self.TIMEFRAME_MAPPINGS.keys())
        
        logger.info(f"Refreshing data for symbols: {symbols_to_refresh}, timeframes: {timeframes_to_refresh}")
        
        for sym in symbols_to_refresh:
            for tf in timeframes_to_refresh:
                try:
                    self.fetch_data(sym, tf)
                except Exception as e:
                    logger.error(f"Failed to refresh {sym} {tf}: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    # Example initialization
    orchestrator = MetaStrategyOrchestrator(
        db_connection_string='data/trading_data_BTC.db',  # Will be auto-formatted
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        lookback_period=500
    )
    
    # Run setup
    orchestrator.setup()
    
    # Get current regimes
    current_regimes = orchestrator.get_current_regimes()
    print("\nCurrent Market Regimes:")
    for symbol, regimes in current_regimes.items():
        print(f"\n{symbol}:")
        for timeframe, (regime, metrics) in regimes.items():
            print(f"  {timeframe}: {regime}")
    
    # Get strategy recommendations
    print("\nStrategy Recommendations:")
    for symbol in orchestrator.symbols:
        recommendations = orchestrator.get_strategy_recommendations(symbol)
        print(f"\n{symbol}:")
        for key, strategies in recommendations.items():
            print(f"  {key}: {strategies}")
