"""
Strategy Bridge Module

This module provides a bridge between the old strategy descriptors
and the new executable strategies, allowing both to be used together.

"""

import logging
from typing import Dict, List, Optional, Any, Type
import pandas as pd
import sqlite3

# Import old strategy descriptors
from backend.sql_workflow.strategies import (
    RSIMomentumDivergenceSwingStrategy,
    BollingerBandsMeanReversionStrategy,
    MACDMomentumCrossoverStrategy,
    SMAGoldenCrossStrategy,
    IchimokuCloudBreakoutStrategy,
    ParabolicSARTrendFollowingStrategy,
    FibonacciRetracementSupportResistanceStrategy,
    GaussianChannelBreakoutMeanReversionStrategy
)

# Import new executable strategies
from backend.executable_workflow.strategies import (
    RSIMomentumDivergence,
    BollingerBandsMeanReversion,
    MACDMomentumCrossover,
    SMAGoldenCross,
    IchimokuCloudBreakout,
    ParabolicSARTrendFollowing,
    FibonacciRetracementSupportResistance,
    GaussianChannelBreakoutMeanReversion
)

from backend.executable_workflow.interfaces.trading_strategy_interface import TradingStrategy

logger = logging.getLogger(__name__)


class StrategyBridge:
    """
    Bridge between old strategy descriptors and new executable strategies.
    
    This class provides a unified interface to work with both implementations,
    allowing gradual migration while maintaining backward compatibility.
    """
    
    # Mapping between descriptor and executable classes
    STRATEGY_MAPPING = {
        'RSIMomentumDivergenceSwingStrategy': RSIMomentumDivergence,
        'BollingerBandsMeanReversionStrategy': BollingerBandsMeanReversion,
        'MACDMomentumCrossoverStrategy': MACDMomentumCrossover,
        'SMAGoldenCrossStrategy': SMAGoldenCross,
        'IchimokuCloudBreakoutStrategy': IchimokuCloudBreakout,
        'ParabolicSARTrendFollowingStrategy': ParabolicSARTrendFollowing,
        'FibonacciRetracementSupportResistanceStrategy': FibonacciRetracementSupportResistance,
        'GaussianChannelBreakoutMeanReversionStrategy': GaussianChannelBreakoutMeanReversion
    }
    
    def __init__(self, strategy_name: str, db_path: str = None):
        """
        Initialize the bridge with a strategy.
        
        Args:
            strategy_name: Name of the strategy to use
            db_path: Path to SQLite database for historical queries
        """
        self.strategy_name = strategy_name
        self.db_path = db_path or 'data/trading_data_BTC.db'
        
        # Initialize descriptor and executable
        self.descriptor = self._get_descriptor(strategy_name)
        self.executable_class = self._get_executable_class(strategy_name)
        self.executable: Optional[TradingStrategy] = None
        
    def _get_descriptor(self, strategy_name: str):
        """Get the descriptor instance for a strategy."""
        descriptor_classes = {
            'RSI_Momentum': RSIMomentumDivergenceSwingStrategy,
            'Bollinger_Bands': BollingerBandsMeanReversionStrategy,
            'MACD_Momentum': MACDMomentumCrossoverStrategy,
            'SMA_Golden_Cross': SMAGoldenCrossStrategy,
            'Ichimoku_Cloud': IchimokuCloudBreakoutStrategy,
            'Parabolic_SAR': ParabolicSARTrendFollowingStrategy,
            'Fibonacci_Retracement': FibonacciRetracementSupportResistanceStrategy,
            'Gaussian_Channel': GaussianChannelBreakoutMeanReversionStrategy
        }
        
        for key, cls in descriptor_classes.items():
            if key in strategy_name:
                return cls()
        
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def _get_executable_class(self, strategy_name: str) -> Type[TradingStrategy]:
        """Get the executable class for a strategy."""
        descriptor_name = self.descriptor.__class__.__name__
        
        if descriptor_name in self.STRATEGY_MAPPING:
            return self.STRATEGY_MAPPING[descriptor_name]
        
        raise ValueError(f"No executable mapping for {descriptor_name}")
    
    def initialize_executable(self, data: pd.DataFrame, config: dict = None) -> None:
        """
        Initialize the executable strategy with data.
        
        Args:
            data: OHLCV DataFrame
            config: Optional configuration parameters
        """
        self.executable = self.executable_class(data, config)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get combined information from both implementations.
        
        Returns:
            Dictionary with strategy information
        """
        info = {
            'descriptor': {
                'description': self.descriptor.get_strategy_description(),
                'entry_conditions': self.descriptor.get_entry_conditions(),
                'exit_conditions': self.descriptor.get_exit_conditions(),
                'risk_rules': self.descriptor.get_risk_management_rules()
            }
        }
        
        if self.executable:
            info['executable'] = {
                'name': self.executable.name,
                'version': self.executable.version,
                'has_sufficient_data': self.executable.has_sufficient_data(),
                'required_data_points': self.executable.get_required_data_points()
            }
        
        return info
    
    def get_historical_signals(self, limit: int = 100) -> pd.DataFrame:
        """
        Get historical signals using the descriptor's SQL query.
        
        Args:
            limit: Maximum number of signals to return
            
        Returns:
            DataFrame with historical signals
        """
        try:
            query = self.descriptor.get_sql_query()
            # Modify query to limit results
            query = query.replace("LIMIT 100", f"LIMIT {limit}")
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
                return df
                
        except Exception as e:
            logger.error(f"Error getting historical signals: {e}")
            return pd.DataFrame()
    
    def get_live_signal(self, data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Get live signal using the executable strategy.
        
        Args:
            data: Optional new OHLCV data
            
        Returns:
            Signal dictionary
        """
        if self.executable is None:
            if data is None:
                raise ValueError("Executable not initialized. Provide data.")
            self.initialize_executable(data)
        
        if data is not None and self.executable:
            self.executable.data = data
            self.executable._calculate_indicators()
        
        return self.executable.calculate_signal()
    
    def is_strategy_allowed(self, market_bias: str) -> bool:
        """
        Check if strategy is suitable for current market regime.
        
        Args:
            market_bias: Current market regime
            
        Returns:
            True if strategy is suitable
        """
        if self.executable:
            return self.executable.is_strategy_allowed(market_bias)
        
        # Fallback logic based on descriptor
        strategy_type = self.descriptor.get_strategy_description().get('type', '')
        
        if 'Mean Reversion' in strategy_type:
            return market_bias in ['Neutral', 'Ranging']
        elif 'Trend Following' in strategy_type:
            return market_bias not in ['Neutral', 'Ranging']
        
        return True  # Default to allowed
    
    def compare_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare signals from SQL-based and executable approaches.
        
        Useful for validation and debugging.
        
        Args:
            data: OHLCV data
            
        Returns:
            Comparison results
        """
        # Get historical signals from SQL
        historical = self.get_historical_signals(limit=10)
        
        # Get live signal from executable
        self.initialize_executable(data)
        live_signal = self.get_live_signal()
        
        # Get the most recent historical signal
        if not historical.empty:
            recent_historical = historical.iloc[0]
            
            comparison = {
                'historical': {
                    'timestamp': recent_historical.get('timestamp'),
                    'signal': recent_historical.get('signal_type'),
                    'price': recent_historical.get('price', recent_historical.get('close'))
                },
                'live': {
                    'signal': live_signal.get('signal'),
                    'confidence': live_signal.get('confidence'),
                    'reason': live_signal.get('reason')
                },
                'data_info': {
                    'data_points': len(data),
                    'latest_close': data['close'].iloc[-1] if not data.empty else None,
                    'latest_time': data.index[-1] if not data.empty else None
                }
            }
        else:
            comparison = {
                'historical': {'message': 'No historical signals found'},
                'live': live_signal,
                'data_info': {
                    'data_points': len(data),
                    'latest_close': data['close'].iloc[-1] if not data.empty else None
                }
            }
        
        return comparison


class UnifiedStrategyFactory:
    """
    Factory for creating strategy bridges with unified interface.
    """
    
    @staticmethod
    def create_strategy(strategy_name: str, data: pd.DataFrame = None, 
                       config: dict = None, db_path: str = None) -> StrategyBridge:
        """
        Create a strategy bridge instance.
        
        Args:
            strategy_name: Name of the strategy
            data: Optional OHLCV data for executable initialization
            config: Optional configuration
            db_path: Optional database path
            
        Returns:
            StrategyBridge instance
        """
        bridge = StrategyBridge(strategy_name, db_path)
        
        if data is not None:
            bridge.initialize_executable(data, config)
        
        return bridge
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategies."""
        return [
            'RSI_Momentum_Divergence',
            'Bollinger_Bands_Mean_Reversion',
            'MACD_Momentum_Crossover',
            'SMA_Golden_Cross',
            'Ichimoku_Cloud_Breakout',
            'Parabolic_SAR_Trend_Following',
            'Fibonacci_Retracement_Support_Resistance',
            'Gaussian_Channel_Breakout_Mean_Reversion'
        ]
    
    @staticmethod
    def create_all_strategies(data: pd.DataFrame, config: dict = None) -> Dict[str, StrategyBridge]:
        """
        Create bridge instances for all available strategies.
        
        Args:
            data: OHLCV data
            config: Optional configuration for all strategies
            
        Returns:
            Dictionary of strategy bridges
        """
        strategies = {}
        
        for strategy_name in UnifiedStrategyFactory.get_available_strategies():
            try:
                bridge = UnifiedStrategyFactory.create_strategy(
                    strategy_name, data, config
                )
                strategies[strategy_name] = bridge
            except Exception as e:
                logger.error(f"Error creating {strategy_name}: {e}")
        
        return strategies
