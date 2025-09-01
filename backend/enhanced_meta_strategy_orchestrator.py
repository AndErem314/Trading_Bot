"""
Enhanced Meta Strategy Orchestrator with Executable Strategies

This module provides an enhanced version of the MetaStrategyOrchestrator
that integrates with the new executable strategy framework.

Author: Trading Bot Team
Date: 2025
"""

import logging
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Import market regime detector
from backend.Indicators import MarketRegimeDetector

# Import executable strategies
from backend.strategies_executable import (
    BollingerBandsMeanReversion,
    RSIMomentumDivergence,
    MACDMomentumCrossover,
    SMAGoldenCross
)

# Import the strategy interface
from backend.trading_strategy_interface import TradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedMetaStrategyOrchestrator:
    """
    Enhanced Meta Strategy Orchestrator with executable strategies.
    
    This version integrates the new TradingStrategy interface and provides
    improved strategy management, dynamic activation, and better performance.
    """
    
    # Available strategy classes
    STRATEGY_REGISTRY = {
        'BollingerBandsMeanReversion': BollingerBandsMeanReversion,
        'RSIMomentumDivergence': RSIMomentumDivergence,
        'MACDMomentumCrossover': MACDMomentumCrossover,
        'SMAGoldenCross': SMAGoldenCross
    }
    
    # Default timeframes
    TIMEFRAMES = ['D1', 'H4', 'H1']
    TIMEFRAME_MAP = {
        'D1': '1d',
        'H4': '4h', 
        'H1': '1h'
    }
    
    def __init__(
        self, 
        db_connection_string: str,
        symbols: List[str] = None,
        config_path: str = None
    ):
        """
        Initialize the Enhanced Meta Strategy Orchestrator.
        
        Args:
            db_connection_string: SQLAlchemy database connection string
            symbols: List of trading symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            config_path: Path to strategy configuration JSON file
        """
        self.db_connection_string = db_connection_string
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT']
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize containers
        self.regime_detectors: Dict[str, Dict[str, MarketRegimeDetector]] = {}
        self.strategies: Dict[str, Dict[str, TradingStrategy]] = {}
        self.data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        # Strategy state tracking
        self.strategy_weights: Dict[str, Dict[str, float]] = {}
        self.active_strategies: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.signal_history: List[Dict] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize structure
        for symbol in self.symbols:
            self.regime_detectors[symbol] = {}
            self.strategies[symbol] = {}
            self.data_cache[symbol] = {}
            self.active_strategies[symbol] = []
        
        # Create database engine
        self.engine = create_engine(self.db_connection_string)
        
        logger.info(
            f"Enhanced orchestrator initialized with symbols: {self.symbols}"
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load strategy configuration from JSON file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Use default configuration
            return {
                "strategies": {
                    "bollinger_bands": {
                        "class": "BollingerBandsMeanReversion",
                        "enabled": True,
                        "parameters": {}
                    },
                    "rsi_divergence": {
                        "class": "RSIMomentumDivergence",
                        "enabled": True,
                        "parameters": {}
                    },
                    "macd_momentum": {
                        "class": "MACDMomentumCrossover",
                        "enabled": True,
                        "parameters": {}
                    },
                    "sma_golden_cross": {
                        "class": "SMAGoldenCross",
                        "enabled": True,
                        "parameters": {}
                    }
                },
                "signal_filters": {
                    "min_signal_strength": 0.6,
                    "min_confidence": 0.7
                }
            }
    
    def setup(self, primary_timeframe: str = 'H1') -> None:
        """
        Initialize regime detectors and strategies for all symbols.
        
        Args:
            primary_timeframe: Primary timeframe for trading
        """
        logger.info("Setting up enhanced orchestrator...")
        
        # Load data for all symbols and timeframes
        self._load_all_data()
        
        # Initialize regime detectors
        for symbol in self.symbols:
            for tf in self.TIMEFRAMES:
                if tf in self.data_cache[symbol]:
                    data = self.data_cache[symbol][tf]
                    if not data.empty:
                        self.regime_detectors[symbol][tf] = MarketRegimeDetector(
                            symbol=symbol,
                            timeframe=tf
                        )
                        self.regime_detectors[symbol][tf].data = data
        
        # Initialize strategies for primary timeframe
        self._initialize_strategies(primary_timeframe)
        
        logger.info("Setup complete")
    
    def _load_all_data(self) -> None:
        """Load market data for all symbols and timeframes."""
        for symbol in self.symbols:
            for tf, tf_str in self.TIMEFRAME_MAP.items():
                try:
                    # This is a placeholder - implement your data loading logic
                    # For now, we'll assume data is loaded from database
                    data = self._fetch_ohlcv_data(symbol, tf_str)
                    if data is not None and not data.empty:
                        self.data_cache[symbol][tf] = data
                        logger.info(f"Loaded {len(data)} bars for {symbol} {tf}")
                except Exception as e:
                    logger.error(f"Error loading data for {symbol} {tf}: {e}")
    
    def _fetch_ohlcv_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch OHLCV data from database.
        
        This is a placeholder - implement based on your database schema.
        """
        # Example query - adjust based on your schema
        query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_data
        WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
        ORDER BY timestamp DESC
        LIMIT 500
        """
        
        try:
            df = pd.read_sql(query, self.engine, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def _initialize_strategies(self, timeframe: str) -> None:
        """Initialize executable strategies for each symbol."""
        for symbol in self.symbols:
            if timeframe not in self.data_cache[symbol]:
                continue
                
            data = self.data_cache[symbol][timeframe]
            if data.empty:
                continue
            
            # Initialize enabled strategies
            for strategy_name, strategy_config in self.config['strategies'].items():
                if not strategy_config.get('enabled', False):
                    continue
                
                strategy_class_name = strategy_config.get('class')
                if strategy_class_name not in self.STRATEGY_REGISTRY:
                    logger.warning(f"Strategy class {strategy_class_name} not found")
                    continue
                
                try:
                    # Get strategy class
                    strategy_class = self.STRATEGY_REGISTRY[strategy_class_name]
                    
                    # Get parameters
                    parameters = strategy_config.get('parameters', {})
                    
                    # Create strategy instance
                    strategy = strategy_class(data=data, config=parameters)
                    
                    # Check if strategy has sufficient data
                    if strategy.has_sufficient_data():
                        self.strategies[symbol][strategy_name] = strategy
                        self.active_strategies[symbol].append(strategy_name)
                        logger.info(f"Initialized {strategy_name} for {symbol}")
                    else:
                        logger.warning(
                            f"Insufficient data for {strategy_name} on {symbol} "
                            f"(need {strategy.get_required_data_points()} points)"
                        )
                        
                except Exception as e:
                    logger.error(f"Error initializing {strategy_name}: {e}")
    
    def detect_market_regimes(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect market regimes across all symbols and timeframes.
        
        Returns:
            Nested dictionary of market regimes by symbol and timeframe
        """
        regimes = {}
        
        for symbol in self.symbols:
            regimes[symbol] = {}
            
            for tf in self.TIMEFRAMES:
                if tf in self.regime_detectors[symbol]:
                    try:
                        detector = self.regime_detectors[symbol][tf]
                        regime_data = detector.get_market_regime()
                        
                        if regime_data:
                            regimes[symbol][tf] = {
                                'bias': regime_data.get('market_bias', 'Unknown'),
                                'strength': regime_data.get('bias_strength', 0),
                                'volatility': regime_data.get('volatility_regime', 'Normal'),
                                'trend_quality': regime_data.get('trend_quality', 0)
                            }
                    except Exception as e:
                        logger.error(f"Error detecting regime for {symbol} {tf}: {e}")
        
        return regimes
    
    def calculate_overall_bias(self, regimes: Dict[str, Dict[str, Any]]) -> str:
        """
        Calculate overall market bias from multi-timeframe regimes.
        
        Args:
            regimes: Market regimes by symbol and timeframe
            
        Returns:
            Overall market bias
        """
        # Weight regimes by timeframe (higher timeframes have more weight)
        timeframe_weights = {'D1': 0.5, 'H4': 0.3, 'H1': 0.2}
        
        bias_scores = {
            'Strong Bullish': 2,
            'Bullish': 1,
            'Neutral': 0,
            'Bearish': -1,
            'Strong Bearish': -2
        }
        
        weighted_score = 0
        total_weight = 0
        
        for symbol in regimes:
            for tf, regime in regimes[symbol].items():
                bias = regime.get('bias', 'Neutral')
                weight = timeframe_weights.get(tf, 0.1)
                
                if bias in bias_scores:
                    weighted_score += bias_scores[bias] * weight
                    total_weight += weight
        
        if total_weight == 0:
            return 'Neutral'
        
        avg_score = weighted_score / total_weight
        
        if avg_score >= 1.5:
            return 'Strong Bullish'
        elif avg_score >= 0.5:
            return 'Bullish'
        elif avg_score <= -1.5:
            return 'Strong Bearish'
        elif avg_score <= -0.5:
            return 'Bearish'
        else:
            return 'Neutral'
    
    def update_strategy_weights(self, market_bias: str) -> None:
        """
        Update strategy weights based on market bias and strategy suitability.
        
        Args:
            market_bias: Current overall market bias
        """
        # Base weights for different market regimes
        base_weights = {
            'Strong Bullish': {
                'macd_momentum': 0.9,
                'sma_golden_cross': 0.8,
                'rsi_divergence': 0.3,
                'bollinger_bands': 0.1
            },
            'Bullish': {
                'macd_momentum': 0.7,
                'sma_golden_cross': 0.6,
                'rsi_divergence': 0.4,
                'bollinger_bands': 0.2
            },
            'Neutral': {
                'macd_momentum': 0.2,
                'sma_golden_cross': 0.1,
                'rsi_divergence': 0.7,
                'bollinger_bands': 0.8
            },
            'Bearish': {
                'macd_momentum': 0.7,
                'sma_golden_cross': 0.6,
                'rsi_divergence': 0.4,
                'bollinger_bands': 0.2
            },
            'Strong Bearish': {
                'macd_momentum': 0.9,
                'sma_golden_cross': 0.8,
                'rsi_divergence': 0.3,
                'bollinger_bands': 0.1
            }
        }
        
        weights = base_weights.get(market_bias, base_weights['Neutral'])
        
        # Apply dynamic strategy activation
        for symbol in self.symbols:
            self.strategy_weights[symbol] = {}
            
            for strategy_name, strategy in self.strategies[symbol].items():
                # Check if strategy is allowed in current regime
                if strategy.is_strategy_allowed(market_bias):
                    # Check if strategy has sufficient data
                    if strategy.has_sufficient_data():
                        self.strategy_weights[symbol][strategy_name] = weights.get(
                            strategy_name, 0.5
                        )
                    else:
                        self.strategy_weights[symbol][strategy_name] = 0.0
                        logger.debug(f"{strategy_name} disabled for {symbol}: insufficient data")
                else:
                    self.strategy_weights[symbol][strategy_name] = 0.0
                    logger.debug(f"{strategy_name} disabled for {symbol}: unsuitable regime")
    
    def generate_signals(self) -> Dict[str, Any]:
        """
        Generate trading signals from all active strategies.
        
        Returns:
            Dictionary of signals by symbol
        """
        signals = {}
        
        for symbol in self.symbols:
            symbol_signals = {}
            
            for strategy_name, strategy in self.strategies[symbol].items():
                weight = self.strategy_weights[symbol].get(strategy_name, 0)
                
                if weight > 0:
                    try:
                        # Get signal from strategy
                        signal_data = strategy.calculate_signal()
                        
                        # Apply minimum thresholds
                        min_strength = self.config['signal_filters']['min_signal_strength']
                        min_confidence = self.config['signal_filters'].get('min_confidence', 0.7)
                        
                        signal_value = signal_data.get('signal', 0)
                        confidence = signal_data.get('confidence', 0)
                        
                        if abs(signal_value) >= min_strength and confidence >= min_confidence:
                            symbol_signals[strategy_name] = {
                                'signal': signal_value,
                                'weight': weight,
                                'weighted_signal': signal_value * weight,
                                'confidence': confidence,
                                'metadata': signal_data
                            }
                        
                    except Exception as e:
                        logger.error(f"Error generating signal for {strategy_name}: {e}")
            
            signals[symbol] = symbol_signals
        
        return signals
    
    def calculate_composite_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate composite signals from weighted strategy signals.
        
        Args:
            signals: Raw signals from strategies
            
        Returns:
            Composite signals by symbol
        """
        composite = {}
        
        for symbol, strategy_signals in signals.items():
            if not strategy_signals:
                composite[symbol] = {'signal': 0, 'confidence': 0}
                continue
            
            # Calculate weighted average
            total_weighted_signal = sum(
                s['weighted_signal'] for s in strategy_signals.values()
            )
            total_weight = sum(s['weight'] for s in strategy_signals.values())
            
            if total_weight > 0:
                avg_signal = total_weighted_signal / total_weight
                avg_confidence = sum(
                    s['confidence'] * s['weight'] for s in strategy_signals.values()
                ) / total_weight
                
                composite[symbol] = {
                    'signal': round(avg_signal, 3),
                    'confidence': round(avg_confidence, 3),
                    'contributing_strategies': list(strategy_signals.keys()),
                    'strategy_count': len(strategy_signals)
                }
            else:
                composite[symbol] = {'signal': 0, 'confidence': 0}
        
        return composite
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete orchestration cycle.
        
        Returns:
            Dictionary containing signals, regimes, and metadata
        """
        try:
            # 1. Detect market regimes
            regimes = self.detect_market_regimes()
            
            # 2. Calculate overall market bias
            overall_bias = self.calculate_overall_bias(regimes)
            
            # 3. Update strategy weights
            self.update_strategy_weights(overall_bias)
            
            # 4. Generate signals from active strategies
            raw_signals = self.generate_signals()
            
            # 5. Calculate composite signals
            composite_signals = self.calculate_composite_signals(raw_signals)
            
            # 6. Record results
            result = {
                'timestamp': datetime.now().isoformat(),
                'market_regimes': regimes,
                'overall_bias': overall_bias,
                'strategy_weights': self.strategy_weights,
                'raw_signals': raw_signals,
                'composite_signals': composite_signals
            }
            
            # Store in history
            self.signal_history.append(result)
            
            # Log summary
            logger.info(f"Orchestration complete - Market bias: {overall_bias}")
            for symbol, signal_data in composite_signals.items():
                logger.info(
                    f"{symbol}: Signal={signal_data['signal']}, "
                    f"Confidence={signal_data['confidence']}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in orchestration cycle: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'composite_signals': {}
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics for the orchestrator.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.signal_history:
            return {}
        
        # Calculate signal statistics
        total_signals = len(self.signal_history)
        
        # Signal direction distribution
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        
        for record in self.signal_history:
            for symbol, signal_data in record.get('composite_signals', {}).items():
                signal = signal_data.get('signal', 0)
                if signal > 0.3:
                    buy_signals += 1
                elif signal < -0.3:
                    sell_signals += 1
                else:
                    neutral_signals += 1
        
        # Strategy participation
        strategy_participation = {}
        for record in self.signal_history:
            for symbol, signals in record.get('raw_signals', {}).items():
                for strategy in signals:
                    strategy_participation[strategy] = strategy_participation.get(strategy, 0) + 1
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'neutral_signals': neutral_signals,
            'strategy_participation': strategy_participation,
            'last_update': self.signal_history[-1]['timestamp'] if self.signal_history else None
        }
