"""
Refined Meta Strategy Orchestrator with Enhanced Regime Detection

This module implements the refined orchestrator with:
- ADX-based regime detection
- Optimized strategy weights
- Integration of the new Volatility Breakout Short strategy
- Removal of SMA Golden Cross as a standalone strategy
"""

import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Import the enhanced market regime detector
from backend.executable_workflow.orchestration.enhanced_market_regime_detector import EnhancedMarketRegimeDetector

# Import executable strategies
from backend.executable_workflow.strategies import (
    BollingerBandsMeanReversion,
    RSIMomentumDivergence,
    MACDMomentumCrossover,
    IchimokuCloudBreakout,
    ParabolicSARTrendFollowing,
    FibonacciRetracementSupportResistance,
    GaussianChannelBreakoutMeanReversion,
    VolatilityBreakoutShort
)

# Import the strategy interface
from backend.executable_workflow.interfaces.trading_strategy_interface import TradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RefinedMetaStrategyOrchestrator:
    """
    Refined Meta Strategy Orchestrator with enhanced regime detection.
    
    This version implements:
    - ADX-based market regime detection
    - Optimized strategy weights per regime
    - Proper mean reversion logic
    - Dedicated crash/panic strategy
    """
    
    # Available strategy classes (SMA Golden Cross removed)
    STRATEGY_REGISTRY = {
        'BollingerBandsMeanReversion': BollingerBandsMeanReversion,
        'RSIMomentumDivergence': RSIMomentumDivergence,
        'MACDMomentumCrossover': MACDMomentumCrossover,
        'IchimokuCloudBreakout': IchimokuCloudBreakout,
        'ParabolicSARTrendFollowing': ParabolicSARTrendFollowing,
        'FibonacciRetracementSupportResistance': FibonacciRetracementSupportResistance,
        'GaussianChannelBreakoutMeanReversion': GaussianChannelBreakoutMeanReversion,
        'VolatilityBreakoutShort': VolatilityBreakoutShort
    }
    
    # Refined strategy weights by market regime
    STRATEGY_WEIGHTS = {
        'STRONG_BULLISH': {
            'macd_momentum': 0.4,  # Effective but redundant with regime detection
            'ichimoku_cloud': 0.7,  # Premier trend-following system
            'parabolic_sar': 0.6,   # Best for trailing stop management
            'bollinger_bands': 0.0,  # Do not fight the trend
            'rsi_divergence': 0.0,   # Mean reversion ineffective
            'gaussian_channel': 0.0,  # Do not fight the trend
            'fibonacci_retracement': 0.0,  # Support levels break in strong trends
            'volatility_breakout_short': 0.0  # Long-only in bull markets
        },
        'STRONG_BEARISH': {
            'macd_momentum': 0.4,  # For short signals
            'ichimoku_cloud': 0.7,  # For short signals
            'parabolic_sar': 0.6,   # For short signals
            'bollinger_bands': 0.0,  # Do not fight the trend
            'rsi_divergence': 0.0,   # Mean reversion ineffective
            'gaussian_channel': 0.0,  # Do not fight the trend
            'fibonacci_retracement': 0.0,  # Resistance levels break
            'volatility_breakout_short': 0.0  # Not extreme enough for this strategy
        },
        'NEUTRAL_RANGING': {
            'bollinger_bands': 0.7,  # Core mean reversion strategy
            'rsi_divergence': 0.8,   # Highest weight: excels at reversals
            'gaussian_channel': 0.6,  # Alternative mean reversion
            'fibonacci_retracement': 0.5,  # Key levels hold in ranges
            'macd_momentum': 0.1,    # Low probability in choppy markets
            'ichimoku_cloud': 0.1,   # Generates false signals
            'parabolic_sar': 0.1,    # Whipsaws in ranging markets
            'volatility_breakout_short': 0.0  # No extreme volatility
        },
        'CRASH_PANIC': {
            'volatility_breakout_short': 1.0,  # REQUIRED: Dedicated crash strategy
            'bollinger_bands': 0.0,
            'rsi_divergence': 0.0,
            'gaussian_channel': 0.0,
            'fibonacci_retracement': 0.0,
            'macd_momentum': 0.0,
            'ichimoku_cloud': 0.0,
            'parabolic_sar': 0.0
        }
    }
    
    # Trading configuration
    SIGNAL_THRESHOLD = 0.6  # Minimum absolute weighted signal to trade
    MAX_RISK_PER_TRADE = 0.01  # 1% of portfolio per trade
    CRASH_RISK_PER_TRADE = 0.005  # 0.5% in crash conditions
    
    def __init__(
        self, 
        db_connection_string: str,
        symbols: List[str] = None,
        config_path: str = None
    ):
        """
        Initialize the Refined Meta Strategy Orchestrator.
        
        Args:
            db_connection_string: SQLAlchemy database connection string
            symbols: List of trading symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            config_path: Path to strategy configuration JSON file
        """
        self.db_connection_string = db_connection_string
        self.symbols = symbols or ['BTC/USDT']
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize containers
        self.regime_detectors: Dict[str, EnhancedMarketRegimeDetector] = {}
        self.strategies: Dict[str, Dict[str, TradingStrategy]] = {}
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # Current regime tracking
        self.current_regimes: Dict[str, str] = {}
        self.regime_metrics: Dict[str, Dict] = {}
        
        # Performance tracking
        self.signal_history: List[Dict] = []
        self.active_positions: Dict[str, Dict] = {}
        
        # Create database engine
        self.engine = create_engine(self.db_connection_string)
        
        logger.info(
            f"Refined orchestrator initialized with symbols: {self.symbols}"
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load strategy configuration from JSON file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Use updated default configuration
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
                    "ichimoku_cloud": {
                        "class": "IchimokuCloudBreakout",
                        "enabled": True,
                        "parameters": {}
                    },
                    "parabolic_sar": {
                        "class": "ParabolicSARTrendFollowing",
                        "enabled": True,
                        "parameters": {}
                    },
                    "fibonacci_retracement": {
                        "class": "FibonacciRetracementSupportResistance",
                        "enabled": True,
                        "parameters": {}
                    },
                    "gaussian_channel": {
                        "class": "GaussianChannelBreakoutMeanReversion",
                        "enabled": True,
                        "parameters": {}
                    },
                    "volatility_breakout_short": {
                        "class": "VolatilityBreakoutShort",
                        "enabled": True,
                        "parameters": {}
                    }
                },
                "signal_filters": {
                    "min_signal_strength": 0.6,
                    "min_confidence": 0.7
                }
            }
    
    def detect_market_regime(self, symbol: str) -> Tuple[str, Dict[str, float]]:
        """
        Detect market regime for a symbol using enhanced ADX-based logic.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (regime_name, regime_metrics)
        """
        if symbol not in self.regime_detectors:
            logger.warning(f"No regime detector for {symbol}")
            return "NEUTRAL_RANGING", {}
        
        return self.regime_detectors[symbol].detect_market_regime()
    
    def setup(self, timeframe: str = '1h') -> None:
        """
        Initialize regime detectors and strategies for all symbols.
        
        Args:
            timeframe: Timeframe for analysis (e.g., '1h', '4h')
        """
        logger.info("Setting up refined orchestrator...")
        
        # Load data for all symbols
        self._load_market_data(timeframe)
        
        # Initialize regime detectors
        for symbol in self.symbols:
            if symbol in self.data_cache and not self.data_cache[symbol].empty:
                self.regime_detectors[symbol] = EnhancedMarketRegimeDetector(
                    self.data_cache[symbol]
                )
                
                # Detect initial regime
                regime, metrics = self.detect_market_regime(symbol)
                self.current_regimes[symbol] = regime
                self.regime_metrics[symbol] = metrics
                logger.info(f"{symbol} initial regime: {regime}")
        
        # Initialize strategies
        self._initialize_strategies()
        
        logger.info("Setup complete")
    
    def _load_market_data(self, timeframe: str) -> None:
        """Load market data for all symbols."""
        for symbol in self.symbols:
            try:
                # Load OHLCV data (implement your data loading logic)
                data = self._fetch_ohlcv_data(symbol, timeframe)
                if data is not None and not data.empty:
                    self.data_cache[symbol] = data
                    logger.info(f"Loaded {len(data)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
    
    def _fetch_ohlcv_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch OHLCV data from database.
        
        This is a placeholder - implement your actual data fetching logic.
        """
        # Example query structure
        query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_data
        WHERE symbol = '{symbol}'
        AND timeframe = '{timeframe}'
        ORDER BY timestamp DESC
        LIMIT 500
        """
        
        try:
            df = pd.read_sql(query, self.engine, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return pd.DataFrame()
    
    def _initialize_strategies(self) -> None:
        """Initialize all enabled strategies for each symbol."""
        for symbol in self.symbols:
            if symbol not in self.data_cache:
                continue
                
            self.strategies[symbol] = {}
            data = self.data_cache[symbol]
            
            for strategy_name, strategy_config in self.config['strategies'].items():
                if not strategy_config.get('enabled', True):
                    continue
                
                class_name = strategy_config['class']
                if class_name not in self.STRATEGY_REGISTRY:
                    logger.warning(f"Unknown strategy class: {class_name}")
                    continue
                
                try:
                    strategy_class = self.STRATEGY_REGISTRY[class_name]
                    parameters = strategy_config.get('parameters', {})
                    
                    # Initialize strategy
                    strategy = strategy_class(data, parameters)
                    self.strategies[symbol][strategy_name] = strategy
                    logger.info(f"Initialized {strategy_name} for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize {strategy_name} for {symbol}: {e}")
    
    def calculate_composite_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate composite signal for a symbol based on regime and strategy signals.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with composite signal and metadata
        """
        # Get current regime
        regime, regime_metrics = self.detect_market_regime(symbol)
        self.current_regimes[symbol] = regime
        self.regime_metrics[symbol] = regime_metrics
        
        # Get strategy weights for current regime
        regime_weights = self.STRATEGY_WEIGHTS.get(regime, self.STRATEGY_WEIGHTS['NEUTRAL_RANGING'])
        
        # Collect signals from active strategies
        strategy_signals = []
        total_weight = 0
        
        for strategy_name, strategy in self.strategies[symbol].items():
            weight = regime_weights.get(strategy_name, 0)
            
            # Skip if weight is 0 or strategy not allowed in regime
            if weight == 0:
                continue
            
            if not strategy.is_strategy_allowed(regime):
                continue
            
            try:
                # Get signal from strategy
                signal_data = strategy.calculate_signal()
                
                if signal_data.get('signal', 0) != 0:
                    strategy_signals.append({
                        'strategy': strategy_name,
                        'signal': signal_data['signal'],
                        'confidence': signal_data.get('confidence', 0.5),
                        'weight': weight,
                        'reason': signal_data.get('reason', ''),
                        'metadata': signal_data
                    })
                    total_weight += weight
                    
            except Exception as e:
                logger.error(f"Error getting signal from {strategy_name}: {e}")
        
        # Calculate weighted composite signal
        if not strategy_signals or total_weight == 0:
            return {
                'symbol': symbol,
                'signal': 0,
                'confidence': 0,
                'regime': regime,
                'reason': 'No valid signals',
                'strategies': []
            }
        
        # Normalize weights
        for sig in strategy_signals:
            sig['normalized_weight'] = sig['weight'] / total_weight
        
        # Calculate composite
        composite_signal = sum(
            sig['signal'] * sig['normalized_weight'] * sig['confidence']
            for sig in strategy_signals
        )
        
        # Average confidence
        avg_confidence = sum(
            sig['confidence'] * sig['normalized_weight']
            for sig in strategy_signals
        )
        
        # Determine primary reason
        primary_strategy = max(strategy_signals, key=lambda x: abs(x['signal'] * x['normalized_weight']))
        
        # Adjust risk based on regime
        risk_per_trade = self.CRASH_RISK_PER_TRADE if regime == 'CRASH_PANIC' else self.MAX_RISK_PER_TRADE
        
        return {
            'symbol': symbol,
            'signal': round(composite_signal, 3),
            'confidence': round(avg_confidence, 3),
            'regime': regime,
            'regime_metrics': regime_metrics,
            'primary_strategy': primary_strategy['strategy'],
            'reason': primary_strategy['reason'],
            'risk_per_trade': risk_per_trade,
            'strategies': strategy_signals,
            'timestamp': datetime.now()
        }
    
    def should_execute_trade(self, composite_signal: Dict[str, Any]) -> bool:
        """
        Determine if a trade should be executed based on composite signal.
        
        Args:
            composite_signal: Composite signal data
            
        Returns:
            True if trade should be executed
        """
        signal_strength = abs(composite_signal['signal'])
        confidence = composite_signal['confidence']
        
        # Check signal thresholds
        min_signal = self.config['signal_filters']['min_signal_strength']
        min_confidence = self.config['signal_filters']['min_confidence']
        
        if signal_strength < min_signal or confidence < min_confidence:
            return False
        
        # Additional checks for crash regime
        if composite_signal['regime'] == 'CRASH_PANIC':
            # Only allow shorts in crash
            if composite_signal['signal'] > 0:
                return False
            
            # Require higher confidence in crash
            if confidence < 0.8:
                return False
        
        return True
    
    def run(self) -> Dict[str, Any]:
        """
        Run the orchestrator and generate signals for all symbols.
        
        Returns:
            Dictionary with signals and analysis
        """
        results = {
            'timestamp': datetime.now(),
            'regimes': self.current_regimes,
            'signals': {},
            'executable_trades': []
        }
        
        for symbol in self.symbols:
            if symbol not in self.strategies:
                continue
            
            # Calculate composite signal
            composite = self.calculate_composite_signal(symbol)
            results['signals'][symbol] = composite
            
            # Check if trade should be executed
            if self.should_execute_trade(composite):
                results['executable_trades'].append({
                    'symbol': symbol,
                    'direction': 'LONG' if composite['signal'] > 0 else 'SHORT',
                    'signal_strength': abs(composite['signal']),
                    'confidence': composite['confidence'],
                    'regime': composite['regime'],
                    'risk_per_trade': composite['risk_per_trade']
                })
                
                # Log signal
                self.signal_history.append(composite)
        
        return results
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current market regimes and metrics."""
        return {
            'current_regimes': self.current_regimes,
            'regime_metrics': self.regime_metrics,
            'strategy_weights': {
                symbol: self.STRATEGY_WEIGHTS.get(regime, {})
                for symbol, regime in self.current_regimes.items()
            }
        }
