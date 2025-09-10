"""
Strategy Loader Utility

This module handles dynamically loading and initializing trading strategies
for backtesting.
"""

import importlib
import inspect
import sys
import os
from typing import Dict, Any, Optional
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Ensure numpy compatibility for pandas_ta
try:
    # Try absolute import first
    from backend.core.indicators.numpy_compat import *
except ImportError:
    try:
        # Try relative import
        from ...core.indicators.numpy_compat import *
    except ImportError:
        # If all else fails, patch numpy directly here
        if not hasattr(np, 'NaN'):
            np.NaN = np.nan
        sys.modules['numpy'].NaN = np.nan

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Base class for all trading strategies
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize strategy with parameters
        
        Args:
            parameters: Strategy-specific parameters
        """
        self.parameters = parameters
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with columns:
                - timestamp: Signal timestamp
                - signal: 1 for buy, -1 for sell, 0 for hold
                - strength: Signal strength (0-1)
                - price: Current price
        """
        pass
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators needed by the strategy
        
        Args:
            data: Market data
            
        Returns:
            Data with added indicator columns
        """
        return data


class StrategyLoader:
    """
    Dynamically loads and initializes trading strategies
    """
    
    def __init__(self, strategies_path: Optional[str] = None):
        """
        Initialize strategy loader
        
        Args:
            strategies_path: Path to strategies directory
        """
        if strategies_path is None:
            # Default to backend/executable_workflow/strategies
            strategies_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'executable_workflow',
                'strategies'
            )
        
        self.strategies_path = strategies_path
        self.loaded_strategies = {}
        
        # Add strategies path to Python path
        if self.strategies_path not in sys.path:
            sys.path.insert(0, self.strategies_path)
    
    def load_strategy(self, strategy_name: str, parameters: Dict[str, Any]) -> BaseStrategy:
        """
        Load and initialize a strategy
        
        Args:
            strategy_name: Name of the strategy (e.g., 'bollinger_bands')
            parameters: Strategy parameters
            
        Returns:
            Initialized strategy instance
        """
        # Check if strategy is already loaded
        if strategy_name in self.loaded_strategies:
            strategy_class = self.loaded_strategies[strategy_name]
        else:
            strategy_class = self._import_strategy(strategy_name)
            self.loaded_strategies[strategy_name] = strategy_class
        
        # Create strategy instance
        strategy = strategy_class(parameters)
        logger.info(f"Loaded strategy: {strategy_name} with parameters: {parameters}")
        
        return strategy
    
    def _import_strategy(self, strategy_name: str):
        """
        Import a strategy module and return the strategy class
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy class
        """
        # Map strategy names to class names
        strategy_map = {
            'bollinger_bands': 'BollingerBandsMeanReversion',
            'rsi_divergence': 'RSIMomentumDivergence',
            'macd_momentum': 'MACDMomentumCrossover',
            'volatility_breakout_short': 'VolatilityBreakoutShort',
            'ichimoku_cloud': 'IchimokuCloudBreakout',
            'parabolic_sar': 'ParabolicSARTrendFollowing',
            'fibonacci_retracement': 'FibonacciRetracementSupportResistance',
            'gaussian_channel': 'GaussianChannelBreakoutMeanReversion'
        }
        
        class_name = strategy_map.get(strategy_name)
        if not class_name:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Try to import from existing strategies
        try:
            # Import from backend.executable_workflow.strategies
            module = importlib.import_module(f'executable_workflow.strategies.{strategy_name}')
            strategy_class = getattr(module, class_name)
            return strategy_class
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not import existing strategy {strategy_name}: {e}")
            
        # If not found, create a wrapper using the strategy definition
        return self._create_strategy_wrapper(strategy_name, class_name)
    
    def _create_strategy_wrapper(self, strategy_name: str, class_name: str):
        """
        Create a strategy wrapper that adapts existing strategies to backtesting format
        
        Args:
            strategy_name: Strategy identifier
            class_name: Strategy class name
            
        Returns:
            Wrapped strategy class
        """
        class StrategyWrapper(BaseStrategy):
            def __init__(self, parameters):
                super().__init__(parameters)
                self.strategy_name = strategy_name
                
            def generate_signals(self, data):
                """Generate signals based on strategy logic"""
                df = data.copy()
                
                # Ensure we have a timestamp column and reset index to avoid duplicates
                if df.index.name == 'timestamp':
                    df = df.reset_index()
                elif 'timestamp' not in df.columns:
                    df['timestamp'] = df.index
                    df = df.reset_index(drop=True)
                
                # Initialize signal columns
                df['signal'] = 0
                df['strength'] = 0.0
                df['price'] = df['close']
                
                # Apply strategy-specific logic
                if strategy_name == 'bollinger_bands':
                    df = self._bollinger_bands_signals(df)
                elif strategy_name == 'rsi_divergence':
                    df = self._rsi_divergence_signals(df)
                elif strategy_name == 'macd_momentum':
                    df = self._macd_momentum_signals(df)
                elif strategy_name == 'volatility_breakout_short':
                    df = self._volatility_breakout_signals(df)
                elif strategy_name == 'ichimoku_cloud':
                    df = self._ichimoku_cloud_signals(df)
                elif strategy_name == 'parabolic_sar':
                    df = self._parabolic_sar_signals(df)
                elif strategy_name == 'fibonacci_retracement':
                    df = self._fibonacci_retracement_signals(df)
                elif strategy_name == 'gaussian_channel':
                    df = self._gaussian_channel_signals(df)
                
                # Return only signal columns with regular index (not timestamp)
                result_df = df[['timestamp', 'signal', 'strength', 'price']].copy()
                return result_df
            
            def _bollinger_bands_signals(self, df):
                """Bollinger Bands Mean Reversion strategy"""
                # Calculate Bollinger Bands
                bb_length = self.parameters.get('bb_length', 20)
                bb_std = self.parameters.get('bb_std', 2.0)
                
                df['bb_middle'] = df['close'].rolling(window=bb_length).mean()
                bb_std_dev = df['close'].rolling(window=bb_length).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
                df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
                
                # Calculate RSI
                rsi_length = self.parameters.get('rsi_length', 14)
                rsi_oversold = self.parameters.get('rsi_oversold', 35)
                rsi_overbought = self.parameters.get('rsi_overbought', 65)
                
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_length).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_length).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # Generate signals
                # Buy when price touches lower band and RSI is oversold
                buy_condition = (df['close'] <= df['bb_lower']) & (df['rsi'] < rsi_oversold)
                # Sell when price touches upper band and RSI is overbought
                sell_condition = (df['close'] >= df['bb_upper']) & (df['rsi'] > rsi_overbought)
                
                df.loc[buy_condition, 'signal'] = 1
                df.loc[sell_condition, 'signal'] = -1
                
                # Calculate signal strength based on how far price is from bands
                df.loc[buy_condition, 'strength'] = (df['bb_lower'] - df['close']) / df['bb_lower']
                df.loc[sell_condition, 'strength'] = (df['close'] - df['bb_upper']) / df['bb_upper']
                df['strength'] = df['strength'].clip(0, 1)
                
                return df
            
            def _rsi_divergence_signals(self, df):
                """RSI Momentum Divergence strategy"""
                # Calculate RSI
                rsi_length = self.parameters.get('rsi_length', 14)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_length).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_length).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # RSI SMAs
                rsi_sma_fast = self.parameters.get('rsi_sma_fast', 5)
                rsi_sma_slow = self.parameters.get('rsi_sma_slow', 10)
                df['rsi_sma_fast'] = df['rsi'].rolling(window=rsi_sma_fast).mean()
                df['rsi_sma_slow'] = df['rsi'].rolling(window=rsi_sma_slow).mean()
                
                # Divergence detection (simplified)
                lookback = self.parameters.get('divergence_lookback', 20)
                
                # Bullish divergence: price making lower lows, RSI making higher lows
                price_lows = df['low'].rolling(window=lookback).min()
                rsi_lows = df['rsi'].rolling(window=lookback).min()
                
                price_lower = df['low'] < price_lows.shift(1)
                rsi_higher = df['rsi'] > rsi_lows.shift(1)
                bullish_div = price_lower & rsi_higher
                
                # Bearish divergence: price making higher highs, RSI making lower highs
                price_highs = df['high'].rolling(window=lookback).max()
                rsi_highs = df['rsi'].rolling(window=lookback).max()
                
                price_higher = df['high'] > price_highs.shift(1)
                rsi_lower = df['rsi'] < rsi_highs.shift(1)
                bearish_div = price_higher & rsi_lower
                
                # Generate signals
                rsi_oversold = self.parameters.get('rsi_oversold', 30)
                rsi_overbought = self.parameters.get('rsi_overbought', 70)
                
                df.loc[bullish_div & (df['rsi'] < rsi_oversold), 'signal'] = 1
                df.loc[bearish_div & (df['rsi'] > rsi_overbought), 'signal'] = -1
                df.loc[df['signal'] != 0, 'strength'] = 0.8
                
                return df
            
            def _macd_momentum_signals(self, df):
                """MACD Momentum Crossover strategy"""
                # Calculate MACD
                macd_fast = self.parameters.get('macd_fast', 12)
                macd_slow = self.parameters.get('macd_slow', 26)
                macd_signal = self.parameters.get('macd_signal', 9)
                
                df['ema_fast'] = df['close'].ewm(span=macd_fast).mean()
                df['ema_slow'] = df['close'].ewm(span=macd_slow).mean()
                df['macd'] = df['ema_fast'] - df['ema_slow']
                df['macd_signal'] = df['macd'].ewm(span=macd_signal).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
                
                # Momentum
                momentum_period = self.parameters.get('momentum_period', 14)
                df['momentum'] = df['close'] - df['close'].shift(momentum_period)
                
                # Volume filter
                volume_threshold = self.parameters.get('volume_threshold', 1.5)
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                high_volume = df['volume'] > (df['volume_sma'] * volume_threshold)
                
                # Generate signals
                # Buy: MACD crosses above signal with positive momentum and high volume
                macd_cross_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
                buy_condition = macd_cross_up & (df['momentum'] > 0) & high_volume
                
                # Sell: MACD crosses below signal with negative momentum
                macd_cross_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
                sell_condition = macd_cross_down & (df['momentum'] < 0)
                
                df.loc[buy_condition, 'signal'] = 1
                df.loc[sell_condition, 'signal'] = -1
                df.loc[df['signal'] != 0, 'strength'] = 0.7
                
                return df
            
            def _volatility_breakout_signals(self, df):
                """Volatility Breakout Short strategy"""
                # Calculate ATR
                atr_period = self.parameters.get('atr_period', 14)
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift(1))
                low_close = np.abs(df['low'] - df['close'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr'] = true_range.rolling(window=atr_period).mean()
                
                # Volatility expansion
                lookback = self.parameters.get('lookback_period', 20)
                df['volatility_percentile'] = df['atr'].rolling(window=lookback).apply(
                    lambda x: (x.iloc[-1] > x.quantile(0.8)).astype(int)
                )
                
                # RSI extreme
                rsi_period = self.parameters.get('rsi_period', 14)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                rsi_extreme = self.parameters.get('rsi_extreme', 20)
                
                # Volume spike
                volume_multiplier = self.parameters.get('volume_multiplier', 2.0)
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                volume_spike = df['volume'] > (df['volume_sma'] * volume_multiplier)
                
                # Breakout detection
                df['high_20'] = df['high'].rolling(window=lookback).max()
                df['low_20'] = df['low'].rolling(window=lookback).min()
                
                # Short signals when breaking down with high volatility
                breakdown = df['close'] < df['low_20'].shift(1)
                short_condition = breakdown & (df['volatility_percentile'] == 1) & volume_spike & (df['rsi'] < rsi_extreme)
                
                # Cover signals when RSI becomes oversold
                cover_condition = df['rsi'] < 20
                
                df.loc[short_condition, 'signal'] = -1  # Short
                df.loc[cover_condition & (df['signal'].shift(1) == -1), 'signal'] = 1  # Cover
                df.loc[df['signal'] != 0, 'strength'] = 0.8
                
                return df
            
            def _ichimoku_cloud_signals(self, df):
                """Ichimoku Cloud Breakout strategy"""
                # Ichimoku parameters
                tenkan_period = self.parameters.get('tenkan_period', 9)
                kijun_period = self.parameters.get('kijun_period', 26)
                senkou_b_period = self.parameters.get('senkou_b_period', 52)
                displacement = self.parameters.get('displacement', 26)
                
                # Tenkan-sen (Conversion Line)
                high_tenkan = df['high'].rolling(window=tenkan_period).max()
                low_tenkan = df['low'].rolling(window=tenkan_period).min()
                df['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
                
                # Kijun-sen (Base Line)
                high_kijun = df['high'].rolling(window=kijun_period).max()
                low_kijun = df['low'].rolling(window=kijun_period).min()
                df['kijun_sen'] = (high_kijun + low_kijun) / 2
                
                # Senkou Span A (Leading Span A)
                df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(displacement)
                
                # Senkou Span B (Leading Span B)
                high_senkou = df['high'].rolling(window=senkou_b_period).max()
                low_senkou = df['low'].rolling(window=senkou_b_period).min()
                df['senkou_span_b'] = ((high_senkou + low_senkou) / 2).shift(displacement)
                
                # Cloud signals
                df['cloud_top'] = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
                df['cloud_bottom'] = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
                
                # Buy: Price breaks above cloud
                above_cloud = (df['close'] > df['cloud_top']) & (df['close'].shift(1) <= df['cloud_top'].shift(1))
                tenkan_above_kijun = df['tenkan_sen'] > df['kijun_sen']
                
                # Sell: Price breaks below cloud
                below_cloud = (df['close'] < df['cloud_bottom']) & (df['close'].shift(1) >= df['cloud_bottom'].shift(1))
                tenkan_below_kijun = df['tenkan_sen'] < df['kijun_sen']
                
                df.loc[above_cloud & tenkan_above_kijun, 'signal'] = 1
                df.loc[below_cloud & tenkan_below_kijun, 'signal'] = -1
                df.loc[df['signal'] != 0, 'strength'] = 0.8
                
                return df
            
            def _parabolic_sar_signals(self, df):
                """Parabolic SAR Trend Following strategy"""
                # SAR parameters
                start = self.parameters.get('start', 0.02)
                increment = self.parameters.get('increment', 0.02)
                maximum = self.parameters.get('maximum', 0.2)
                
                # Simple SAR calculation (simplified version)
                df['sar'] = df['close'].copy()
                df['ep'] = df['high'].copy()  # Extreme point
                df['af'] = start  # Acceleration factor
                df['uptrend'] = True
                
                for i in range(1, len(df)):
                    if df['uptrend'].iloc[i-1]:
                        df.loc[df.index[i], 'sar'] = df['sar'].iloc[i-1] + df['af'].iloc[i-1] * (df['ep'].iloc[i-1] - df['sar'].iloc[i-1])
                        df.loc[df.index[i], 'sar'] = min(df['sar'].iloc[i], df['low'].iloc[i-1])
                        
                        if df['high'].iloc[i] > df['ep'].iloc[i-1]:
                            df.loc[df.index[i], 'ep'] = df['high'].iloc[i]
                            df.loc[df.index[i], 'af'] = min(df['af'].iloc[i-1] + increment, maximum)
                        else:
                            df.loc[df.index[i], 'ep'] = df['ep'].iloc[i-1]
                            df.loc[df.index[i], 'af'] = df['af'].iloc[i-1]
                            
                        if df['low'].iloc[i] < df['sar'].iloc[i]:
                            df.loc[df.index[i], 'uptrend'] = False
                            df.loc[df.index[i], 'sar'] = df['ep'].iloc[i-1]
                            df.loc[df.index[i], 'ep'] = df['low'].iloc[i]
                            df.loc[df.index[i], 'af'] = start
                    else:
                        df.loc[df.index[i], 'sar'] = df['sar'].iloc[i-1] + df['af'].iloc[i-1] * (df['ep'].iloc[i-1] - df['sar'].iloc[i-1])
                        df.loc[df.index[i], 'sar'] = max(df['sar'].iloc[i], df['high'].iloc[i-1])
                        
                        if df['low'].iloc[i] < df['ep'].iloc[i-1]:
                            df.loc[df.index[i], 'ep'] = df['low'].iloc[i]
                            df.loc[df.index[i], 'af'] = min(df['af'].iloc[i-1] + increment, maximum)
                        else:
                            df.loc[df.index[i], 'ep'] = df['ep'].iloc[i-1]
                            df.loc[df.index[i], 'af'] = df['af'].iloc[i-1]
                            
                        if df['high'].iloc[i] > df['sar'].iloc[i]:
                            df.loc[df.index[i], 'uptrend'] = True
                            df.loc[df.index[i], 'sar'] = df['ep'].iloc[i-1]
                            df.loc[df.index[i], 'ep'] = df['high'].iloc[i]
                            df.loc[df.index[i], 'af'] = start
                
                # Generate signals
                sar_flip_up = (df['close'] > df['sar']) & (df['close'].shift(1) <= df['sar'].shift(1))
                sar_flip_down = (df['close'] < df['sar']) & (df['close'].shift(1) >= df['sar'].shift(1))
                
                df.loc[sar_flip_up, 'signal'] = 1
                df.loc[sar_flip_down, 'signal'] = -1
                df.loc[df['signal'] != 0, 'strength'] = 0.7
                
                return df
            
            def _fibonacci_retracement_signals(self, df):
                """Fibonacci Retracement Support/Resistance strategy"""
                lookback_period = self.parameters.get('lookback_period', 50)
                fib_levels = self.parameters.get('fib_levels', [0.236, 0.382, 0.5, 0.618, 0.786])
                
                # Find recent high and low
                recent_high = df['high'].rolling(window=lookback_period).max()
                recent_low = df['low'].rolling(window=lookback_period).min()
                
                # Calculate Fibonacci levels
                price_range = recent_high - recent_low
                
                for level in fib_levels:
                    df[f'fib_{level}'] = recent_high - (price_range * level)
                
                # Detect bounces off Fibonacci levels
                for level in fib_levels:
                    fib_price = df[f'fib_{level}']
                    
                    # Buy signal when price bounces off Fibonacci support
                    touch_support = (df['low'] <= fib_price * 1.01) & (df['close'] > fib_price)
                    
                    # Sell signal when price rejects Fibonacci resistance
                    touch_resistance = (df['high'] >= fib_price * 0.99) & (df['close'] < fib_price)
                    
                    # Weight signals by Fibonacci level importance
                    if level in [0.382, 0.5, 0.618]:  # Golden ratios
                        strength = 0.8
                    else:
                        strength = 0.6
                    
                    df.loc[touch_support & (df['signal'] == 0), 'signal'] = 1
                    df.loc[touch_support & (df['signal'] == 1), 'strength'] = strength
                    
                    df.loc[touch_resistance & (df['signal'] == 0), 'signal'] = -1
                    df.loc[touch_resistance & (df['signal'] == -1), 'strength'] = strength
                
                return df
            
            def _gaussian_channel_signals(self, df):
                """Gaussian Channel Breakout/Mean Reversion strategy"""
                period = self.parameters.get('period', 20)
                std_dev = self.parameters.get('std_dev', 2.0)
                adaptive = self.parameters.get('adaptive', True)
                
                # Calculate Gaussian Channel
                df['gc_middle'] = df['close'].rolling(window=period).mean()
                
                if adaptive:
                    # Adaptive standard deviation based on market volatility
                    df['volatility'] = df['close'].pct_change().rolling(window=period).std()
                    df['volatility_percentile'] = df['volatility'].rolling(window=100).rank(pct=True)
                    
                    # Adjust channel width based on volatility
                    dynamic_std = std_dev * (1 + df['volatility_percentile'])
                    channel_std = df['close'].rolling(window=period).std() * dynamic_std
                else:
                    channel_std = df['close'].rolling(window=period).std() * std_dev
                
                df['gc_upper'] = df['gc_middle'] + channel_std
                df['gc_lower'] = df['gc_middle'] - channel_std
                
                # Mean reversion signals
                # Buy when price touches lower channel in low volatility
                low_vol = df.get('volatility_percentile', 0.5) < 0.5 if adaptive else True
                touch_lower = df['low'] <= df['gc_lower']
                
                # Sell when price touches upper channel in low volatility
                touch_upper = df['high'] >= df['gc_upper']
                
                # Breakout signals in high volatility
                high_vol = df.get('volatility_percentile', 0.5) > 0.7 if adaptive else False
                break_upper = (df['close'] > df['gc_upper']) & (df['close'].shift(1) <= df['gc_upper'].shift(1))
                break_lower = (df['close'] < df['gc_lower']) & (df['close'].shift(1) >= df['gc_lower'].shift(1))
                
                # Mean reversion mode
                df.loc[touch_lower & low_vol, 'signal'] = 1
                df.loc[touch_upper & low_vol, 'signal'] = -1
                df.loc[(touch_lower & low_vol) | (touch_upper & low_vol), 'strength'] = 0.7
                
                # Breakout mode
                df.loc[break_upper & high_vol, 'signal'] = 1
                df.loc[break_lower & high_vol, 'signal'] = -1
                df.loc[(break_upper & high_vol) | (break_lower & high_vol), 'strength'] = 0.9
                
                return df
        
        # Set the class name dynamically
        StrategyWrapper.__name__ = class_name
        return StrategyWrapper
    
    def get_available_strategies(self) -> Dict[str, str]:
        """
        Get list of available strategies
        
        Returns:
            Dictionary mapping strategy names to descriptions
        """
        strategies = {
            'bollinger_bands': 'Bollinger Bands Mean Reversion - Trade bounces off bands with RSI confirmation',
            'rsi_divergence': 'RSI Momentum Divergence - Detect price/RSI divergences for reversal trades',
            'macd_momentum': 'MACD Momentum Crossover - Trade MACD signal crossovers with momentum filter',
            'volatility_breakout_short': 'Volatility Breakout Short - Short volatility expansions at resistance',
            'ichimoku_cloud': 'Ichimoku Cloud Breakout - Trade cloud breakouts with trend confirmation',
            'parabolic_sar': 'Parabolic SAR Trend Following - Follow trends using SAR indicator',
            'fibonacci_retracement': 'Fibonacci Retracement - Trade bounces off key Fibonacci levels',
            'gaussian_channel': 'Gaussian Channel - Adaptive channel trading based on volatility'
        }
        
        return strategies
