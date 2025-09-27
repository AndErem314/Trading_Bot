"""
Strategy Configuration Manager

This module provides a comprehensive configuration management system for trading strategies.
It supports loading, validating, and managing strategy configurations from YAML/JSON files,
allowing flexible definition of trading rules, parameters, and risk management settings.

Features:
- Multiple strategy configurations support
- Signal condition combinations using Ichimoku characteristics
- Configurable indicator parameters
- Risk management settings (stop loss, take profit)
- Position sizing rules
- Configuration validation and error handling
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import copy

# Configure logging
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


class SignalCondition(Enum):
    """Available signal conditions matching IchimokuSignalDetector."""
    PRICE_ABOVE_CLOUD = "PriceAboveCloud"
    PRICE_BELOW_CLOUD = "PriceBelowCloud"
    TENKAN_ABOVE_KIJUN = "TenkanAboveKijun"
    TENKAN_BELOW_KIJUN = "TenkanBelowKijun"
    CHIKOU_ABOVE_PRICE = "ChikouAbovePrice"
    CHIKOU_BELOW_PRICE = "ChikouBelowPrice"
    CHIKOU_ABOVE_CLOUD = "ChikouAboveCloud"
    CHIKOU_BELOW_CLOUD = "ChikouBelowCloud"
    CHIKOU_CROSS_ABOVE_SENKOU_B = "ChikouCrossAboveSenkouB"
    CHIKOU_CROSS_BELOW_SENKOU_B = "ChikouCrossBelowSenkouB"


@dataclass
class IchimokuParameters:
    """Ichimoku indicator parameters."""
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_b_period: int = 52
    chikou_offset: int = 26
    senkou_offset: int = 26


@dataclass
class RiskManagement:
    """Risk management settings."""
    stop_loss_pct: float = 2.0  # Percentage
    take_profit_pct: float = 6.0  # Percentage
    trailing_stop: bool = False
    trailing_stop_pct: float = 1.5
    max_position_size_pct: float = 100.0  # Percentage of capital
    risk_per_trade_pct: float = 2.0  # Percentage of capital to risk


@dataclass
class PositionSizing:
    """Position sizing rules."""
    method: str = "fixed"  # fixed, risk_based, volatility_based
    fixed_size: float = 1.0  # For fixed method
    use_kelly_criterion: bool = False
    max_leverage: float = 1.0
    min_position_size: float = 0.01
    max_position_size: float = 10.0


@dataclass
class SignalConditions:
    """Signal conditions for entry/exit."""
    buy_conditions: List[str]
    sell_conditions: List[str]
    buy_logic: str = "AND"  # AND or OR
    sell_logic: str = "AND"  # AND or OR


@dataclass
class StrategyConfig:
    """Complete strategy configuration."""
    name: str
    description: str
    enabled: bool
    signal_conditions: SignalConditions
    ichimoku_parameters: IchimokuParameters
    risk_management: RiskManagement
    position_sizing: PositionSizing
    timeframe: str = "1h"
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC/USDT"]


class StrategyConfigManager:
    """
    Manages strategy configurations from YAML/JSON files.
    
    This class handles loading, validating, and accessing strategy configurations,
    ensuring all parameters are properly set and valid before use.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to config directory relative to this file
            self.config_dir = Path(__file__).parent
        
        self.strategies: Dict[str, StrategyConfig] = {}
        self.valid_signal_conditions = {cond.value for cond in SignalCondition}
    
    def load_config(self, config_file: str) -> Dict[str, StrategyConfig]:
        """
        Load strategy configurations from a YAML or JSON file.
        
        Args:
            config_file: Path to configuration file (relative to config_dir or absolute)
            
        Returns:
            Dictionary of strategy configurations
            
        Raises:
            ConfigError: If file cannot be loaded or parsed
        """
        # Determine file path
        if os.path.isabs(config_file):
            file_path = Path(config_file)
        else:
            file_path = self.config_dir / config_file
        
        if not file_path.exists():
            raise ConfigError(f"Configuration file not found: {file_path}")
        
        try:
            # Load file based on extension
            if file_path.suffix in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ConfigError(f"Unsupported file format: {file_path.suffix}")
            
            # Parse strategies
            if 'strategies' not in config_data:
                raise ConfigError("No 'strategies' section found in configuration")
            
            strategies = {}
            for strategy_id, strategy_data in config_data['strategies'].items():
                strategy_config = self._parse_strategy_config(strategy_id, strategy_data)
                strategies[strategy_id] = strategy_config
                
            # Validate all strategies
            for strategy_id, strategy_config in strategies.items():
                self.validate_config(strategy_config)
            
            # Store loaded strategies
            self.strategies.update(strategies)
            
            logger.info(f"Loaded {len(strategies)} strategies from {file_path}")
            return strategies
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML file: {e}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Error parsing JSON file: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading configuration: {e}")
    
    def _parse_strategy_config(self, strategy_id: str, data: Dict[str, Any]) -> StrategyConfig:
        """Parse a single strategy configuration."""
        try:
            # Parse signal conditions
            signal_data = data.get('signal_conditions', {})
            signal_conditions = SignalConditions(
                buy_conditions=signal_data.get('buy_conditions', []),
                sell_conditions=signal_data.get('sell_conditions', []),
                buy_logic=signal_data.get('buy_logic', 'AND'),
                sell_logic=signal_data.get('sell_logic', 'AND')
            )
            
            # Parse Ichimoku parameters
            ichimoku_data = data.get('ichimoku_parameters', {})
            ichimoku_params = IchimokuParameters(
                tenkan_period=ichimoku_data.get('tenkan_period', 9),
                kijun_period=ichimoku_data.get('kijun_period', 26),
                senkou_b_period=ichimoku_data.get('senkou_b_period', 52),
                chikou_offset=ichimoku_data.get('chikou_offset', 26),
                senkou_offset=ichimoku_data.get('senkou_offset', 26)
            )
            
            # Parse risk management
            risk_data = data.get('risk_management', {})
            risk_management = RiskManagement(
                stop_loss_pct=risk_data.get('stop_loss_pct', 2.0),
                take_profit_pct=risk_data.get('take_profit_pct', 6.0),
                trailing_stop=risk_data.get('trailing_stop', False),
                trailing_stop_pct=risk_data.get('trailing_stop_pct', 1.5),
                max_position_size_pct=risk_data.get('max_position_size_pct', 100.0),
                risk_per_trade_pct=risk_data.get('risk_per_trade_pct', 2.0)
            )
            
            # Parse position sizing
            sizing_data = data.get('position_sizing', {})
            position_sizing = PositionSizing(
                method=sizing_data.get('method', 'fixed'),
                fixed_size=sizing_data.get('fixed_size', 1.0),
                use_kelly_criterion=sizing_data.get('use_kelly_criterion', False),
                max_leverage=sizing_data.get('max_leverage', 1.0),
                min_position_size=sizing_data.get('min_position_size', 0.01),
                max_position_size=sizing_data.get('max_position_size', 10.0)
            )
            
            # Create strategy config
            strategy_config = StrategyConfig(
                name=data.get('name', strategy_id),
                description=data.get('description', f"Strategy {strategy_id}"),
                enabled=data.get('enabled', True),
                signal_conditions=signal_conditions,
                ichimoku_parameters=ichimoku_params,
                risk_management=risk_management,
                position_sizing=position_sizing,
                timeframe=data.get('timeframe', '1h'),
                symbols=data.get('symbols', ['BTC/USDT'])
            )
            
            return strategy_config
            
        except Exception as e:
            raise ConfigError(f"Error parsing strategy '{strategy_id}': {e}")
    
    def validate_config(self, config: StrategyConfig) -> bool:
        """
        Validate a strategy configuration.
        
        Args:
            config: Strategy configuration to validate
            
        Returns:
            True if valid
            
        Raises:
            ConfigError: If configuration is invalid
        """
        errors = []
        
        # Validate signal conditions
        for condition in config.signal_conditions.buy_conditions:
            if condition not in self.valid_signal_conditions:
                errors.append(f"Invalid buy condition: {condition}")
        
        for condition in config.signal_conditions.sell_conditions:
            if condition not in self.valid_signal_conditions:
                errors.append(f"Invalid sell condition: {condition}")
        
        if not config.signal_conditions.buy_conditions:
            errors.append("No buy conditions specified")
        
        if not config.signal_conditions.sell_conditions:
            errors.append("No sell conditions specified")
        
        if config.signal_conditions.buy_logic not in ['AND', 'OR']:
            errors.append(f"Invalid buy logic: {config.signal_conditions.buy_logic}")
        
        if config.signal_conditions.sell_logic not in ['AND', 'OR']:
            errors.append(f"Invalid sell logic: {config.signal_conditions.sell_logic}")
        
        # Validate Ichimoku parameters
        if config.ichimoku_parameters.tenkan_period <= 0:
            errors.append("Tenkan period must be positive")
        
        if config.ichimoku_parameters.kijun_period <= 0:
            errors.append("Kijun period must be positive")
        
        if config.ichimoku_parameters.senkou_b_period <= 0:
            errors.append("Senkou B period must be positive")
        
        # Validate risk management
        if config.risk_management.stop_loss_pct < 0:
            errors.append("Stop loss percentage cannot be negative")
        
        if config.risk_management.take_profit_pct < 0:
            errors.append("Take profit percentage cannot be negative")
        
        if config.risk_management.risk_per_trade_pct < 0 or config.risk_management.risk_per_trade_pct > 100:
            errors.append("Risk per trade must be between 0 and 100%")
        
        # Validate position sizing
        if config.position_sizing.method not in ['fixed', 'risk_based', 'volatility_based']:
            errors.append(f"Invalid position sizing method: {config.position_sizing.method}")
        
        if config.position_sizing.max_leverage < 1:
            errors.append("Max leverage must be at least 1")
        
        if config.position_sizing.min_position_size <= 0:
            errors.append("Minimum position size must be positive")
        
        if config.position_sizing.min_position_size > config.position_sizing.max_position_size:
            errors.append("Min position size cannot exceed max position size")
        
        # Validate timeframe
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
        if config.timeframe not in valid_timeframes:
            errors.append(f"Invalid timeframe: {config.timeframe}")
        
        # Validate symbols
        if not config.symbols:
            errors.append("No symbols specified")
        
        if errors:
            raise ConfigError(f"Configuration validation failed:\n" + "\n".join(errors))
        
        return True
    
    def get_strategy_params(self, strategy_name: str) -> Optional[StrategyConfig]:
        """
        Get parameters for a specific strategy.
        
        Args:
            strategy_name: Name or ID of the strategy
            
        Returns:
            Strategy configuration or None if not found
        """
        # First check by strategy ID
        if strategy_name in self.strategies:
            return copy.deepcopy(self.strategies[strategy_name])
        
        # Then check by strategy name
        for strategy_id, config in self.strategies.items():
            if config.name == strategy_name:
                return copy.deepcopy(config)
        
        logger.warning(f"Strategy '{strategy_name}' not found")
        return None
    
    def list_strategies(self) -> List[str]:
        """Get list of loaded strategy names."""
        return [config.name for config in self.strategies.values()]
    
    def list_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategy names."""
        return [config.name for config in self.strategies.values() if config.enabled]
    
    def save_config(self, config: Union[StrategyConfig, Dict[str, StrategyConfig]], 
                   output_file: str, format: str = 'yaml') -> None:
        """
        Save strategy configuration(s) to file.
        
        Args:
            config: Single strategy config or dictionary of configs
            output_file: Output file path
            format: Output format ('yaml' or 'json')
        """
        # Prepare data
        if isinstance(config, StrategyConfig):
            data = {'strategies': {'custom_strategy': self._config_to_dict(config)}}
        else:
            data = {'strategies': {
                strategy_id: self._config_to_dict(cfg) 
                for strategy_id, cfg in config.items()
            }}
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved configuration to {output_path}")
    
    def _config_to_dict(self, config: StrategyConfig) -> Dict[str, Any]:
        """Convert StrategyConfig to dictionary for serialization."""
        return {
            'name': config.name,
            'description': config.description,
            'enabled': config.enabled,
            'timeframe': config.timeframe,
            'symbols': config.symbols,
            'signal_conditions': {
                'buy_conditions': config.signal_conditions.buy_conditions,
                'sell_conditions': config.signal_conditions.sell_conditions,
                'buy_logic': config.signal_conditions.buy_logic,
                'sell_logic': config.signal_conditions.sell_logic
            },
            'ichimoku_parameters': asdict(config.ichimoku_parameters),
            'risk_management': asdict(config.risk_management),
            'position_sizing': asdict(config.position_sizing)
        }
    
    def merge_configs(self, *config_files: str) -> Dict[str, StrategyConfig]:
        """
        Merge multiple configuration files.
        
        Later files override earlier ones for duplicate strategy IDs.
        
        Args:
            config_files: Configuration file paths
            
        Returns:
            Merged strategy configurations
        """
        merged = {}
        
        for file_path in config_files:
            strategies = self.load_config(file_path)
            merged.update(strategies)
        
        self.strategies = merged
        return merged


# Example usage and configuration creation
if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create example configurations
    manager = StrategyConfigManager()
    
    # Strategy 01: Conservative Ichimoku
    strategy01 = StrategyConfig(
        name="Conservative Ichimoku",
        description="Conservative strategy requiring multiple confirmations",
        enabled=True,
        signal_conditions=SignalConditions(
            buy_conditions=["PriceAboveCloud", "TenkanAboveKijun", "ChikouAboveCloud"],
            sell_conditions=["PriceBelowCloud", "TenkanBelowKijun", "ChikouBelowPrice"],
            buy_logic="AND",
            sell_logic="AND"
        ),
        ichimoku_parameters=IchimokuParameters(),  # Use defaults
        risk_management=RiskManagement(
            stop_loss_pct=2.0,
            take_profit_pct=6.0,
            trailing_stop=True,
            trailing_stop_pct=1.5,
            risk_per_trade_pct=2.0
        ),
        position_sizing=PositionSizing(
            method="risk_based",
            max_leverage=1.0
        ),
        timeframe="1h",
        symbols=["BTC/USDT", "ETH/USDT"]
    )
    
    # Strategy 02: Aggressive Ichimoku
    strategy02 = StrategyConfig(
        name="Aggressive Ichimoku",
        description="More aggressive strategy with relaxed conditions",
        enabled=True,
        signal_conditions=SignalConditions(
            buy_conditions=["PriceAboveCloud", "TenkanAboveKijun", "ChikouAbovePrice"],
            sell_conditions=["PriceBelowCloud", "TenkanBelowKijun", "ChikouBelowCloud"],
            buy_logic="AND",
            sell_logic="AND"
        ),
        ichimoku_parameters=IchimokuParameters(
            tenkan_period=7,  # Faster
            kijun_period=22   # Faster
        ),
        risk_management=RiskManagement(
            stop_loss_pct=3.0,  # Wider stop
            take_profit_pct=9.0,  # Higher target
            risk_per_trade_pct=3.0  # Higher risk
        ),
        position_sizing=PositionSizing(
            method="fixed",
            fixed_size=2.0  # Larger positions
        ),
        timeframe="4h",
        symbols=["BTC/USDT"]
    )
    
    # Save example configurations
    configs = {
        "strategy01": strategy01,
        "strategy02": strategy02
    }
    
    # Save as YAML
    manager.save_config(configs, "strategies.yaml", format='yaml')
    
    # Test loading
    loaded_configs = manager.load_config("strategies.yaml")
    
    print(f"Loaded {len(loaded_configs)} strategies:")
    for strategy_id, config in loaded_configs.items():
        print(f"\n{strategy_id}: {config.name}")
        print(f"  Buy conditions: {', '.join(config.signal_conditions.buy_conditions)}")
        print(f"  Sell conditions: {', '.join(config.signal_conditions.sell_conditions)}")
        print(f"  Stop loss: {config.risk_management.stop_loss_pct}%")
        print(f"  Take profit: {config.risk_management.take_profit_pct}%")