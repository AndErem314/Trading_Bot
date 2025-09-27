"""
Example Usage of Strategy Configuration System

This script demonstrates how to use the StrategyConfigManager to:
1. Load configurations from YAML/JSON files
2. Access strategy parameters
3. Validate configurations
4. Create new strategy configurations programmatically
"""

import logging
from pathlib import Path
from strategy_config_manager import StrategyConfigManager, StrategyConfig, SignalConditions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Demonstrate strategy configuration usage."""
    
    # Initialize configuration manager
    config_manager = StrategyConfigManager()
    
    print("=" * 60)
    print("STRATEGY CONFIGURATION MANAGER DEMO")
    print("=" * 60)
    
    # Example 1: Load configurations from YAML
    print("\n1. Loading configurations from YAML file...")
    try:
        yaml_strategies = config_manager.load_config("strategies.yaml")
        print(f"✅ Loaded {len(yaml_strategies)} strategies from YAML")
        
        # List loaded strategies
        print("\nLoaded strategies:")
        for strategy_id, config in yaml_strategies.items():
            print(f"  - {strategy_id}: {config.name} ({'Enabled' if config.enabled else 'Disabled'})")
    except Exception as e:
        print(f"❌ Error loading YAML: {e}")
    
    # Example 2: Load configurations from JSON
    print("\n2. Loading configurations from JSON file...")
    try:
        json_strategies = config_manager.load_config("strategies_advanced.json")
        print(f"✅ Loaded {len(json_strategies)} strategies from JSON")
        
        for strategy_id, config in json_strategies.items():
            print(f"  - {strategy_id}: {config.name}")
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
    
    # Example 3: Get specific strategy parameters
    print("\n3. Accessing specific strategy parameters...")
    strategy_name = "Conservative Ichimoku"
    strategy_config = config_manager.get_strategy_params(strategy_name)
    
    if strategy_config:
        print(f"\nStrategy: {strategy_config.name}")
        print(f"Description: {strategy_config.description}")
        print(f"Timeframe: {strategy_config.timeframe}")
        print(f"Symbols: {', '.join(strategy_config.symbols)}")
        
        print("\nSignal Conditions:")
        print(f"  Buy: {' AND '.join(strategy_config.signal_conditions.buy_conditions)}")
        print(f"  Sell: {' AND '.join(strategy_config.signal_conditions.sell_conditions)}")
        
        print("\nRisk Management:")
        print(f"  Stop Loss: {strategy_config.risk_management.stop_loss_pct}%")
        print(f"  Take Profit: {strategy_config.risk_management.take_profit_pct}%")
        print(f"  Trailing Stop: {'Yes' if strategy_config.risk_management.trailing_stop else 'No'}")
        
        print("\nPosition Sizing:")
        print(f"  Method: {strategy_config.position_sizing.method}")
        print(f"  Max Leverage: {strategy_config.position_sizing.max_leverage}x")
    
    # Example 4: List enabled strategies
    print("\n4. Listing enabled strategies...")
    enabled = config_manager.list_enabled_strategies()
    print(f"Enabled strategies: {', '.join(enabled)}")
    
    # Example 5: Validate configuration
    print("\n5. Validating configurations...")
    for strategy_name in config_manager.list_strategies():
        config = config_manager.get_strategy_params(strategy_name)
        try:
            config_manager.validate_config(config)
            print(f"✅ {strategy_name}: Valid")
        except Exception as e:
            print(f"❌ {strategy_name}: {e}")
    
    # Example 6: Create and save new configuration
    print("\n6. Creating new strategy configuration...")
    
    # Create a custom strategy
    from strategy_config_manager import (
        SignalConditions, IchimokuParameters, 
        RiskManagement, PositionSizing
    )
    
    custom_strategy = StrategyConfig(
        name="Custom Test Strategy",
        description="A custom strategy for testing",
        enabled=True,
        signal_conditions=SignalConditions(
            buy_conditions=["PriceAboveCloud", "ChikouAbovePrice"],
            sell_conditions=["PriceBelowCloud", "ChikouBelowPrice"],
            buy_logic="AND",
            sell_logic="AND"
        ),
        ichimoku_parameters=IchimokuParameters(
            tenkan_period=8,
            kijun_period=24
        ),
        risk_management=RiskManagement(
            stop_loss_pct=1.5,
            take_profit_pct=4.5
        ),
        position_sizing=PositionSizing(
            method="fixed",
            fixed_size=0.5
        ),
        timeframe="30m",
        symbols=["BTC/USDT"]
    )
    
    # Save the custom strategy
    config_manager.save_config(custom_strategy, "custom_strategy.yaml")
    print("✅ Saved custom strategy to custom_strategy.yaml")
    
    # Example 7: Merge multiple configuration files
    print("\n7. Merging configuration files...")
    all_configs = config_manager.merge_configs(
        "strategies.yaml",
        "strategies_advanced.json",
        "custom_strategy.yaml"
    )
    print(f"✅ Total strategies after merge: {len(all_configs)}")
    
    # Example 8: Using configurations with signal detector
    print("\n8. Integration example with IchimokuSignalDetector...")
    print("(This shows how to use config with actual trading logic)")
    
    # Get a strategy config
    momentum_strategy = config_manager.get_strategy_params("Momentum Ichimoku")
    if momentum_strategy:
        print(f"\nUsing strategy: {momentum_strategy.name}")
        print(f"Buy conditions to check: {momentum_strategy.signal_conditions.buy_conditions}")
        print(f"Ichimoku Tenkan period: {momentum_strategy.ichimoku_parameters.tenkan_period}")
        print(f"Stop loss: {momentum_strategy.risk_management.stop_loss_pct}%")
        
        # In real usage, you would:
        # 1. Use ichimoku_parameters to calculate indicators
        # 2. Use signal_conditions to detect signals
        # 3. Use risk_management for position management
        # 4. Use position_sizing to calculate trade size


if __name__ == "__main__":
    main()