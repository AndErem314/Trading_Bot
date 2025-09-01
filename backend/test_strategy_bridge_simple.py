"""
Simple Strategy Bridge Test

This script tests the strategy bridge functionality focusing on
the descriptor access and structure without requiring executable strategies.

"""

import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.strategy_bridge import StrategyBridge, UnifiedStrategyFactory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_strategy_descriptors():
    """Test accessing strategy descriptors through the bridge."""
    logger.info("Testing Strategy Descriptors")
    logger.info("=" * 80)
    
    # Get all available strategies
    strategies = UnifiedStrategyFactory.get_available_strategies()
    logger.info(f"Found {len(strategies)} strategies available:")
    for i, strategy in enumerate(strategies, 1):
        logger.info(f"  {i}. {strategy}")
    
    # Test each strategy descriptor
    for strategy_name in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing descriptor for: {strategy_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create bridge
            bridge = StrategyBridge(strategy_name)
            logger.info(f"✓ Created bridge for {strategy_name}")
            
            # Get strategy info from descriptor
            info = bridge.get_strategy_info()
            descriptor_info = info['descriptor']
            
            # Display strategy details
            desc = descriptor_info['description']
            logger.info(f"\nStrategy Type: {desc.get('type', 'N/A')}")
            logger.info(f"Market Regime: {desc.get('market_regime', 'N/A')}")
            logger.info(f"Time Frame: {desc.get('time_frame', 'N/A')}")
            
            # Entry conditions
            logger.info(f"\nEntry Conditions:")
            entry = descriptor_info['entry_conditions']
            for key, value in entry.items():
                if isinstance(value, list):
                    logger.info(f"  {key}:")
                    for item in value:
                        logger.info(f"    - {item}")
                else:
                    logger.info(f"  {key}: {value}")
            
            # Exit conditions
            logger.info(f"\nExit Conditions:")
            exit_cond = descriptor_info['exit_conditions']
            for key, value in exit_cond.items():
                if isinstance(value, list):
                    logger.info(f"  {key}:")
                    for item in value:
                        logger.info(f"    - {item}")
                else:
                    logger.info(f"  {key}: {value}")
            
            # Risk management
            logger.info(f"\nRisk Management:")
            risk = descriptor_info['risk_rules']
            logger.info(f"  Stop Loss: {risk.get('stop_loss', 'N/A')}")
            logger.info(f"  Take Profit: {risk.get('take_profit', 'N/A')}")
            logger.info(f"  Position Size: {risk.get('position_size', 'N/A')}")
            
            # Get SQL query
            sql_query = bridge.descriptor.get_sql_query()
            logger.info(f"\nSQL Query Length: {len(sql_query)} characters")
            logger.info(f"SQL Query Preview: {sql_query[:100]}...")
            
            # Test market regime suitability (using bridge logic)
            logger.info(f"\nMarket Regime Suitability:")
            for regime in ['Bullish', 'Bearish', 'Neutral', 'Ranging']:
                suitable = bridge.is_strategy_allowed(regime)
                logger.info(f"  {regime}: {'✓' if suitable else '✗'}")
            
            logger.info(f"\n✓ Successfully tested {strategy_name}")
            
        except Exception as e:
            logger.error(f"✗ Error testing {strategy_name}: {str(e)}")
            import traceback
            traceback.print_exc()


def test_strategy_mapping():
    """Test the mapping between descriptors and executable classes."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Strategy Mapping")
    logger.info("=" * 80)
    
    # Check the mapping
    logger.info("\nStrategy Mapping:")
    for descriptor_class, executable_class in StrategyBridge.STRATEGY_MAPPING.items():
        logger.info(f"  {descriptor_class} -> {executable_class.__name__}")
    
    logger.info(f"\nTotal mappings: {len(StrategyBridge.STRATEGY_MAPPING)}")


def test_factory_methods():
    """Test the factory methods."""
    logger.info("\n" + "=" * 80)
    logger.info("Testing Factory Methods")
    logger.info("=" * 80)
    
    # Test creating individual strategies
    test_strategies = [
        'RSI_Momentum_Divergence',
        'Bollinger_Bands_Mean_Reversion',
        'MACD_Momentum_Crossover'
    ]
    
    for strategy_name in test_strategies:
        try:
            bridge = UnifiedStrategyFactory.create_strategy(strategy_name)
            logger.info(f"✓ Created {strategy_name} using factory")
            
            # Verify descriptor is loaded
            desc_type = bridge.descriptor.get_strategy_description().get('type', 'Unknown')
            logger.info(f"  Strategy type: {desc_type}")
            
        except Exception as e:
            logger.error(f"✗ Failed to create {strategy_name}: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting Simple Strategy Bridge Tests")
    logger.info("=" * 80)
    
    test_strategy_descriptors()
    test_strategy_mapping()
    test_factory_methods()
    
    logger.info("\n" + "=" * 80)
    logger.info("All tests completed!")
    logger.info("=" * 80)
