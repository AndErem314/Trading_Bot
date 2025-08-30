"""
Fibonacci Retracement Support Resistance Strategy

Strategy Name: Fibonacci Retracement Support Resistance Strategy

Description:
This strategy uses Fibonacci retracement levels to identify potential support and
resistance zones where price is likely to reverse or consolidate. It focuses on
the key Fibonacci ratios (38.2%, 50%, 61.8%) and combines them with trend analysis
to generate high-probability reversal signals at these mathematical levels.

Author: Andrey's Trading Bot
Date: 2025-08-29
"""

class FibonacciRetracementSupportResistanceStrategy:
    """
    Fibonacci-based trading strategy that uses retracement levels for support/resistance trading.
    """
    
    def __init__(self):
        self.name = "Fibonacci Retracement Support Resistance Strategy"
        self.version = "1.0.0"
        self.min_holding_period = 4  # Minimum hours to hold position
        self.max_holding_period = 48  # Maximum hours to hold position (2 days)
        
    def get_strategy_description(self):
        """
        Returns detailed description of the strategy.
        """
        return {
            "name": self.name,
            "type": "Support/Resistance Trading",
            "timeframe": "4H preferred (can work with 1D)",
            "indicators": ["Fib 23.6%", "Fib 38.2%", "Fib 50%", "Fib 61.8%", "Fib 78.6%"],
            "risk_level": "Medium-Low",
            "expected_trade_duration": "4-48 hours",
            "market_conditions": "Works in all markets, best after significant moves",
            "description": """
            This strategy identifies trading opportunities by:
            1. Key Fibonacci levels acting as support/resistance
            2. Price reactions at 38.2%, 50%, and 61.8% levels
            3. Trend direction to trade with the overall bias
            4. Confluence of multiple Fib levels for stronger signals
            5. Risk/reward optimization using Fib level targets
            """
        }
    
    def get_entry_conditions(self):
        """
        Returns the precise entry conditions for the strategy.
        """
        return {
            "LONG_ENTRY": {
                "primary_conditions": [
                    "Price touches or approaches key Fib support (38.2%, 50%, or 61.8%)",
                    "Trend direction is 'uptrend'",
                    "Fib signal is 'buy_support' or 'strong_buy'"
                ],
                "confirmation_conditions": [
                    "Price shows rejection from Fib level (wick/pin bar)",
                    "Nearest Fib level distance < 1% (very close to level)",
                    "Support/resistance shows level acting as support"
                ],
                "description": "Enter long at Fibonacci support levels in uptrends"
            },
            "SHORT_ENTRY": {
                "primary_conditions": [
                    "Price touches or approaches key Fib resistance (38.2%, 50%, or 61.8%)",
                    "Trend direction is 'downtrend'",
                    "Fib signal is 'sell_resistance' or 'strong_sell'"
                ],
                "confirmation_conditions": [
                    "Price shows rejection from Fib level (wick/pin bar)",
                    "Nearest Fib level distance < 1% (very close to level)",
                    "Support/resistance shows level acting as resistance"
                ],
                "description": "Enter short at Fibonacci resistance levels in downtrends"
            }
        }
    
    def get_exit_conditions(self):
        """
        Returns the precise exit conditions for the strategy.
        """
        return {
            "LONG_EXIT": {
                "take_profit": [
                    "Price reaches next Fibonacci resistance level",
                    "Price reaches 0% retracement (full recovery)",
                    "Fib signal changes to 'sell_resistance'",
                    "Risk/reward of 1:2 achieved"
                ],
                "stop_loss": [
                    "Price closes below entered Fib support level",
                    "Price reaches 78.6% retracement (deep pullback)",
                    "Maximum holding period reached (48 hours)"
                ],
                "description": "Exit long at next Fib resistance or breakdown of support"
            },
            "SHORT_EXIT": {
                "take_profit": [
                    "Price reaches next Fibonacci support level",
                    "Price reaches 100% retracement (full pullback)",
                    "Fib signal changes to 'buy_support'",
                    "Risk/reward of 1:2 achieved"
                ],
                "stop_loss": [
                    "Price closes above entered Fib resistance level",
                    "Price breaks above 23.6% retracement (shallow pullback)",
                    "Maximum holding period reached (48 hours)"
                ],
                "description": "Exit short at next Fib support or breakout above resistance"
            }
        }
    
    def get_risk_management_rules(self):
        """
        Returns risk management rules for the strategy.
        """
        return {
            "position_sizing": {
                "method": "Level-based sizing",
                "risk_per_trade": "1.5% of account",
                "max_concurrent_positions": 3,
                "scaling": "Larger positions at 61.8% (golden ratio)"
            },
            "stop_loss": {
                "initial": "Just beyond the Fib level entered",
                "trailing": "Move to breakeven at next Fib level",
                "time_based": "Exit after 48 hours"
            },
            "filters": {
                "level_strength": "Prefer 38.2%, 50%, 61.8% levels",
                "trend_alignment": "Trade with overall trend direction",
                "confluence": "Better signals with multiple levels nearby"
            },
            "additional_rules": [
                "61.8% is the most important retracement level",
                "50% is psychological but very effective",
                "38.2% for shallow retracements in strong trends",
                "Use multiple timeframe Fib levels for confluence",
                "Combine with other indicators for confirmation"
            ]
        }
    
    def get_sql_query(self):
        """
        Returns SQL query to identify trading signals based on this strategy.
        """
        return """
        WITH fib_analysis AS (
            SELECT 
                o.timestamp,
                o.close,
                o.high,
                o.low,
                f.level_0,
                f.level_23_6,
                f.level_38_2,
                f.level_50_0,
                f.level_61_8,
                f.level_78_6,
                f.level_100,
                f.trend_direction,
                f.nearest_fib_level,
                f.fib_signal,
                f.support_resistance,
                -- Previous values for level breach detection
                LAG(o.close, 1) OVER (ORDER BY o.timestamp) as prev_close,
                LAG(f.fib_signal, 1) OVER (ORDER BY o.timestamp) as prev_signal,
                -- Detect price touching key levels
                CASE 
                    WHEN ABS(o.close - f.level_38_2) / o.close < 0.01 THEN 1
                    WHEN ABS(o.close - f.level_50_0) / o.close < 0.01 THEN 1
                    WHEN ABS(o.close - f.level_61_8) / o.close < 0.01 THEN 1
                    ELSE 0
                END as touches_key_level,
                -- Detect pin bars/wicks at levels
                CASE 
                    WHEN (o.high - o.close) / (o.high - o.low) > 0.6 THEN 'bearish_pin'
                    WHEN (o.close - o.low) / (o.high - o.low) > 0.6 THEN 'bullish_pin'
                    ELSE 'none'
                END as pin_bar
            FROM fibonacci_retracement_indicator f
            JOIN ohlcv_data o ON f.ohlcv_id = o.id
            WHERE o.timestamp >= datetime('now', '-3 months')
        ),
        signals AS (
            SELECT 
                timestamp,
                close,
                level_38_2,
                level_50_0,
                level_61_8,
                trend_direction,
                nearest_fib_level,
                fib_signal,
                support_resistance,
                CASE 
                    -- LONG ENTRY CONDITIONS
                    WHEN 
                        fib_signal IN ('buy_support', 'strong_buy')
                        AND trend_direction = 'uptrend'
                        AND nearest_fib_level < 1.0
                        AND touches_key_level = 1
                        AND (pin_bar = 'bullish_pin' OR support_resistance LIKE '%_support')
                    THEN 1  -- BUY SIGNAL
                    
                    -- SHORT ENTRY CONDITIONS
                    WHEN 
                        fib_signal IN ('sell_resistance', 'strong_sell')
                        AND trend_direction = 'downtrend'
                        AND nearest_fib_level < 1.0
                        AND touches_key_level = 1
                        AND (pin_bar = 'bearish_pin' OR support_resistance LIKE '%_resistance')
                    THEN -1  -- SELL SIGNAL
                    
                    -- EXIT LONG CONDITIONS
                    WHEN 
                        (close > level_23_6 AND prev_close <= level_23_6)  -- Reached next resistance
                        OR close < level_78_6  -- Deep retracement
                        OR fib_signal = 'sell_resistance'
                    THEN -2  -- EXIT LONG SIGNAL
                    
                    -- EXIT SHORT CONDITIONS
                    WHEN 
                        (close < level_78_6 AND prev_close >= level_78_6)  -- Reached next support
                        OR close > level_23_6  -- Shallow retracement broken
                        OR fib_signal = 'buy_support'
                    THEN 2  -- EXIT SHORT SIGNAL
                    
                    ELSE 0  -- HOLD
                END as signal
            FROM fib_analysis
            WHERE prev_close IS NOT NULL
        )
        SELECT 
            timestamp,
            close as price,
            level_38_2 as fib_38_2,
            level_50_0 as fib_50_0,
            level_61_8 as fib_61_8,
            trend_direction,
            fib_signal,
            support_resistance,
            signal,
            CASE signal
                WHEN 1 THEN 'BUY'
                WHEN -1 THEN 'SELL'
                WHEN 2 THEN 'EXIT_SHORT'
                WHEN -2 THEN 'EXIT_LONG'
                ELSE 'HOLD'
            END as signal_name
        FROM signals
        WHERE signal != 0
        ORDER BY timestamp DESC
        LIMIT 100;
        """


# Example usage and testing
if __name__ == "__main__":
    strategy = FibonacciRetracementSupportResistanceStrategy()
    
    print(f"Strategy Name: {strategy.name}")
    print(f"Version: {strategy.version}")
    print("\n" + "="*50 + "\n")
    
    # Print strategy description
    desc = strategy.get_strategy_description()
    print("STRATEGY DESCRIPTION:")
    for key, value in desc.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # Print entry conditions
    print("ENTRY CONDITIONS:")
    entries = strategy.get_entry_conditions()
    for signal_type, conditions in entries.items():
        print(f"\n{signal_type}:")
        for condition_type, condition_list in conditions.items():
            print(f"  {condition_type}:")
            if isinstance(condition_list, list):
                for condition in condition_list:
                    print(f"    - {condition}")
            else:
                print(f"    {condition_list}")
