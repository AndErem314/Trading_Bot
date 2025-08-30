"""
Gaussian Channel Breakout Mean Reversion Strategy

Strategy Name: Gaussian Channel Breakout Mean Reversion Strategy

Description:
This strategy uses the Gaussian Channel indicator to identify breakout opportunities
and mean reversion trades. It combines channel breakouts for trend following with
mean reversion trades when price returns to the channel after false breakouts.
The strategy adapts to market volatility through dynamic channel width.

Author: Andrey's Trading Bot
Date: 2025-08-29
"""

class GaussianChannelBreakoutMeanReversionStrategy:
    """
    Gaussian Channel-based trading strategy that uses channel breakouts and mean reversion.
    """
    
    def __init__(self):
        self.name = "Gaussian Channel Breakout Mean Reversion Strategy"
        self.version = "1.0.0"
        self.min_holding_period = 3  # Minimum hours to hold position
        self.max_holding_period = 36  # Maximum hours to hold position (1.5 days)
        
    def get_strategy_description(self):
        """
        Returns detailed description of the strategy.
        """
        return {
            "name": self.name,
            "type": "Breakout & Mean Reversion Hybrid",
            "timeframe": "4H preferred (can work with 1H)",
            "indicators": ["GC Upper", "GC Middle", "GC Lower", "Channel Position", "Channel Width"],
            "risk_level": "Medium",
            "expected_trade_duration": "3-36 hours",
            "market_conditions": "Works in both trending and ranging markets",
            "description": """
            This strategy identifies trading opportunities by:
            1. Channel breakouts for trend continuation trades
            2. False breakout reversals for mean reversion trades
            3. Channel position for optimal entry points
            4. Dynamic channel width for volatility adaptation
            5. Middle line as dynamic support/resistance
            """
        }
    
    def get_entry_conditions(self):
        """
        Returns the precise entry conditions for the strategy.
        """
        return {
            "LONG_ENTRY": {
                "primary_conditions": [
                    "Price breaks above upper channel (gc_signal = 'strong_buy')",
                    "OR Price bounces from lower channel (gc_signal = 'buy')",
                    "Breakout signal is 'bullish_breakout' OR 'bullish_reversal'"
                ],
                "confirmation_conditions": [
                    "Channel position < 0.2 for mean reversion entry",
                    "Volume above average on breakout",
                    "Channel width not extreme (< 5% of middle)"
                ],
                "description": "Enter long on upper channel breakout or lower channel bounce"
            },
            "SHORT_ENTRY": {
                "primary_conditions": [
                    "Price breaks below lower channel (gc_signal = 'strong_sell')",
                    "OR Price rejected from upper channel (gc_signal = 'sell')",
                    "Breakout signal is 'bearish_breakout' OR 'bearish_reversal'"
                ],
                "confirmation_conditions": [
                    "Channel position > 0.8 for mean reversion entry",
                    "Volume above average on breakout",
                    "Channel width not extreme (< 5% of middle)"
                ],
                "description": "Enter short on lower channel breakout or upper channel rejection"
            }
        }
    
    def get_exit_conditions(self):
        """
        Returns the precise exit conditions for the strategy.
        """
        return {
            "LONG_EXIT": {
                "take_profit": [
                    "Price reaches opposite channel (upper if mean reversion)",
                    "Price returns to middle line (50% profit target)",
                    "Channel position reaches > 0.9",
                    "Breakout signal turns 'bearish_reversal'"
                ],
                "stop_loss": [
                    "Price closes below lower channel",
                    "Distance from middle exceeds -3%",
                    "Maximum holding period reached (36 hours)"
                ],
                "description": "Exit long at opposite channel or middle line target"
            },
            "SHORT_EXIT": {
                "take_profit": [
                    "Price reaches opposite channel (lower if mean reversion)",
                    "Price returns to middle line (50% profit target)",
                    "Channel position reaches < 0.1",
                    "Breakout signal turns 'bullish_reversal'"
                ],
                "stop_loss": [
                    "Price closes above upper channel",
                    "Distance from middle exceeds 3%",
                    "Maximum holding period reached (36 hours)"
                ],
                "description": "Exit short at opposite channel or middle line target"
            }
        }
    
    def get_risk_management_rules(self):
        """
        Returns risk management rules for the strategy.
        """
        return {
            "position_sizing": {
                "method": "Channel width based",
                "risk_per_trade": "1.5% of account",
                "max_concurrent_positions": 3,
                "scaling": "Smaller positions when channel is wide (high volatility)"
            },
            "stop_loss": {
                "initial": "Outside channel by 0.5%",
                "trailing": "Trail to middle line after profit",
                "time_based": "Exit after 36 hours"
            },
            "filters": {
                "volatility": "Avoid when channel width > 6%",
                "volume": "Require volume confirmation",
                "channel_quality": "Clear channel definition needed"
            },
            "additional_rules": [
                "Best breakout trades after channel squeeze",
                "Mean reversion works best in wide channels",
                "Middle line acts as profit target and support/resistance",
                "Watch for failed breakouts as reversal signals",
                "Combine with trend analysis for direction bias"
            ]
        }
    
    def get_sql_query(self):
        """
        Returns SQL query to identify trading signals based on this strategy.
        """
        return """
        WITH gc_analysis AS (
            SELECT 
                o.timestamp,
                o.close,
                o.volume,
                g.gc_upper,
                g.gc_middle,
                g.gc_lower,
                -- Calculate channel metrics
                (o.close - g.gc_lower) / NULLIF(g.gc_upper - g.gc_lower, 0) as channel_position,
                ((g.gc_upper - g.gc_lower) / g.gc_middle * 100) as channel_width_pct,
                ((o.close - g.gc_middle) / g.gc_middle * 100) as distance_from_middle,
                -- Detect breakouts
                CASE 
                    WHEN o.close > g.gc_upper THEN 'above_upper'
                    WHEN o.close < g.gc_lower THEN 'below_lower'
                    WHEN o.close > g.gc_middle THEN 'above_middle'
                    ELSE 'below_middle'
                END as price_position,
                -- Previous values for signal detection
                LAG(o.close, 1) OVER (ORDER BY o.timestamp) as prev_close,
                LAG(g.gc_upper, 1) OVER (ORDER BY o.timestamp) as prev_upper,
                LAG(g.gc_lower, 1) OVER (ORDER BY o.timestamp) as prev_lower,
                -- Average volume for comparison
                AVG(o.volume) OVER (ORDER BY o.timestamp ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as avg_volume
            FROM gaussian_channel_indicator g
            JOIN ohlcv_data o ON g.ohlcv_id = o.id
            WHERE o.timestamp >= datetime('now', '-3 months')
        ),
        signals AS (
            SELECT 
                timestamp,
                close,
                gc_upper,
                gc_middle,
                gc_lower,
                channel_position,
                channel_width_pct,
                distance_from_middle,
                CASE 
                    -- LONG ENTRY CONDITIONS
                    WHEN 
                        -- Breakout above upper channel
                        (close > gc_upper AND prev_close <= prev_upper AND volume > avg_volume * 1.2)
                        -- OR bounce from lower channel
                        OR (close > gc_lower AND prev_close <= prev_lower AND channel_position < 0.2)
                        -- Channel not too wide
                        AND channel_width_pct < 5
                    THEN 1  -- BUY SIGNAL
                    
                    -- SHORT ENTRY CONDITIONS
                    WHEN 
                        -- Breakout below lower channel
                        (close < gc_lower AND prev_close >= prev_lower AND volume > avg_volume * 1.2)
                        -- OR rejection from upper channel
                        OR (close < gc_upper AND prev_close >= prev_upper AND channel_position > 0.8)
                        -- Channel not too wide
                        AND channel_width_pct < 5
                    THEN -1  -- SELL SIGNAL
                    
                    -- EXIT LONG CONDITIONS
                    WHEN 
                        channel_position > 0.9  -- Near upper channel
                        OR distance_from_middle < 0  -- Back below middle
                        OR close < gc_lower  -- Below lower channel
                    THEN -2  -- EXIT LONG SIGNAL
                    
                    -- EXIT SHORT CONDITIONS
                    WHEN 
                        channel_position < 0.1  -- Near lower channel
                        OR distance_from_middle > 0  -- Back above middle
                        OR close > gc_upper  -- Above upper channel
                    THEN 2  -- EXIT SHORT SIGNAL
                    
                    ELSE 0  -- HOLD
                END as signal
            FROM gc_analysis
            WHERE prev_close IS NOT NULL
        )
        SELECT 
            timestamp,
            close as price,
            gc_upper,
            gc_middle,
            gc_lower,
            channel_position,
            channel_width_pct,
            distance_from_middle,
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
    strategy = GaussianChannelBreakoutMeanReversionStrategy()
    
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
