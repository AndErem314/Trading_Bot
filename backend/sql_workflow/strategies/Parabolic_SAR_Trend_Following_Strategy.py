"""
Parabolic SAR Trend Following Strategy

Strategy Name: Parabolic SAR Trend Following Strategy

Description:
This strategy uses the Parabolic SAR (Stop and Reverse) indicator to identify and follow
trends. It generates signals when the SAR flips from one side of price to the other,
indicating potential trend reversals. The strategy is designed for riding trends
with built-in stop-loss levels provided by the SAR itself.

Author: Andrey's Trading Bot
Date: 2025-08-29
"""

class ParabolicSARTrendFollowingStrategy:
    """
    Parabolic SAR-based trading strategy that uses SAR flips for trend following.
    """
    
    def __init__(self):
        self.name = "Parabolic SAR Trend Following Strategy"
        self.version = "1.0.0"
        self.min_holding_period = 4  # Minimum hours to hold position
        self.max_holding_period = 72  # Maximum hours to hold position (3 days)
        
    def get_strategy_description(self):
        """
        Returns detailed description of the strategy.
        """
        return {
            "name": self.name,
            "type": "Trend Following",
            "timeframe": "4H preferred (can work with 1H)",
            "indicators": ["Parabolic SAR", "Trend Direction", "Signal Strength", "Acceleration Factor"],
            "risk_level": "Medium",
            "expected_trade_duration": "4-72 hours",
            "market_conditions": "Works best in trending markets",
            "description": """
            This strategy identifies trading opportunities by:
            1. SAR flips (dots switching sides) for trend reversal entries
            2. Signal strength based on price-SAR distance
            3. Acceleration factor showing trend momentum
            4. Built-in stop-loss levels using SAR values
            5. Trend continuation signals when SAR accelerates
            """
        }
    
    def get_entry_conditions(self):
        """
        Returns the precise entry conditions for the strategy.
        """
        return {
            "LONG_ENTRY": {
                "primary_conditions": [
                    "SAR flips from above to below price (trend = 'up')",
                    "Reversal signal is TRUE",
                    "Price is above the new SAR level"
                ],
                "confirmation_conditions": [
                    "Volume is above average on reversal",
                    "Signal strength is positive",
                    "Previous trend lasted at least 5 periods"
                ],
                "description": "Enter long when SAR flips below price indicating uptrend start"
            },
            "SHORT_ENTRY": {
                "primary_conditions": [
                    "SAR flips from below to above price (trend = 'down')",
                    "Reversal signal is TRUE",
                    "Price is below the new SAR level"
                ],
                "confirmation_conditions": [
                    "Volume is above average on reversal",
                    "Signal strength is positive",
                    "Previous trend lasted at least 5 periods"
                ],
                "description": "Enter short when SAR flips above price indicating downtrend start"
            }
        }
    
    def get_exit_conditions(self):
        """
        Returns the precise exit conditions for the strategy.
        """
        return {
            "LONG_EXIT": {
                "take_profit": [
                    "SAR flips from below to above price (trend reversal)",
                    "Signal strength drops below 1% (weak trend)",
                    "Acceleration factor reaches maximum (0.2)",
                    "Target profit of 3% reached"
                ],
                "stop_loss": [
                    "Price touches or crosses below SAR level",
                    "Reversal signal detected (SAR flip)",
                    "Maximum holding period reached (72 hours)"
                ],
                "description": "Exit long on SAR flip or when price hits SAR stop level"
            },
            "SHORT_EXIT": {
                "take_profit": [
                    "SAR flips from above to below price (trend reversal)",
                    "Signal strength drops below 1% (weak trend)",
                    "Acceleration factor reaches maximum (0.2)",
                    "Target profit of 3% reached"
                ],
                "stop_loss": [
                    "Price touches or crosses above SAR level",
                    "Reversal signal detected (SAR flip)",
                    "Maximum holding period reached (72 hours)"
                ],
                "description": "Exit short on SAR flip or when price hits SAR stop level"
            }
        }
    
    def get_risk_management_rules(self):
        """
        Returns risk management rules for the strategy.
        """
        return {
            "position_sizing": {
                "method": "SAR distance based",
                "risk_per_trade": "1.5% of account",
                "max_concurrent_positions": 3,
                "scaling": "Smaller positions when SAR is far from price"
            },
            "stop_loss": {
                "initial": "SAR level (dynamic stop)",
                "trailing": "SAR automatically trails the trend",
                "time_based": "Exit after 72 hours"
            },
            "filters": {
                "volatility": "Avoid entries when SAR flips too frequently",
                "trend_length": "Prefer reversals after established trends",
                "volume": "Require volume confirmation on flips"
            },
            "additional_rules": [
                "SAR provides automatic trailing stop",
                "Best signals after extended trends",
                "Avoid choppy markets with frequent SAR flips",
                "Consider higher timeframe SAR for confirmation",
                "Watch acceleration factor for trend exhaustion"
            ]
        }
    
    def get_sql_query(self):
        """
        Returns SQL query to identify trading signals based on this strategy.
        """
        return """
        WITH sar_analysis AS (
            SELECT 
                o.timestamp,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume,
                p.parabolic_sar,
                p.trend,
                p.reversal_signal,
                p.signal_strength,
                p.acceleration_factor,
                p.sar_signal,
                -- Previous values for trend analysis
                LAG(p.trend, 1) OVER (ORDER BY o.timestamp) as prev_trend,
                LAG(p.parabolic_sar, 1) OVER (ORDER BY o.timestamp) as prev_sar,
                -- Count consecutive periods in same trend
                SUM(CASE WHEN p.trend = LAG(p.trend, 1) OVER (ORDER BY o.timestamp) THEN 0 ELSE 1 END) 
                    OVER (ORDER BY o.timestamp) as trend_change_count,
                -- Average volume for comparison
                AVG(o.volume) OVER (ORDER BY o.timestamp ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as avg_volume
            FROM parabolic_sar_indicator p
            JOIN ohlcv_data o ON p.ohlcv_id = o.id
            WHERE o.timestamp >= datetime('now', '-3 months')
        ),
        trend_length AS (
            SELECT 
                *,
                -- Calculate how long current trend has lasted
                ROW_NUMBER() OVER (PARTITION BY trend_change_count ORDER BY timestamp) as periods_in_trend
            FROM sar_analysis
        ),
        signals AS (
            SELECT 
                timestamp,
                close,
                parabolic_sar,
                trend,
                reversal_signal,
                signal_strength,
                acceleration_factor,
                sar_signal,
                CASE 
                    -- LONG ENTRY CONDITIONS
                    WHEN 
                        trend = 'up'
                        AND reversal_signal = 1
                        AND close > parabolic_sar
                        AND volume > avg_volume * 1.1
                        AND periods_in_trend >= 5  -- Previous trend lasted at least 5 periods
                    THEN 1  -- BUY SIGNAL
                    
                    -- SHORT ENTRY CONDITIONS
                    WHEN 
                        trend = 'down'
                        AND reversal_signal = 1
                        AND close < parabolic_sar
                        AND volume > avg_volume * 1.1
                        AND periods_in_trend >= 5  -- Previous trend lasted at least 5 periods
                    THEN -1  -- SELL SIGNAL
                    
                    -- EXIT LONG CONDITIONS
                    WHEN 
                        prev_trend = 'up' AND trend = 'down'  -- SAR flip
                        OR (trend = 'up' AND close <= parabolic_sar)  -- Price hits SAR
                        OR (trend = 'up' AND signal_strength < 1)  -- Weak trend
                    THEN -2  -- EXIT LONG SIGNAL
                    
                    -- EXIT SHORT CONDITIONS
                    WHEN 
                        prev_trend = 'down' AND trend = 'up'  -- SAR flip
                        OR (trend = 'down' AND close >= parabolic_sar)  -- Price hits SAR
                        OR (trend = 'down' AND signal_strength < 1)  -- Weak trend
                    THEN 2  -- EXIT SHORT SIGNAL
                    
                    ELSE 0  -- HOLD
                END as signal
            FROM trend_length
            WHERE prev_trend IS NOT NULL
        )
        SELECT 
            timestamp,
            close as price,
            parabolic_sar,
            trend,
            reversal_signal,
            signal_strength,
            acceleration_factor,
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
    strategy = ParabolicSARTrendFollowingStrategy()
    
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
