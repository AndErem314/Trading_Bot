"""
MACD Momentum Crossover Trading Strategy

Strategy Name: MACD Momentum Crossover Strategy

Description:
This strategy uses the MACD (Moving Average Convergence Divergence) indicator to identify
momentum shifts and trend changes. It focuses on MACD line and signal line crossovers,
histogram momentum analysis, and divergence patterns to generate trading signals.
The strategy is designed for trend following with momentum confirmation.

Author: Andrey's Trading Bot
Date: 2025-08-29
"""

class MACDMomentumCrossoverStrategy:
    """
    MACD-based trading strategy that uses crossovers and momentum for entry/exit signals.
    """
    
    def __init__(self):
        self.name = "MACD Momentum Crossover Strategy"
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
            "indicators": ["MACD Line", "Signal Line", "Histogram", "Zero Line Cross", "Divergence"],
            "risk_level": "Medium",
            "expected_trade_duration": "4-72 hours",
            "market_conditions": "Works best in trending markets",
            "description": """
            This strategy identifies trading opportunities by:
            1. MACD line crossing above/below signal line for primary signals
            2. Histogram momentum shifts for confirmation
            3. Zero line crosses for trend strength
            4. Divergence patterns for potential reversals
            5. Strong trends confirmed by both lines above/below zero
            """
        }
    
    def get_entry_conditions(self):
        """
        Returns the precise entry conditions for the strategy.
        """
        return {
            "LONG_ENTRY": {
                "primary_conditions": [
                    "MACD line crosses above Signal line (bullish crossover)",
                    "Histogram is positive and increasing",
                    "MACD signal is 'bullish' or 'strong_bullish'"
                ],
                "confirmation_conditions": [
                    "Both MACD and Signal lines are above zero (strong trend) OR",
                    "MACD line is crossing above zero (trend reversal)",
                    "No bearish divergence present"
                ],
                "description": "Enter long when MACD shows bullish momentum with trend confirmation"
            },
            "SHORT_ENTRY": {
                "primary_conditions": [
                    "MACD line crosses below Signal line (bearish crossover)",
                    "Histogram is negative and decreasing",
                    "MACD signal is 'bearish' or 'strong_bearish'"
                ],
                "confirmation_conditions": [
                    "Both MACD and Signal lines are below zero (strong trend) OR",
                    "MACD line is crossing below zero (trend reversal)",
                    "No bullish divergence present"
                ],
                "description": "Enter short when MACD shows bearish momentum with trend confirmation"
            }
        }
    
    def get_exit_conditions(self):
        """
        Returns the precise exit conditions for the strategy.
        """
        return {
            "LONG_EXIT": {
                "take_profit": [
                    "MACD line crosses below Signal line (bearish crossover)",
                    "Histogram turns negative",
                    "Bearish divergence detected",
                    "Both lines cross below zero"
                ],
                "stop_loss": [
                    "Price drops 2.5% from entry",
                    "MACD signal becomes 'strong_bearish'",
                    "Maximum holding period reached (72 hours)"
                ],
                "description": "Exit long on bearish momentum shift or trend exhaustion"
            },
            "SHORT_EXIT": {
                "take_profit": [
                    "MACD line crosses above Signal line (bullish crossover)",
                    "Histogram turns positive",
                    "Bullish divergence detected",
                    "Both lines cross above zero"
                ],
                "stop_loss": [
                    "Price rises 2.5% from entry",
                    "MACD signal becomes 'strong_bullish'",
                    "Maximum holding period reached (72 hours)"
                ],
                "description": "Exit short on bullish momentum shift or trend exhaustion"
            }
        }
    
    def get_risk_management_rules(self):
        """
        Returns risk management rules for the strategy.
        """
        return {
            "position_sizing": {
                "method": "Fixed percentage risk",
                "risk_per_trade": "1.5% of account",
                "max_concurrent_positions": 3,
                "scaling": "Can add to position on histogram expansion"
            },
            "stop_loss": {
                "initial": "2.5% from entry price",
                "trailing": "Activate at 3% profit, trail by 1.5%",
                "time_based": "Exit after 72 hours if no profit target hit"
            },
            "filters": {
                "volatility": "Avoid entries when histogram changes direction rapidly",
                "volume": "Prefer entries with above-average volume",
                "correlation": "Check correlation with market trend"
            },
            "additional_rules": [
                "Wait for candle close confirmation on crossovers",
                "Avoid trading during major news events",
                "Stronger signals when crossover occurs away from zero line",
                "Consider reducing position size in ranging markets",
                "Monitor for false signals in choppy conditions"
            ]
        }
    
    def get_sql_query(self):
        """
        Returns SQL query to identify trading signals based on this strategy.
        """
        return """
        WITH macd_analysis AS (
            SELECT 
                o.timestamp,
                o.close,
                m.macd_line,
                m.signal_line,
                m.histogram,
                m.macd_signal,
                LAG(m.macd_line, 1) OVER (ORDER BY o.timestamp) as prev_macd_line,
                LAG(m.signal_line, 1) OVER (ORDER BY o.timestamp) as prev_signal_line,
                LAG(m.histogram, 1) OVER (ORDER BY o.timestamp) as prev_histogram,
                -- Detect crossovers
                CASE 
                    WHEN m.macd_line > m.signal_line AND 
                         LAG(m.macd_line, 1) OVER (ORDER BY o.timestamp) <= LAG(m.signal_line, 1) OVER (ORDER BY o.timestamp)
                    THEN 1 ELSE 0 
                END as bullish_crossover,
                CASE 
                    WHEN m.macd_line < m.signal_line AND 
                         LAG(m.macd_line, 1) OVER (ORDER BY o.timestamp) >= LAG(m.signal_line, 1) OVER (ORDER BY o.timestamp)
                    THEN 1 ELSE 0 
                END as bearish_crossover
            FROM macd_indicator m
            JOIN ohlcv_data o ON m.ohlcv_id = o.id
            WHERE o.timestamp >= datetime('now', '-3 months')
        ),
        signals AS (
            SELECT 
                timestamp,
                close,
                macd_line,
                signal_line,
                histogram,
                macd_signal,
                CASE 
                    -- LONG ENTRY CONDITIONS
                    WHEN 
                        bullish_crossover = 1
                        AND histogram > 0
                        AND histogram > prev_histogram
                        AND macd_signal IN ('bullish', 'strong_bullish')
                    THEN 1  -- BUY SIGNAL
                    
                    -- SHORT ENTRY CONDITIONS
                    WHEN 
                        bearish_crossover = 1
                        AND histogram < 0
                        AND histogram < prev_histogram
                        AND macd_signal IN ('bearish', 'strong_bearish')
                    THEN -1  -- SELL SIGNAL
                    
                    -- EXIT LONG CONDITIONS
                    WHEN 
                        bearish_crossover = 1
                        OR (histogram < 0 AND prev_histogram >= 0)
                    THEN -2  -- EXIT LONG SIGNAL
                    
                    -- EXIT SHORT CONDITIONS
                    WHEN 
                        bullish_crossover = 1
                        OR (histogram > 0 AND prev_histogram <= 0)
                    THEN 2  -- EXIT SHORT SIGNAL
                    
                    ELSE 0  -- HOLD
                END as signal
            FROM macd_analysis
            WHERE prev_macd_line IS NOT NULL
        )
        SELECT 
            timestamp,
            close as price,
            macd_line,
            signal_line,
            histogram,
            macd_signal,
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
    strategy = MACDMomentumCrossoverStrategy()
    
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
