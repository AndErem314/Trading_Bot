"""
SMA Golden Cross Trading Strategy

Strategy Name: SMA Golden Cross Strategy

Description:
This strategy uses Simple Moving Averages (SMA) to identify trend changes and momentum.
It focuses on the classic Golden Cross (SMA 50 crossing above SMA 200) and Death Cross
(SMA 50 crossing below SMA 200) signals, combined with price position relative to the SMAs.
The strategy is designed for trend following in medium to long-term timeframes.

Author: Andrey's Trading Bot
Date: 2025-08-29
"""

class SMAGoldenCrossStrategy:
    """
    SMA-based trading strategy that uses moving average crossovers and trend alignment.
    """
    
    def __init__(self):
        self.name = "SMA Golden Cross Strategy"
        self.version = "1.0.0"
        self.min_holding_period = 8  # Minimum hours to hold position
        self.max_holding_period = 120  # Maximum hours to hold position (5 days)
        
    def get_strategy_description(self):
        """
        Returns detailed description of the strategy.
        """
        return {
            "name": self.name,
            "type": "Trend Following",
            "timeframe": "1D preferred (can work with 4H)",
            "indicators": ["SMA 50", "SMA 200", "Golden Cross", "Death Cross", "Trend Strength"],
            "risk_level": "Low-Medium",
            "expected_trade_duration": "8-120 hours",
            "market_conditions": "Works best in trending markets",
            "description": """
            This strategy identifies trading opportunities by:
            1. Golden Cross and Death Cross major trend reversals
            2. Price position relative to both SMAs for trend confirmation
            3. SMA alignment (50 above/below 200) for trend direction
            4. Trend strength measurement for position sizing
            5. Multiple timeframe confirmation for stronger signals
            """
        }
    
    def get_entry_conditions(self):
        """
        Returns the precise entry conditions for the strategy.
        """
        return {
            "LONG_ENTRY": {
                "primary_conditions": [
                    "Golden Cross occurs (SMA 50 crosses above SMA 200)",
                    "Price is above SMA 50",
                    "Trend strength is positive (> 2)"
                ],
                "confirmation_conditions": [
                    "Volume is above average on crossover",
                    "Price has pulled back to SMA 50 after crossover (better entry)",
                    "SMA 50 slope is positive (rising)"
                ],
                "description": "Enter long on Golden Cross with price above short-term SMA"
            },
            "SHORT_ENTRY": {
                "primary_conditions": [
                    "Death Cross occurs (SMA 50 crosses below SMA 200)",
                    "Price is below SMA 50",
                    "Trend strength is negative (< -2)"
                ],
                "confirmation_conditions": [
                    "Volume is above average on crossover",
                    "Price has pulled back to SMA 50 after crossover (better entry)",
                    "SMA 50 slope is negative (falling)"
                ],
                "description": "Enter short on Death Cross with price below short-term SMA"
            }
        }
    
    def get_exit_conditions(self):
        """
        Returns the precise exit conditions for the strategy.
        """
        return {
            "LONG_EXIT": {
                "take_profit": [
                    "Death Cross occurs (opposite signal)",
                    "Price closes below SMA 50 for 2 consecutive periods",
                    "Trend strength turns negative (< -5)",
                    "Price reaches 2x initial risk (2:1 R/R)"
                ],
                "stop_loss": [
                    "Price closes below SMA 200",
                    "Price drops 3% from entry",
                    "Maximum holding period reached (120 hours)"
                ],
                "description": "Exit long on trend reversal or breakdown below key SMAs"
            },
            "SHORT_EXIT": {
                "take_profit": [
                    "Golden Cross occurs (opposite signal)",
                    "Price closes above SMA 50 for 2 consecutive periods",
                    "Trend strength turns positive (> 5)",
                    "Price reaches 2x initial risk (2:1 R/R)"
                ],
                "stop_loss": [
                    "Price closes above SMA 200",
                    "Price rises 3% from entry",
                    "Maximum holding period reached (120 hours)"
                ],
                "description": "Exit short on trend reversal or breakout above key SMAs"
            }
        }
    
    def get_risk_management_rules(self):
        """
        Returns risk management rules for the strategy.
        """
        return {
            "position_sizing": {
                "method": "Trend strength based",
                "risk_per_trade": "2% of account",
                "max_concurrent_positions": 2,
                "scaling": "Increase size when trend strength > 10"
            },
            "stop_loss": {
                "initial": "3% from entry or opposite SMA",
                "trailing": "Trail to SMA 50 in profit",
                "time_based": "Exit after 120 hours"
            },
            "filters": {
                "volatility": "Avoid entries during high volatility spikes",
                "correlation": "Check market trend alignment",
                "volume": "Require above-average volume on crosses"
            },
            "additional_rules": [
                "Wait for candle close confirmation on crossovers",
                "Stronger signals when price tests SMA after cross",
                "Reduce position size in choppy markets",
                "Consider higher timeframe trend alignment",
                "Best entries after pullback to SMA support/resistance"
            ]
        }
    
    def get_sql_query(self):
        """
        Returns SQL query to identify trading signals based on this strategy.
        """
        return """
        WITH sma_analysis AS (
            SELECT 
                o.timestamp,
                o.close,
                o.volume,
                s.sma_50,
                s.sma_200,
                s.sma_ratio,
                s.price_vs_sma50,
                s.price_vs_sma200,
                s.trend_strength,
                s.sma_signal,
                s.cross_signal,
                -- Previous values for crossover detection
                LAG(s.sma_50, 1) OVER (ORDER BY o.timestamp) as prev_sma_50,
                LAG(s.sma_200, 1) OVER (ORDER BY o.timestamp) as prev_sma_200,
                LAG(o.close, 1) OVER (ORDER BY o.timestamp) as prev_close,
                -- Average volume for comparison
                AVG(o.volume) OVER (ORDER BY o.timestamp ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as avg_volume
            FROM sma_indicator s
            JOIN ohlcv_data o ON s.ohlcv_id = o.id
            WHERE o.timestamp >= datetime('now', '-6 months')
        ),
        signals AS (
            SELECT 
                timestamp,
                close,
                volume,
                sma_50,
                sma_200,
                trend_strength,
                cross_signal,
                CASE 
                    -- LONG ENTRY CONDITIONS
                    WHEN 
                        cross_signal = 'golden_cross'
                        AND close > sma_50
                        AND trend_strength > 2
                        AND volume > avg_volume * 1.2
                    THEN 1  -- BUY SIGNAL
                    
                    -- SHORT ENTRY CONDITIONS
                    WHEN 
                        cross_signal = 'death_cross'
                        AND close < sma_50
                        AND trend_strength < -2
                        AND volume > avg_volume * 1.2
                    THEN -1  -- SELL SIGNAL
                    
                    -- EXIT LONG CONDITIONS
                    WHEN 
                        cross_signal = 'death_cross'
                        OR (close < sma_50 AND prev_close < sma_50)  -- 2 consecutive closes below SMA 50
                        OR trend_strength < -5
                        OR close < sma_200
                    THEN -2  -- EXIT LONG SIGNAL
                    
                    -- EXIT SHORT CONDITIONS
                    WHEN 
                        cross_signal = 'golden_cross'
                        OR (close > sma_50 AND prev_close > sma_50)  -- 2 consecutive closes above SMA 50
                        OR trend_strength > 5
                        OR close > sma_200
                    THEN 2  -- EXIT SHORT SIGNAL
                    
                    ELSE 0  -- HOLD
                END as signal
            FROM sma_analysis
            WHERE prev_sma_50 IS NOT NULL
        )
        SELECT 
            timestamp,
            close as price,
            sma_50,
            sma_200,
            trend_strength,
            cross_signal,
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
    strategy = SMAGoldenCrossStrategy()
    
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
