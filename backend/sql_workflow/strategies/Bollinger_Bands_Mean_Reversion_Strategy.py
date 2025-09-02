"""
Bollinger Bands Mean Reversion Trading Strategy

Strategy Name: Bollinger Bands Mean Reversion Strategy

Description:
This strategy uses Bollinger Bands to identify overbought and oversold conditions,
volatility squeezes, and band breakouts. It combines mean reversion principles
with volatility-based signals to capture price movements when they deviate from
the statistical mean. The strategy works well in ranging markets and volatility expansions.

Author: Andrey's Trading Bot
Date: 2025-08-29
"""

class BollingerBandsMeanReversionStrategy:
    """
    Bollinger Bands-based trading strategy that uses band touches, squeezes, and volatility patterns.
    """
    
    def __init__(self):
        self.name = "Bollinger Bands Mean Reversion Strategy"
        self.version = "1.0.0"
        self.min_holding_period = 2  # Minimum hours to hold position
        self.max_holding_period = 48  # Maximum hours to hold position (2 days)
        
    def get_strategy_description(self):
        """
        Returns detailed description of the strategy.
        """
        return {
            "name": self.name,
            "type": "Mean Reversion",
            "timeframe": "4H preferred (can work with 1H)",
            "indicators": ["BB Upper", "BB Lower", "BB Middle", "BB Width", "BB Percent"],
            "risk_level": "Medium",
            "expected_trade_duration": "2-48 hours",
            "market_conditions": "Works best in ranging markets and volatility expansions",
            "description": """
            This strategy identifies trading opportunities by:
            1. Band touches and rejections for mean reversion trades
            2. Volatility squeezes for breakout anticipation
            3. Band expansions for trend continuation
            4. %B extremes for overbought/oversold conditions
            5. Walking the bands patterns for strong trends
            """
        }
    
    def get_entry_conditions(self):
        """
        Returns the precise entry conditions for the strategy.
        """
        return {
            "LONG_ENTRY": {
                "primary_conditions": [
                    "Price touches or breaks below lower Bollinger Band (close <= bb_lower)",
                    "%B is below 0.2 (oversold condition)",
                    "NOT in a strong downtrend (price not walking the lower band)"
                ],
                "confirmation_conditions": [
                    "BB Width is expanding (volatility increasing) OR",
                    "Coming out of a squeeze (bb_squeeze was true recently)",
                    "Price shows rejection from lower band (wick below, close above)"
                ],
                "description": "Enter long when price is oversold at lower band with reversal signs"
            },
            "SHORT_ENTRY": {
                "primary_conditions": [
                    "Price touches or breaks above upper Bollinger Band (close >= bb_upper)",
                    "%B is above 0.8 (overbought condition)",
                    "NOT in a strong uptrend (price not walking the upper band)"
                ],
                "confirmation_conditions": [
                    "BB Width is expanding (volatility increasing) OR",
                    "Coming out of a squeeze (bb_squeeze was true recently)",
                    "Price shows rejection from upper band (wick above, close below)"
                ],
                "description": "Enter short when price is overbought at upper band with reversal signs"
            }
        }
    
    def get_exit_conditions(self):
        """
        Returns the precise exit conditions for the strategy.
        """
        return {
            "LONG_EXIT": {
                "take_profit": [
                    "Price reaches middle band (bb_middle) for conservative exit",
                    "Price reaches upper band (bb_upper) for aggressive exit",
                    "%B rises above 0.8 (from oversold to overbought)",
                    "Band width starts contracting after expansion"
                ],
                "stop_loss": [
                    "Price closes below lower band by more than 1%",
                    "%B falls below -0.1 (strong breakdown)",
                    "Maximum holding period reached (48 hours)"
                ],
                "description": "Exit long on mean reversion to middle/upper band or breakdown"
            },
            "SHORT_EXIT": {
                "take_profit": [
                    "Price reaches middle band (bb_middle) for conservative exit",
                    "Price reaches lower band (bb_lower) for aggressive exit",
                    "%B falls below 0.2 (from overbought to oversold)",
                    "Band width starts contracting after expansion"
                ],
                "stop_loss": [
                    "Price closes above upper band by more than 1%",
                    "%B rises above 1.1 (strong breakout)",
                    "Maximum holding period reached (48 hours)"
                ],
                "description": "Exit short on mean reversion to middle/lower band or breakout"
            }
        }
    
    def get_risk_management_rules(self):
        """
        Returns risk management rules for the strategy.
        """
        return {
            "position_sizing": {
                "method": "Volatility-based sizing",
                "risk_per_trade": "1.5% of account",
                "max_concurrent_positions": 4,
                "scaling": "Reduce size in high volatility (wide bands)"
            },
            "stop_loss": {
                "initial": "2% from entry or outside bands by 1%",
                "trailing": "Trail to middle band after reaching target",
                "time_based": "Exit after 48 hours if no profit"
            },
            "filters": {
                "volatility": "Avoid entries during extreme band width",
                "squeeze": "Prefer entries after squeeze release",
                "volume": "Require above-average volume on band touches"
            },
            "additional_rules": [
                "Don't fight strong trends (price walking the bands)",
                "Reduce position size during news events",
                "Best signals come from multiple band touches",
                "Watch for failed breakouts as reversal signals",
                "Consider market regime (trending vs ranging)"
            ]
        }
    
    def get_sql_query(self):
        """
        Returns SQL query to identify trading signals based on this strategy.
        """
        return """
        WITH bb_analysis AS (
            SELECT 
                o.timestamp,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume,
                b.bb_upper,
                b.bb_middle,
                b.bb_lower,
                b.bb_width,
                b.bb_percent,
                -- Detect band touches
                CASE WHEN o.close <= b.bb_lower THEN 1 ELSE 0 END as touches_lower,
                CASE WHEN o.close >= b.bb_upper THEN 1 ELSE 0 END as touches_upper,
                -- Detect squeeze
                CASE WHEN b.bb_width < LAG(b.bb_width, 20) OVER (ORDER BY o.timestamp) * 0.75 
                THEN 1 ELSE 0 END as in_squeeze,
                -- Previous values for trend detection
                LAG(o.close, 1) OVER (ORDER BY o.timestamp) as prev_close,
                LAG(b.bb_percent, 1) OVER (ORDER BY o.timestamp) as prev_bb_percent,
                LAG(b.bb_width, 1) OVER (ORDER BY o.timestamp) as prev_bb_width,
                -- Check if walking the bands (3 consecutive touches)
                SUM(CASE WHEN o.close >= b.bb_upper THEN 1 ELSE 0 END) 
                    OVER (ORDER BY o.timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as upper_band_count,
                SUM(CASE WHEN o.close <= b.bb_lower THEN 1 ELSE 0 END) 
                    OVER (ORDER BY o.timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as lower_band_count
            FROM bollinger_bands_indicator b
            JOIN ohlcv_data o ON b.ohlcv_id = o.id
            WHERE o.timestamp >= datetime('now', '-3 months')
        ),
        signals AS (
            SELECT 
                timestamp,
                close,
                bb_upper,
                bb_middle,
                bb_lower,
                bb_width,
                bb_percent,
                CASE 
                    -- LONG ENTRY CONDITIONS
                    WHEN 
                        touches_lower = 1
                        AND bb_percent < 0.2
                        AND lower_band_count < 3  -- Not walking the lower band
                        AND (bb_width > prev_bb_width OR in_squeeze = 1)  -- Volatility condition
                    THEN 1  -- BUY SIGNAL
                    
                    -- SHORT ENTRY CONDITIONS
                    WHEN 
                        touches_upper = 1
                        AND bb_percent > 0.8
                        AND upper_band_count < 3  -- Not walking the upper band
                        AND (bb_width > prev_bb_width OR in_squeeze = 1)  -- Volatility condition
                    THEN -1  -- SELL SIGNAL
                    
                    -- EXIT LONG CONDITIONS
                    WHEN 
                        bb_percent >= 0.5  -- Reached middle band
                        OR bb_percent >= 0.8  -- Reached overbought
                        OR (bb_percent < -0.1 AND prev_bb_percent >= -0.1)  -- Breakdown
                    THEN -2  -- EXIT LONG SIGNAL
                    
                    -- EXIT SHORT CONDITIONS
                    WHEN 
                        bb_percent <= 0.5  -- Reached middle band
                        OR bb_percent <= 0.2  -- Reached oversold
                        OR (bb_percent > 1.1 AND prev_bb_percent <= 1.1)  -- Breakout
                    THEN 2  -- EXIT SHORT SIGNAL
                    
                    ELSE 0  -- HOLD
                END as signal
            FROM bb_analysis
            WHERE prev_close IS NOT NULL
        )
        SELECT 
            timestamp,
            close as price,
            bb_upper,
            bb_middle,
            bb_lower,
            bb_width,
            bb_percent,
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
    strategy = BollingerBandsMeanReversionStrategy()
    
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
