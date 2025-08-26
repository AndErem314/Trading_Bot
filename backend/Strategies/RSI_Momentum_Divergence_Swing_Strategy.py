"""
RSI Momentum Divergence Swing Trading Strategy

Strategy Name: RSI Momentum Divergence Swing Strategy

Description:
This swing trading strategy leverages RSI (Relative Strength Index) patterns, momentum shifts,
divergence signals, and RSI moving averages to identify high-probability swing trading opportunities.
The strategy focuses on capturing medium-term price swings by identifying oversold/overbought
conditions with momentum confirmation. Designed for 24/7 cryptocurrency markets with positions
that can be held over weekends.

Author: Andrey's Trading Bot
Date: 2025-08-26
"""

class RSIMomentumDivergenceSwingStrategy:
    """
    RSI-based swing trading strategy that combines multiple RSI signals for robust entry/exit points.
    """
    
    def __init__(self):
        self.name = "RSI Momentum Divergence Swing Strategy"
        self.version = "1.1.0"  # Updated for 24/7 trading
        self.min_holding_period = 4  # Minimum hours to hold position
        self.max_holding_period = 168  # Maximum hours to hold position (1 week)
        
    def get_strategy_description(self):
        """
        Returns detailed description of the strategy.
        """
        return {
            "name": self.name,
            "type": "Swing Trading",
            "timeframe": "1H preferred (can work with 4H)",
            "indicators": ["RSI", "RSI SMA 5", "RSI SMA 10", "Momentum Shift", "Divergence"],
            "risk_level": "Medium",
            "expected_trade_duration": "4-168 hours",
            "market_conditions": "Works best in ranging and trending markets",
            "description": """
            This strategy identifies swing trading opportunities by combining:
            1. RSI extreme levels (oversold/overbought)
            2. RSI moving average crossovers for trend confirmation
            3. Momentum shifts for entry timing
            4. Divergence signals for additional confirmation
            5. Trend strength filters to avoid false signals
            """
        }
    
    def get_entry_conditions(self):
        """
        Returns the precise entry conditions for the strategy.
        """
        return {
            "LONG_ENTRY": {
                "primary_conditions": [
                    "RSI crosses above 30 from oversold territory (RSI was < 30 and now > 30)",
                    "RSI SMA 5 crosses above RSI SMA 10 (bullish crossover)",
                    "Momentum shift is TRUE (indicating positive momentum change)"
                ],
                "confirmation_conditions": [
                    "Divergence signal is 'bullish' OR 'none' (not bearish)",
                    "Trend strength is not 'strong_bearish'"
                ],
                "description": "Enter long when RSI recovers from oversold with momentum confirmation"
            },
            "SHORT_ENTRY": {
                "primary_conditions": [
                    "RSI crosses below 70 from overbought territory (RSI was > 70 and now < 70)",
                    "RSI SMA 5 crosses below RSI SMA 10 (bearish crossover)",
                    "Momentum shift is TRUE (indicating negative momentum change)"
                ],
                "confirmation_conditions": [
                    "Divergence signal is 'bearish' OR 'none' (not bullish)",
                    "Trend strength is not 'strong_bullish'"
                ],
                "description": "Enter short when RSI falls from overbought with momentum confirmation"
            }
        }
    
    def get_exit_conditions(self):
        """
        Returns the precise exit conditions for the strategy.
        """
        return {
            "LONG_EXIT": {
                "take_profit": [
                    "RSI reaches 70 or above (overbought)",
                    "RSI SMA 5 crosses below RSI SMA 10 (bearish crossover)",
                    "Divergence signal turns 'bearish'"
                ],
                "stop_loss": [
                    "RSI falls back below 25",
                    "Trend strength becomes 'strong_bearish'",
                    "Maximum holding period reached (168 hours)"
                ],
                "description": "Exit long on overbought conditions or trend reversal signals"
            },
            "SHORT_EXIT": {
                "take_profit": [
                    "RSI reaches 30 or below (oversold)",
                    "RSI SMA 5 crosses above RSI SMA 10 (bullish crossover)",
                    "Divergence signal turns 'bullish'"
                ],
                "stop_loss": [
                    "RSI rises back above 75",
                    "Trend strength becomes 'strong_bullish'",
                    "Maximum holding period reached (168 hours)"
                ],
                "description": "Exit short on oversold conditions or trend reversal signals"
            }
        }
    
    def get_risk_management_rules(self):
        """
        Returns risk management rules for the strategy.
        """
        return {
            "position_sizing": {
                "method": "Fixed percentage risk",
                "risk_per_trade": "2% of account",
                "max_concurrent_positions": 3,
                "scaling": "Can scale into positions when RSI < 25 or > 75"
            },
            "stop_loss": {
                "initial": "3% from entry price",
                "trailing": "Activate at 5% profit, trail by 2%",
                "time_based": "Exit after 168 hours if no profit target hit"
            },
            "filters": {
                "volatility": "Avoid entries when RSI changes > 20 points in 1 hour",
                "time": "No new positions 1 hour before major news events",
                "correlation": "Check correlation with market index before entry"
            },
            "additional_rules": [
                "24/7 trading enabled - positions can be held over weekends",
                "Reduce position size by 50% if consecutive losses > 3",
                "Wait for at least 4 candles between trades on same asset",
                "Use limit orders for entries to avoid slippage",
                "Monitor volume - prefer entries with above-average volume",
                "Consider reducing position size during low-liquidity periods (holidays)",
                "Monitor for exchange maintenance windows and have contingency orders"
            ]
        }
    
    def get_sql_query(self):
        """
        Returns SQL query to identify trading signals based on this strategy.
        """
        return """
        WITH rsi_analysis AS (
            SELECT 
                o.timestamp,
                o.close,
                r.rsi,
                r.rsi_sma_5,
                r.rsi_sma_10,
                r.overbought,
                r.oversold,
                r.trend_strength,
                r.divergence_signal,
                r.momentum_shift,
                LAG(r.rsi, 1) OVER (ORDER BY o.timestamp) as prev_rsi,
                LAG(r.rsi_sma_5, 1) OVER (ORDER BY o.timestamp) as prev_rsi_sma_5,
                LAG(r.rsi_sma_10, 1) OVER (ORDER BY o.timestamp) as prev_rsi_sma_10,
                LAG(r.oversold, 1) OVER (ORDER BY o.timestamp) as prev_oversold,
                LAG(r.overbought, 1) OVER (ORDER BY o.timestamp) as prev_overbought
            FROM rsi_indicator r
            JOIN ohlcv_data o ON r.ohlcv_id = o.id
            WHERE o.timestamp >= datetime('now', '-6 months')  -- Analyze last 6 months
        ),
        signals AS (
            SELECT 
                timestamp,
                close,
                rsi,
                rsi_sma_5,
                rsi_sma_10,
                trend_strength,
                divergence_signal,
                momentum_shift,
                CASE 
                    -- LONG ENTRY CONDITIONS
                    WHEN 
                        -- RSI crosses above 30 from oversold
                        prev_rsi <= 30 AND rsi > 30
                        -- RSI SMA 5 crosses above RSI SMA 10
                        AND prev_rsi_sma_5 <= prev_rsi_sma_10 AND rsi_sma_5 > rsi_sma_10
                        -- Momentum shift is true
                        AND momentum_shift = 1
                        -- Divergence is not bearish
                        AND (divergence_signal IN ('bullish', 'none'))
                        -- Trend is not strongly bearish
                        AND trend_strength != 'strong_bearish'
                    THEN 1  -- BUY SIGNAL
                    
                    -- SHORT ENTRY CONDITIONS
                    WHEN 
                        -- RSI crosses below 70 from overbought
                        prev_rsi >= 70 AND rsi < 70
                        -- RSI SMA 5 crosses below RSI SMA 10
                        AND prev_rsi_sma_5 >= prev_rsi_sma_10 AND rsi_sma_5 < rsi_sma_10
                        -- Momentum shift is true
                        AND momentum_shift = 1
                        -- Divergence is not bullish
                        AND (divergence_signal IN ('bearish', 'none'))
                        -- Trend is not strongly bullish
                        AND trend_strength != 'strong_bullish'
                    THEN -1  -- SELL SIGNAL
                    
                    -- EXIT CONDITIONS FOR LONGS (when in long position)
                    WHEN 
                        -- Take profit: RSI overbought or bearish crossover
                        (rsi >= 70 OR 
                         (prev_rsi_sma_5 >= prev_rsi_sma_10 AND rsi_sma_5 < rsi_sma_10) OR
                         divergence_signal = 'bearish')
                    THEN -2  -- EXIT LONG SIGNAL
                    
                    -- EXIT CONDITIONS FOR SHORTS (when in short position)
                    WHEN 
                        -- Take profit: RSI oversold or bullish crossover
                        (rsi <= 30 OR 
                         (prev_rsi_sma_5 <= prev_rsi_sma_10 AND rsi_sma_5 > rsi_sma_10) OR
                         divergence_signal = 'bullish')
                    THEN 2  -- EXIT SHORT SIGNAL
                    
                    ELSE 0  -- HOLD
                END as signal
            FROM rsi_analysis
            WHERE prev_rsi IS NOT NULL  -- Ensure we have previous values
        )
        SELECT 
            timestamp,
            close as price,
            rsi,
            rsi_sma_5,
            rsi_sma_10,
            trend_strength,
            divergence_signal,
            momentum_shift,
            signal,
            CASE signal
                WHEN 1 THEN 'BUY'
                WHEN -1 THEN 'SELL'
                WHEN 2 THEN 'EXIT_SHORT'
                WHEN -2 THEN 'EXIT_LONG'
                ELSE 'HOLD'
            END as signal_name
        FROM signals
        WHERE signal != 0  -- Only show actual signals, not holds
        ORDER BY timestamp DESC
        LIMIT 100;  -- Last 100 signals
        """
    


# Example usage and testing
if __name__ == "__main__":
    strategy = RSIMomentumDivergenceSwingStrategy()
    
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
    
    print("\n" + "="*50 + "\n")
    
    # Print SQL query
    print("SQL QUERY FOR SIGNAL IDENTIFICATION:")
    print(strategy.get_sql_query())
