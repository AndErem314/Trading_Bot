"""
Ichimoku Cloud Breakout Trading Strategy

Strategy Name: Ichimoku Cloud Breakout Strategy

Description:
This strategy uses the Ichimoku Cloud system to identify strong trend breakouts,
momentum shifts, and support/resistance levels. It combines cloud breakouts,
TK (Tenkan-Kijun) crosses, and lagging span confirmations to generate high-probability
trading signals. The strategy is designed for trend following in all market conditions.

Author: Andrey's Trading Bot
Date: 2025-08-29
"""

class IchimokuCloudBreakoutStrategy:
    """
    Ichimoku Cloud-based trading strategy that uses cloud breakouts and line crosses.
    """
    
    def __init__(self):
        self.name = "Ichimoku Cloud Breakout Strategy"
        self.version = "1.0.0"
        self.min_holding_period = 6  # Minimum hours to hold position
        self.max_holding_period = 96  # Maximum hours to hold position (4 days)
        
    def get_strategy_description(self):
        """
        Returns detailed description of the strategy.
        """
        return {
            "name": self.name,
            "type": "Trend Following",
            "timeframe": "4H preferred (can work with 1D)",
            "indicators": ["Tenkan-sen", "Kijun-sen", "Senkou Span A/B", "Chikou Span", "Cloud Color"],
            "risk_level": "Medium",
            "expected_trade_duration": "6-96 hours",
            "market_conditions": "Works in all markets but best in trending conditions",
            "description": """
            This strategy identifies trading opportunities by:
            1. Cloud breakouts (price breaking above/below cloud)
            2. TK crosses (Tenkan crossing Kijun) for momentum
            3. Cloud color changes (future trend indication)
            4. Chikou span position for trend confirmation
            5. Strong signals when all components align
            """
        }
    
    def get_entry_conditions(self):
        """
        Returns the precise entry conditions for the strategy.
        """
        return {
            "LONG_ENTRY": {
                "primary_conditions": [
                    "Price breaks above the cloud (above both Senkou spans)",
                    "Tenkan-sen is above Kijun-sen (bullish TK cross)",
                    "Cloud is green (Senkou A > Senkou B) or turning green"
                ],
                "confirmation_conditions": [
                    "Chikou span is above its corresponding price 26 periods ago",
                    "Ichimoku signal is 'bullish' or 'strong_bullish'",
                    "Price has closed above cloud for at least 2 periods"
                ],
                "description": "Enter long on cloud breakout with bullish alignment"
            },
            "SHORT_ENTRY": {
                "primary_conditions": [
                    "Price breaks below the cloud (below both Senkou spans)",
                    "Tenkan-sen is below Kijun-sen (bearish TK cross)",
                    "Cloud is red (Senkou A < Senkou B) or turning red"
                ],
                "confirmation_conditions": [
                    "Chikou span is below its corresponding price 26 periods ago",
                    "Ichimoku signal is 'bearish' or 'strong_bearish'",
                    "Price has closed below cloud for at least 2 periods"
                ],
                "description": "Enter short on cloud breakdown with bearish alignment"
            }
        }
    
    def get_exit_conditions(self):
        """
        Returns the precise exit conditions for the strategy.
        """
        return {
            "LONG_EXIT": {
                "take_profit": [
                    "Price reaches opposite side of cloud (strong resistance)",
                    "Tenkan-sen crosses below Kijun-sen (bearish TK cross)",
                    "Chikou span crosses below price",
                    "Cloud color changes from green to red ahead"
                ],
                "stop_loss": [
                    "Price closes back below the cloud",
                    "Price drops below Kijun-sen support",
                    "Maximum holding period reached (96 hours)"
                ],
                "description": "Exit long on trend reversal signals or cloud re-entry"
            },
            "SHORT_EXIT": {
                "take_profit": [
                    "Price reaches opposite side of cloud (strong support)",
                    "Tenkan-sen crosses above Kijun-sen (bullish TK cross)",
                    "Chikou span crosses above price",
                    "Cloud color changes from red to green ahead"
                ],
                "stop_loss": [
                    "Price closes back above the cloud",
                    "Price rises above Kijun-sen resistance",
                    "Maximum holding period reached (96 hours)"
                ],
                "description": "Exit short on trend reversal signals or cloud re-entry"
            }
        }
    
    def get_risk_management_rules(self):
        """
        Returns risk management rules for the strategy.
        """
        return {
            "position_sizing": {
                "method": "Cloud thickness based",
                "risk_per_trade": "1.5% of account",
                "max_concurrent_positions": 3,
                "scaling": "Larger positions when cloud is thick (strong trend)"
            },
            "stop_loss": {
                "initial": "Outside cloud or at Kijun-sen",
                "trailing": "Trail to Tenkan-sen in strong trends",
                "time_based": "Exit after 96 hours"
            },
            "filters": {
                "volatility": "Prefer thick clouds (clear trends)",
                "alignment": "All Ichimoku components should align",
                "time": "Avoid thin cloud periods (consolidation)"
            },
            "additional_rules": [
                "Best signals when breaking out of thick clouds",
                "Avoid trading inside the cloud (uncertainty)",
                "Strong signals when cloud color matches direction",
                "Use Kijun-sen as dynamic support/resistance",
                "Consider multiple timeframe cloud alignment"
            ]
        }
    
    def get_sql_query(self):
        """
        Returns SQL query to identify trading signals based on this strategy.
        """
        return """
        WITH ichimoku_analysis AS (
            SELECT 
                o.timestamp,
                o.close,
                i.tenkan_sen,
                i.kijun_sen,
                i.senkou_span_a,
                i.senkou_span_b,
                i.chikou_span,
                i.cloud_color,
                i.ichimoku_signal,
                -- Calculate cloud boundaries
                CASE 
                    WHEN i.senkou_span_a > i.senkou_span_b THEN i.senkou_span_a 
                    ELSE i.senkou_span_b 
                END as cloud_top,
                CASE 
                    WHEN i.senkou_span_a < i.senkou_span_b THEN i.senkou_span_a 
                    ELSE i.senkou_span_b 
                END as cloud_bottom,
                -- Previous values for crossover detection
                LAG(o.close, 1) OVER (ORDER BY o.timestamp) as prev_close,
                LAG(i.tenkan_sen, 1) OVER (ORDER BY o.timestamp) as prev_tenkan,
                LAG(i.kijun_sen, 1) OVER (ORDER BY o.timestamp) as prev_kijun,
                LAG(o.close, 26) OVER (ORDER BY o.timestamp) as price_26_ago,
                -- Cloud position
                CASE 
                    WHEN o.close > i.senkou_span_a AND o.close > i.senkou_span_b THEN 'above'
                    WHEN o.close < i.senkou_span_a AND o.close < i.senkou_span_b THEN 'below'
                    ELSE 'inside'
                END as price_vs_cloud
            FROM ichimoku_indicator i
            JOIN ohlcv_data o ON i.ohlcv_id = o.id
            WHERE o.timestamp >= datetime('now', '-3 months')
        ),
        signals AS (
            SELECT 
                timestamp,
                close,
                tenkan_sen,
                kijun_sen,
                cloud_top,
                cloud_bottom,
                cloud_color,
                ichimoku_signal,
                CASE 
                    -- LONG ENTRY CONDITIONS
                    WHEN 
                        price_vs_cloud = 'above'
                        AND LAG(price_vs_cloud, 1) OVER (ORDER BY timestamp) != 'above'
                        AND tenkan_sen > kijun_sen
                        AND cloud_color = 'green'
                        AND ichimoku_signal IN ('bullish', 'strong_bullish')
                        AND chikou_span > price_26_ago
                    THEN 1  -- BUY SIGNAL
                    
                    -- SHORT ENTRY CONDITIONS
                    WHEN 
                        price_vs_cloud = 'below'
                        AND LAG(price_vs_cloud, 1) OVER (ORDER BY timestamp) != 'below'
                        AND tenkan_sen < kijun_sen
                        AND cloud_color = 'red'
                        AND ichimoku_signal IN ('bearish', 'strong_bearish')
                        AND chikou_span < price_26_ago
                    THEN -1  -- SELL SIGNAL
                    
                    -- EXIT LONG CONDITIONS
                    WHEN 
                        (tenkan_sen < kijun_sen AND prev_tenkan >= prev_kijun)  -- TK bearish cross
                        OR price_vs_cloud = 'below'  -- Back below cloud
                        OR close < kijun_sen  -- Below Kijun support
                    THEN -2  -- EXIT LONG SIGNAL
                    
                    -- EXIT SHORT CONDITIONS
                    WHEN 
                        (tenkan_sen > kijun_sen AND prev_tenkan <= prev_kijun)  -- TK bullish cross
                        OR price_vs_cloud = 'above'  -- Back above cloud
                        OR close > kijun_sen  -- Above Kijun resistance
                    THEN 2  -- EXIT SHORT SIGNAL
                    
                    ELSE 0  -- HOLD
                END as signal
            FROM ichimoku_analysis
            WHERE prev_close IS NOT NULL
        )
        SELECT 
            timestamp,
            close as price,
            tenkan_sen,
            kijun_sen,
            cloud_top,
            cloud_bottom,
            cloud_color,
            ichimoku_signal,
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
    strategy = IchimokuCloudBreakoutStrategy()
    
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
