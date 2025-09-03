#!/usr/bin/env python3
"""
Demonstration script showing the difference between Optimization and LLM Analysis
This script provides practical examples and explanations
"""

import subprocess
import json
import time
from datetime import datetime

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")

def run_command(cmd):
    """Run a command and display output"""
    print(f"Command: {' '.join(cmd)}")
    print("-" * 40)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    return result.returncode == 0

def demonstrate_optimization():
    """Demonstrate parameter optimization"""
    print_section("PARAMETER OPTIMIZATION DEMONSTRATION")
    
    print("What is Parameter Optimization?")
    print("-" * 40)
    print("Parameter optimization automatically tests different parameter combinations")
    print("to find the values that produce the best trading performance.")
    print()
    print("For example, for Bollinger Bands strategy:")
    print("- bb_length: Should we use 20, 25, or 30 periods?")
    print("- bb_std: Is 2.0, 2.5, or 3.0 standard deviations better?")
    print("- rsi_oversold: Should the threshold be 30, 35, or 40?")
    print()
    print("The optimizer will test ALL these combinations and find the best one!")
    
    input("\nPress Enter to see optimization in action...")
    
    print("\nRunning optimization for Bollinger Bands strategy...")
    print("This will test multiple parameter combinations:")
    
    # Show what parameters will be tested
    print("\nParameter ranges being tested:")
    print("  bb_length: 10 to 30 (step 5) -> Tests: 10, 15, 20, 25, 30")
    print("  bb_std: 1.5 to 3.0 (step 0.5) -> Tests: 1.5, 2.0, 2.5, 3.0")
    print("  rsi_length: 10 to 20 (step 2) -> Tests: 10, 12, 14, 16, 18, 20")
    print("  rsi_oversold: 20 to 40 (step 5) -> Tests: 20, 25, 30, 35, 40")
    print("  rsi_overbought: 60 to 80 (step 5) -> Tests: 60, 65, 70, 75, 80")
    print()
    print("Total combinations to test: 5 × 4 × 6 × 5 × 5 = 3,000 combinations!")
    
    # Run the optimization
    cmd = [
        'python', 
        'backend/backtesting/scripts/run_single_strategy.py', 
        'bollinger_bands',
        '--optimize'
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    print("\nNOTE: This may take several minutes as it tests many combinations...")
    
    # In real execution, this would run the command
    # run_command(cmd)
    
    print("\nOptimization Results (example output):")
    print("-" * 40)
    print("Best parameters found:")
    print("  bb_length: 20")
    print("  bb_std: 2.5") 
    print("  rsi_length: 14")
    print("  rsi_oversold: 30")
    print("  rsi_overbought: 70")
    print()
    print("Performance with optimized parameters:")
    print("  Total Return: 45.23%")
    print("  Sharpe Ratio: 1.85")
    print("  Max Drawdown: -12.34%")
    print("  Win Rate: 58.5%")

def demonstrate_llm_analysis():
    """Demonstrate LLM analysis"""
    print_section("LLM ANALYSIS DEMONSTRATION")
    
    print("What is LLM Analysis?")
    print("-" * 40)
    print("LLM (Large Language Model) analysis uses AI to understand and explain")
    print("your strategy's performance. It doesn't change parameters, but provides")
    print("insights about WHY the strategy performed the way it did.")
    print()
    print("The AI analyzes:")
    print("- Trade patterns and timing")
    print("- Market conditions during wins/losses")
    print("- Risk exposure and drawdown periods")
    print("- Suggestions for improvements")
    
    input("\nPress Enter to see LLM analysis in action...")
    
    print("\nRunning strategy with LLM analysis...")
    
    # First run a basic backtest
    cmd = [
        'python',
        'backend/backtesting/scripts/run_single_strategy.py',
        'rsi_divergence',
        '--params', '{"symbol": "BTC/USDT"}',
        '--analyze'
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    print("\nThe LLM will analyze the backtest results...")
    
    # In real execution, this would run the command
    # run_command(cmd)
    
    print("\nLLM Analysis Report (example output):")
    print("-" * 40)
    print("""
RSI Divergence Strategy Analysis Report

**Performance Overview:**
The RSI Divergence strategy showed moderate performance with a total return of 23.4% 
and a Sharpe ratio of 1.2. The strategy correctly identified several major trend 
reversals but struggled during choppy market conditions.

**Key Strengths:**
1. Successfully caught 3 major bottoms with divergence signals
2. Low false signal rate during strong trends (only 15% false positives)
3. Good risk management with average loss of -1.2% vs average win of 2.8%

**Areas for Improvement:**
1. The strategy underperformed during sideways markets (Jun-Aug period)
2. Consider adding a trend filter to avoid signals in ranging markets
3. The RSI oversold level of 30 might be too conservative for crypto markets

**Recommendations:**
- Test with RSI oversold level of 25 for more aggressive entries
- Add volume confirmation to reduce false signals
- Consider shorter RSI periods (10-12) for faster markets like crypto
- Implement a market regime filter to pause trading during low volatility

**Risk Analysis:**
Maximum drawdown occurred during the May correction (-18.5%). The strategy
held positions too long during the initial decline. Consider tighter stops
or faster exit signals when divergence fails to materialize.
    """)

def demonstrate_combined():
    """Demonstrate combined optimization and analysis"""
    print_section("COMBINED OPTIMIZATION + ANALYSIS")
    
    print("The Power of Using Both Together:")
    print("-" * 40)
    print("1. First, optimization finds the best parameters")
    print("2. Then, LLM analysis explains WHY those parameters work best")
    print("3. You get both optimal performance AND understanding")
    
    input("\nPress Enter to see combined example...")
    
    cmd = [
        'python',
        'backend/backtesting/scripts/run_single_strategy.py',
        'macd_momentum',
        '--optimize',
        '--analyze'
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    print("\nThis will:")
    print("1. Test hundreds of parameter combinations")
    print("2. Find the best performing parameters")
    print("3. Run final backtest with optimized parameters")
    print("4. Analyze the results with AI for insights")
    
    # In real execution, this would run the command
    # run_command(cmd)
    
    print("\nCombined Results (example):")
    print("-" * 40)
    print("Optimization found: macd_fast=10, macd_slow=24, macd_signal=8")
    print("\nLLM Analysis explains: 'The shorter MACD periods (10/24 vs standard 12/26)")
    print("work better for crypto markets due to higher volatility and faster cycles.'")

def show_practical_examples():
    """Show practical command examples"""
    print_section("PRACTICAL COMMAND EXAMPLES")
    
    examples = [
        {
            "desc": "Quick test with default parameters",
            "cmd": "python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --params '{\"symbol\": \"BTC/USDT\"}'"
        },
        {
            "desc": "Find best parameters (may take time)",
            "cmd": "python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --optimize"
        },
        {
            "desc": "Get AI insights on existing results",
            "cmd": "python backend/backtesting/scripts/run_single_strategy.py bollinger_bands --params '{\"symbol\": \"BTC/USDT\"}' --analyze"
        },
        {
            "desc": "Fast optimization with random search",
            "cmd": "python backend/backtesting/scripts/run_single_strategy.py macd_momentum --optimize --method random_search"
        },
        {
            "desc": "Smart optimization with Bayesian method",
            "cmd": "python backend/backtesting/scripts/run_single_strategy.py rsi_divergence --optimize --method bayesian"
        },
        {
            "desc": "Full workflow: optimize then analyze",
            "cmd": "python backend/backtesting/scripts/run_single_strategy.py ichimoku_cloud --optimize --analyze"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['desc']}:")
        print(f"   {example['cmd']}")
        print()

def main():
    """Main demonstration flow"""
    print("\n" + "="*80)
    print(" TRADING BOT: OPTIMIZATION vs ANALYSIS - INTERACTIVE DEMO")
    print("="*80)
    
    while True:
        print("\nWhat would you like to learn about?")
        print("1. Parameter Optimization (--optimize)")
        print("2. LLM Analysis (--analyze)")
        print("3. Combined Usage (both)")
        print("4. Show Command Examples")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            demonstrate_optimization()
        elif choice == '2':
            demonstrate_llm_analysis()
        elif choice == '3':
            demonstrate_combined()
        elif choice == '4':
            show_practical_examples()
        elif choice == '5':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    # Quick summary
    print("\nQUICK SUMMARY:")
    print("-" * 40)
    print("OPTIMIZATION: Finds the BEST parameter values (what numbers work best)")
    print("LLM ANALYSIS: Explains WHY the strategy works (or doesn't work)")
    print("\nUse --optimize to find best parameters")
    print("Use --analyze to understand performance")
    print("Use both together for maximum insight!")
    
    # Run interactive demo
    main()
