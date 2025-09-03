"""
LLM Analyzer using Google Gemini API and OpenAI API

This module provides AI-powered analysis of backtesting results
to suggest parameter optimizations and strategy improvements.
Supports both Google Gemini and OpenAI models.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import AI libraries conditionally
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class StrategyAnalysis:
    """Container for strategy analysis results"""
    strategy_name: str
    current_performance: Dict[str, float]
    suggested_parameters: Dict[str, Any]
    optimization_reasoning: str
    market_conditions: List[str]
    risk_assessment: str
    improvement_potential: float
    confidence_score: float  # 0-100 confidence in the analysis


class BaseLLMAnalyzer(ABC):
    """Base class for LLM analyzers"""
    
    @abstractmethod
    def generate_analysis(self, prompt: str) -> str:
        """Generate analysis from the LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available"""
        pass


class GeminiLLMAnalyzer(BaseLLMAnalyzer):
    """Gemini-specific LLM analyzer"""
    
    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
            
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def generate_analysis(self, prompt: str) -> str:
        """Generate analysis using Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
            
    def is_available(self) -> bool:
        """Check if Gemini is available"""
        return GEMINI_AVAILABLE and self.api_key is not None


class OpenAILLMAnalyzer(BaseLLMAnalyzer):
    """OpenAI-specific LLM analyzer"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")
            
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client with v1.0+ syntax
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
    def generate_analysis(self, prompt: str) -> str:
        """Generate analysis using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert quantitative trading analyst specializing in algorithmic trading strategy optimization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
            
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return OPENAI_AVAILABLE and self.api_key is not None


class LLMAnalyzer:
    """
    Main LLM Analyzer that can use either Gemini or OpenAI
    """
    
    def __init__(
        self,
        provider: Literal["gemini", "openai", "auto"] = "auto",
        api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini"
    ):
        """
        Initialize the LLM analyzer
        
        Args:
            provider: Which LLM provider to use ("gemini", "openai", or "auto")
            api_key: API key (if not provided, reads from environment)
            openai_model: OpenAI model to use (default: gpt-4)
        """
        self.provider = provider
        self.llm: Optional[BaseLLMAnalyzer] = None
        
        if provider == "auto":
            # Try OpenAI first, then Gemini
            try:
                self.llm = OpenAILLMAnalyzer(api_key, openai_model)
                self.active_provider = "openai"
                logger.info("Using OpenAI for LLM analysis")
            except (ImportError, ValueError):
                try:
                    self.llm = GeminiLLMAnalyzer(api_key)
                    self.active_provider = "gemini"
                    logger.info("Using Gemini for LLM analysis")
                except (ImportError, ValueError):
                    logger.warning("No LLM provider available. Analysis will use fallback methods.")
                    self.active_provider = None
        elif provider == "openai":
            self.llm = OpenAILLMAnalyzer(api_key, openai_model)
            self.active_provider = "openai"
        elif provider == "gemini":
            self.llm = GeminiLLMAnalyzer(api_key)
            self.active_provider = "gemini"
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def analyze_strategy_performance(
        self,
        strategy_name: str,
        performance_metrics: Dict[str, float],
        current_parameters: Dict[str, Any],
        optimization_ranges: Dict[str, Dict],
        trade_history: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> StrategyAnalysis:
        """
        Analyze strategy performance and suggest improvements
        
        Args:
            strategy_name: Name of the strategy
            performance_metrics: Current performance metrics
            current_parameters: Current strategy parameters
            optimization_ranges: Valid parameter ranges
            trade_history: DataFrame of historical trades
            market_data: Market data used in backtesting
            
        Returns:
            StrategyAnalysis object with recommendations
        """
        # Prepare context for the LLM
        context = self._prepare_analysis_context(
            strategy_name,
            performance_metrics,
            current_parameters,
            optimization_ranges,
            trade_history,
            market_data
        )
        
        # Generate analysis prompt
        prompt = self._create_analysis_prompt(context)
        
        # Get analysis from LLM or use fallback
        if self.llm and self.llm.is_available():
            try:
                response_text = self.llm.generate_analysis(prompt)
                analysis = self._parse_llm_response(response_text, strategy_name, performance_metrics)
                analysis.confidence_score = 85.0  # High confidence for LLM analysis
                logger.info(f"Successfully analyzed {strategy_name} strategy using {self.active_provider}")
                return analysis
            except Exception as e:
                logger.error(f"Error analyzing strategy with LLM: {e}")
        
        # Fallback to heuristic analysis
        logger.info(f"Using heuristic analysis for {strategy_name} strategy")
        analysis = self._create_heuristic_analysis(
            strategy_name,
            performance_metrics,
            current_parameters,
            optimization_ranges,
            trade_history,
            market_data
        )
        analysis.confidence_score = 65.0  # Lower confidence for heuristic analysis
        return analysis
    
    def _prepare_analysis_context(
        self,
        strategy_name: str,
        performance_metrics: Dict[str, float],
        current_parameters: Dict[str, Any],
        optimization_ranges: Dict[str, Dict],
        trade_history: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Prepare context data for LLM analysis"""
        
        # Calculate additional trade statistics
        trade_stats = {}
        if len(trade_history) > 0:
            winning_trades = trade_history[trade_history['pnl'] > 0]
            losing_trades = trade_history[trade_history['pnl'] <= 0]
            
            # Add duration calculation if timestamps exist
            if 'entry_timestamp' in trade_history.columns and 'exit_timestamp' in trade_history.columns:
                try:
                    trade_history['duration_hours'] = (
                        pd.to_datetime(trade_history['exit_timestamp']) - 
                        pd.to_datetime(trade_history['entry_timestamp'])
                    ).dt.total_seconds() / 3600
                    
                    if len(winning_trades) > 0 and 'duration_hours' in winning_trades.columns:
                        trade_stats['avg_win_duration'] = winning_trades['duration_hours'].mean()
                    else:
                        trade_stats['avg_win_duration'] = 0
                        
                    if len(losing_trades) > 0 and 'duration_hours' in losing_trades.columns:
                        trade_stats['avg_loss_duration'] = losing_trades['duration_hours'].mean()
                    else:
                        trade_stats['avg_loss_duration'] = 0
                except Exception as e:
                    logger.debug(f"Could not calculate trade durations: {e}")
                    trade_stats['avg_win_duration'] = 0
                    trade_stats['avg_loss_duration'] = 0
            
            trade_stats['best_trade_conditions'] = self._analyze_best_trades(trade_history, market_data)
            trade_stats['worst_trade_conditions'] = self._analyze_worst_trades(trade_history, market_data)
        
        # Analyze market conditions
        market_analysis = self._analyze_market_conditions(market_data)
        
        return {
            'strategy_name': strategy_name,
            'performance_metrics': performance_metrics,
            'current_parameters': current_parameters,
            'optimization_ranges': optimization_ranges,
            'trade_stats': trade_stats,
            'market_analysis': market_analysis
        }
    
    def _create_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Create a detailed prompt for LLM analysis"""
        
        prompt = f"""
You are an expert quantitative trading analyst. Analyze the following backtesting results 
for the {context['strategy_name']} trading strategy and provide specific recommendations 
for parameter optimization.

CURRENT PERFORMANCE METRICS:
- Total Return: {context['performance_metrics'].get('total_return', 0):.2f}%
- Sharpe Ratio: {context['performance_metrics'].get('sharpe_ratio', 0):.2f}
- Max Drawdown: {context['performance_metrics'].get('max_drawdown', 0):.2f}%
- Win Rate: {context['performance_metrics'].get('win_rate', 0):.2f}%
- Profit Factor: {context['performance_metrics'].get('profit_factor', 0):.2f}
- Total Trades: {context['performance_metrics'].get('total_trades', 0)}

CURRENT PARAMETERS:
{json.dumps(context['current_parameters'], indent=2)}

PARAMETER OPTIMIZATION RANGES:
{json.dumps(context['optimization_ranges'], indent=2)}

MARKET CONDITIONS ANALYSIS:
{json.dumps(context['market_analysis'], indent=2)}

Please provide your analysis in the following JSON structure:

{{
    "suggested_parameters": {{
        // Specific parameter values within the optimization ranges
        // Example: "bb_length": 25, "bb_std": 2.5
    }},
    "market_conditions": [
        // List of market conditions where strategy performs best
        // Example: "High volatility trending markets", "Range-bound consolidation"
    ],
    "risk_assessment": "Detailed risk assessment text",
    "optimization_reasoning": "Detailed explanation of why these parameters would improve performance",
    "improvement_potential": 15.0,  // Estimated percentage improvement
    "specific_recommendations": [
        // List of specific actionable recommendations
    ]
}}

Focus on:
1. Parameters that will improve Sharpe ratio while maintaining acceptable returns
2. Reducing maximum drawdown without sacrificing too much profit
3. Identifying the ideal market conditions for this strategy
4. Practical recommendations that can be implemented immediately
"""
        
        return prompt
    
    def _analyze_best_trades(self, trade_history: pd.DataFrame, market_data: pd.DataFrame) -> Dict:
        """Analyze conditions during best performing trades"""
        if len(trade_history) == 0:
            return {}
        
        # Get top 20% of trades by P&L percentage
        top_trades = trade_history.nlargest(max(1, len(trade_history) // 5), 'pnl_percentage')
        
        conditions = {
            'avg_volatility': 0,
            'trend_strength': 0,
            'avg_volume_ratio': 1.0,
            'price_momentum': 0
        }
        
        # Analyze market conditions during these trades
        for _, trade in top_trades.iterrows():
            # Get market data during trade period
            if 'entry_timestamp' in trade and 'exit_timestamp' in trade:
                mask = (market_data.index >= trade['entry_timestamp']) & \
                       (market_data.index <= trade['exit_timestamp'])
                period_data = market_data[mask]
                
                if len(period_data) > 0:
                    # Calculate volatility
                    returns = period_data['close'].pct_change().dropna()
                    conditions['avg_volatility'] += returns.std() * np.sqrt(252) * 100
                    
                    # Calculate trend strength (simple linear regression slope)
                    if len(period_data) > 1:
                        x = np.arange(len(period_data))
                        y = period_data['close'].values
                        slope = np.polyfit(x, y, 1)[0]
                        conditions['trend_strength'] += slope / period_data['close'].mean()
                    
                    # Volume analysis
                    if 'volume' in period_data.columns:
                        avg_volume = period_data['volume'].mean()
                        overall_avg = market_data['volume'].mean()
                        conditions['avg_volume_ratio'] += avg_volume / overall_avg if overall_avg > 0 else 1
        
        # Average the conditions
        num_trades = len(top_trades)
        for key in conditions:
            conditions[key] /= num_trades
            
        return conditions
    
    def _analyze_worst_trades(self, trade_history: pd.DataFrame, market_data: pd.DataFrame) -> Dict:
        """Analyze conditions during worst performing trades"""
        if len(trade_history) == 0:
            return {}
        
        # Get bottom 20% of trades by P&L percentage
        worst_trades = trade_history.nsmallest(max(1, len(trade_history) // 5), 'pnl_percentage')
        
        conditions = {
            'avg_volatility': 0,
            'trend_strength': 0,
            'avg_volume_ratio': 1.0,
            'price_momentum': 0
        }
        
        # Similar analysis as best trades
        for _, trade in worst_trades.iterrows():
            if 'entry_timestamp' in trade and 'exit_timestamp' in trade:
                mask = (market_data.index >= trade['entry_timestamp']) & \
                       (market_data.index <= trade['exit_timestamp'])
                period_data = market_data[mask]
                
                if len(period_data) > 0:
                    returns = period_data['close'].pct_change().dropna()
                    conditions['avg_volatility'] += returns.std() * np.sqrt(252) * 100
                    
                    if len(period_data) > 1:
                        x = np.arange(len(period_data))
                        y = period_data['close'].values
                        slope = np.polyfit(x, y, 1)[0]
                        conditions['trend_strength'] += slope / period_data['close'].mean()
                    
                    if 'volume' in period_data.columns:
                        avg_volume = period_data['volume'].mean()
                        overall_avg = market_data['volume'].mean()
                        conditions['avg_volume_ratio'] += avg_volume / overall_avg if overall_avg > 0 else 1
                        
        num_trades = len(worst_trades)
        for key in conditions:
            conditions[key] /= num_trades
            
        return conditions
    
    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> Dict:
        """Analyze overall market conditions during backtesting period"""
        if len(market_data) == 0:
            return {}
        
        returns = market_data['close'].pct_change().dropna()
        
        # Calculate various market metrics
        analysis = {
            'volatility': {
                'annual': returns.std() * np.sqrt(252) * 100,
                'regime': 'high' if returns.std() > 0.02 else 'normal' if returns.std() > 0.01 else 'low'
            },
            'trend': {
                'direction': 'bullish' if returns.mean() > 0 else 'bearish',
                'strength': abs(returns.mean()) * 252 * 100,
                'consistency': 1 - (returns.std() / abs(returns.mean())) if returns.mean() != 0 else 0
            },
            'price_range': {
                'high': float(market_data['high'].max()),
                'low': float(market_data['low'].min()),
                'range_pct': float(((market_data['high'].max() - market_data['low'].min()) / 
                             market_data['low'].min() * 100))
            },
            'volume_profile': {
                'avg_volume': float(market_data['volume'].mean()) if 'volume' in market_data else 0,
                'volume_trend': 'increasing' if market_data['volume'].iloc[-30:].mean() > 
                               market_data['volume'].iloc[:30].mean() else 'decreasing'
            } if 'volume' in market_data else {}
        }
        
        return analysis
    
    def _parse_llm_response(
        self,
        response_text: str,
        strategy_name: str,
        performance_metrics: Dict[str, float]
    ) -> StrategyAnalysis:
        """Parse the LLM response into a structured format"""
        
        try:
            # Try to parse as JSON first
            # Find JSON content in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                parsed_data = json.loads(json_text)
                
                return StrategyAnalysis(
                    strategy_name=strategy_name,
                    current_performance=performance_metrics,
                    suggested_parameters=parsed_data.get('suggested_parameters', {}),
                    optimization_reasoning=parsed_data.get('optimization_reasoning', ''),
                    market_conditions=parsed_data.get('market_conditions', []),
                    risk_assessment=parsed_data.get('risk_assessment', ''),
                    improvement_potential=float(parsed_data.get('improvement_potential', 10.0)),
                    confidence_score=85.0
                )
            else:
                # Fallback to text parsing
                return self._parse_text_response(response_text, strategy_name, performance_metrics)
                
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from LLM response, using text parsing")
            return self._parse_text_response(response_text, strategy_name, performance_metrics)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_fallback_analysis(strategy_name, performance_metrics, {})
    
    def _parse_text_response(
        self,
        response_text: str,
        strategy_name: str,
        performance_metrics: Dict[str, float]
    ) -> StrategyAnalysis:
        """Parse non-JSON text response from LLM"""
        
        # Simple text parsing logic
        suggested_parameters = {}
        market_conditions = []
        
        # Look for parameter suggestions in the text
        lines = response_text.split('\n')
        for line in lines:
            if ':' in line and any(param in line.lower() for param in ['length', 'period', 'threshold', 'multiplier']):
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    try:
                        value = float(parts[1].strip().split()[0])
                        suggested_parameters[key] = value
                    except:
                        pass
        
        # Extract market conditions (look for bullet points or numbered lists)
        for line in lines:
            if any(marker in line for marker in ['•', '-', '*']) and 'market' in line.lower():
                condition = line.strip('•-* ').strip()
                if condition:
                    market_conditions.append(condition)
        
        return StrategyAnalysis(
            strategy_name=strategy_name,
            current_performance=performance_metrics,
            suggested_parameters=suggested_parameters,
            optimization_reasoning=response_text[:500],  # First 500 chars as summary
            market_conditions=market_conditions if market_conditions else ["General market conditions"],
            risk_assessment="See full analysis for risk details",
            improvement_potential=15.0,  # Default estimate
            confidence_score=70.0
        )
    
    def _create_heuristic_analysis(
        self,
        strategy_name: str,
        performance_metrics: Dict[str, float],
        current_parameters: Dict[str, Any],
        optimization_ranges: Dict[str, Dict],
        trade_history: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> StrategyAnalysis:
        """Create analysis using heuristic rules when LLM is unavailable"""
        
        suggested_parameters = current_parameters.copy()
        recommendations = []
        
        # Analyze performance and suggest improvements
        sharpe = performance_metrics.get('sharpe_ratio', 0)
        win_rate = performance_metrics.get('win_rate', 0)
        max_dd = performance_metrics.get('max_drawdown', 0)
        profit_factor = performance_metrics.get('profit_factor', 0)
        
        # Win rate optimization
        if win_rate < 40:
            recommendations.append("Consider tightening entry conditions to improve win rate")
            # Adjust relevant parameters based on strategy type
            if 'rsi_oversold' in suggested_parameters:
                suggested_parameters['rsi_oversold'] = min(
                    suggested_parameters['rsi_oversold'] + 5,
                    optimization_ranges.get('rsi_oversold', {}).get('max', 40)
                )
            if 'bb_std' in suggested_parameters:
                suggested_parameters['bb_std'] = min(
                    suggested_parameters['bb_std'] + 0.5,
                    optimization_ranges.get('bb_std', {}).get('max', 3.0)
                )
        
        # Drawdown optimization
        if max_dd > 20:
            recommendations.append("High drawdown detected - implement tighter risk management")
            if 'atr_stop_multiplier' in suggested_parameters:
                suggested_parameters['atr_stop_multiplier'] = max(
                    suggested_parameters['atr_stop_multiplier'] - 0.5,
                    optimization_ranges.get('atr_stop_multiplier', {}).get('min', 1.5)
                )
        
        # Sharpe ratio optimization
        if sharpe < 1.0:
            recommendations.append("Low Sharpe ratio - focus on reducing volatility of returns")
            if 'lookback_period' in suggested_parameters:
                suggested_parameters['lookback_period'] = min(
                    suggested_parameters['lookback_period'] + 5,
                    optimization_ranges.get('lookback_period', {}).get('max', 30)
                )
        
        # Market condition analysis
        if len(market_data) > 0:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            if volatility > 30:
                market_conditions = ["High volatility markets", "Trending conditions preferred"]
            elif volatility < 15:
                market_conditions = ["Low volatility markets", "Range-bound conditions"]
            else:
                market_conditions = ["Normal volatility markets", "Mixed conditions"]
        else:
            market_conditions = ["Unable to determine market conditions"]
        
        # Risk assessment
        if max_dd > 25:
            risk_assessment = "High risk - significant drawdowns observed. Recommend position sizing reduction."
        elif max_dd > 15:
            risk_assessment = "Moderate risk - acceptable drawdowns but room for improvement."
        else:
            risk_assessment = "Low risk - drawdowns well controlled."
        
        # Improvement potential estimation
        improvement_potential = 0.0
        if win_rate < 45:
            improvement_potential += 10.0
        if sharpe < 1.0:
            improvement_potential += 10.0
        if max_dd > 20:
            improvement_potential += 5.0
        
        optimization_reasoning = f"""
Based on heuristic analysis:
- Current Sharpe ratio of {sharpe:.2f} suggests {'good' if sharpe > 1.5 else 'room for improvement in'} risk-adjusted returns
- Win rate of {win_rate:.1f}% is {'acceptable' if win_rate > 45 else 'below optimal levels'}
- Maximum drawdown of {max_dd:.1f}% indicates {'high' if max_dd > 20 else 'acceptable'} risk levels

Recommendations:
{chr(10).join(f"- {rec}" for rec in recommendations)}
"""
        
        return StrategyAnalysis(
            strategy_name=strategy_name,
            current_performance=performance_metrics,
            suggested_parameters=suggested_parameters,
            optimization_reasoning=optimization_reasoning.strip(),
            market_conditions=market_conditions,
            risk_assessment=risk_assessment,
            improvement_potential=min(improvement_potential, 30.0),
            confidence_score=65.0
        )
    
    def _create_fallback_analysis(
        self,
        strategy_name: str,
        performance_metrics: Dict[str, float],
        current_parameters: Dict[str, Any]
    ) -> StrategyAnalysis:
        """Create a basic analysis when all methods fail"""
        
        return StrategyAnalysis(
            strategy_name=strategy_name,
            current_performance=performance_metrics,
            suggested_parameters=current_parameters,
            optimization_reasoning="Unable to generate detailed analysis. Please check configuration.",
            market_conditions=["Analysis unavailable"],
            risk_assessment="Risk assessment unavailable",
            improvement_potential=0.0,
            confidence_score=0.0
        )
    
    def generate_optimization_report(
        self,
        analyses: List[StrategyAnalysis],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive optimization report for multiple strategies
        
        Args:
            analyses: List of strategy analyses
            output_path: Optional path to save the report
            
        Returns:
            Formatted report as string
        """
        report = f"""
# Trading Strategy Optimization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Provider: {self.active_provider or 'Heuristic'} 

## Executive Summary

This report analyzes {len(analyses)} trading strategies and provides {'AI-powered' if self.active_provider else 'heuristic-based'} 
recommendations for parameter optimization to improve performance.

### Overall Performance Summary

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Improvement Potential |
|----------|-------------|--------------|--------------|----------|---------------------|
"""
        
        for analysis in analyses:
            report += f"| {analysis.strategy_name} | "
            report += f"{analysis.current_performance.get('total_return', 0):.2f}% | "
            report += f"{analysis.current_performance.get('sharpe_ratio', 0):.2f} | "
            report += f"{analysis.current_performance.get('max_drawdown', 0):.2f}% | "
            report += f"{analysis.current_performance.get('win_rate', 0):.1f}% | "
            report += f"{analysis.improvement_potential:.1f}% |\n"
        
        report += "\n---\n"
        
        # Detailed analysis for each strategy
        for analysis in analyses:
            report += f"""
## {analysis.strategy_name} Strategy

### Current Performance
- **Total Return**: {analysis.current_performance.get('total_return', 0):.2f}%
- **Sharpe Ratio**: {analysis.current_performance.get('sharpe_ratio', 0):.2f}
- **Max Drawdown**: {analysis.current_performance.get('max_drawdown', 0):.2f}%
- **Win Rate**: {analysis.current_performance.get('win_rate', 0):.2f}%
- **Profit Factor**: {analysis.current_performance.get('profit_factor', 0):.2f}
- **Total Trades**: {analysis.current_performance.get('total_trades', 0)}

### Optimization Recommendations

{analysis.optimization_reasoning}

### Suggested Parameter Adjustments

```json
{json.dumps(analysis.suggested_parameters, indent=2)}
```

### Optimal Market Conditions
{chr(10).join(f"- {condition}" for condition in analysis.market_conditions)}

### Risk Assessment
{analysis.risk_assessment}

### Performance Improvement Potential
- **Estimated Improvement**: {analysis.improvement_potential:.1f}%
- **Confidence Score**: {analysis.confidence_score:.1f}%

---
"""
        
        # Add footer with disclaimer
        report += f"""
## Disclaimer

This analysis is based on historical data and {'AI-generated insights' if self.active_provider else 'heuristic rules'}. 
Past performance does not guarantee future results. Always validate recommendations through 
thorough backtesting before implementing in live trading.

Analysis confidence scores indicate the reliability of the recommendations:
- 80-100%: High confidence (AI-based analysis with good data)
- 60-80%: Moderate confidence (Heuristic analysis or limited data)  
- Below 60%: Low confidence (Insufficient data or analysis failure)
"""
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Optimization report saved to {output_path}")
                
        return report
    
    def compare_providers(
        self,
        strategy_name: str,
        performance_metrics: Dict[str, float],
        current_parameters: Dict[str, Any],
        optimization_ranges: Dict[str, Dict],
        trade_history: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, StrategyAnalysis]:
        """
        Compare analysis results from different LLM providers
        
        Useful for validating recommendations across different AI models
        """
        results = {}
        
        # Try Gemini
        try:
            gemini_analyzer = LLMAnalyzer(provider="gemini")
            results['gemini'] = gemini_analyzer.analyze_strategy_performance(
                strategy_name, performance_metrics, current_parameters,
                optimization_ranges, trade_history, market_data
            )
        except Exception as e:
            logger.warning(f"Gemini analysis failed: {e}")
            
        # Try OpenAI
        try:
            openai_analyzer = LLMAnalyzer(provider="openai")
            results['openai'] = openai_analyzer.analyze_strategy_performance(
                strategy_name, performance_metrics, current_parameters,
                optimization_ranges, trade_history, market_data
            )
        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {e}")
            
        # Always include heuristic
        results['heuristic'] = self._create_heuristic_analysis(
            strategy_name, performance_metrics, current_parameters,
            optimization_ranges, trade_history, market_data
        )
        
        return results
