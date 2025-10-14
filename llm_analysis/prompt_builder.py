import json
from typing import Dict, Any

_ANALYST_TMPL = """
System:
You are a senior quantitative researcher specialized in Ichimoku-based crypto trading strategies. Your task is to recommend concrete, minimal, high-impact adjustments to strategy settings to improve risk-adjusted performance. Focus on Ichimoku parameters, signal logic, and risk management. You must not describe backtest methodology; only deliver optimization guidance.

Constraints:
- Base your analysis strictly on the provided payload (aggregated metrics, compact trade summaries, and current settings).
- Assume the backtest coverage and data quality are correct.
- The output must be compact and directly actionable.

User:
Here is the compact backtest summary:
{payload_json}

Required output (two parts):
1) JSON object ONLY with this shape (no extra fields):
{{
  "parameter_changes": {{
    "ichimoku": {{
      "tenkan_period": {{"current": <int>, "suggested": <int>, "rationale": "<1-2 sentences>"}},
      "kijun_period": {{"current": <int>, "suggested": <int>, "rationale": "<1-2 sentences>"}},
      "senkou_b_period": {{"current": <int>, "suggested": <int>, "rationale": "<1-2 sentences>"}},
      "chikou_offset": {{"current": <int>, "suggested": <int>, "rationale": "<1-2 sentences>"}},
      "senkou_offset": {{"current": <int>, "suggested": <int>, "rationale": "<1-2 sentences>"}}
    }},
    "signal_logic": {{
      "buy_logic": {{"current": "AND|OR", "suggested": "AND|OR", "rationale": "<1-2 sentences>"}},
      "sell_logic": {{"current": "AND|OR", "suggested": "AND|OR", "rationale": "<1-2 sentences>"}},
      "add_conditions": ["SignalType", ...],
      "remove_conditions": ["SignalType", ...]
    }},
    "risk_management": {{
      "stop_loss_pct": {{"current": <float or null>, "suggested": <float>, "rationale": "<1-2 sentences>"}},
      "take_profit_pct": {{"current": <float or null>, "suggested": <float>, "rationale": "<1-2 sentences>"}},
      "position_sizing": {{"current": "fixed", "suggested": "fixed|volatility", "rationale": "<1-2 sentences>"}}
    }}
  }},
  "experiments": [
    {{"name": "Walk-forward grid", "description": "Brief", "params_to_sweep": {{"tenkan_period": [..], "kijun_period": [..]}}}},
    {{"name": "Trend filter A/B", "description": "Brief", "params_to_sweep": {{"add_conditions": ["ChikouAbovePrice"]}}}}
  ]
}}
2) A short PDF-ready memo (max ~350 words) titled: “Strategy Settings Optimization — Executive Summary”. Use bullet points. Do not restate metrics; focus on adjustments and rationale.

Output formatting rules:
- All suggested numeric values must be single numbers only (no ranges). Do not output ranges like [min,max].
- First print the JSON object (and nothing else).
- Then print the memo text after a line with exactly: ---MEMO---
"""

_RISK_TMPL = """
System:
You are a seasoned risk manager optimizing Ichimoku strategies for robust, drawdown-aware performance. Only propose changes to strategy settings (parameters, signals, risk). Minimize churn: prefer small, defensible adjustments.

User:
Backtest summary (compact):
{payload_json}

Guidelines:
- Use the distribution cues (win/loss percentiles, P&L hist bins, day-of-week totals) rather than raw trades.
- If win rate is low but profit factor is acceptable, aim to preserve edge while trimming tail losses (stop-loss/position sizing).
- If drawdowns or tail losses are heavy, prioritize risk controls over aggressiveness.
- If trade frequency is too low, consider slightly shorter Tenkan/Kijun; if whipsaw is high, consider longer.
- Only recommend changes you can justify using the provided aggregates.

Required output (two parts):
1) STRICT JSON (same schema as Version A), keeping suggested values tight (prefer single numbers or narrow ranges).
2) After the JSON, print a brief PDF-ready memo (<= 250 words) titled: “Risk-Focused Optimization”. Bulleted and concise.

Output formatting rules:
- All suggested numeric values must be single numbers only (no ranges). Do not output ranges like [min,max].
- First the JSON object.
- Then a line with exactly: ---MEMO---
- Then the memo text.
"""


def build_prompt(payload: Dict[str, Any], variant: str = "analyst") -> str:
    pj = json.dumps(payload, ensure_ascii=False)
    if variant == "risk" or variant == "risk_manager":
        return _RISK_TMPL.format(payload_json=pj)
    return _ANALYST_TMPL.format(payload_json=pj)
