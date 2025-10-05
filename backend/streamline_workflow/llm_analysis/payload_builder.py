import json
import math
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime


def _filter_by_time(df: pd.DataFrame, start: Optional[str], end: Optional[str], time_cols=('entry_time','exit_time')) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    sdf = df.copy()
    # Try to handle both index-based and column-based timestamps
    if 'timestamp' in sdf.columns:
        idx = pd.to_datetime(sdf['timestamp'])
    elif isinstance(sdf.index, pd.DatetimeIndex):
        idx = sdf.index
    else:
        # fall back to entry_time for trades
        if time_cols[0] in sdf.columns:
            idx = pd.to_datetime(sdf[time_cols[0]])
        else:
            return sdf
    mask = pd.Series(True, index=sdf.index)
    if start:
        mask &= (idx >= pd.to_datetime(start))
    if end:
        mask &= (idx <= pd.to_datetime(end))
    return sdf.loc[mask]


def _top_k(df: pd.DataFrame, by: str, k: int, ascending=False, cols=None) -> list:
    if df is None or df.empty or by not in df.columns:
        return []
    sub = df.sort_values(by=by, ascending=ascending).head(k)
    if cols:
        sub = sub[[c for c in cols if c in sub.columns]]
    recs = sub.to_dict(orient='records')
    # Convert any timestamps to ISO strings
    for r in recs:
        for key in ("entry_time", "exit_time", "timestamp"):
            if key in r and r[key] is not None:
                try:
                    r[key] = pd.to_datetime(r[key]).isoformat()
                except Exception:
                    r[key] = str(r[key])
    return recs


def _histogram(series: pd.Series, bins: int = 11) -> Dict[str, Any]:
    if series is None or len(series) == 0:
        return {"bins": [], "counts": []}
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(s.values, bins=bins)
    edges = [float(x) for x in edges.tolist()]
    counts = [int(x) for x in counts.tolist()]
    return {"bins": edges, "counts": counts}


def _to_jsonable(obj: Union[Dict[str, Any], list, tuple, np.generic, pd.Timestamp, datetime, Any]):
    """Recursively convert objects to JSON-serializable Python primitives."""
    # Pandas Timestamp or numpy datetime
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        # numpy scalar to native python type
        val = obj.item()
        if isinstance(val, float) and (np.isnan(val) or val == float('inf') or val == float('-inf')):
            return None
        return val
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    # Normalize plain Python floats for NaN/Inf
    if isinstance(obj, float):
        if math.isnan(obj) or obj == float('inf') or obj == float('-inf'):
            return None
    # Fallback for other types that json can't handle
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def build_llm_payload(
    *,
    result_metrics: Dict[str, Any],
    trades_df: Optional[pd.DataFrame],
    equity_df: Optional[pd.DataFrame],
    strategy_config: Dict[str, Any],
    analysis_start: Optional[str] = None,
    analysis_end: Optional[str] = None,
    budget: str = "standard"
) -> Dict[str, Any]:
    """Build a compact, token-efficient payload for the LLM.

    This function intentionally avoids dumping raw series. It aggregates and trims.
    """
    # Filter by analysis window
    tdf = _filter_by_time(trades_df, analysis_start, analysis_end)
    edf = _filter_by_time(equity_df, analysis_start, analysis_end, time_cols=("timestamp","timestamp"))

    # Metrics summary: pick a concise subset and rename to conventional forms
    m = result_metrics or {}
    metrics_summary = {
        "win_rate": float(m.get("win_rate_pct", m.get("win_rate", 0))) / (100.0 if m.get("win_rate_pct") is not None else 1.0),
        "profit_factor": float(m.get("profit_factor", 0) or 0),
        "sharpe_ratio": float(m.get("sharpe_ratio", 0) or 0),
        "max_drawdown_pct": float(m.get("max_drawdown_pct", m.get("max_drawdown", 0)) or 0),
        "total_trades": int(m.get("total_trades", 0) or 0),
        "avg_win": float(m.get("avg_winning_trade", 0) or 0),
        "avg_loss": float(abs(m.get("avg_losing_trade", 0) or 0)),
        "largest_win": float(m.get("largest_win", 0) or 0),
        "largest_loss": float(m.get("largest_loss", 0) or 0),
        "net_profit": float(m.get("net_profit", 0) or 0),
        "total_commission": float(m.get("total_commission", 0) or 0),
        "total_slippage": float(m.get("total_slippage", 0) or 0),
    }

    # Trades aggregates
    if tdf is not None and not tdf.empty:
        # Ensure numeric columns are numeric
        for col in ("pnl", "return_pct", "bars_held"):
            if col in tdf.columns:
                tdf[col] = pd.to_numeric(tdf[col], errors='coerce')
        top_wins = _top_k(tdf, by="pnl", k=10, ascending=False,
                          cols=["pnl","return_pct","bars_held","entry_time","exit_time","exit_reason"])
        top_losses = _top_k(tdf, by="pnl", k=10, ascending=True,
                            cols=["pnl","return_pct","bars_held","entry_time","exit_time","exit_reason"])
        pnl_hist = _histogram(tdf.get("pnl"), bins=11) if "pnl" in tdf.columns else {"bins":[],"counts":[]}
        dur_hist = _histogram(tdf.get("bars_held"), bins=11) if "bars_held" in tdf.columns else {"bins":[],"counts":[]}
        # Day-of-week pnl totals
        dow = []
        if "entry_time" in tdf.columns and "pnl" in tdf.columns:
            dd = tdf.copy()
            dd["entry_time"] = pd.to_datetime(dd["entry_time"], errors='coerce')
            dd = dd.dropna(subset=["entry_time"])  # keep only valid times
            if not dd.empty:
                dd["dow"] = dd["entry_time"].dt.dayofweek
                g = dd.groupby("dow")["pnl"].sum()
                dow = [float(g.get(i, 0.0)) for i in range(7)]
    else:
        top_wins, top_losses, pnl_hist, dur_hist, dow = [], [], {"bins":[],"counts":[]}, {"bins":[],"counts":[]}, []

    # Equity summary from edf
    equity_summary = {}
    if edf is not None and not edf.empty:
        if "drawdown" in edf.columns:
            equity_summary["avg_drawdown"] = float(pd.to_numeric(edf["drawdown"], errors='coerce').mean() or 0)
            equity_summary["max_drawdown"] = float(pd.to_numeric(edf["drawdown"], errors='coerce').min() or 0)
        if "returns" in edf.columns:
            rv = pd.to_numeric(edf["returns"], errors='coerce')
            equity_summary["rolling_vol_mean"] = float((rv.rolling(30).std() * np.sqrt(252)).mean() or 0)
        # drawdown durations (very compact): count stretches below 0
        ddur = 0
        current = 0
        durations = []
        if "drawdown" in edf.columns:
            for val in edf["drawdown"].fillna(0).values:
                if val < 0:
                    current += 1
                elif current > 0:
                    durations.append(current)
                    current = 0
            if current > 0:
                durations.append(current)
        if durations:
            equity_summary["drawdown_duration_max"] = int(max(durations))
            equity_summary["drawdown_duration_median"] = float(np.median(durations))

    # Strategy summary (only necessary fields)
    sconf = strategy_config or {}
    strategy_summary = {
        "name": sconf.get("name"),
        "timeframe": sconf.get("timeframes", sconf.get("timeframe")),
        "symbols": sconf.get("symbols"),
        "ichimoku_parameters": sconf.get("ichimoku_parameters", {}),
        "signal_conditions": sconf.get("signal_conditions", {}),
        "position_sizing": sconf.get("position_sizing", {}),
        "risk_management": sconf.get("risk_management", {}),
    }

    payload = {
        "strategy_summary": strategy_summary,
        "metrics_summary": metrics_summary,
        "trades_summary": {
            "top_k_winners": top_wins,
            "top_k_losers": top_losses,
            "pnl_histogram": pnl_hist,
            "duration_histogram": dur_hist,
            "dow_pnl": dow,
        },
        "equity_summary": equity_summary,
        "time_scope": {"analysis_start": analysis_start, "analysis_end": analysis_end},
        "token_budget_hint": budget,
    }
    # Ensure JSON-serializable types
    payload = _to_jsonable(payload)
    return payload
