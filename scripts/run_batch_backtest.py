#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.ichimoku_backtester import IchimokuBacktester, StrategyBacktestRunner  # noqa: E402


def load_strategy_keys(yaml_path: Path) -> list[str]:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    strategies = data.get("strategies", {}) or {}
    return list(strategies.keys())


essential_cols = [
    "strategy_key",
    "name",
    "total_trades",
    "win_rate_pct",
    "profit_factor",
    "max_drawdown_pct",
    "total_return_pct",
    "net_profit",
]


def main():
    # Batch parameters per request
    symbol_short = "BTC"           # BTC/USDT
    timeframe = "4h"               # 4-hour timeframe
    start_date = "2023-01-01"      # starting from

    yaml_path = ROOT / "config" / "strategies.yaml"
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    strategy_keys = load_strategy_keys(yaml_path)
    if not strategy_keys:
        print(f"No strategies found in {yaml_path}")
        return

    backtester = IchimokuBacktester()
    runner = StrategyBacktestRunner(backtester)

    rows: list[dict] = []

    print(f"Running batch backtest for {symbol_short}/USDT {timeframe} from {start_date}")
    print(f"Strategies: {len(strategy_keys)} found\n")

    # Recompute Ichimoku with each strategy's parameters for fair comparison
    force_recompute_ichimoku = True

    for key in strategy_keys:
        try:
            outcome = runner.run_from_json(
                strategy_key=key,
                symbol_short=symbol_short,
                timeframe=timeframe,
                start=start_date,
                end=None,
                initial_capital=10000.0,
                report_formats="pdf",
                output_dir=str(reports_dir),
                force_recompute_ichimoku=force_recompute_ichimoku,
            )
            metrics = outcome["result"].metrics or {}
            sc = outcome.get("strategy_config", {}) or {}

            rows.append(
                {
                    "strategy_key": key,
                    "name": sc.get("name", key),
                    "total_trades": metrics.get("total_trades"),
                    "win_rate_pct": metrics.get("win_rate_pct"),
                    "profit_factor": metrics.get("profit_factor"),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                    "total_return_pct": metrics.get("total_return_pct"),
                    "net_profit": metrics.get("net_profit"),
                }
            )

            pf = metrics.get("profit_factor")
            tr = metrics.get("total_return_pct")
            tt = metrics.get("total_trades")
            print(f"✓ {key:>20} — trades={tt} return={tr:.2f}% PF={(pf if pd.notna(pf) else float('nan')):.2f}")
        except Exception as e:
            print(f"✗ {key:>20} — error: {e}")
            rows.append({"strategy_key": key, "name": None, "error": str(e)})

    # Write batch summary CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"batch_summary_{symbol_short}_{timeframe}_{start_date}_{ts}.csv"
    df = pd.DataFrame(rows)

    # Ensure consistent column order
    cols = [c for c in essential_cols if c in df.columns] + [c for c in df.columns if c not in essential_cols]
    df = df.reindex(columns=cols)

    out_path = reports_dir / csv_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved batch summary to {out_path}")


if __name__ == "__main__":
    main()
