#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import sys

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting.ichimoku_backtester import IchimokuBacktester, StrategyBacktestRunner


def main():
    symbol_short = "BTC"
    timeframe = "4h"
    start_date = "2020-01-01"

    # Only these two strategies
    strategy_keys = ["strategy_01", "strategy_02"]

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    backtester = IchimokuBacktester()
    runner = StrategyBacktestRunner(backtester)

    rows: list[dict] = []

    print(f"Comparing strategies for {symbol_short}/USDT {timeframe} from {start_date}")
    print(f"Strategies: {', '.join(strategy_keys)}\n")

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
                force_recompute_ichimoku=True,
            )
            metrics = outcome["result"].metrics or {}
            sc = outcome.get("strategy_config", {}) or {}
            pf = metrics.get("profit_factor")
            tr = metrics.get("total_return_pct")
            tt = metrics.get("total_trades")
            print(f"✓ {key:>12} — trades={tt} return={(tr if tr is not None else float('nan')):.2f}% PF={(pf if pf not in (None, float('inf')) else float('nan')):.2f}")

            rows.append(
                {
                    "strategy_key": key,
                    "name": sc.get("name", key),
                    "total_trades": tt,
                    "win_rate_pct": metrics.get("win_rate_pct"),
                    "profit_factor": pf,
                    "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                    "total_return_pct": tr,
                    "net_profit": metrics.get("net_profit"),
                }
            )
        except Exception as e:
            print(f"✗ {key:>12} — error: {e}")
            rows.append({"strategy_key": key, "name": None, "error": str(e)})

    # Write comparison CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"compare_{symbol_short}_{timeframe}_{start_date}_{ts}.csv"
    df = pd.DataFrame(rows)
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
    cols = [c for c in essential_cols if c in df.columns] + [c for c in df.columns if c not in essential_cols]
    df = df.reindex(columns=cols)
    out_path = reports_dir / csv_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved comparison CSV to {out_path}")


if __name__ == "__main__":
    main()
