#!/usr/bin/env python3
"""
Batch backtest runner for Ichimoku strategies.

Runs strategies 01–15 for BTC/USDT on 4h timeframe starting from 2023-01-01.
Generates per-strategy PDF reports and a CSV summary.
"""
import os
import sys
import csv
import logging
from datetime import datetime

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.insert(0, project_root_dir)
from backtesting.ichimoku_backtester import IchimokuBacktester, StrategyBacktestRunner


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Config
    symbol_short = 'BTC'
    timeframe = '4h'
    start_date = '2023-01-01'
    initial_capital = 10000.0
    output_dir = 'reports'  # ensure it matches your preferred reports folder
    report_formats = 'pdf'

    os.makedirs(output_dir, exist_ok=True)

    # Strategy keys 01..15 (skip 16 as requested)
    strategy_keys = [
        'strategy_01_tk_sell',
        'strategy_02_tk_sell',
        'strategy_03_tk_sell',
        'strategy_04_tk_sell',
        'strategy_05_tk_sell',
        'strategy_06_span_sell',
        'strategy_07_span_sell',
        'strategy_08_span_sell',
        'strategy_09_span_sell',
        'strategy_10_span_sell',
        'strategy_11_tk_span_sell',
        'strategy_12_tk_span_sell',
        'strategy_13_tk_span_sell',
        'strategy_14_tk_span_sell',
        'strategy_15_tk_span_sell',
    ]

    backtester = IchimokuBacktester(
        commission_rate=0.001,
        slippage_rate=0.0003,
        pyramiding=1,
    )
    runner = StrategyBacktestRunner(backtester)

    # CSV summary
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = os.path.join(output_dir, f'batch_summary_{symbol_short}_{timeframe}_{start_date}_{ts}.csv')
    fieldnames = [
        'strategy_key', 'strategy_name', 'symbol', 'timeframe', 'start',
        'total_trades', 'win_rate_pct', 'profit_factor', 'total_return_pct',
        'max_drawdown_pct', 'net_profit', 'pdf'
    ]
    rows = []

    for key in strategy_keys:
        logging.info(f'Running {key} ...')
        try:
            result_bundle = runner.run_from_json(
                strategy_key=key,
                symbol_short=symbol_short,
                timeframe=timeframe,
                start=start_date,
                end=None,
                initial_capital=initial_capital,
                report_formats=report_formats,
                output_dir=output_dir,
                with_llm_optimization=False,
            )
            res = result_bundle.get('result')
            strat_cfg = result_bundle.get('strategy_config', {})
            reports = result_bundle.get('reports', {}) or {}
            pdf_path = reports.get('pdf') if isinstance(reports, dict) else None

            metrics = res.metrics if hasattr(res, 'metrics') else {}
            rows.append({
                'strategy_key': key,
                'strategy_name': strat_cfg.get('name', ''),
                'symbol': f'{symbol_short}/USDT',
                'timeframe': timeframe,
                'start': start_date,
                'total_trades': metrics.get('total_trades', 0),
                'win_rate_pct': round(metrics.get('win_rate_pct', 0), 2),
                'profit_factor': metrics.get('profit_factor', 0),
                'total_return_pct': round(metrics.get('total_return_pct', 0), 2),
                'max_drawdown_pct': round(metrics.get('max_drawdown_pct', 0), 2),
                'net_profit': round(metrics.get('net_profit', 0), 2),
                'pdf': pdf_path or '',
            })
        except Exception as e:
            logging.exception(f'Failed {key}: {e}')
            rows.append({
                'strategy_key': key,
                'strategy_name': '',
                'symbol': f'{symbol_short}/USDT',
                'timeframe': timeframe,
                'start': start_date,
                'total_trades': 0,
                'win_rate_pct': 0,
                'profit_factor': 0,
                'total_return_pct': 0,
                'max_drawdown_pct': 0,
                'net_profit': 0,
                'pdf': '',
            })
            continue

    # Write CSV
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f'Batch complete. Summary: {summary_path}')


if __name__ == '__main__':
    sys.exit(main())
