"""
Crypto Market Regime Detection Module

This module provides a MarketRegimeDetector class specialized for cryptocurrency
markets (e.g., BTC, ETH, SOL). It classifies market states into:
- Bull Trend
- Bear Trend
- Ranging / Accumulation
- High Volatility / Breakout
- Crypto Crash / Panic

It also optionally evaluates Relative Strength vs a benchmark (e.g., BTC) and
can visualize regimes for backtests.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd

# NumPy 2.x compatibility for pandas_ta
try:
    from . import numpy_compat
except ImportError:
    pass

import pandas_ta as ta


class MarketRegimeDetector:
    """
    Crypto-specific market regime detector using pandas_ta indicators.

    Regimes:
        - Bull Trend: Strong, sustained upward momentum
        - Bear Trend: Strong, sustained downward momentum
        - Ranging / Accumulation: Low volatility, sideways price
        - High Volatility / Breakout: Volatility expansion periods
        - Crypto Crash / Panic: Extreme downward momentum with panic selling

    Enhanced features:
        - Volatility z-scores for ATR(14) and Volume(14) vs 100-period stats
        - Crash detection with SMA200, RSI(14), ROC(7)
        - Relative strength vs benchmark (ratio-based ADX & SMA logic)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame for the target asset with a DatetimeIndex.
    asset_name : str, optional
        Symbol identifier (e.g., 'BTC', 'ETH', 'SOL'). Used for DB checks and
        parameter tuning context. Default is 'BTC'.
    benchmark_df : pd.DataFrame, optional
        Benchmark OHLCV DataFrame (e.g., BTC) to compute relative strength.
    price_column : str, optional
        Column to use for price-based indicators. Default is 'close'.
    """

    # Minimum lookback requirements
    _ADX_LEN = 14
    _ATR_LEN = 14
    _RSI_LEN = 14
    _ROC_LEN = 7
    _SMA_SHORT = 50
    _SMA_LONG = 200
    _Z_ROLL = 100

    def __init__(
        self,
        df: pd.DataFrame,
        asset_name: str = 'BTC',
        benchmark_df: Optional[pd.DataFrame] = None,
        price_column: str = 'close',
    ):
        # Basic validations
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("df cannot be empty")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex")

        required_cols = ['high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in df: {missing_cols}")
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found in df")

        # Minimum data length: ensure SMA200 and z-scores compute
        min_periods = max(self._SMA_LONG, self._Z_ROLL) + self._ADX_LEN
        if len(df) < min_periods:
            raise ValueError(
                f"Insufficient data: need at least {min_periods} rows, got {len(df)}"
            )

        self.asset_name = str(asset_name).upper()
        self.price_column = price_column

        # Ensure sorted by index
        self.df = df.sort_index().copy()

        # Optional benchmark
        self.benchmark_df = None
        if benchmark_df is not None:
            if not isinstance(benchmark_df, pd.DataFrame):
                raise ValueError("benchmark_df must be a pandas DataFrame if provided")
            if not isinstance(benchmark_df.index, pd.DatetimeIndex):
                raise ValueError("benchmark_df must have a DatetimeIndex")
            for c in ['high', 'low', 'close']:
                if c not in benchmark_df.columns:
                    raise ValueError(f"benchmark_df missing required column '{c}'")
            self.benchmark_df = benchmark_df.sort_index().copy()

        # Precompute indicators for efficiency and plotting
        self.indicators_df = self._calculate_indicators(self.df)

        # Precompute relative strength series if benchmark provided
        self._rs_df: Optional[pd.DataFrame] = None
        if self.benchmark_df is not None:
            self._rs_df = self._calculate_relative_strength_df(
                self.indicators_df, self.benchmark_df
            )

    # -------------------------- Indicator Calculations -------------------------
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and append all required indicators for classification.

        Returns
        -------
        pd.DataFrame
            Copy of input with indicator columns appended.
        """
        df = data.copy()

        # Trend indicators
        df.ta.adx(high='high', low='low', close='close', length=self._ADX_LEN, append=True)
        df['sma50'] = df.ta.sma(close=df[self.price_column], length=self._SMA_SHORT)
        df['sma200'] = df.ta.sma(close=df[self.price_column], length=self._SMA_LONG)

        # Volatility
        df['atr'] = df.ta.atr(high='high', low='low', close='close', length=self._ATR_LEN)

        # Momentum / Oversold
        df['rsi14'] = df.ta.rsi(close=df[self.price_column], length=self._RSI_LEN)
        df['roc7'] = df.ta.roc(close=df[self.price_column], length=self._ROC_LEN)

        # Z-scores (use min_periods to handle NaNs gracefully)
        # ATR Z-score vs 100-period mean/std
        atr_mean_100 = df['atr'].rolling(self._Z_ROLL, min_periods=50).mean()
        atr_std_100 = df['atr'].rolling(self._Z_ROLL, min_periods=50).std(ddof=0)
        df['atr_z'] = (df['atr'] - atr_mean_100) / atr_std_100.replace(0, np.nan)

        # Volume Z-score vs 100-period mean/std (use raw 'volume')
        vol_mean_100 = df['volume'].rolling(self._Z_ROLL, min_periods=50).mean()
        vol_std_100 = df['volume'].rolling(self._Z_ROLL, min_periods=50).std(ddof=0)
        df['vol_z'] = (df['volume'] - vol_mean_100) / vol_std_100.replace(0, np.nan)

        # Volatility regime flag
        df['volatility_high'] = (df['atr_z'] > 2.5) | (df['vol_z'] > 3.0)

        # Crash / Panic conditions
        df['crash_flag'] = (
            (df[self.price_column] < df['sma200'])
            & (df['rsi14'] < 25)
            & (df['roc7'] <= -20)
        )

        # Prepare regime classification for the entire series (vectorized)
        # Trend strength and direction
        adx = df[f'ADX_{self._ADX_LEN}']
        plus_di = df[f'DMP_{self._ADX_LEN}']
        minus_di = df[f'DMN_{self._ADX_LEN}']

        strong_bull = (adx > 30) & (plus_di > minus_di)
        strong_bear = (adx > 30) & (minus_di > plus_di)
        ranging = (adx < 20) & (~df['volatility_high'])
        breakout = (adx < 25) & (df['volatility_high'])

        regime = np.where(df['crash_flag'], 'Crypto Crash / Panic',
                  np.where(strong_bull, 'Bull Trend',
                  np.where(strong_bear, 'Bear Trend',
                  np.where(ranging, 'Ranging / Accumulation',
                  np.where(breakout, 'High Volatility / Breakout', 'Transitioning / Unknown')))))
        df['regime'] = pd.Series(regime, index=df.index)

        return df

    def _calculate_relative_strength_df(
        self, asset_ind_df: pd.DataFrame, benchmark_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute relative strength series (asset vs benchmark) and indicators using
        ratio OHLC to allow ADX calculations.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by the intersection of both, including:
            - ratio_high, ratio_low, ratio_close
            - RS ADX/DI: RS_ADX_14, RS_DMP_14, RS_DMN_14
            - RS SMA200 on ratio_close
        """
        # Align on common index
        a = asset_ind_df[['high', 'low', 'close']].copy()
        b = benchmark_df[['high', 'low', 'close']].copy()
        both_idx = a.index.intersection(b.index)
        a = a.loc[both_idx]
        b = b.loc[both_idx]

        rs = pd.DataFrame(index=both_idx)
        rs['ratio_high'] = a['high'] / b['high']
        rs['ratio_low'] = a['low'] / b['low']
        rs['ratio_close'] = a['close'] / b['close']

        # ADX on ratio OHLC (directional strength of out/under performance)
        rs.ta.adx(high='ratio_high', low='ratio_low', close='ratio_close', length=self._ADX_LEN, append=True)
        rs['rs_sma200'] = rs.ta.sma(close=rs['ratio_close'], length=self._SMA_LONG)

        return rs

    # ----------------------------- Classification -----------------------------
    def classify_regime(self) -> Tuple[str, Dict[str, float]]:
        """
        Classify the current (most recent) market regime and return metrics.

        Returns
        -------
        tuple[str, dict]
            (primary_regime, metrics_dict)
        """
        latest = self.indicators_df.iloc[-1]

        # Validate key indicator availability
        needed = [
            f'ADX_{self._ADX_LEN}', f'DMP_{self._ADX_LEN}', f'DMN_{self._ADX_LEN}',
            'sma200', 'atr', 'atr_z', 'vol_z', 'rsi14', 'roc7', 'volatility_high', 'crash_flag'
        ]
        for col in needed:
            if pd.isna(latest.get(col, np.nan)):
                raise ValueError(f"Insufficient data to calculate indicator '{col}' at the last index")

        adx = float(latest[f'ADX_{self._ADX_LEN}'])
        plus_di = float(latest[f'DMP_{self._ADX_LEN}'])
        minus_di = float(latest[f'DMN_{self._ADX_LEN}'])
        sma200 = float(latest['sma200'])
        price = float(latest[self.price_column])
        atr = float(latest['atr'])
        atr_z = float(latest['atr_z']) if not pd.isna(latest['atr_z']) else 0.0
        vol_z = float(latest['vol_z']) if not pd.isna(latest['vol_z']) else 0.0
        rsi = float(latest['rsi14'])
        roc7 = float(latest['roc7'])
        vol_high = bool(latest['volatility_high'])
        crash_flag = bool(latest['crash_flag'])

        # Primary regime classification per rules (crash has precedence)
        if crash_flag:
            primary_regime = 'Crypto Crash / Panic'
        elif adx > 30 and plus_di > minus_di:
            primary_regime = 'Bull Trend'
        elif adx > 30 and minus_di > plus_di:
            primary_regime = 'Bear Trend'
        elif adx < 20 and not vol_high:
            primary_regime = 'Ranging / Accumulation'
        elif adx < 25 and vol_high:
            primary_regime = 'High Volatility / Breakout'
        else:
            primary_regime = 'Transitioning / Unknown'

        # Relative strength vs benchmark (if available)
        relative_strength = 'Unavailable'
        if self._rs_df is not None and not self._rs_df.empty:
            rs_latest = self._rs_df.iloc[-1]
            rs_adx = rs_latest.get(f'ADX_{self._ADX_LEN}', np.nan)
            rs_pdi = rs_latest.get(f'DMP_{self._ADX_LEN}', np.nan)
            rs_mdi = rs_latest.get(f'DMN_{self._ADX_LEN}', np.nan)
            rs_close = rs_latest.get('ratio_close', np.nan)
            rs_sma200 = rs_latest.get('rs_sma200', np.nan)

            if not any(pd.isna([rs_adx, rs_pdi, rs_mdi, rs_close, rs_sma200])):
                if rs_adx > 20 and rs_pdi > rs_mdi and rs_close > rs_sma200:
                    relative_strength = 'Outperforming'
                elif rs_adx > 20 and rs_mdi > rs_pdi and rs_close < rs_sma200:
                    relative_strength = 'Underperforming'
                else:
                    relative_strength = 'Neutral'

        metrics: Dict[str, float] = {
            'primary_regime': primary_regime,
            'asset': self.asset_name,
            'price': round(price, 8),
            'adx': round(adx, 4),
            'plus_di': round(plus_di, 4),
            'minus_di': round(minus_di, 4),
            'sma200': round(sma200, 8),
            'atr': round(atr, 8),
            'atr_zscore': round(atr_z, 4),
            'volume_zscore': round(vol_z, 4),
            'volatility_regime': 'High' if vol_high else 'Normal',
            'rsi': round(rsi, 4),
            'roc_7': round(roc7, 4),
            'crash_flag': crash_flag,
            'relative_strength': relative_strength,
        }

        return primary_regime, metrics

    # -------------------------- Visualization Utilities -----------------------
    def plot_regime(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot price action color-coded by detected regime for backtest analysis.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size for matplotlib.

        Notes
        -----
        This method computes the regime series across the entire dataset (already
        available in self.indicators_df['regime']) and draws colored line
        segments per regime.
        """
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError(
                "matplotlib is required for plot_regime(). Install it via 'pip install matplotlib'"
            ) from e

        df = self.indicators_df[[self.price_column, 'regime']].copy()

        # Color map for regimes
        colors = {
            'Bull Trend': '#2ca02c',
            'Bear Trend': '#d62728',
            'Ranging / Accumulation': '#7f7f7f',
            'High Volatility / Breakout': '#ff7f0e',
            'Crypto Crash / Panic': '#8c1b13',
            'Transitioning / Unknown': '#1f77b4',
        }

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{self.asset_name} Regime Chart")
        ax.set_xlabel("Date")
        ax.set_ylabel(self.price_column.capitalize())

        # Plot segments per regime by masking others with NaN
        for regime_name, color in colors.items():
            segment = df[self.price_column].where(df['regime'] == regime_name)
            ax.plot(segment.index, segment.values, color=color, label=regime_name)

        ax.legend(loc='best', ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    # ----------------------------- DB Connectivity ----------------------------
    def verify_db_connection(self, data_dir: str = '../../data', timeout: int = 5) -> bool:
        """
        Verify that the SQLite database for the given asset exists and is
        accessible. Expects files named like 'trading_data_BTC.db' in the data
        directory.

        Parameters
        ----------
        data_dir : str
            Relative or absolute path to the folder containing DB files.
        timeout : int
            SQLite connection timeout in seconds.

        Returns
        -------
        bool
            True if the DB exists and is readable, else False.
        """
        asset_key = self.asset_name.split('/')[0].upper()
        default_name = f"trading_data_{asset_key}.db"
        # Explicitly handle BTC, ETH, SOL naming
        db_file_map = {
            'BTC': 'trading_data_BTC.db',
            'ETH': 'trading_data_ETH.db',
            'SOL': 'trading_data_SOL.db',
        }
        db_name = db_file_map.get(asset_key, default_name)
        
        # Handle relative path from backend/Indicators to data folder
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # Go up to Trading_Bot
        db_path = project_root / 'data' / db_name

        if not db_path.exists():
            return False

        try:
            with sqlite3.connect(db_path.as_posix(), timeout=timeout) as conn:
                # Minimal query to ensure readability
                _ = conn.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;").fetchone()
            return True
        except Exception:
            return False


__all__ = ["MarketRegimeDetector"]
