"""
Parabolic SAR (PSAR) indicator computation utilities.

Returns PSAR value, trend (+1 up / -1 down), and reversal flag per bar.
Default parameters: step=0.02, max_step=0.2
"""
from typing import Tuple
import pandas as pd
import numpy as np


def compute_psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    """Compute Parabolic SAR for an OHLC DataFrame.

    Args:
        df: DataFrame with columns ['high','low','close'] indexed by timestamp
        step: acceleration factor increment
        max_step: maximum acceleration factor

    Returns:
        DataFrame with columns ['psar','psar_trend','psar_reversal'] aligned to df index
    """
    if not set(['high', 'low', 'close']).issubset(df.columns):
        raise ValueError("compute_psar requires 'high','low','close' columns")
    if len(df) == 0:
        return pd.DataFrame(index=df.index, columns=['psar', 'psar_trend', 'psar_reversal'])

    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)

    n = len(df)
    psar = np.full(n, np.nan, dtype=float)
    trend = np.full(n, 0, dtype=int)
    reversal = np.full(n, 0, dtype=int)

    # Initialization based on first two bars
    if n >= 2:
        uptrend = df['close'].iloc[1] >= df['close'].iloc[0]
    else:
        uptrend = True
    af = step
    ep = highs[0] if uptrend else lows[0]
    sar = lows[0] if uptrend else highs[0]

    psar[0] = sar
    trend[0] = 1 if uptrend else -1

    for i in range(1, n):
        prev = i - 1
        prev2 = i - 2 if i - 2 >= 0 else None

        # Calculate SAR for current bar based on previous state
        sar = sar + af * (ep - sar)
        if uptrend:
            # Enforce SAR not above prior two lows
            max_allowed = lows[prev]
            if prev2 is not None:
                max_allowed = min(max_allowed, lows[prev2])
            sar = min(sar, max_allowed)
            # Check for reversal
            if lows[i] < sar:
                uptrend = False
                reversal[i] = 1
                sar = ep  # On reversal, SAR is set to prior EP
                ep = lows[i]
                af = step
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + step, max_step)
        else:
            # Downtrend
            min_allowed = highs[prev]
            if prev2 is not None:
                min_allowed = max(min_allowed, highs[prev2])
            sar = max(sar, min_allowed)
            # Check for reversal
            if highs[i] > sar:
                uptrend = True
                reversal[i] = 1
                sar = ep
                ep = highs[i]
                af = step
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + step, max_step)

        psar[i] = sar
        trend[i] = 1 if uptrend else -1

    out = pd.DataFrame({
        'psar': psar,
        'psar_trend': trend,
        'psar_reversal': reversal.astype(int)
    }, index=df.index)
    return out
