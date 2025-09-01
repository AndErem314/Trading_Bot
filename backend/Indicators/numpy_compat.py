"""
NumPy compatibility patch for pandas_ta with NumPy 2.x
"""

import numpy as np
import sys

# Patch numpy to add back NaN alias for compatibility
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Also add to numpy module in sys.modules
sys.modules['numpy'].NaN = np.nan
