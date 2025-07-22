#!/usr/bin/env python3
"""
Entry point script to collect historical data.
This script allows running the data collection functionality from the root directory.
"""
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import and run the collect historical data module
from collect_historical_data import collect_all_historical_data_for_all_pairs

if __name__ == '__main__':
    collect_all_historical_data_for_all_pairs()
