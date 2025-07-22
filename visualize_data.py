#!/usr/bin/env python3
"""
Entry point script to run data visualization.
This script allows running the data visualization functionality from the root directory.
"""
import sys
import os

# Add the frontend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frontend'))

# Import and run the data visualizer module
from data_visualizer import main

if __name__ == '__main__':
    main()
