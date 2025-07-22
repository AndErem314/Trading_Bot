#!/usr/bin/env python3
"""
Entry point script to run the Trading Bot from the root directory.
This script allows running the main backend functionality while keeping files organized.
"""
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import and run the main module
from main import main

if __name__ == '__main__':
    main()
