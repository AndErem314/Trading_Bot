"""
Analytics module for trading performance analysis.

This module provides comprehensive performance metrics calculation,
visualization, and reporting capabilities for trading strategies.
"""

from .performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceMetrics
)

__all__ = [
    'PerformanceAnalyzer',
    'PerformanceMetrics'
]