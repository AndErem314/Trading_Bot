#!/usr/bin/env python3
"""
Crypto Market Regime Detector Module

This module provides sophisticated market regime detection specifically designed for cryptocurrency markets.
It identifies various market states including bull/bear trends, ranging markets, high volatility periods,
and crypto-specific crash conditions. The detector uses a combination of traditional technical indicators
and crypto-specific metrics to classify market conditions.

Author: Andrey's Trading Bot
Date: 2025-08-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class CryptoMarketRegimeDetector:
    """
    Sophisticated market regime detector specifically designed for cryptocurrency markets.
    
    This class analyzes OHLCV data to classify the current market state into distinct regimes,
    accounting for the unique characteristics of crypto assets like extreme volatility,
    24/7 trading, and sentiment-driven moves.
    
    Attributes:
        df (pd.DataFrame): Price data for the target asset
        asset_name (str): Name of the asset being analyzed (e.g., 'BTC', 'ETH')
        benchmark_df (pd.DataFrame): Optional benchmark data for relative strength analysis
        price_column (str): Column name to use for price calculations
        
    Regimes:
        - Bull Trend: Strong, sustained upward momentum
        - Bear Trend: Strong, sustained downward momentum
        - Ranging / Accumulation: Low volatility, sideways price action
        - High Volatility / Breakout: Periods of rapidly expanding volatility
        - Crypto Crash / Panic: Extreme downward momentum with panic selling
    """
    
    # Crypto-specific parameters tuned for different assets
    ASSET_PARAMS = {
        'BTC': {
            'adx_threshold_trend': 30,
            'adx_threshold_range': 20,
            'volatility_zscore_threshold': 2.5,
            'volume_zscore_threshold': 3.0,
            'crash_rsi_threshold': 25,
            'crash_roc_threshold': -20,
            'sma_period': 200
        },
        'ETH': {
            'adx_threshold_trend': 28,
            'adx_threshold_range': 18,
            'volatility_zscore_threshold': 2.3,
            'volume_zscore_threshold': 2.8,
            'crash_rsi_threshold': 23,
            'crash_roc_threshold': -22,
            'sma_period': 200
        },
        'DEFAULT': {
            'adx_threshold_trend': 25,
            'adx_threshold_range': 20,
            'volatility_zscore_threshold': 2.0,
            'volume_zscore_threshold': 2.5,
            'crash_rsi_threshold': 20,
            'crash_roc_threshold': -25,
            'sma_period': 200
        }
    }
    
    def __init__(self, df: pd.DataFrame, asset_name: str = 'BTC', 
                 benchmark_df: Optional[pd.DataFrame] = None, 
                 price_column: str = 'close'):
        """
        Initialize the Crypto Market Regime Detector.
        
        Args:
            df: pandas DataFrame with OHLCV data and datetime index
            asset_name: String identifier for asset-specific parameter tuning
            benchmark_df: Optional DataFrame for benchmark (e.g., BTC) for relative strength
            price_column: Column name to use for price calculations (default: 'close')
        """
        self.df = df.copy()
        self.asset_name = asset_name.upper()
        self.benchmark_df = benchmark_df.copy() if benchmark_df is not None else None
        self.price_column = price_column
        
        # Get asset-specific parameters
        self.params = self.ASSET_PARAMS.get(self.asset_name, self.ASSET_PARAMS['DEFAULT'])
        
        # Calculate all indicators on initialization
        self._calculate_indicators()
        
    def _calculate_indicators(self) -> None:
        """Calculate all technical indicators and metrics needed for regime detection."""
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 1. ADX and Directional Indicators
        self._calculate_adx(14)
        
        # 2. Simple Moving Averages
        self.df['SMA_50'] = self.df[self.price_column].rolling(window=50).mean()
        self.df['SMA_200'] = self.df[self.price_column].rolling(window=self.params['sma_period']).mean()
        
        # 3. ATR (Average True Range)
        self._calculate_atr(14)
        
        # 4. RSI (Relative Strength Index)
        self._calculate_rsi(14)
        
        # 5. Rate of Change
        self.df['ROC_7'] = ((self.df[self.price_column] - self.df[self.price_column].shift(7)) / self.df[self.price_column].shift(7)) * 100
        
        # 6. Volume metrics
        self.df['Volume_SMA'] = self.df['volume'].rolling(window=20).mean()
        
        # 7. Z-Score calculations for volatility regime
        self._calculate_volatility_metrics()
        
        # 8. Relative strength if benchmark provided
        if self.benchmark_df is not None:
            self._calculate_relative_strength()
    
    def _calculate_rsi(self, period: int = 14) -> None:
        """Calculate RSI (Relative Strength Index)."""
        delta = self.df[self.price_column].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        self.df['RSI'] = 100 - (100 / (1 + rs))
        self.df['RSI'].fillna(50, inplace=True)  # Fill NaN with neutral value
    
    def _calculate_atr(self, period: int = 14) -> None:
        """Calculate ATR (Average True Range)."""
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift()).abs()
        low_close = (self.df['low'] - self.df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['ATR'] = true_range.rolling(window=period).mean()
    
    def _calculate_adx(self, period: int = 14) -> None:
        """Calculate ADX (Average Directional Index) and DI+/DI-."""
        # Calculate directional movement
        high_diff = self.df['high'].diff()
        low_diff = -self.df['low'].diff()
        
        # Positive and negative directional movement
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR for this calculation
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift()).abs()
        low_close = (self.df['low'] - self.df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        # Calculate DI+ and DI-
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di))
        adx = dx.rolling(window=period).mean()
        
        # Store results
        self.df['DI+'] = pos_di.fillna(0)
        self.df['DI-'] = neg_di.fillna(0)
        self.df['ADX'] = adx.fillna(25)  # Fill NaN with neutral value
    
    def _calculate_volatility_metrics(self) -> None:
        """Calculate volatility-based metrics including Z-scores."""
        
        # ATR Z-Score calculation
        atr_mean = self.df['ATR'].rolling(window=100, min_periods=50).mean()
        atr_std = self.df['ATR'].rolling(window=100, min_periods=50).std()
        self.df['ATR_ZScore'] = (self.df['ATR'] - atr_mean) / atr_std.replace(0, np.nan)
        
        # Volume Z-Score calculation
        volume_mean = self.df['volume'].rolling(window=100, min_periods=50).mean()
        volume_std = self.df['volume'].rolling(window=100, min_periods=50).std()
        self.df['Volume_ZScore'] = (self.df['volume'] - volume_mean) / volume_std.replace(0, np.nan)
        
        # Handle NaN values
        self.df['ATR_ZScore'].fillna(0, inplace=True)
        self.df['Volume_ZScore'].fillna(0, inplace=True)
        
    def _calculate_relative_strength(self) -> None:
        """Calculate relative strength metrics against benchmark."""
        
        if self.benchmark_df is None:
            return
        
        # Ensure benchmark has same index as main df
        benchmark_aligned = self.benchmark_df.reindex(self.df.index, method='ffill')
        
        # Calculate price ratio
        self.df['Price_Ratio'] = self.df[self.price_column] / benchmark_aligned[self.price_column]
        
        # Calculate simple relative strength indicators
        # Use rate of change on the ratio
        self.df['Ratio_ROC'] = ((self.df['Price_Ratio'] - self.df['Price_Ratio'].shift(14)) / self.df['Price_Ratio'].shift(14)) * 100
        
        # Simple directional indicators based on ratio movement
        ratio_diff = self.df['Price_Ratio'].diff()
        self.df['Ratio_DI+'] = ratio_diff.where(ratio_diff > 0, 0).rolling(window=14).mean() * 100
        self.df['Ratio_DI-'] = -ratio_diff.where(ratio_diff < 0, 0).rolling(window=14).mean() * 100
        
        # Simple ADX-like measure for ratio
        dx = 100 * (abs(self.df['Ratio_DI+'] - self.df['Ratio_DI-']) / (self.df['Ratio_DI+'] + self.df['Ratio_DI-'] + 0.0001))
        self.df['Ratio_ADX'] = dx.rolling(window=14).mean()
        
        # Calculate ratio SMA
        self.df['Ratio_SMA_50'] = self.df['Price_Ratio'].rolling(window=50).mean()
        
    def _detect_high_volatility(self) -> bool:
        """
        Detect if market is in high volatility regime.
        
        Returns:
            bool: True if high volatility conditions are met
        """
        latest = self.df.iloc[-1]
        
        # Check both ATR and Volume Z-scores
        atr_high = latest['ATR_ZScore'] > self.params['volatility_zscore_threshold']
        volume_high = latest['Volume_ZScore'] > self.params['volume_zscore_threshold']
        
        return atr_high or volume_high
    
    def _detect_crypto_crash(self) -> bool:
        """
        Detect if market is in crypto crash/panic regime.
        
        Returns:
            bool: True if crash conditions are met
        """
        latest = self.df.iloc[-1]
        
        # Check all crash conditions
        below_long_ma = latest[self.price_column] < latest['SMA_200']
        deeply_oversold = latest['RSI'] < self.params['crash_rsi_threshold']
        sharp_decline = latest['ROC_7'] < self.params['crash_roc_threshold']
        
        # All conditions must be true
        return below_long_ma and deeply_oversold and sharp_decline
    
    def _classify_relative_strength(self) -> str:
        """
        Classify relative strength against benchmark.
        
        Returns:
            str: 'Outperforming', 'Underperforming', or 'Neutral'
        """
        if self.benchmark_df is None or 'Ratio_ADX' not in self.df.columns:
            return 'No Benchmark'
        
        latest = self.df.iloc[-1]
        
        # Check if ratio is trending
        if pd.notna(latest['Ratio_ADX']) and latest['Ratio_ADX'] > 25:
            if latest['Ratio_DI+'] > latest['Ratio_DI-']:
                return 'Outperforming'
            else:
                return 'Underperforming'
        
        # Check simple ratio position
        if latest['Price_Ratio'] > latest['Ratio_SMA_50']:
            return 'Outperforming'
        else:
            return 'Underperforming'
    
    def classify_regime(self) -> Tuple[str, Dict]:
        """
        Classify the current market regime based on all indicators.
        
        Returns:
            Tuple containing:
                - Primary regime string (e.g., 'Bull Trend')
                - Dictionary with all calculated values and classifications
        """
        latest = self.df.iloc[-1]
        
        # Extract key values
        adx = latest['ADX']
        di_plus = latest['DI+']
        di_minus = latest['DI-']
        rsi = latest['RSI']
        atr_zscore = latest['ATR_ZScore']
        volume_zscore = latest['Volume_ZScore']
        
        # Detect specific conditions
        high_volatility = self._detect_high_volatility()
        is_crash = self._detect_crypto_crash()
        relative_strength = self._classify_relative_strength()
        
        # Classification logic
        if is_crash:
            primary_regime = 'Crypto Crash'
        elif pd.notna(adx) and adx > self.params['adx_threshold_trend'] and di_plus > di_minus:
            primary_regime = 'Bull Trend'
        elif pd.notna(adx) and adx > self.params['adx_threshold_trend'] and di_minus > di_plus:
            primary_regime = 'Bear Trend'
        elif pd.notna(adx) and adx < self.params['adx_threshold_range'] and not high_volatility:
            primary_regime = 'Ranging / Accumulation'
        elif pd.notna(adx) and adx < 25 and high_volatility:
            primary_regime = 'High Volatility / Breakout'
        else:
            primary_regime = 'Transitioning / Unknown'
        
        # Compile all metrics
        metrics = {
            'primary_regime': primary_regime,
            'adx': round(adx, 2) if pd.notna(adx) else None,
            'di_plus': round(di_plus, 2) if pd.notna(di_plus) else None,
            'di_minus': round(di_minus, 2) if pd.notna(di_minus) else None,
            'rsi': round(rsi, 2) if pd.notna(rsi) else None,
            'atr_zscore': round(atr_zscore, 2) if pd.notna(atr_zscore) else None,
            'volume_zscore': round(volume_zscore, 2) if pd.notna(volume_zscore) else None,
            'volatility_regime': 'High' if high_volatility else 'Normal',
            'relative_strength': relative_strength,
            'price_vs_sma200': 'Above' if latest[self.price_column] > latest['SMA_200'] else 'Below',
            'roc_7': round(latest['ROC_7'], 2) if pd.notna(latest['ROC_7']) else None
        }
        
        return primary_regime, metrics
    
    def get_regime_history(self) -> pd.DataFrame:
        """
        Calculate regime classification for entire historical data.
        
        Returns:
            pd.DataFrame: DataFrame with regime classifications for each timestamp
        """
        regime_history = []
        
        # Need minimum data points for indicators
        min_periods = 200
        
        for i in range(min_periods, len(self.df)):
            # Create temporary detector with data up to current point
            temp_df = self.df.iloc[:i+1].copy()
            temp_benchmark = None
            if self.benchmark_df is not None:
                temp_benchmark = self.benchmark_df.iloc[:i+1].copy()
            
            # Create temporary detector
            temp_detector = CryptoMarketRegimeDetector(
                temp_df, 
                self.asset_name, 
                temp_benchmark, 
                self.price_column
            )
            
            # Classify regime
            regime, metrics = temp_detector.classify_regime()
            
            # Store results
            regime_history.append({
                'timestamp': self.df.index[i],
                'regime': regime,
                **metrics
            })
        
        return pd.DataFrame(regime_history).set_index('timestamp')
    
    def plot_regime(self, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None,
                    use_plotly: bool = True) -> None:
        """
        Visualize price action color-coded by detected regime.
        
        Args:
            start_date: Start date for visualization (format: 'YYYY-MM-DD')
            end_date: End date for visualization (format: 'YYYY-MM-DD')
            use_plotly: If True, use plotly for interactive chart, else matplotlib
        """
        # Get regime history
        regime_df = self.get_regime_history()
        
        # Filter date range if specified
        plot_df = self.df.copy()
        if start_date:
            plot_df = plot_df[plot_df.index >= start_date]
            regime_df = regime_df[regime_df.index >= start_date]
        if end_date:
            plot_df = plot_df[plot_df.index <= end_date]
            regime_df = regime_df[regime_df.index <= end_date]
        
        # Merge price data with regime
        plot_df = plot_df.join(regime_df[['regime']], how='left')
        plot_df['regime'].fillna('Unknown', inplace=True)
        
        # Define colors for each regime
        regime_colors = {
            'Bull Trend': '#00D775',           # Green
            'Bear Trend': '#FF3B30',           # Red
            'Ranging / Accumulation': '#FFD60A', # Yellow
            'High Volatility / Breakout': '#FF9500', # Orange
            'Crypto Crash': '#8B0000',         # Dark Red
            'Transitioning / Unknown': '#8E8E93'  # Gray
        }
        
        if use_plotly:
            self._plot_regime_plotly(plot_df, regime_colors)
        else:
            self._plot_regime_matplotlib(plot_df, regime_colors)
    
    def _plot_regime_plotly(self, plot_df: pd.DataFrame, regime_colors: Dict) -> None:
        """Create interactive plotly visualization."""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{self.asset_name} Price with Market Regime', 
                           'ADX & Directional Indicators', 
                           'RSI & Volatility Z-Score'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Plot 1: Price with regime colors
        for regime, color in regime_colors.items():
            regime_data = plot_df[plot_df['regime'] == regime]
            if len(regime_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=regime_data.index,
                        y=regime_data[self.price_column],
                        mode='markers+lines',
                        name=regime,
                        line=dict(color=color, width=2),
                        marker=dict(size=3),
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='blue', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df['SMA_200'],
                mode='lines',
                name='SMA 200',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Plot 2: ADX and DI
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df['ADX'],
                mode='lines',
                name='ADX',
                line=dict(color='black', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df['DI+'],
                mode='lines',
                name='DI+',
                line=dict(color='green', width=1)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df['DI-'],
                mode='lines',
                name='DI-',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Add ADX threshold lines
        fig.add_hline(y=self.params['adx_threshold_trend'], line_dash="dash", 
                     line_color="gray", row=2, col=1)
        fig.add_hline(y=self.params['adx_threshold_range'], line_dash="dot", 
                     line_color="gray", row=2, col=1)
        
        # Plot 3: RSI and Volatility
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=plot_df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2),
                yaxis='y5'
            ),
            row=3, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=self.params['crash_rsi_threshold'], line_dash="dot", 
                     line_color="darkred", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{self.asset_name} Market Regime Analysis',
            xaxis_title='Date',
            height=1000,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='ADX/DI', row=2, col=1)
        fig.update_yaxes(title_text='RSI', row=3, col=1)
        
        fig.show()
    
    def _plot_regime_matplotlib(self, plot_df: pd.DataFrame, regime_colors: Dict) -> None:
        """Create matplotlib visualization."""
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), 
                                           gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot 1: Price with regime colors
        for regime, color in regime_colors.items():
            regime_data = plot_df[plot_df['regime'] == regime]
            if len(regime_data) > 0:
                ax1.plot(regime_data.index, regime_data[self.price_column], 
                        color=color, label=regime, linewidth=2)
        
        # Add moving averages
        ax1.plot(plot_df.index, plot_df['SMA_50'], 'b--', alpha=0.5, label='SMA 50')
        ax1.plot(plot_df.index, plot_df['SMA_200'], 'r--', alpha=0.5, label='SMA 200')
        
        ax1.set_ylabel('Price')
        ax1.set_title(f'{self.asset_name} Price with Market Regime')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ADX and DI
        ax2.plot(plot_df.index, plot_df['ADX'], 'k-', linewidth=2, label='ADX')
        ax2.plot(plot_df.index, plot_df['DI+'], 'g-', label='DI+')
        ax2.plot(plot_df.index, plot_df['DI-'], 'r-', label='DI-')
        ax2.axhline(y=self.params['adx_threshold_trend'], color='gray', 
                   linestyle='--', alpha=0.5)
        ax2.axhline(y=self.params['adx_threshold_range'], color='gray', 
                   linestyle=':', alpha=0.5)
        
        ax2.set_ylabel('ADX/DI')
        ax2.set_title('ADX & Directional Indicators')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: RSI
        ax3.plot(plot_df.index, plot_df['RSI'], 'purple', linewidth=2)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.axhline(y=self.params['crash_rsi_threshold'], color='darkred', 
                   linestyle=':', alpha=0.5)
        
        ax3.set_ylabel('RSI')
        ax3.set_xlabel('Date')
        ax3.set_title('RSI Indicator')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_regime_statistics(self) -> pd.DataFrame:
        """
        Calculate statistics for each regime including duration and frequency.
        
        Returns:
            pd.DataFrame: Statistics for each regime type
        """
        regime_history = self.get_regime_history()
        
        # Calculate regime statistics
        stats = []
        for regime in regime_history['regime'].unique():
            regime_data = regime_history[regime_history['regime'] == regime]
            
            # Calculate consecutive regime periods
            regime_changes = (regime_history['regime'] != regime_history['regime'].shift()).cumsum()
            regime_groups = regime_history[regime_history['regime'] == regime].groupby(regime_changes)
            
            durations = []
            for _, group in regime_groups:
                if len(group) > 0:
                    duration = (group.index[-1] - group.index[0]).total_seconds() / 3600  # Hours
                    durations.append(duration)
            
            stats.append({
                'regime': regime,
                'count': len(regime_data),
                'percentage': len(regime_data) / len(regime_history) * 100,
                'avg_duration_hours': np.mean(durations) if durations else 0,
                'max_duration_hours': max(durations) if durations else 0,
                'min_duration_hours': min(durations) if durations else 0,
                'avg_rsi': regime_data['rsi'].mean(),
                'avg_adx': regime_data['adx'].mean(),
                'avg_volatility_zscore': regime_data['atr_zscore'].mean()
            })
        
        return pd.DataFrame(stats).sort_values('percentage', ascending=False)


# Example usage and testing
if __name__ == "__main__":
    # This is just an example - in production, data would come from your database
    print("CryptoMarketRegimeDetector module loaded successfully.")
    print("\nExample usage:")
    print("```python")
    print("# Load your OHLCV data")
    print("df = pd.read_csv('btc_data.csv', index_col='timestamp', parse_dates=True)")
    print()
    print("# Create detector instance") 
    print("detector = CryptoMarketRegimeDetector(df, asset_name='BTC')")
    print()
    print("# Classify current regime")
    print("regime, metrics = detector.classify_regime()")
    print("print(f'Current regime: {regime}')")
    print("print(f'Metrics: {metrics}')")
    print()
    print("# Plot regime visualization")
    print("detector.plot_regime()")
    print("```")
