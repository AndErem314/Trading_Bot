"""
Results Visualization System

This module provides comprehensive visualization capabilities for backtesting results,
including interactive charts using Plotly for professional-grade analysis.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import os
import logging

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """
    Comprehensive visualization system for trading backtest results.
    
    This class creates interactive visualizations using Plotly, including:
    - Price charts with Ichimoku cloud
    - Trade entry/exit markers
    - Equity curves with drawdown
    - Performance dashboards
    - HTML reports
    """
    
    def __init__(self, output_dir: str = "frontend/backtest_results"):
        """
        Initialize the ResultsVisualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'cloud_green': 'rgba(76, 175, 80, 0.2)',
            'cloud_red': 'rgba(244, 67, 54, 0.2)',
            'background': '#f8f9fa',
            'grid': '#e0e0e0'
        }
        
        # Chart template
        self.template = self._create_custom_template()
    
    def plot_trading_chart(self, data: pd.DataFrame, trades: pd.DataFrame, 
                          ichimoku_data: pd.DataFrame) -> go.Figure:
        """
        Create an interactive trading chart with Ichimoku cloud and trade markers.
        
        Args:
            data: OHLCV price data
            trades: DataFrame with trade entries/exits
            ichimoku_data: DataFrame with Ichimoku indicator values
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating interactive trading chart...")
        
        # Create subplots: main chart, volume, and indicators
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Ichimoku Cloud', 'Volume', 'Trade Performance'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # 1. Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color=self.colors['success'],
                decreasing_line_color=self.colors['danger']
            ),
            row=1, col=1
        )
        
        # 2. Ichimoku Cloud
        if all(col in ichimoku_data.columns for col in ['senkou_span_a', 'senkou_span_b']):
            # Add cloud fill
            fig.add_trace(
                go.Scatter(
                    x=ichimoku_data.index,
                    y=ichimoku_data['senkou_span_a'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=ichimoku_data.index,
                    y=ichimoku_data['senkou_span_b'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=self.colors['cloud_green'],
                    name='Ichimoku Cloud',
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # Add cloud color based on span relationship
            cloud_colors = ichimoku_data['senkou_span_a'] > ichimoku_data['senkou_span_b']
            
            # Tenkan-sen (Conversion Line)
            fig.add_trace(
                go.Scatter(
                    x=ichimoku_data.index,
                    y=ichimoku_data['tenkan_sen'],
                    mode='lines',
                    name='Tenkan-sen',
                    line=dict(color='#FF6B6B', width=1.5)
                ),
                row=1, col=1
            )
            
            # Kijun-sen (Base Line)
            fig.add_trace(
                go.Scatter(
                    x=ichimoku_data.index,
                    y=ichimoku_data['kijun_sen'],
                    mode='lines',
                    name='Kijun-sen',
                    line=dict(color='#4ECDC4', width=1.5)
                ),
                row=1, col=1
            )
            
            # Chikou Span (Lagging Span)
            if 'chikou_span' in ichimoku_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ichimoku_data.index,
                        y=ichimoku_data['chikou_span'],
                        mode='lines',
                        name='Chikou Span',
                        line=dict(color='#9B59B6', width=1, dash='dot')
                    ),
                    row=1, col=1
                )
        
        # 3. Trade markers
        if trades is not None and not trades.empty:
            # Entry points
            entry_trades = trades[trades['entry_time'].notna()]
            for _, trade in entry_trades.iterrows():
                if trade['entry_time'] in data.index:
                    fig.add_trace(
                        go.Scatter(
                            x=[trade['entry_time']],
                            y=[trade['entry_price']],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color=self.colors['success'],
                                line=dict(width=2, color='white')
                            ),
                            name='Buy',
                            showlegend=False,
                            hovertext=f"Buy @ {trade['entry_price']:.2f}"
                        ),
                        row=1, col=1
                    )
            
            # Exit points
            exit_trades = trades[trades['exit_time'].notna()]
            for _, trade in exit_trades.iterrows():
                if trade['exit_time'] in data.index:
                    color = self.colors['success'] if trade.get('net_pnl', 0) > 0 else self.colors['danger']
                    fig.add_trace(
                        go.Scatter(
                            x=[trade['exit_time']],
                            y=[trade['exit_price']],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color=color,
                                line=dict(width=2, color='white')
                            ),
                            name='Sell',
                            showlegend=False,
                            hovertext=f"Sell @ {trade['exit_price']:.2f} (PnL: {trade.get('net_pnl', 0):.2f})"
                        ),
                        row=1, col=1
                    )
        
        # 4. Volume bars
        colors = ['red' if row['close'] < row['open'] else 'green' 
                 for idx, row in data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 5. Trade performance (cumulative PnL)
        if trades is not None and not trades.empty and 'net_pnl' in trades.columns:
            trades_sorted = trades.sort_values('exit_time')
            cumulative_pnl = trades_sorted['net_pnl'].cumsum()
            
            fig.add_trace(
                go.Scatter(
                    x=trades_sorted['exit_time'],
                    y=cumulative_pnl,
                    mode='lines+markers',
                    name='Cumulative PnL',
                    line=dict(color=self.colors['primary'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.2)'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Trading Chart with Ichimoku Analysis',
            template=self.template,
            height=900,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            margin=dict(l=70, r=70, t=50, b=50)
        )
        
        # Update x-axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="PnL ($)", row=3, col=1)
        
        # Remove range slider from candlestick
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        
        return fig
    
    def plot_performance_dashboard(self, metrics: Dict[str, Any]) -> go.Figure:
        """
        Create a comprehensive performance dashboard.
        
        Args:
            metrics: Dictionary containing performance metrics
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating performance dashboard...")
        
        # Create subplots for dashboard
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Equity Curve', 'Monthly Returns', 'Trade Distribution',
                'Drawdown', 'Returns Distribution', 'Risk Metrics',
                'Win/Loss Analysis', 'Trade Duration', 'Rolling Sharpe'
            ),
            specs=[
                [{"colspan": 2}, None, {"type": "table"}],
                [{"colspan": 2}, None, {}],
                [{}, {}, {}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Equity Curve
        if 'equity_curve' in metrics:
            equity = metrics['equity_curve']
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity['total_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color=self.colors['primary'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.1)'
                ),
                row=1, col=1
            )
        
        # 2. Key Metrics Table
        if 'performance_metrics' in metrics:
            perf = metrics['performance_metrics']
            table_data = [
                ['Metric', 'Value'],
                ['Total Return', f"{perf.get('total_return', 0):.2f}%"],
                ['Sharpe Ratio', f"{perf.get('sharpe_ratio', 0):.2f}"],
                ['Win Rate', f"{perf.get('win_rate', 0):.2f}%"],
                ['Profit Factor', f"{perf.get('profit_factor', 0):.2f}"],
                ['Max Drawdown', f"{perf.get('max_drawdown_pct', 0):.2f}%"],
                ['Total Trades', f"{perf.get('total_trades', 0)}"]
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['<b>Metric</b>', '<b>Value</b>'],
                        fill_color=self.colors['primary'],
                        font=dict(color='white', size=12),
                        align='left'
                    ),
                    cells=dict(
                        values=list(zip(*table_data[1:])),
                        fill_color=[['white', self.colors['background']] * len(table_data[1:])],
                        align='left'
                    )
                ),
                row=1, col=3
            )
        
        # 3. Drawdown Chart
        if 'equity_curve' in metrics:
            equity = metrics['equity_curve']['total_value']
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color=self.colors['danger'], width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(214, 39, 40, 0.2)'
                ),
                row=2, col=1
            )
        
        # 4. Returns Distribution
        if 'returns' in metrics:
            returns = metrics['returns']
            fig.add_trace(
                go.Histogram(
                    x=returns * 100,
                    nbinsx=50,
                    name='Returns',
                    marker_color=self.colors['info'],
                    opacity=0.7
                ),
                row=2, col=3
            )
            
            # Add normal distribution overlay
            mean = returns.mean() * 100
            std = returns.std() * 100
            x = np.linspace(mean - 4*std, mean + 4*std, 100)
            y = (1/np.sqrt(2*np.pi*std**2)) * np.exp(-0.5*((x-mean)/std)**2)
            y = y * len(returns) * (returns.max() - returns.min()) * 100 / 50  # Scale to histogram
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name='Normal Dist',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=2, col=3
            )
        
        # 5. Win/Loss Analysis
        if 'trades' in metrics:
            trades_df = metrics['trades']
            wins = len(trades_df[trades_df['net_pnl'] > 0])
            losses = len(trades_df[trades_df['net_pnl'] <= 0])
            
            fig.add_trace(
                go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[wins, losses],
                    hole=0.3,
                    marker=dict(colors=[self.colors['success'], self.colors['danger']]),
                    textposition='inside',
                    textinfo='label+percent'
                ),
                row=3, col=1
            )
        
        # 6. Trade Duration Distribution
        if 'trades' in metrics and 'bars_held' in metrics['trades'].columns:
            fig.add_trace(
                go.Box(
                    y=metrics['trades']['bars_held'],
                    name='Trade Duration',
                    marker_color=self.colors['warning'],
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=3, col=2
            )
        
        # 7. Rolling Sharpe Ratio
        if 'rolling_sharpe' in metrics:
            rolling_sharpe = metrics['rolling_sharpe'].dropna()
            fig.add_trace(
                go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe,
                    mode='lines',
                    name='Rolling Sharpe',
                    line=dict(color=self.colors['secondary'], width=2)
                ),
                row=3, col=3
            )
            
            # Add reference line
            if 'sharpe_ratio' in metrics.get('performance_metrics', {}):
                fig.add_hline(
                    y=metrics['performance_metrics']['sharpe_ratio'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Overall Sharpe",
                    row=3, col=3
                )
        
        # Update layout
        fig.update_layout(
            title='Performance Dashboard',
            template=self.template,
            height=1000,
            showlegend=False,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Returns (%)", row=2, col=3)
        fig.update_yaxes(title_text="Frequency", row=2, col=3)
        fig.update_yaxes(title_text="Bars Held", row=3, col=2)
        fig.update_xaxes(title_text="Date", row=3, col=3)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=3)
        
        return fig
    
    def plot_monthly_returns_heatmap(self, returns: pd.Series) -> go.Figure:
        """
        Create a monthly returns heatmap.
        
        Args:
            returns: Series of daily returns
            
        Returns:
            Plotly figure object
        """
        # Calculate monthly returns
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # Create matrix for heatmap
        years = monthly.index.year.unique()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        matrix = np.full((len(years), 12), np.nan)
        
        for i, year in enumerate(years):
            year_data = monthly[monthly.index.year == year]
            for data_point in year_data:
                month_idx = year_data.index[year_data == data_point][0].month - 1
                matrix[i, month_idx] = data_point
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=months,
            y=years,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(matrix, 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Monthly Return (%)")
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Month',
            yaxis_title='Year',
            template=self.template,
            height=400
        )
        
        return fig
    
    def generate_html_report(self, results: Dict[str, Any], 
                           filename: str = "backtest_report.html") -> str:
        """
        Generate a comprehensive HTML report with all visualizations.
        
        Args:
            results: Dictionary containing all backtest results
            filename: Output filename
            
        Returns:
            Path to generated HTML file
        """
        logger.info("Generating comprehensive HTML report...")
        
        # Create all visualizations
        charts = []
        
        # 1. Trading Chart
        if all(key in results for key in ['data', 'trades', 'ichimoku_data']):
            trading_chart = self.plot_trading_chart(
                results['data'], 
                results['trades'], 
                results['ichimoku_data']
            )
            charts.append(('trading_chart', trading_chart))
        
        # 2. Performance Dashboard
        if 'metrics' in results:
            dashboard = self.plot_performance_dashboard(results['metrics'])
            charts.append(('dashboard', dashboard))
        
        # 3. Monthly Returns Heatmap
        if 'returns' in results:
            heatmap = self.plot_monthly_returns_heatmap(results['returns'])
            charts.append(('heatmap', heatmap))
        
        # 4. Additional custom charts
        if 'trades' in results:
            # Trade PnL distribution
            trade_pnl_chart = self._create_trade_pnl_distribution(results['trades'])
            charts.append(('trade_pnl', trade_pnl_chart))
            
            # Cumulative returns comparison
            if 'equity_curve' in results:
                comparison_chart = self._create_returns_comparison(results['equity_curve'])
                charts.append(('returns_comparison', comparison_chart))
        
        # Generate HTML
        html_content = self._generate_html_template(results, charts)
        
        # Save to file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {output_path}")
        return output_path
    
    def _create_trade_pnl_distribution(self, trades: pd.DataFrame) -> go.Figure:
        """Create trade PnL distribution chart."""
        fig = go.Figure()
        
        # Separate winning and losing trades
        wins = trades[trades['net_pnl'] > 0]['net_pnl']
        losses = trades[trades['net_pnl'] <= 0]['net_pnl']
        
        # Add histograms
        fig.add_trace(go.Histogram(
            x=wins,
            name='Winning Trades',
            marker_color=self.colors['success'],
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.add_trace(go.Histogram(
            x=losses,
            name='Losing Trades',
            marker_color=self.colors['danger'],
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.update_layout(
            title='Trade PnL Distribution',
            xaxis_title='PnL ($)',
            yaxis_title='Frequency',
            template=self.template,
            barmode='overlay',
            height=400
        )
        
        return fig
    
    def _create_returns_comparison(self, equity_curve: pd.DataFrame) -> go.Figure:
        """Create cumulative returns comparison chart."""
        fig = go.Figure()
        
        # Calculate cumulative returns
        returns = equity_curve['returns'].fillna(0)
        cum_returns = (1 + returns).cumprod() - 1
        
        # Add cumulative returns
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns * 100,
            mode='lines',
            name='Strategy Returns',
            line=dict(color=self.colors['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))
        
        # Add benchmark (buy and hold)
        if 'benchmark_returns' in equity_curve.columns:
            benchmark = (1 + equity_curve['benchmark_returns']).cumprod() - 1
            fig.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark * 100,
                mode='lines',
                name='Benchmark',
                line=dict(color=self.colors['secondary'], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            template=self.template,
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def _generate_html_template(self, results: Dict[str, Any], 
                              charts: List[Tuple[str, go.Figure]]) -> str:
        """Generate the HTML template with embedded charts."""
        
        # Convert charts to HTML
        chart_htmls = {}
        for chart_name, fig in charts:
            chart_htmls[chart_name] = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        
        # Extract key metrics
        metrics = results.get('metrics', {}).get('performance_metrics', {})
        
        # Strategy information
        strategy_info = results.get('strategy_config', {})
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Results Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .metric-value.positive {{
            color: #2ca02c;
        }}
        .metric-value.negative {{
            color: #d62728;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .section-title {{
            font-size: 1.8em;
            color: #333;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .info-section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        .info-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        .info-label {{
            font-weight: 600;
            color: #666;
        }}
        .info-value {{
            color: #333;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Backtest Results Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Strategy: {strategy_info.get('name', 'Ichimoku Strategy')}</p>
        </div>
        
        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {('positive' if metrics.get('total_return', 0) > 0 else 'negative')}">
                    {metrics.get('total_return', 0):.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">
                    {metrics.get('sharpe_ratio', 0):.2f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">
                    {metrics.get('win_rate', 0):.1f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">
                    {metrics.get('profit_factor', 0):.2f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative">
                    {metrics.get('max_drawdown_pct', 0):.2f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">
                    {int(metrics.get('total_trades', 0))}
                </div>
            </div>
        </div>
        
        <!-- Trading Chart -->
        <h2 class="section-title">Trading Analysis</h2>
        <div class="chart-container">
            {chart_htmls.get('trading_chart', '<p>Trading chart not available</p>')}
        </div>
        
        <!-- Performance Dashboard -->
        <h2 class="section-title">Performance Dashboard</h2>
        <div class="chart-container">
            {chart_htmls.get('dashboard', '<p>Dashboard not available</p>')}
        </div>
        
        <!-- Monthly Returns Heatmap -->
        <h2 class="section-title">Monthly Returns Analysis</h2>
        <div class="chart-container">
            {chart_htmls.get('heatmap', '<p>Heatmap not available</p>')}
        </div>
        
        <!-- Trade Analysis -->
        <h2 class="section-title">Trade Analysis</h2>
        <div class="info-section">
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">Average Win:</span>
                    <span class="info-value">${metrics.get('avg_win', 0):.2f}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Average Loss:</span>
                    <span class="info-value">${metrics.get('avg_loss', 0):.2f}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Largest Win:</span>
                    <span class="info-value">${metrics.get('largest_win', 0):.2f}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Largest Loss:</span>
                    <span class="info-value">${metrics.get('largest_loss', 0):.2f}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Win Streak (Max):</span>
                    <span class="info-value">{int(metrics.get('win_streak_max', 0))}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Loss Streak (Max):</span>
                    <span class="info-value">{int(metrics.get('loss_streak_max', 0))}</span>
                </div>
            </div>
        </div>
        
        <!-- Additional Charts -->
        <div class="chart-container">
            {chart_htmls.get('trade_pnl', '')}
        </div>
        <div class="chart-container">
            {chart_htmls.get('returns_comparison', '')}
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Report generated by Trading Bot Backtest System</p>
            <p>© 2024 - Powered by Plotly</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _create_custom_template(self) -> go.layout.Template:
        """Create a custom Plotly template for consistent styling."""
        template = go.layout.Template()
        
        # Layout settings
        template.layout = go.Layout(
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=60, t=60, b=60),
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial, sans-serif"
            )
        )
        
        # Update axes
        template.layout.xaxis = dict(
            gridcolor='#e0e0e0',
            zerolinecolor='#e0e0e0',
            showgrid=True,
            zeroline=True
        )
        template.layout.yaxis = dict(
            gridcolor='#e0e0e0',
            zerolinecolor='#e0e0e0',
            showgrid=True,
            zeroline=True
        )
        
        return template


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Results Visualizer Module")
    print("-" * 50)
    print("Features:")
    print("- Interactive trading charts with Ichimoku cloud")
    print("- Trade entry/exit markers")
    print("- Performance dashboards")
    print("- Monthly returns heatmap")
    print("- Comprehensive HTML reports")
    print("\nUse with backtest results for professional visualization")