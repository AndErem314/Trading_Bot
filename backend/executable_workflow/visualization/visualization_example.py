"""
Visualization Example

This module demonstrates the complete visualization system for backtest results,
including all chart types and HTML report generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

from backend.executable_workflow.visualization import ResultsVisualizer
from backend.executable_workflow.analytics import PerformanceAnalyzer
from backend.executable_workflow.backtesting import IchimokuBacktester
from backend.executable_workflow.data_fetching import OHLCVDataFetcher, DataPreprocessor
from backend.executable_workflow.indicators import IchimokuCalculator

logger = logging.getLogger(__name__)


class VisualizationExample:
    """
    Example class demonstrating all visualization capabilities
    for backtest results.
    """
    
    def __init__(self):
        self.visualizer = ResultsVisualizer()
        self.analyzer = PerformanceAnalyzer()
        
    def run_complete_visualization_example(self):
        """
        Run a complete example demonstrating all visualization features.
        """
        logger.info("Starting Visualization Example")
        
        # 1. Generate or load sample data
        sample_data = self._generate_sample_data()
        
        # 2. Create trading chart with Ichimoku cloud
        logger.info("\n" + "="*60)
        logger.info("1. CREATING TRADING CHART WITH ICHIMOKU CLOUD")
        logger.info("="*60)
        
        trading_chart = self.visualizer.plot_trading_chart(
            data=sample_data['ohlcv'],
            trades=sample_data['trades'],
            ichimoku_data=sample_data['ichimoku']
        )
        
        # Save as standalone HTML
        trading_chart.write_html("frontend/backtest_results/trading_chart.html")
        logger.info("Trading chart saved to frontend/backtest_results/trading_chart.html")
        
        # 3. Create performance dashboard
        logger.info("\n" + "="*60)
        logger.info("2. CREATING PERFORMANCE DASHBOARD")
        logger.info("="*60)
        
        # Calculate metrics first
        metrics = self.analyzer.calculate_all_metrics(
            trades=sample_data['trades'],
            equity_curve=sample_data['equity_curve']
        )
        
        # Prepare metrics dictionary
        metrics_dict = {
            'equity_curve': sample_data['equity_curve'],
            'returns': sample_data['equity_curve']['returns'],
            'trades': sample_data['trades'],
            'performance_metrics': {
                'total_return': metrics.total_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'max_drawdown_pct': metrics.max_drawdown_pct,
                'total_trades': metrics.total_trades
            },
            'rolling_sharpe': metrics.rolling_sharpe
        }
        
        dashboard = self.visualizer.plot_performance_dashboard(metrics_dict)
        dashboard.write_html("frontend/backtest_results/performance_dashboard.html")
        logger.info("Dashboard saved to frontend/backtest_results/performance_dashboard.html")
        
        # 4. Create monthly returns heatmap
        logger.info("\n" + "="*60)
        logger.info("3. CREATING MONTHLY RETURNS HEATMAP")
        logger.info("="*60)
        
        heatmap = self.visualizer.plot_monthly_returns_heatmap(
            sample_data['equity_curve']['returns']
        )
        heatmap.write_html("frontend/backtest_results/monthly_heatmap.html")
        logger.info("Heatmap saved to frontend/backtest_results/monthly_heatmap.html")
        
        # 5. Generate comprehensive HTML report
        logger.info("\n" + "="*60)
        logger.info("4. GENERATING COMPREHENSIVE HTML REPORT")
        logger.info("="*60)
        
        # Prepare complete results dictionary
        complete_results = {
            'data': sample_data['ohlcv'],
            'trades': sample_data['trades'],
            'ichimoku_data': sample_data['ichimoku'],
            'equity_curve': sample_data['equity_curve'],
            'returns': sample_data['equity_curve']['returns'],
            'metrics': metrics_dict,
            'strategy_config': {
                'name': 'Example Ichimoku Strategy',
                'timeframe': '1h',
                'symbols': ['BTC/USDT']
            }
        }
        
        report_path = self.visualizer.generate_html_report(
            results=complete_results,
            filename="complete_backtest_report.html"
        )
        
        logger.info(f"Complete HTML report generated: {report_path}")
        
        # 6. Demonstrate custom visualizations
        logger.info("\n" + "="*60)
        logger.info("5. CUSTOM VISUALIZATIONS")
        logger.info("="*60)
        
        self._create_custom_visualizations(sample_data)
        
        return report_path
    
    def _generate_sample_data(self):
        """Generate sample data for visualization demonstration."""
        logger.info("Generating sample data...")
        
        # Generate OHLCV data
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='1H')
        n_bars = len(dates)
        
        # Generate realistic price movement
        np.random.seed(42)
        price_base = 50000
        returns = np.random.normal(0.0001, 0.01, n_bars)
        prices = price_base * (1 + returns).cumprod()
        
        # Add some trends
        trend = np.linspace(0, 0.15, n_bars)
        prices = prices * (1 + trend)
        
        # Generate OHLCV
        ohlcv_data = pd.DataFrame(index=dates)
        ohlcv_data['close'] = prices
        ohlcv_data['open'] = prices * (1 + np.random.normal(0, 0.001, n_bars))
        ohlcv_data['high'] = np.maximum(ohlcv_data['open'], ohlcv_data['close']) * (1 + np.abs(np.random.normal(0, 0.002, n_bars)))
        ohlcv_data['low'] = np.minimum(ohlcv_data['open'], ohlcv_data['close']) * (1 - np.abs(np.random.normal(0, 0.002, n_bars)))
        ohlcv_data['volume'] = np.random.uniform(100, 1000, n_bars)
        
        # Generate Ichimoku data
        calc = IchimokuCalculator()
        ichimoku_data = calc.calculate(ohlcv_data)
        
        # Generate sample trades
        n_trades = 30
        trade_indices = np.sort(np.random.choice(range(100, n_bars-100), n_trades*2, replace=False))
        
        trades_list = []
        for i in range(0, len(trade_indices), 2):
            entry_idx = trade_indices[i]
            exit_idx = trade_indices[i+1]
            
            entry_price = ohlcv_data.iloc[entry_idx]['close']
            exit_price = ohlcv_data.iloc[exit_idx]['close']
            
            # Add some randomness to make some losing trades
            if np.random.random() < 0.3:  # 30% losing trades
                exit_price *= (1 - np.random.uniform(0.005, 0.02))
            else:
                exit_price *= (1 + np.random.uniform(0.005, 0.03))
            
            pnl = (exit_price - entry_price) * 0.001  # Position size
            
            trades_list.append({
                'entry_time': dates[entry_idx],
                'exit_time': dates[exit_idx],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': 0.001,
                'net_pnl': pnl,
                'return_pct': (exit_price / entry_price - 1) * 100,
                'bars_held': exit_idx - entry_idx
            })
        
        trades_df = pd.DataFrame(trades_list)
        
        # Generate equity curve
        initial_capital = 10000
        equity_values = [initial_capital]
        
        for i in range(1, len(dates)):
            # Add returns from trades
            trades_today = trades_df[trades_df['exit_time'] == dates[i]]
            daily_pnl = trades_today['net_pnl'].sum() if not trades_today.empty else 0
            
            # Add some random daily volatility
            random_return = np.random.normal(0.0001, 0.002)
            
            new_value = equity_values[-1] * (1 + random_return) + daily_pnl
            equity_values.append(new_value)
        
        equity_curve = pd.DataFrame({
            'total_value': equity_values,
            'returns': pd.Series(equity_values).pct_change()
        }, index=dates)
        
        return {
            'ohlcv': ohlcv_data,
            'ichimoku': ichimoku_data,
            'trades': trades_df,
            'equity_curve': equity_curve
        }
    
    def _create_custom_visualizations(self, sample_data):
        """Create additional custom visualizations."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 1. Multi-panel analysis chart
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=(
                'Price Action with Ichimoku',
                'Volume Analysis',
                'Trade Performance',
                'Drawdown Analysis'
            )
        )
        
        data = sample_data['ohlcv']
        ichimoku = sample_data['ichimoku']
        equity = sample_data['equity_curve']
        
        # Price with Ichimoku
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add Ichimoku lines
        fig.add_trace(
            go.Scatter(
                x=ichimoku.index,
                y=ichimoku['tenkan_sen'],
                name='Tenkan-sen',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=ichimoku.index,
                y=ichimoku['kijun_sen'],
                name='Kijun-sen',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Volume with color coding
        colors = ['red' if data.iloc[i]['close'] < data.iloc[i]['open'] else 'green' 
                 for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Cumulative returns
        cum_returns = (1 + equity['returns'].fillna(0)).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns * 100,
                fill='tozeroy',
                name='Cumulative Returns'
            ),
            row=3, col=1
        )
        
        # Drawdown
        running_max = equity['total_value'].expanding().max()
        drawdown_pct = (equity['total_value'] - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown_pct.index,
                y=drawdown_pct,
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.3)',
                name='Drawdown'
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Multi-Panel Trading Analysis',
            height=1000,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Return %", row=3, col=1)
        fig.update_yaxes(title_text="DD %", row=4, col=1)
        
        fig.write_html("frontend/backtest_results/custom_analysis.html")
        logger.info("Custom analysis saved to frontend/backtest_results/custom_analysis.html")
        
        # 2. 3D surface plot of returns
        self._create_3d_returns_surface(equity['returns'])
    
    def _create_3d_returns_surface(self, returns):
        """Create a 3D surface plot of returns over time."""
        import plotly.graph_objects as go
        
        # Prepare data for 3D visualization
        returns_clean = returns.fillna(0)
        
        # Create rolling windows
        windows = [5, 10, 20, 30, 60]
        surface_data = []
        
        for window in windows:
            rolling_returns = returns_clean.rolling(window).mean() * 100
            surface_data.append(rolling_returns.values)
        
        # Create surface plot
        fig = go.Figure(data=[go.Surface(
            z=surface_data,
            x=returns_clean.index,
            y=windows,
            colorscale='RdYlGn',
            colorbar=dict(title="Returns %")
        )])
        
        fig.update_layout(
            title='Rolling Returns Surface (3D)',
            scene=dict(
                xaxis_title='Date',
                yaxis_title='Window Size',
                zaxis_title='Returns %'
            ),
            height=600
        )
        
        fig.write_html("frontend/backtest_results/returns_surface_3d.html")
        logger.info("3D returns surface saved to frontend/backtest_results/returns_surface_3d.html")
    
    def demonstrate_real_backtest_visualization(self):
        """
        Demonstrate visualization with real backtest data.
        """
        logger.info("\n" + "="*60)
        logger.info("REAL BACKTEST VISUALIZATION EXAMPLE")
        logger.info("="*60)
        
        # This would integrate with actual backtest results
        # For demonstration, we'll use the sample data
        
        # 1. Fetch real data (simplified)
        fetcher = OHLCVDataFetcher()
        preprocessor = DataPreprocessor()
        
        logger.info("Fetching real market data...")
        # In a real scenario:
        # raw_data = fetcher.fetch_ohlcv('binance', 'BTC/USDT', '1h', limit=500)
        # clean_data = preprocessor.process(raw_data)
        
        # 2. Run backtest (simplified)
        logger.info("Running backtest...")
        # backtester = IchimokuBacktester()
        # results = backtester.run_backtest(strategy_config, clean_data)
        
        # 3. Analyze performance
        logger.info("Analyzing performance...")
        # analyzer = PerformanceAnalyzer()
        # metrics = analyzer.calculate_all_metrics(results['trades'], results['equity_curve'])
        
        # 4. Generate visualizations
        logger.info("Generating visualizations...")
        # visualizer = ResultsVisualizer()
        # report_path = visualizer.generate_html_report(results)
        
        logger.info("Real backtest visualization complete!")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create example instance
    example = VisualizationExample()
    
    # Run complete visualization example
    print("\n1. Running Complete Visualization Example:")
    report_path = example.run_complete_visualization_example()
    
    print(f"\n✓ Visualization example complete!")
    print(f"✓ HTML report generated: {report_path}")
    print("\nGenerated files in frontend/backtest_results/:")
    print("- trading_chart.html (Interactive trading chart with Ichimoku)")
    print("- performance_dashboard.html (Comprehensive performance metrics)")
    print("- monthly_heatmap.html (Monthly returns heatmap)")
    print("- custom_analysis.html (Multi-panel analysis)")
    print("- returns_surface_3d.html (3D returns visualization)")
    print("- complete_backtest_report.html (Full report with all charts)")
    
    print("\n2. Real Backtest Visualization (Demo):")
    example.demonstrate_real_backtest_visualization()