#!/usr/bin/env python3
"""
Trading Bot Web Interface
A Flask-based frontend for the trading bot with interactive features.
"""
from flask import Flask, render_template, jsonify, request, redirect, url_for
import sqlite3
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import subprocess
import threading
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.Strategies import RSIMomentumDivergenceSwingStrategy
from backend.data_manager import DataManager
from backend.data_fetcher import DataFetcher
from backend.Indicators import (
    SimpleMovingAverageCalculator, BollingerBandsCalculator,
    IchimokuCloudCalculator, MACDCalculator, ParabolicSARCalculator,
    FibonacciRetracementCalculator, GaussianChannelCalculator,
    calculate_rsi_for_symbol_timeframe
)
from backend.enhanced_market_regime_detector import EnhancedMarketRegimeDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variable to store background task status
task_status = {
    'running': False,
    'task': None,
    'progress': 0,
    'message': ''
}

@app.route('/')
def index():
    """Main menu page"""
    return render_template('index.html')

@app.route('/status')
def status():
    """Get database status and summary"""
    try:
        # Get available symbols and timeframes with counts
        with sqlite3.connect('data/trading_data_BTC.db') as conn:
            # Get total records
            total_query = "SELECT COUNT(*) FROM ohlcv_data"
            total_records = conn.execute(total_query).fetchone()[0]
            
            # Get symbols
            symbols_query = "SELECT DISTINCT symbol FROM symbols ORDER BY symbol"
            symbols_df = pd.read_sql_query(symbols_query, conn)
            symbols_list = symbols_df['symbol'].tolist() if not symbols_df.empty else []
            
            # Get timeframes
            timeframes_query = "SELECT DISTINCT timeframe FROM timeframes ORDER BY timeframe"
            timeframes_df = pd.read_sql_query(timeframes_query, conn)
            timeframes_list = timeframes_df['timeframe'].tolist() if not timeframes_df.empty else []
            
            # Get detailed data info
            query = """
            SELECT DISTINCT s.symbol, t.timeframe, COUNT(o.id) as records,
                   MIN(o.timestamp) as start_date, MAX(o.timestamp) as end_date
            FROM symbols s
            JOIN ohlcv_data o ON s.id = o.symbol_id
            JOIN timeframes t ON o.timeframe_id = t.id
            GROUP BY s.symbol, t.timeframe
            ORDER BY s.symbol, t.timeframe
            """
            df = pd.read_sql_query(query, conn)
        
        return jsonify({
            'database': 'data/trading_data_BTC.db',
            'total_records': total_records,
            'symbols': symbols_list,
            'timeframes': timeframes_list,
            'integrity': True,  # Simplified for now
            'data_details': df.to_dict('records') if not df.empty else []
        })
    except Exception as e:
        # Return error response
        return jsonify({
            'error': str(e),
            'database': 'data/trading_data_BTC.db',
            'total_records': 0,
            'symbols': [],
            'timeframes': [],
            'integrity': False,
            'data_details': []
        }), 500

@app.route('/update_data', methods=['GET', 'POST'])
def update_data():
    """Run data update with new and historical data"""
    if request.method == 'POST':
        # Get parameters from form
        symbols = request.form.getlist('symbols[]')
        timeframes = request.form.getlist('timeframes[]')
        start_date = request.form.get('start_date', '2020-01-01')
        update_type = request.form.get('update_type', 'both')  # 'new', 'historical', or 'both'
        
        # Start background task
        def run_update():
            global task_status
            task_status['running'] = True
            task_status['task'] = 'data_update'
            task_status['progress'] = 0
            
            try:
                total_tasks = len(symbols) * len(timeframes)
                completed = 0
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        task_status['message'] = f'Updating {symbol} ({timeframe})...'
                        
                        # Run data collection
                        cmd = [
                            sys.executable, 'run_trading_bot.py',
                            '--mode', 'collect',
                            '--symbols', symbol,
                            '--timeframes', timeframe,
                            '--start-date', start_date
                        ]
                        subprocess.run(cmd, capture_output=True, text=True)
                        
                        completed += 1
                        task_status['progress'] = int((completed / total_tasks) * 100)
                
                task_status['message'] = 'Data update completed!'
                task_status['progress'] = 100
            except Exception as e:
                task_status['message'] = f'Error: {str(e)}'
            finally:
                task_status['running'] = False
        
        # Start thread
        thread = threading.Thread(target=run_update)
        thread.start()
        
        return jsonify({'status': 'started'})
    
    # GET request - show form
    return render_template('update_data.html')

@app.route('/run_indicators', methods=['GET', 'POST'])
def run_indicators():
    """Run all indicators with new data"""
    if request.method == 'POST':
        symbols = request.form.getlist('symbols[]')
        timeframes = request.form.getlist('timeframes[]')
        indicators = request.form.getlist('indicators[]')
        
        def run_indicators_task():
            global task_status
            task_status['running'] = True
            task_status['task'] = 'indicators'
            task_status['progress'] = 0
            
            try:
                # Calculate indicators
                indicator_calculators = {
                    'sma': SimpleMovingAverageCalculator(),
                    'bollinger': BollingerBandsCalculator(),
                    'ichimoku': IchimokuCloudCalculator(),
                    'macd': MACDCalculator(),
                    'parabolic': ParabolicSARCalculator(),
                    'fibonacci': FibonacciRetracementCalculator(),
                    'gaussian': GaussianChannelCalculator()
                }
                
                total_tasks = len(symbols) * len(timeframes) * len(indicators)
                completed = 0
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        for indicator_name in indicators:
                            task_status['message'] = f'Calculating {indicator_name} for {symbol} ({timeframe})...'
                            
                            if indicator_name == 'rsi':
                                calculate_rsi_for_symbol_timeframe(symbol, timeframe)
                            else:
                                calc = indicator_calculators.get(indicator_name)
                                if calc:
                                    df_raw = calc.fetch_raw_data(symbol, timeframe)
                                    if not df_raw.empty:
                                        # Calculate based on indicator type
                                        if indicator_name == 'sma':
                                            df_result = calc.calculate_sma(df_raw)
                                            calc.save_sma_data(df_result, symbol, timeframe)
                                        elif indicator_name == 'bollinger':
                                            df_result = calc.calculate_bollinger_bands(df_raw)
                                            calc.save_bollinger_bands_data(df_result, symbol, timeframe)
                                        # Add other indicators...
                            
                            completed += 1
                            task_status['progress'] = int((completed / total_tasks) * 100)
                
                task_status['message'] = 'Indicators calculation completed!'
                task_status['progress'] = 100
            except Exception as e:
                task_status['message'] = f'Error: {str(e)}'
            finally:
                task_status['running'] = False
        
        thread = threading.Thread(target=run_indicators_task)
        thread.start()
        
        return jsonify({'status': 'started'})
    
    return render_template('run_indicators.html')

@app.route('/run_strategy')
def run_strategy():
    """Run the RSI strategy and show signals"""
    strategy = RSIMomentumDivergenceSwingStrategy()
    
    # Get signals using the strategy's SQL query
    with sqlite3.connect('data/trading_data_BTC.db') as conn:
        query = strategy.get_sql_query()
        df = pd.read_sql_query(query, conn)
    
    # Get current market status
    status_query = """
    SELECT 
        o.timestamp,
        o.close as price,
        r.rsi,
        r.rsi_sma_5,
        r.rsi_sma_10,
        r.overbought,
        r.oversold,
        r.trend_strength,
        r.divergence_signal
    FROM rsi_indicator r
    JOIN ohlcv_data o ON r.ohlcv_id = o.id
    ORDER BY o.timestamp DESC
    LIMIT 1
    """
    
    with sqlite3.connect('data/trading_data_BTC.db') as conn:
        current_status = pd.read_sql_query(status_query, conn)
    
    return render_template('run_strategy.html', 
                         signals=df.to_dict('records'),
                         current_status=current_status.to_dict('records')[0] if not current_status.empty else None,
                         strategy_info=strategy.get_strategy_description())

@app.route('/visualization')
def visualization():
    """Interactive visualization page"""
    return render_template('visualization.html')

@app.route('/visualization_fast')
def visualization_fast():
    """Fast visualization page using Lightweight Charts"""
    return render_template('visualization_fast.html')

@app.route('/api/chart_data')
def get_chart_data():
    """Get chart data with indicators and signals"""
    symbol = request.args.get('symbol', 'BTC/USDT')
    timeframe = request.args.get('timeframe', '1d')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    indicators = request.args.getlist('indicators[]')
    show_signals = request.args.get('show_signals', 'true') == 'true'
    
    # Base query for OHLCV data - normalize timestamps to remove format differences
    # Use GROUP BY to ensure one row per date, taking the first occurrence
    query = """
    SELECT 
        DATE(o.timestamp) as timestamp,
        MAX(o.open) as open, 
        MAX(o.high) as high, 
        MAX(o.low) as low, 
        MAX(o.close) as close, 
        MAX(o.volume) as volume
    FROM ohlcv_data o
    JOIN symbols s ON o.symbol_id = s.id
    JOIN timeframes t ON o.timeframe_id = t.id
    WHERE s.symbol = ? AND t.timeframe = ?
    """
    params = [symbol, timeframe]
    
    if start_date:
        query += " AND o.timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND o.timestamp <= ?"
        params.append(end_date)
    
    query += " GROUP BY DATE(o.timestamp)"
    query += " ORDER BY timestamp ASC"
    
    with sqlite3.connect('data/trading_data_BTC.db') as conn:
        df = pd.read_sql_query(query, conn, params=params)
        
        # Ensure proper data types for OHLCV data
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Get RSI strategy signals if requested
        signals_df = pd.DataFrame()
        if show_signals:
            strategy = RSIMomentumDivergenceSwingStrategy()
            signals_query = strategy.get_sql_query()
            # Modify query to filter by symbol and timeframe
            signals_query = signals_query.replace("LIMIT 100", "")
            signals_df = pd.read_sql_query(signals_query, conn)
            
            # Filter signals by date range if provided
            if not signals_df.empty:
                # Handle different date formats - use format='mixed' for mixed formats
                signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'], format='mixed')
                signals_df['price'] = pd.to_numeric(signals_df['price'], errors='coerce')
                signals_df['rsi'] = pd.to_numeric(signals_df['rsi'], errors='coerce')
                
                if start_date:
                    signals_df = signals_df[signals_df['timestamp'] >= pd.to_datetime(start_date)]
                if end_date:
                    signals_df = signals_df[signals_df['timestamp'] <= pd.to_datetime(end_date)]
    
    # Create Plotly figure
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Price Chart', 'Volume'),
        shared_xaxes=True
    )
    
    # Remove debug logging
    
    # Create custom candlestick chart using scatter for better control
    if not df.empty:
        # Create lists for the candlestick components
        x_coords = []
        y_opens = []
        y_closes = []
        y_highs = []
        y_lows = []
        colors = []
        hover_texts = []
        
        for i in range(len(df)):
            open_price = df.iloc[i]['open']
            close_price = df.iloc[i]['close']
            high_price = df.iloc[i]['high']
            low_price = df.iloc[i]['low']
            
            x_coords.extend([i, i, None])  # For line segments
            y_lows.extend([low_price, None, None])
            y_highs.extend([high_price, None, None])
            
            # Determine color
            color = '#00ff00' if close_price > open_price else '#ff0000'
            colors.append(color)
            
            # Create hover text
            hover_text = (f"Date: {df.iloc[i]['timestamp'].strftime('%Y-%m-%d')}<br>"
                         f"Open: ${open_price:,.2f}<br>"
                         f"High: ${high_price:,.2f}<br>"
                         f"Low: ${low_price:,.2f}<br>"
                         f"Close: ${close_price:,.2f}")
            hover_texts.append(hover_text)
        
        # Add high-low lines (wicks) as a single trace
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=[y if y is not None else (l if l is not None else h) 
                   for y, l, h in zip(y_lows, y_lows, y_highs)],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Add rectangles for candle bodies
        for i in range(len(df)):
            open_price = df.iloc[i]['open']
            close_price = df.iloc[i]['close']
            color = colors[i]
            
            # Calculate bar width based on total data points
            bar_width = 0.8 if len(df) < 50 else 0.6 if len(df) < 100 else 0.4
            
            fig.add_shape(
                type="rect",
                x0=i - bar_width/2,
                x1=i + bar_width/2,
                y0=min(open_price, close_price),
                y1=max(open_price, close_price),
                fillcolor=color,
                line=dict(width=0),
                row=1, col=1
            )
            
            # Add vertical line for wick
            fig.add_shape(
                type="line",
                x0=i,
                x1=i,
                y0=df.iloc[i]['low'],
                y1=df.iloc[i]['high'],
                line=dict(color=color, width=1),
                row=1, col=1
            )
        
        # Add invisible scatter for hover information
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df['close'],
                mode='markers',
                marker=dict(size=1, opacity=0),
                text=hover_texts,
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add a visible trace for legend only
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[df['close'].iloc[0]],
                mode='markers',
                marker=dict(size=8, color='green'),
                name='OHLC',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Update x-axis to show dates on both subplots
        tickvals = list(range(0, len(df), max(1, len(df)//10)))
        ticktext = [df.iloc[i]['timestamp'].strftime('%Y-%m-%d') 
                   for i in range(0, len(df), max(1, len(df)//10))]
        
        fig.update_xaxes(
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            row=1, col=1
        )
        
        fig.update_xaxes(
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            row=2, col=1
        )
    
    # Add volume bars
    if not df.empty:
        fig.add_trace(
            go.Bar(
                x=list(range(len(df))),
                y=df['volume'],
                name='Volume',
                marker_color='rgba(0,0,255,0.3)',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Add buy/sell signals
    if show_signals and not signals_df.empty and not df.empty:
        # Create a mapping from timestamp to x-coordinate
        timestamp_to_x = {ts: i for i, ts in enumerate(df['timestamp'])}
        
        buy_signals = signals_df[signals_df['signal_name'].str.contains('BUY')]
        sell_signals = signals_df[signals_df['signal_name'].str.contains('SELL')]
        
        if not buy_signals.empty:
            # Map signal timestamps to x-coordinates
            buy_x_coords = []
            buy_y_coords = []
            buy_texts = []
            for _, row in buy_signals.iterrows():
                # Find the closest date in our chart data
                for i, chart_ts in enumerate(df['timestamp']):
                    if chart_ts.date() == row['timestamp'].date():
                        buy_x_coords.append(i)
                        buy_y_coords.append(row['price'])
                        buy_texts.append(f"Buy: {row['signal_name']}<br>RSI: {row['rsi']:.2f}")
                        break
            
            if buy_x_coords:
                fig.add_trace(
                    go.Scatter(
                        x=buy_x_coords,
                        y=buy_y_coords,
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green'
                        ),
                        text=buy_texts,
                        hoverinfo='text+y'
                    ),
                    row=1, col=1
                )
        
        if not sell_signals.empty:
            # Map signal timestamps to x-coordinates
            sell_x_coords = []
            sell_y_coords = []
            sell_texts = []
            for _, row in sell_signals.iterrows():
                # Find the closest date in our chart data
                for i, chart_ts in enumerate(df['timestamp']):
                    if chart_ts.date() == row['timestamp'].date():
                        sell_x_coords.append(i)
                        sell_y_coords.append(row['price'])
                        sell_texts.append(f"Sell: {row['signal_name']}<br>RSI: {row['rsi']:.2f}")
                        break
            
            if sell_x_coords:
                fig.add_trace(
                    go.Scatter(
                        x=sell_x_coords,
                        y=sell_y_coords,
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red'
                        ),
                        text=sell_texts,
                        hoverinfo='text+y'
                    ),
                    row=1, col=1
                )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} ({timeframe}) - Trading Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark'
    )
    
    # Force proper Y-axis scaling for price chart
    if not df.empty:
        price_min = df[['low']].min().min() * 0.95  # 5% margin
        price_max = df[['high']].max().max() * 1.05  # 5% margin
        
        
        fig.update_yaxes(
            title_text="Price (USDT)",
            range=[price_min, price_max],
            row=1, col=1
        )
    else:
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return jsonify(json.loads(fig.to_json()))

@app.route('/api/chart_data_fast')
def get_chart_data_fast():
    """Get chart data optimized for Lightweight Charts (TradingView)"""
    symbol = request.args.get('symbol', 'BTC/USDT')
    timeframe = request.args.get('timeframe', '1d')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    show_signals = request.args.get('show_signals', 'true') == 'true'
    
    # Base query for OHLCV data
    query = """
    SELECT 
        DATE(o.timestamp) as timestamp,
        MAX(o.open) as open, 
        MAX(o.high) as high, 
        MAX(o.low) as low, 
        MAX(o.close) as close, 
        MAX(o.volume) as volume
    FROM ohlcv_data o
    JOIN symbols s ON o.symbol_id = s.id
    JOIN timeframes t ON o.timeframe_id = t.id
    WHERE s.symbol = ? AND t.timeframe = ?
    """
    params = [symbol, timeframe]
    
    if start_date:
        query += " AND o.timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND o.timestamp <= ?"
        params.append(end_date)
    
    query += " GROUP BY DATE(o.timestamp)"
    query += " ORDER BY timestamp ASC"
    
    with sqlite3.connect('data/trading_data_BTC.db') as conn:
        df = pd.read_sql_query(query, conn, params=params)
        
        # Ensure proper data types
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Prepare candlestick data for Lightweight Charts
        candlesticks = []
        volume_data = []
        
        for _, row in df.iterrows():
            # Convert timestamp to Unix timestamp (seconds)
            unix_timestamp = int(row['timestamp'].timestamp())
            
            candlesticks.append({
                'time': unix_timestamp,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })
            
            volume_data.append({
                'time': unix_timestamp,
                'value': float(row['volume']),
                'color': 'rgba(0, 150, 255, 0.5)'
            })
        
        # Get signals if requested
        markers = []
        if show_signals:
            strategy = RSIMomentumDivergenceSwingStrategy()
            signals_query = strategy.get_sql_query()
            signals_query = signals_query.replace("LIMIT 100", "")
            signals_df = pd.read_sql_query(signals_query, conn)
            
            if not signals_df.empty:
                signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'], format='mixed')
                signals_df['price'] = pd.to_numeric(signals_df['price'], errors='coerce')
                signals_df['rsi'] = pd.to_numeric(signals_df['rsi'], errors='coerce')
                
                if start_date:
                    signals_df = signals_df[signals_df['timestamp'] >= pd.to_datetime(start_date)]
                if end_date:
                    signals_df = signals_df[signals_df['timestamp'] <= pd.to_datetime(end_date)]
                
                # Create markers for signals
                for _, signal in signals_df.iterrows():
                    # Find the corresponding candlestick
                    signal_date = signal['timestamp'].date()
                    for candle in candlesticks:
                        candle_date = datetime.fromtimestamp(candle['time']).date()
                        if candle_date == signal_date:
                            is_buy = 'BUY' in signal['signal_name']
                            markers.append({
                                'time': candle['time'],
                                'position': 'belowBar' if is_buy else 'aboveBar',
                                'color': 'green' if is_buy else 'red',
                                'shape': 'arrowUp' if is_buy else 'arrowDown',
                                'text': f"{signal['signal_name']}\nRSI: {signal['rsi']:.2f}"
                            })
                            break
    
    return jsonify({
        'candlesticks': candlesticks,
        'volume': volume_data,
        'markers': markers
    })

@app.route('/api/task_status')
def get_task_status():
    """Get background task status"""
    return jsonify(task_status)

@app.route('/api/symbols')
def get_symbols():
    """Get available symbols and timeframes"""
    with sqlite3.connect('data/trading_data_BTC.db') as conn:
        symbols_query = "SELECT DISTINCT symbol FROM symbols ORDER BY symbol"
        symbols = pd.read_sql_query(symbols_query, conn)['symbol'].tolist()
        
        timeframes_query = "SELECT DISTINCT timeframe FROM timeframes ORDER BY timeframe"
        timeframes = pd.read_sql_query(timeframes_query, conn)['timeframe'].tolist()
    
    return jsonify({
        'symbols': symbols,
        'timeframes': timeframes
    })

@app.route('/market_regime')
def market_regime():
    """Display market regime for BTC, ETH, and SOL"""
    return render_template('market_regime.html')

@app.route('/api/market_regime')
def get_market_regime():
    """Get current market regime for BTC, ETH, and SOL"""
    try:
        results = {}
        assets = ['BTC', 'ETH', 'SOL']
        timeframe = request.args.get('timeframe', '4h')
        periods = int(request.args.get('periods', '500'))
        
        # Load BTC data for benchmark
        btc_df = None
        with sqlite3.connect('data/trading_data_BTC.db') as conn:
            query = """
            SELECT o.timestamp, o.open, o.high, o.low, o.close, o.volume
            FROM ohlcv_data o
            JOIN symbols s ON o.symbol_id = s.id
            JOIN timeframes t ON o.timeframe_id = t.id
            WHERE s.symbol = 'BTC/USDT' AND t.timeframe = ?
            ORDER BY o.timestamp DESC
            LIMIT ?
            """
            btc_df = pd.read_sql_query(query, conn, params=(timeframe, periods))
            if not btc_df.empty:
                # Handle both ISO format and space-separated format
                btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], format='mixed', dayfirst=False)
                btc_df.set_index('timestamp', inplace=True)
                btc_df.sort_index(inplace=True)
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    btc_df[col] = pd.to_numeric(btc_df[col], errors='coerce')
        
        # Process each asset
        for asset in assets:
            try:
                # Get database path for the asset
                db_path = f'data/trading_data_{asset}.db'
                
                # Load data
                with sqlite3.connect(db_path) as conn:
                    query = """
                    SELECT o.timestamp, o.open, o.high, o.low, o.close, o.volume
                    FROM ohlcv_data o
                    JOIN symbols s ON o.symbol_id = s.id
                    JOIN timeframes t ON o.timeframe_id = t.id
                    WHERE s.symbol = ? AND t.timeframe = ?
                    ORDER BY o.timestamp DESC
                    LIMIT ?
                    """
                    symbol = f'{asset}/USDT'
                    df = pd.read_sql_query(query, conn, params=(symbol, timeframe, periods))
                
                if df.empty:
                    results[asset] = {
                        'error': f'No data available for {asset}',
                        'regime': 'Unknown',
                        'metrics': {}
                    }
                    continue
                
                # Prepare dataframe
                # Handle both ISO format and space-separated format
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=False)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Create detector
                # Use BTC as benchmark for ETH and SOL
                benchmark = btc_df if asset != 'BTC' else None
                detector = EnhancedMarketRegimeDetector(df)
                
                # Get current regime
                regime, metrics = detector.detect_market_regime()
                
                # Get regime statistics (not available in enhanced detector)
                stats = pd.DataFrame()
                
                # Calculate 24-hour price change
                price_change_24h = None
                if not df.empty:
                    current_time = df.index[-1]
                    time_24h_ago = current_time - pd.Timedelta(hours=24)
                    
                    # Find the closest data point to 24 hours ago
                    past_data = df[df.index <= time_24h_ago]
                    if not past_data.empty:
                        past_price = past_data['close'].iloc[-1]
                        current_price = df['close'].iloc[-1]
                        price_change_24h = float((current_price / past_price - 1) * 100)
                
                # Prepare results
                results[asset] = {
                    'symbol': symbol,
                    'regime': regime,
                    'metrics': metrics,
                    'statistics': stats.to_dict('records') if not stats.empty else [],
                    'last_update': df.index[-1].isoformat() if not df.empty else None,
                    'current_price': float(df['close'].iloc[-1]) if not df.empty else None,
                    'price_change_24h': price_change_24h
                }
                
            except Exception as e:
                results[asset] = {
                    'error': str(e),
                    'regime': 'Error',
                    'metrics': {}
                }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market_regime_history/<asset>')
def get_market_regime_history(asset):
    """Get historical regime data for charting"""
    try:
        timeframe = request.args.get('timeframe', '4h')
        periods = int(request.args.get('periods', '1000'))
        
        # Get database path
        db_path = f'data/trading_data_{asset}.db'
        
        # Load data
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT o.timestamp, o.open, o.high, o.low, o.close, o.volume
            FROM ohlcv_data o
            JOIN symbols s ON o.symbol_id = s.id
            JOIN timeframes t ON o.timeframe_id = t.id
            WHERE s.symbol = ? AND t.timeframe = ?
            ORDER BY o.timestamp DESC
            LIMIT ?
            """
            symbol = f'{asset}/USDT'
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, periods))
        
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        # Prepare dataframe
        # Handle both ISO format and space-separated format
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=False)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Load BTC as benchmark if needed
        benchmark = None
        if asset != 'BTC':
            with sqlite3.connect('data/trading_data_BTC.db') as conn:
                btc_df = pd.read_sql_query(query, conn, params=('BTC/USDT', timeframe, periods))
                if not btc_df.empty:
                    # Handle both ISO format and space-separated format
                    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], format='mixed', dayfirst=False)
                    btc_df.set_index('timestamp', inplace=True)
                    btc_df.sort_index(inplace=True)
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        btc_df[col] = pd.to_numeric(btc_df[col], errors='coerce')
                    benchmark = btc_df
        
        # Create detector and get history
        detector = EnhancedMarketRegimeDetector(df)
        
        # Note: Enhanced detector doesn't have get_regime_history method
        # We'll use the history method if available
        regime_history = detector.get_regime_history() if hasattr(detector, 'get_regime_history') else pd.DataFrame()
        
        # Prepare data for frontend
        history_data = []
        for idx, row in regime_history.iterrows():
            history_data.append({
                'timestamp': idx.isoformat(),
                'regime': row['regime'],
                'adx': row.get('adx', None),
                'rsi': row.get('rsi', None),
                'volatility_regime': row.get('volatility_regime', 'Normal')
            })
        
        # Get price data for chart
        price_data = []
        for idx, row in df.iterrows():
            price_data.append({
                'timestamp': idx.isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        return jsonify({
            'regime_history': history_data,
            'price_data': price_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, port=5000)
