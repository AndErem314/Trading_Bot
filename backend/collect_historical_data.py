"""
Script to collect all historical data from August 1st, 2020 for all trading pairs.
"""
from data_fetcher import RawDataCollector
from datetime import datetime


def collect_all_historical_data_for_all_pairs(start_date: str = '2020-08-01'):
    """
    Collect all historical data for all trading pairs since the specified start date.
    """
    # Example trading pairs and timeframes
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SOL/BTC', 'ETH/BTC']
    timeframes = ['4h', '1d']

    # Convert start_date to timestamp in milliseconds
    start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)

    collector = RawDataCollector()

    # Collect historical data for each symbol and timeframe
    for symbol in symbols:
        for timeframe in timeframes:
            collector.collect_all_historical_data(symbol, timeframe, start_time)


if __name__ == '__main__':
    collect_all_historical_data_for_all_pairs()
