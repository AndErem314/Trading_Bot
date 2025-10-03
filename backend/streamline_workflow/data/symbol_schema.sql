-- Per-Symbol Trading Database Schema
-- This schema is applied to each symbol-specific database:
-- trading_data_BTC.db, trading_data_ETH.db, trading_data_SOL.db
-- Purpose: Store OHLCV data and Ichimoku indicators for a specific symbol

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS ichimoku_data;
DROP TABLE IF EXISTS ohlcv_data;

-- Table 1: OHLCV Data (Raw price and volume data)
-- Note: Since each database is symbol-specific, we store the symbol as a constant for reference
CREATE TABLE ohlcv_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    open DECIMAL(20, 8) NOT NULL CHECK(open > 0),
    high DECIMAL(20, 8) NOT NULL CHECK(high > 0),
    low DECIMAL(20, 8) NOT NULL CHECK(low > 0),
    close DECIMAL(20, 8) NOT NULL CHECK(close > 0),
    volume DECIMAL(20, 8) NOT NULL CHECK(volume >= 0),
    timeframe TEXT NOT NULL CHECK(timeframe IN ('1h', '4h', '1d')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure data integrity
    CHECK(high >= low),
    CHECK(high >= open),
    CHECK(high >= close),
    CHECK(low <= open),
    CHECK(low <= close),
    
    -- Prevent duplicate entries for the same timestamp and timeframe
    UNIQUE(timestamp, timeframe)
);

-- Table 2: Ichimoku Data (Calculated Ichimoku Cloud indicators)
CREATE TABLE ichimoku_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ohlcv_id INTEGER NOT NULL,
    tenkan_sen DECIMAL(20, 8),          -- Conversion Line (9-period)
    kijun_sen DECIMAL(20, 8),           -- Base Line (26-period)
    senkou_span_a DECIMAL(20, 8),       -- Leading Span A
    senkou_span_b DECIMAL(20, 8),       -- Leading Span B (52-period)
    chikou_span DECIMAL(20, 8),         -- Lagging Span (26-period)
    cloud_color TEXT CHECK(cloud_color IN ('green', 'red', NULL)),  -- green=bullish, red=bearish
    cloud_thickness DECIMAL(20, 8),     -- Distance between Span A and Span B
    price_position TEXT CHECK(price_position IN ('above_cloud', 'in_cloud', 'below_cloud', NULL)),
    trend_strength TEXT CHECK(trend_strength IN ('strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish', NULL)),
    tk_cross TEXT CHECK(tk_cross IN ('bullish_cross', 'bearish_cross', 'no_cross', NULL)), -- Tenkan/Kijun cross
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraint
    FOREIGN KEY (ohlcv_id) REFERENCES ohlcv_data(id) ON DELETE CASCADE,
    
    -- Ensure one Ichimoku calculation per OHLCV record
    UNIQUE(ohlcv_id)
);

-- Metadata table to store symbol-specific information
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_ohlcv_timeframe ON ohlcv_data(timeframe);
CREATE INDEX idx_ohlcv_timestamp ON ohlcv_data(timestamp);
CREATE INDEX idx_ohlcv_timestamp_timeframe ON ohlcv_data(timestamp, timeframe);
CREATE INDEX idx_ichimoku_ohlcv ON ichimoku_data(ohlcv_id);
CREATE INDEX idx_ichimoku_cloud_color ON ichimoku_data(cloud_color);
CREATE INDEX idx_ichimoku_trend ON ichimoku_data(trend_strength);
CREATE INDEX idx_ichimoku_updated ON ichimoku_data(updated_at);

-- Create views for easy access to combined data
-- View 1: Complete OHLCV and Ichimoku data
CREATE VIEW IF NOT EXISTS ohlcv_ichimoku_view AS
SELECT 
    o.id,
    o.timestamp,
    o.open,
    o.high,
    o.low,
    o.close,
    o.volume,
    o.timeframe,
    i.tenkan_sen,
    i.kijun_sen,
    i.senkou_span_a,
    i.senkou_span_b,
    i.chikou_span,
    i.cloud_color,
    i.cloud_thickness,
    i.price_position,
    i.trend_strength,
    i.tk_cross,
    o.created_at as ohlcv_created_at,
    i.created_at as ichimoku_created_at,
    i.updated_at as ichimoku_updated_at
FROM ohlcv_data o
LEFT JOIN ichimoku_data i ON o.id = i.ohlcv_id
ORDER BY o.timeframe, o.timestamp DESC;

-- View 2: Latest data per timeframe
CREATE VIEW IF NOT EXISTS latest_data_view AS
SELECT 
    o.timeframe,
    MAX(o.timestamp) as latest_timestamp,
    COUNT(o.id) as total_records,
    MIN(o.timestamp) as earliest_timestamp,
    (julianday(MAX(o.timestamp)) - julianday(MIN(o.timestamp))) as days_of_data
FROM ohlcv_data o
GROUP BY o.timeframe;

-- View 3: Ichimoku signals summary
CREATE VIEW IF NOT EXISTS ichimoku_signals_view AS
SELECT 
    o.timestamp,
    o.timeframe,
    o.close as current_price,
    i.trend_strength,
    i.cloud_color,
    i.price_position,
    i.tk_cross,
    CASE 
        WHEN i.trend_strength IN ('strong_bullish', 'bullish') 
             AND i.cloud_color = 'green' 
             AND i.price_position = 'above_cloud' THEN 'STRONG_BUY'
        WHEN i.trend_strength = 'bullish' 
             AND (i.cloud_color = 'green' OR i.price_position = 'above_cloud') THEN 'BUY'
        WHEN i.trend_strength IN ('strong_bearish', 'bearish') 
             AND i.cloud_color = 'red' 
             AND i.price_position = 'below_cloud' THEN 'STRONG_SELL'
        WHEN i.trend_strength = 'bearish' 
             AND (i.cloud_color = 'red' OR i.price_position = 'below_cloud') THEN 'SELL'
        ELSE 'NEUTRAL'
    END as signal
FROM ohlcv_data o
INNER JOIN ichimoku_data i ON o.id = i.ohlcv_id
WHERE i.trend_strength IS NOT NULL
ORDER BY o.timeframe, o.timestamp DESC;

-- Trigger to update the updated_at timestamp in ichimoku_data
CREATE TRIGGER update_ichimoku_timestamp 
AFTER UPDATE ON ichimoku_data
FOR EACH ROW
BEGIN
    UPDATE ichimoku_data SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Trigger to update metadata timestamp
CREATE TRIGGER update_metadata_timestamp 
AFTER UPDATE ON metadata
FOR EACH ROW
BEGIN
    UPDATE metadata SET updated_at = CURRENT_TIMESTAMP WHERE key = NEW.key;
END;