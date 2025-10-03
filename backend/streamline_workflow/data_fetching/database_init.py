"""
Database Initialization Script for Trading Bot
Creates and initializes per-symbol SQLite databases with the new schema
"""
import sqlite3
import os
import logging
from datetime import datetime
from typing import List, Dict


class DatabaseInitializer:
    """Initialize and manage per-symbol trading databases."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the database initializer.
        
        Args:
            data_dir: Directory where databases will be created.
                     Defaults to 'backend/streamline_workflow/data'
        """
        if data_dir is None:
            # Get the project root directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
            data_dir = os.path.join(project_root, 'backend/streamline_workflow/data')
        
        self.data_dir = os.path.abspath(data_dir)
        self.schema_file = os.path.join(self.data_dir, 'symbol_schema.sql')
        
        # Supported symbols
        self.symbols = ['BTC', 'ETH', 'SOL']
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_db_path(self, symbol: str) -> str:
        """Get the database path for a specific symbol."""
        return os.path.join(self.data_dir, f'trading_data_{symbol}.db')
    
    def read_schema(self) -> str:
        """Read the SQL schema from file."""
        try:
            with open(self.schema_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"Schema file not found: {self.schema_file}")
            raise
    
    def initialize_database(self, symbol: str, force_reinit: bool = False) -> bool:
        """
        Initialize a database for a specific symbol.
        
        Args:
            symbol: The cryptocurrency symbol (BTC, ETH, SOL)
            force_reinit: If True, drops existing tables and recreates them
            
        Returns:
            bool: True if successful, False otherwise
        """
        db_path = self.get_db_path(symbol)
        
        try:
            # Check if database already exists
            db_exists = os.path.exists(db_path)
            
            if db_exists and not force_reinit:
                self.logger.info(f"Database already exists for {symbol}: {db_path}")
                return self.verify_database_schema(symbol)
            
            # Create or recreate the database
            self.logger.info(f"{'Recreating' if force_reinit else 'Creating'} database for {symbol}: {db_path}")
            
            # Read schema
            schema_sql = self.read_schema()
            
            # Execute schema
            with sqlite3.connect(db_path) as conn:
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")
                
                # Execute the schema
                conn.executescript(schema_sql)
                
                # Insert metadata
                self.insert_metadata(conn, symbol)
                
                conn.commit()
            
            self.logger.info(f"Successfully initialized database for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database for {symbol}: {e}")
            return False
    
    def insert_metadata(self, conn: sqlite3.Connection, symbol: str):
        """Insert initial metadata into the database."""
        metadata_entries = [
            ('symbol', f'{symbol}/USDT'),
            ('symbol_short', symbol),
            ('created_at', datetime.now().isoformat()),
            ('schema_version', '1.0'),
            ('supported_timeframes', '1h,4h,1d'),
            ('database_type', 'per_symbol'),
            ('description', f'Trading data for {symbol}/USDT pair')
        ]
        
        for key, value in metadata_entries:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value)
            )
    
    def verify_database_schema(self, symbol: str) -> bool:
        """
        Verify that the database has the correct schema.
        
        Args:
            symbol: The cryptocurrency symbol
            
        Returns:
            bool: True if schema is valid, False otherwise
        """
        db_path = self.get_db_path(symbol)
        
        try:
            with sqlite3.connect(db_path) as conn:
                # Check if required tables exist
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('ohlcv_data', 'ichimoku_data', 'metadata')
                    ORDER BY name
                """)
                
                tables = [row[0] for row in cursor.fetchall()]
                required_tables = ['ichimoku_data', 'metadata', 'ohlcv_data']
                
                if tables != required_tables:
                    self.logger.error(f"Missing tables in {symbol} database. Found: {tables}")
                    return False
                
                # Verify OHLCV table structure
                cursor = conn.execute("PRAGMA table_info(ohlcv_data)")
                ohlcv_columns = {row[1] for row in cursor.fetchall()}
                required_ohlcv_columns = {
                    'id', 'timestamp', 'open', 'high', 'low', 
                    'close', 'volume', 'timeframe', 'created_at'
                }
                
                if not required_ohlcv_columns.issubset(ohlcv_columns):
                    self.logger.error(f"Invalid ohlcv_data schema in {symbol} database")
                    return False
                
                # Verify Ichimoku table structure
                cursor = conn.execute("PRAGMA table_info(ichimoku_data)")
                ichimoku_columns = {row[1] for row in cursor.fetchall()}
                required_ichimoku_columns = {
                    'id', 'ohlcv_id', 'tenkan_sen', 'kijun_sen', 
                    'senkou_span_a', 'senkou_span_b', 'chikou_span',
                    'cloud_color', 'cloud_thickness', 'price_position',
                    'trend_strength', 'created_at', 'updated_at'
                }
                
                if not required_ichimoku_columns.issubset(ichimoku_columns):
                    self.logger.error(f"Invalid ichimoku_data schema in {symbol} database")
                    return False
                
                self.logger.info(f"Schema verification successful for {symbol}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to verify schema for {symbol}: {e}")
            return False
    
    def initialize_all_databases(self, force_reinit: bool = False) -> Dict[str, bool]:
        """
        Initialize databases for all supported symbols.
        
        Args:
            force_reinit: If True, drops existing tables and recreates them
            
        Returns:
            Dict mapping symbol to initialization success status
        """
        results = {}
        
        self.logger.info(f"Initializing databases for symbols: {self.symbols}")
        
        for symbol in self.symbols:
            results[symbol] = self.initialize_database(symbol, force_reinit)
        
        # Summary
        successful = [s for s, success in results.items() if success]
        failed = [s for s, success in results.items() if not success]
        
        if successful:
            self.logger.info(f"Successfully initialized: {successful}")
        if failed:
            self.logger.error(f"Failed to initialize: {failed}")
        
        return results
    
    def get_database_info(self, symbol: str) -> Dict:
        """Get information about a specific database."""
        db_path = self.get_db_path(symbol)
        
        if not os.path.exists(db_path):
            return {'exists': False, 'path': db_path}
        
        try:
            info = {
                'exists': True,
                'path': db_path,
                'size_mb': os.path.getsize(db_path) / (1024 * 1024),
                'modified': datetime.fromtimestamp(os.path.getmtime(db_path)).isoformat()
            }
            
            with sqlite3.connect(db_path) as conn:
                # Get record counts
                cursor = conn.execute("SELECT COUNT(*) FROM ohlcv_data")
                info['ohlcv_records'] = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM ichimoku_data")
                info['ichimoku_records'] = cursor.fetchone()[0]
                
                # Get timeframe breakdown
                cursor = conn.execute("""
                    SELECT timeframe, COUNT(*) as count 
                    FROM ohlcv_data 
                    GROUP BY timeframe
                """)
                info['timeframe_counts'] = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Get date range
                cursor = conn.execute("""
                    SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest 
                    FROM ohlcv_data
                """)
                row = cursor.fetchone()
                if row[0]:
                    info['date_range'] = {
                        'earliest': row[0],
                        'latest': row[1]
                    }
                else:
                    info['date_range'] = None
                
                # Get metadata
                cursor = conn.execute("SELECT key, value FROM metadata")
                info['metadata'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get info for {symbol}: {e}")
            return {
                'exists': True,
                'path': db_path,
                'error': str(e)
            }
    
    def print_summary(self):
        """Print a summary of all databases."""
        print("\n" + "="*60)
        print("DATABASE SUMMARY")
        print("="*60)
        
        for symbol in self.symbols:
            info = self.get_database_info(symbol)
            
            print(f"\n{symbol}/USDT Database:")
            print(f"  Path: {info['path']}")
            
            if not info['exists']:
                print("  Status: NOT INITIALIZED")
                continue
            
            if 'error' in info:
                print(f"  Status: ERROR - {info['error']}")
                continue
            
            print(f"  Size: {info['size_mb']:.2f} MB")
            print(f"  Modified: {info['modified']}")
            print(f"  OHLCV Records: {info['ohlcv_records']:,}")
            print(f"  Ichimoku Records: {info['ichimoku_records']:,}")
            
            if info['timeframe_counts']:
                print("  Timeframes:")
                for tf, count in sorted(info['timeframe_counts'].items()):
                    print(f"    - {tf}: {count:,} records")
            
            if info['date_range']:
                print(f"  Date Range: {info['date_range']['earliest']} to {info['date_range']['latest']}")
        
        print("\n" + "="*60)


def main():
    """Main function to initialize all databases."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize trading databases')
    parser.add_argument('--force', action='store_true', 
                       help='Force reinitialize databases (drops existing data)')
    parser.add_argument('--symbol', type=str, 
                       help='Initialize specific symbol only (BTC, ETH, or SOL)')
    parser.add_argument('--verify', action='store_true',
                       help='Only verify existing databases')
    parser.add_argument('--info', action='store_true',
                       help='Show database information')
    
    args = parser.parse_args()
    
    initializer = DatabaseInitializer()
    
    if args.info:
        initializer.print_summary()
    elif args.verify:
        print("Verifying database schemas...")
        for symbol in initializer.symbols:
            if initializer.verify_database_schema(symbol):
                print(f"✅ {symbol}: Schema valid")
            else:
                print(f"❌ {symbol}: Schema invalid or missing")
    elif args.symbol:
        symbol = args.symbol.upper()
        if symbol in initializer.symbols:
            success = initializer.initialize_database(symbol, args.force)
            if success:
                print(f"✅ Successfully initialized {symbol} database")
            else:
                print(f"❌ Failed to initialize {symbol} database")
        else:
            print(f"Error: Invalid symbol. Choose from: {initializer.symbols}")
    else:
        results = initializer.initialize_all_databases(args.force)
        print("\nInitialization complete!")
        initializer.print_summary()


if __name__ == "__main__":
    main()