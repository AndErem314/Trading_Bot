#!/usr/bin/env python3
"""
Indicators Database Separation Script
Splits the current indicators_data.db into separate database files for each indicator:
1. gaussian_channel_data.db
2. bollinger_bands_data.db  
3. sma_data.db
4. ichimoku_data.db
5. macd_data.db
6. rsi_data.db
"""

import sqlite3
import os
import shutil
from datetime import datetime


def backup_indicators_database():
    """Create a backup of the indicators database before splitting."""
    backup_name = f"data/indicators_data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    shutil.copy2('data/indicators_data.db', backup_name)
    print(f"[BACKUP] Indicators database backed up to: {backup_name}")
    return backup_name


def get_indicator_tables():
    """Get all indicator tables from the current database."""
    conn = sqlite3.connect('data/indicators_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def create_individual_indicator_db(table_name, db_filename):
    """Create individual database file for specific indicator."""
    print(f"[INFO] Creating {db_filename} for {table_name} table...")
    
    # Connect to source database
    source_conn = sqlite3.connect('data/indicators_data.db')
    
    # Create new database for this indicator
    indicator_conn = sqlite3.connect(f'data/{db_filename}')
    
    try:
        # Get table structure
        cursor = source_conn.cursor()
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        create_table_sql = cursor.fetchone()[0]
        
        # Create table in new database
        indicator_conn.execute(create_table_sql)
        
        # Copy all data
        cursor.execute(f"SELECT * FROM {table_name}")
        table_data = cursor.fetchall()
        
        if table_data:
            # Get column info for proper insert
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            placeholders = ','.join(['?' for _ in columns])
            
            # Insert data into new database
            indicator_conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", table_data)
            
            # Get record count
            indicator_cursor = indicator_conn.cursor()
            indicator_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            record_count = indicator_cursor.fetchone()[0]
            
            print(f"[SUCCESS] {db_filename}: {record_count} records copied")
            
        indicator_conn.commit()
        
    except Exception as e:
        print(f"[ERROR] Failed to create {db_filename}: {e}")
        
    finally:
        source_conn.close()
        indicator_conn.close()


def verify_separation():
    """Verify that the separation was successful."""
    print("\n[VERIFICATION] Checking individual indicator databases...")
    
    indicator_files = [
        'gaussian_channel_data.db',
        'bollinger_bands_data.db',
        'sma_data.db',
        'ichimoku_data.db',
        'macd_data.db',
        'rsi_data.db'
    ]
    
    total_records = 0
    verification_passed = True
    
    for db_file in indicator_files:
        db_path = f'data/{db_file}'
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = [row[0] for row in cursor.fetchall()]
                
                if tables:
                    table_name = tables[0]  # Should be only one table per database
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    total_records += count
                    
                    print(f"✅ {db_file}: {count} records in {table_name} table")
                else:
                    print(f"❌ {db_file}: No tables found")
                    verification_passed = False
                
                conn.close()
                
            except Exception as e:
                print(f"❌ {db_file}: Error - {e}")
                verification_passed = False
        else:
            print(f"❌ {db_file}: File not found")
            verification_passed = False
    
    print(f"\n📊 Total records across all indicator databases: {total_records}")
    return verification_passed


def main():
    """Main function to execute indicator database separation."""
    print("=== INDICATOR DATABASE SEPARATION STARTED ===")
    print("Splitting indicators_data.db into individual database files")
    print()
    
    try:
        # Create backup
        backup_file = backup_indicators_database()
        
        # Get all indicator tables
        tables = get_indicator_tables()
        print(f"[INFO] Found indicator tables: {', '.join(tables)}")
        print()
        
        # Define mapping of table names to database filenames
        table_to_db_mapping = {
            'gaussian_channel_data': 'gaussian_channel_data.db',
            'bollinger_bands_data': 'bollinger_bands_data.db',
            'sma_data': 'sma_data.db',
            'ichimoku_data': 'ichimoku_data.db',
            'macd_data': 'macd_data.db',
            'rsi_data': 'rsi_data.db'
        }
        
        # Create individual databases
        for table_name in tables:
            if table_name in table_to_db_mapping:
                db_filename = table_to_db_mapping[table_name]
                create_individual_indicator_db(table_name, db_filename)
            else:
                print(f"[WARNING] Unknown table: {table_name}")
        
        # Verify separation
        if verify_separation():
            print("\n✅ INDICATOR DATABASE SEPARATION COMPLETED SUCCESSFULLY")
            print(f"📁 Files created:")
            for db_file in table_to_db_mapping.values():
                print(f"   - data/{db_file}")
            print(f"   - {backup_file} (backup)")
            print("\n🔄 Next steps:")
            print("   1. Update all module database connections")
            print("   2. Test the new database structure")
            print("   3. Remove original indicators_data.db when confident")
        else:
            print("\n❌ INDICATOR DATABASE SEPARATION VERIFICATION FAILED")
            print("Please check the databases manually before proceeding")
            
    except Exception as e:
        print(f"\n❌ INDICATOR DATABASE SEPARATION FAILED: {e}")
        print("Original database remains unchanged")


if __name__ == '__main__':
    main()
