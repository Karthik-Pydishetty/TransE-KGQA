#!/usr/bin/env python3
"""
Monitor the progress of the LC-QuAD preprocessing pipeline.
"""

import os
import json
import time
from datetime import datetime
from config import Config

def monitor_progress():
    """Monitor preprocessing progress by checking output files."""
    
    print("🔍 LC-QuAD Preprocessing Progress Monitor")
    print("=" * 50)
    
    # File paths
    triples_file = Config.get_output_path(Config.OUTPUT_FILES['triples'])
    stats_file = Config.get_output_path(Config.OUTPUT_FILES['statistics'])
    log_file = Config.get_output_path('preprocessing.log')
    
    while True:
        print(f"\n📊 Progress Check - {datetime.now().strftime('%H:%M:%S')}")
        
        # Check triples file
        if os.path.exists(triples_file):
            with open(triples_file, 'r') as f:
                triples_count = sum(1 for _ in f)
            print(f"📈 Triples extracted: {triples_count:,}")
        else:
            print("📈 Triples file: Not created yet")
        
        # Check statistics file
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                query_stats = stats['query_statistics']
                print(f"🔄 Queries processed: {query_stats['total']:,}")
                print(f"✅ Successful: {query_stats['successful']:,}")
                print(f"❌ Failed: {query_stats['failed']:,}")
                print(f"🔗 SSL errors: {query_stats['ssl_errors']:,}")
                print(f"⏰ Timeout errors: {query_stats['timeout_errors']:,}")
                
                if query_stats['total'] > 0:
                    success_rate = (query_stats['successful'] / query_stats['total']) * 100
                    print(f"📊 Success rate: {success_rate:.1f}%")
                
            except Exception as e:
                print(f"📊 Statistics: Error reading - {e}")
        else:
            print("📊 Statistics: Not available yet")
        
        # Check log file for recent activity
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                if lines:
                    last_log = lines[-1].strip()
                    print(f"📝 Last log: {last_log}")
            except Exception:
                pass
        
        # Check if processing is complete
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                # If we have processed both train and test data (roughly 5000 total)
                if stats['query_statistics']['total'] >= 4500:
                    print("\n🎉 Preprocessing appears to be complete!")
                    break
            except Exception:
                pass
        
        print("\n⏳ Checking again in 30 seconds... (Ctrl+C to stop)")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break

def show_final_summary():
    """Show final summary of preprocessing results."""
    
    print("\n" + "=" * 60)
    print("🎯 PHASE A: DATA PREPROCESSING - FINAL SUMMARY")
    print("=" * 60)
    
    triples_file = Config.get_output_path(Config.OUTPUT_FILES['triples'])
    stats_file = Config.get_output_path(Config.OUTPUT_FILES['statistics'])
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"📊 Total queries processed: {stats['query_statistics']['total']:,}")
        print(f"✅ Successful queries: {stats['query_statistics']['successful']:,}")
        print(f"❌ Failed queries: {stats['query_statistics']['failed']:,}")
        print(f"📈 Total unique triples: {stats['total_unique_triples']:,}")
        print(f"🔗 SSL errors: {stats['query_statistics']['ssl_errors']:,}")
        print(f"⏰ Timeout errors: {stats['query_statistics']['timeout_errors']:,}")
        
        success_rate = (stats['query_statistics']['successful'] / stats['query_statistics']['total']) * 100
        print(f"📊 Overall success rate: {success_rate:.1f}%")
    
    if os.path.exists(triples_file):
        with open(triples_file, 'r') as f:
            triples_count = sum(1 for _ in f)
        print(f"💾 Triples saved to: {triples_file}")
        print(f"📝 Ready for TransE training!")
    
    print("\n🎯 Next Steps:")
    print("1. Review the extracted triples quality")
    print("2. Proceed to Phase B: Training TransE model")
    print("3. Use the triples file with your existing TransE code")

if __name__ == "__main__":
    try:
        monitor_progress()
        show_final_summary()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")
        show_final_summary() 