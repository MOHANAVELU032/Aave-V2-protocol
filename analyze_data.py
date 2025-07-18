#!/usr/bin/env python3
"""
Data Analysis Script for Aave V2 Transaction Data
"""

import json
import pandas as pd
from collections import Counter

def analyze_data_structure(file_path):
    """Analyze the structure of the transaction data"""
    print(f"Analyzing data structure from {file_path}...")
    
    # Read first few records to understand structure
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total records: {len(data)}")
    
    if len(data) > 0:
        # Analyze first record
        first_record = data[0]
        print(f"\nFirst record keys: {list(first_record.keys())}")
        print(f"First record sample: {first_record}")
        
        # Analyze action types
        actions = [record.get('action', 'unknown') for record in data[:1000]]  # Sample first 1000
        action_counts = Counter(actions)
        print(f"\nAction types (sample): {dict(action_counts)}")
        
        # Analyze unique wallets
        wallets = [record.get('from', 'unknown') for record in data[:1000]]
        unique_wallets = set(wallets)
        print(f"Unique wallets in sample: {len(unique_wallets)}")
        
        # Check for timestamp format
        if 'timestamp' in first_record:
            print(f"Timestamp format: {type(first_record['timestamp'])}")
            print(f"Sample timestamp: {first_record['timestamp']}")
    
    return data

if __name__ == "__main__":
    analyze_data_structure("user-wallet-transactions.json") 