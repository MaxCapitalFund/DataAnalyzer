#!/usr/bin/env python3

import json
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import our API handler
from api.analyze import handler

# Create a mock request object
class MockRequest:
    def __init__(self, method, body):
        self.method = method
        self.body = body

# Test with a sample CSV file
def test_api():
    # Read a sample CSV file
    csv_file = "StrategyReports /StrategyReports_MESXCME_81425.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    with open(csv_file, 'r') as f:
        csv_data = f.read()
    
    # Create request body
    request_body = {
        "timeframe": "180d:15m",
        "capital": 2500.0,
        "commission": 4.04,
        "csvData": csv_data
    }
    
    # Create mock request
    request = MockRequest("POST", json.dumps(request_body))
    
    print("🧪 Testing Python API locally...")
    print(f"📁 Using CSV file: {csv_file}")
    print(f"📊 CSV size: {len(csv_data)} characters")
    
    try:
        # Call the handler
        result = handler(request)
        
        print(f"✅ API Response Status: {result['statusCode']}")
        
        if result['statusCode'] == 200:
            response_data = json.loads(result['body'])
            print(f"📈 Strategy: {response_data.get('strategy_name', 'Unknown')}")
            print(f"💰 Net Profit: ${response_data.get('metrics', {}).get('net_profit', 0):.2f}")
            print(f"📊 Total Trades: {response_data.get('trades_count', 0)}")
            print(f"📈 Charts Generated: {len([k for k, v in response_data.get('charts', {}).items() if v])}")
            print("🎉 API test successful!")
        else:
            print(f"❌ API Error: {result['body']}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api()
