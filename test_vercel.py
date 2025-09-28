#!/usr/bin/env python3
"""
Test script to verify Vercel deployment setup
"""

import json
import os
import sys
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

def test_vercel_setup():
    """Test the Vercel serverless function setup"""
    
    # Create a minimal test CSV data
    test_csv_data = """Strategy report
Symbol: /MES:XCME
Work Time: 11/27/24 3:45 AM - 8/14/25 5:30 AM

Id;Strategy;Side;Quantity;Amount;Price;Date/Time;Trade P/L;P/L;Position;
1;TestStrategy(BTO 1);Buy to Open;1;5.0;$6,029.00;11/27/24 3:45 AM;;$2.50;5.0;
2;TestStrategy(STO -1);Sell to Close;-1;-5.0;$6,033.75;11/27/24 8:15 AM;$23.75;$23.75;0.0;"""

    # Create request body
    request_body = {
        "timeframe": "180d:15m",
        "capital": 2500.0,
        "commission": 4.04,
        "csvData": test_csv_data
    }
    
    # Create mock request
    request = MockRequest("POST", json.dumps(request_body))
    
    print("🧪 Testing Vercel serverless function...")
    print(f"📊 CSV size: {len(test_csv_data)} characters")
    
    try:
        # Call the handler
        result = handler(request)
        
        print(f"✅ API Response Status: {result['statusCode']}")
        
        if result['statusCode'] == 200:
            response_data = json.loads(result['body'])
            print(f"📈 Strategy: {response_data.get('strategy_name', 'Unknown')}")
            print(f"💰 Net Profit: ${response_data.get('metrics', {}).get('net_profit', 0):.2f}")
            print(f"📊 Total Trades: {response_data.get('trades_count', 0)}")
            print("🎉 Vercel setup test successful!")
            return True
        else:
            print(f"❌ API Error: {result['body']}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vercel_setup()
    sys.exit(0 if success else 1)
