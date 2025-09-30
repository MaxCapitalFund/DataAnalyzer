import json

def handler(request):
    """Simple test handler to verify serverless function works"""
    print(f"Test handler called with method: {request.method}")
    
    # Handle CORS
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json'
    }
    
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    if request.method != 'POST':
        return {
            'statusCode': 405,
            'headers': headers,
            'body': '{"error": "Method not allowed"}'
        }
    
    try:
        # Simple test response
        test_data = {
            'status': 'success',
            'message': 'Serverless function is working',
            'method': request.method,
            'timestamp': '2024-01-01T00:00:00Z'
        }
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(test_data)
        }
        
    except Exception as e:
        print(f"Error in test handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': str(e)})
        }
