#!/usr/bin/env python3
"""
Python handler called by Node.js to process FastAPI requests
"""
import sys
import os
import json
import asyncio
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

async def handle_request(request_data):
    """Handle the request using FastAPI"""
    try:
        # Import FastAPI app
        from app import app as fastapi_app
        from fastapi.testclient import TestClient
        
        # Create test client
        client = TestClient(fastapi_app)
        
        method = request_data.get('method', 'GET').upper()
        url = request_data.get('url', '/')
        headers = request_data.get('headers', {})
        body = request_data.get('body', {})
        
        # Make request to FastAPI app
        if method == 'GET':
            response = client.get(url, headers=headers)
        elif method == 'POST':
            response = client.post(url, json=body, headers=headers)
        elif method == 'OPTIONS':
            response = client.options(url, headers=headers)
        else:
            response = client.request(method, url, json=body, headers=headers)
        
        # Return response data
        return {
            'statusCode': response.status_code,
            'headers': dict(response.headers),
            'body': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': {'error': f'Internal server error: {str(e)}'}
        }

def main():
    """Main function to read from stdin and process request"""
    try:
        # Read request data from stdin
        input_data = sys.stdin.read()
        request_data = json.loads(input_data)
        
        # Process request
        response = asyncio.run(handle_request(request_data))
        
        # Output response as JSON
        print(json.dumps(response))
        
    except Exception as e:
        error_response = {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': {'error': f'Handler error: {str(e)}'}
        }
        print(json.dumps(error_response))

if __name__ == '__main__':
    main()
