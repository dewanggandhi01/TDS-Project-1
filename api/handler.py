from http.server import BaseHTTPRequestHandler
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._process_request()
    
    def do_POST(self):
        self._process_request()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _process_request(self):
        try:
            # Only import when handling request
            from app import app as fastapi_app
            from fastapi.testclient import TestClient
            
            client = TestClient(fastapi_app)
            
            # Get request data
            method = self.command
            path = self.path
            headers = dict(self.headers)
            
            # Read body for POST requests
            content_length = int(headers.get('content-length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b''
            
            # Make request to FastAPI
            if method == 'GET':
                response = client.get(path, headers=headers)
            elif method == 'POST':
                response = client.post(path, content=body, headers=headers)
            else:
                response = client.request(method, path, content=body, headers=headers)
            
            # Send response
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(response.content)
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_msg = json.dumps({"error": str(e)}).encode()
            self.wfile.write(error_msg)
