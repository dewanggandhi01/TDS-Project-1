#!/usr/bin/env python3
"""
Simple Vercel handler for FastAPI application
"""
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Global handler instance
_handler = None

def get_handler():
    """Lazy initialization of the FastAPI handler"""
    global _handler
    if _handler is None:
        try:
            # Import only when needed
            from app import app as fastapi_app
            from mangum import Mangum
            _handler = Mangum(fastapi_app, lifespan="off")
        except Exception as e:
            print(f"Error initializing handler: {e}")
            raise
    return _handler

def app(event, context):
    """Main Vercel handler function"""
    handler = get_handler()
    return handler(event, context)

# Also export as handler for compatibility
handler = app
