import sys
import os
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Import the FastAPI app
from app import app

# Export the app directly - Vercel should handle ASGI
# This is the simplest approach for FastAPI on Vercel
