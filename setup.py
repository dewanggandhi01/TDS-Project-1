#!/usr/bin/env python3
"""
TDS Project 1 - RAG Query API Setup Script

This script sets up the complete environment for the RAG Query API project:
1. Installs all required Python packages
2. Creates/updates the .env file with default configuration
3. Verifies the database file exists
4. Provides instructions for completing the setup

Usage:
    python setup.py

Requirements:
    - Python 3.8 or higher
    - pip package manager
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Project configuration
PROJECT_NAME = "TDS RAG Query API"
REQUIRED_PYTHON_VERSION = (3, 8)
CURRENT_DIR = Path(__file__).parent.absolute()

# Required packages
REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "python-dotenv",
    "aiohttp",
    "numpy",
    "pydantic",
    "beautifulsoup4",
    "html2text",
    "tqdm",
    "markdown", 
    "python-multipart",
    "setuptools",
    "mangum",
    "httpx"
]

# Default environment configuration
DEFAULT_ENV_CONFIG = {
    "# AIPipe Configuration": "",
    "# Get your API key from https://aipipe.org/login": "",
    "API_KEY": "YOUR_AIPIPE_API_KEY_HERE",
    "": "",
    "# Model Configuration": "",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "CHAT_MODEL": "gpt-4o-mini",
    "": "# ",
    "# Search Configuration": "",
    "SIMILARITY_THRESHOLD": "0.68",
    "MAX_RESULTS": "15",
    "MAX_CONTEXT_CHUNKS": "4",
    "": "# ",
    "# Performance Configuration": "",
    "REQUEST_TIMEOUT": "30",
    "MAX_RETRIES": "3",
    "": "# ",
    "# Server Configuration": "",
    "HOST": "0.0.0.0",
    "PORT": "8000",
    "RELOAD": "True",
    "WORKERS": "1",
    "LOG_LEVEL": "info",
    "": "# ",
    "# Database Configuration (optional - will use default if not set)": "",
    "# DB_PATH": "knowledge_base_compressed.db"
}

def print_header():
    """Print setup script header"""
    print("=" * 60)
    print(f"🚀 {PROJECT_NAME} - Setup Script")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version meets requirements"""
    print("🐍 Checking Python version...")
    current_version = sys.version_info[:2]
    
    if current_version < REQUIRED_PYTHON_VERSION:
        print(f"❌ Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}+ required, but {current_version[0]}.{current_version[1]} found")
        print(f"   Please upgrade Python to {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]} or higher")
        return False
    
    print(f"✅ Python {current_version[0]}.{current_version[1]} detected (meets requirements)")
    return True

def check_pip():
    """Check if pip is available"""
    print("\n📦 Checking pip availability...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        print("   Please install pip first")
        return False

def install_packages():
    """Install required packages"""
    print("\n📚 Installing required packages...")
    
    # Create requirements.txt if it doesn't exist
    requirements_file = CURRENT_DIR / "requirements.txt"
    if not requirements_file.exists():
        print("📝 Creating requirements.txt...")
        with open(requirements_file, 'w') as f:
            for package in REQUIRED_PACKAGES:
                f.write(f"{package}\n")
    
    try:
        # Install packages
        print("   Installing packages from requirements.txt...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True, capture_output=True, text=True)
        
        print("✅ All packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        print(f"   Error output: {e.stderr}")
        return False

def setup_env_file():
    """Create or update .env file with default configuration"""
    print("\n⚙️  Setting up environment configuration...")
    
    env_file = CURRENT_DIR / ".env"
    
    # Check if .env already exists and has API_KEY
    existing_api_key = None
    if env_file.exists():
        print("   Found existing .env file, preserving API_KEY...")
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('API_KEY=') and not line.strip().endswith('YOUR_AIPIPE_API_KEY_HERE'):
                        existing_api_key = line.split('=', 1)[1].strip()
                        break
        except Exception as e:
            print(f"   Warning: Could not read existing .env file: {e}")
    
    # Create .env file with default configuration
    try:
        with open(env_file, 'w') as f:
            for key, value in DEFAULT_ENV_CONFIG.items():
                if key.startswith('#') or key == "":
                    f.write(f"{key}\n")
                elif key == "API_KEY" and existing_api_key:
                    f.write(f"API_KEY={existing_api_key}\n")
                else:
                    f.write(f"{key}={value}\n")
        
        if existing_api_key:
            print("✅ .env file updated (preserved existing API_KEY)")
        else:
            print("✅ .env file created with default configuration")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_database():
    """Check if database file exists"""
    print("\n💾 Checking database file...")
    
    db_paths = [
        CURRENT_DIR / "data" / "knowledge_base_compressed.db",
        CURRENT_DIR / "knowledge_base_compressed.db"
    ]
    
    for db_path in db_paths:
        if db_path.exists():
            print(f"✅ Database found: {db_path}")
            return True
    
    print("⚠️  Database file not found in expected locations:")
    for db_path in db_paths:
        print(f"   - {db_path}")
    print("   Make sure the database file is in the correct location")
    return False

def check_project_structure():
    """Check if project structure is correct"""
    print("\n📁 Checking project structure...")
    
    required_files = [
        "src/app.py",
        "api/handler.py",
        "vercel.json",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = CURRENT_DIR / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Some project files are missing:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ Project structure looks good")
    return True

def print_completion_instructions():
    """Print instructions for completing the setup"""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print()
    print("1. 🔑 Set your AIPipe API Key:")
    print("   - Open the .env file")
    print("   - Replace 'YOUR_AIPIPE_API_KEY_HERE' with your actual API key")
    print("   - Get your API key from: https://aipipe.org/login")
    print()
    print("2. 💾 Ensure database is in place:")
    print("   - Make sure 'knowledge_base_compressed.db' is in the 'data/' folder")
    print()
    print("3. 🧪 Test the setup:")
    print("   Local testing:")
    print("   - Run: python src/app.py")
    print("   - Visit: http://localhost:8000/docs")
    print()
    print("   Vercel deployment:")
    print("   - Run: vercel --prod")
    print()
    print("4. 🔍 Run Promptfoo evaluation:")
    print("   - Update promptfooconfig.yaml with your Vercel URL")
    print("   - Run: npx promptfoo eval")
    print()
    print("📚 Documentation:")
    print("   - Check README.md for detailed instructions")
    print("   - See DEPLOYMENT.md for deployment guide")
    print()

def main():
    """Main setup function"""
    print_header()
    
    # Pre-flight checks
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Setup steps
    success = True
    
    # Install packages
    if not install_packages():
        success = False
    
    # Setup environment
    if not setup_env_file():
        success = False
    
    # Check project structure
    check_project_structure()  # Non-critical
    
    # Check database
    check_database()  # Non-critical
    
    if success:
        print_completion_instructions()
    else:
        print("\n❌ Setup completed with some errors")
        print("   Please review the error messages above and fix any issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
