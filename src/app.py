import os
import sys
import json
import sqlite3
import numpy as np
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
import aiohttp
import asyncio
import logging
import traceback
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from datetime import datetime
import signal

# ─────────────────────────────────────────────────────────────────────────────
# 1) Project path setup - needs to be first
# ─────────────────────────────────────────────────────────────────────────────
# Get the project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from project root first
load_dotenv(PROJECT_ROOT / ".env")

# Define paths and constants
# Handle both local and serverless environments
if os.getenv("VERCEL"):
    # In Vercel serverless environment - try different path approaches
    possible_paths = [
        Path("/var/task/api/knowledge_base_compressed.db"),
        Path("/var/task/data/knowledge_base_compressed.db"),
        PROJECT_ROOT / "api" / "knowledge_base_compressed.db",
        PROJECT_ROOT / "data" / "knowledge_base_compressed.db",
        Path("./api/knowledge_base_compressed.db"),
        Path("./data/knowledge_base_compressed.db"),
        Path("api/knowledge_base_compressed.db"),
        Path("data/knowledge_base_compressed.db")
    ]
    
    DB_PATH = None
    for path in possible_paths:
        if path.exists():
            DB_PATH = path
            break
    
    if DB_PATH is None:
        # Default to the expected path even if it doesn't exist yet
        DB_PATH = PROJECT_ROOT / "data" / "knowledge_base_compressed.db"
else:
    # Local development
    DB_PATH = PROJECT_ROOT / "data" / "knowledge_base_compressed.db"

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.68"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "15"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "4"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# API configuration
API_KEY = os.getenv("API_KEY")
AIPIPE_BASE_URL = "https://aipipe.org"
AIPIPE_OPENAI_URL = f"{AIPIPE_BASE_URL}/openai/v1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Enhanced logging configuration with file and console output
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging():
    """Setup comprehensive logging with file rotation and structured format"""
    # Remove all existing handlers to avoid duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s() | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    
    # Only add file handlers in non-serverless environments
    if not os.getenv("VERCEL"):
        # Create logs directory relative to project root
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # File handler for detailed logging
        file_handler = logging.FileHandler(
            log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Error file handler for errors only
        error_handler = logging.FileHandler(
            log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
    
    return root_logger

logger = setup_logging()

# Validate paths and log configuration
logger.info("🚀 Starting RAG Query API application")
logger.info(f"📁 Project root: {PROJECT_ROOT}")
logger.info(f"📊 Database path: {DB_PATH}")
logger.info(f"📝 Database exists: {DB_PATH.exists()}")

# Ensure data directory exists
DB_PATH.parent.mkdir(exist_ok=True)
logger.info(f"📂 Data directory: {DB_PATH.parent}")

logger.info(f"📊 Configuration loaded:")
logger.info(f"   - Similarity threshold: {SIMILARITY_THRESHOLD}")
logger.info(f"   - Max results: {MAX_RESULTS}")
logger.info(f"   - Max context chunks: {MAX_CONTEXT_CHUNKS}")
logger.info(f"   - Request timeout: {REQUEST_TIMEOUT}")
logger.info(f"   - Embedding model: {EMBEDDING_MODEL}")
logger.info(f"   - Chat model: {CHAT_MODEL}")

# ─────────────────────────────────────────────────────────────────────────────
# 3) Request and response models
# ─────────────────────────────────────────────────────────────────────────────

logger.info(f"� Project root: {PROJECT_ROOT}")
logger.info(f"📊 Database path: {DB_PATH}")
logger.info(f"📝 Database exists: {DB_PATH.exists()}")

# Ensure data directory exists
DB_PATH.parent.mkdir(exist_ok=True)
logger.info(f"📂 Data directory: {DB_PATH.parent}")

# ─────────────────────────────────────────────────────────────────────────────
logger.info(f"   - Similarity threshold: {SIMILARITY_THRESHOLD}")
logger.info(f"   - Max results: {MAX_RESULTS}")
logger.info(f"   - Max context chunks: {MAX_CONTEXT_CHUNKS}")
logger.info(f"   - Request timeout: {REQUEST_TIMEOUT}s")
logger.info(f"   - Max retries: {MAX_RETRIES}")
logger.info(f"   - Embedding model: {EMBEDDING_MODEL}")
logger.info(f"   - Chat model: {CHAT_MODEL}")

if not API_KEY:
    logger.error("❌ CRITICAL: API_KEY environment variable not set!")
    logger.error("   Please set your AIPipe API token in the API_KEY environment variable")
    logger.error("   Get your token from: https://aipipe.org/login")
    raise SystemExit("API_KEY environment variable is required")
else:
    # Mask the API key for logging
    masked_key = f"{API_KEY[:8]}...{API_KEY[-4:]}" if len(API_KEY) > 12 else "***"
    logger.info(f"✅ AIPipe API key configured: {masked_key}")

# ─────────────────────────────────────────────────────────────────────────────
# 3) Enhanced Pydantic models with validation
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    image: Optional[str] = Field(None, description="Base64 encoded image (without data URL prefix)")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or just whitespace')
        return v.strip()
    
    @validator('image')
    def validate_image(cls, v):
        if v is not None:
            # Basic validation for base64 string
            if not v or len(v) < 100:  # Very basic check
                raise ValueError('Image must be a valid base64 string')
        return v

class LinkInfo(BaseModel):
    url: str = Field(..., description="URL of the source")
    text: str = Field(..., description="Descriptive text for the link")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    links: List[LinkInfo] = Field(default_factory=list, description="Source links")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

class HealthResponse(BaseModel):
    status: str
    database: str
    api_key_set: bool
    aipipe_connection: str
    discourse_chunks: int = 0
    markdown_chunks: int = 0
    discourse_embeddings: int = 0
    markdown_embeddings: int = 0
    timestamp: str
    uptime_seconds: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 4) Initialize FastAPI app with enhanced configuration
# ─────────────────────────────────────────────────────────────────────────────
app_start_time = datetime.now()

app = FastAPI(
    title="RAG Knowledge Base API",
    description="Enhanced RAG API with AIPipe integration for knowledge base queries",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"]
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    
    # Log request
    logger.info(f"📨 {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    
    try:
        response = await call_next(request)
        process_time = (datetime.now() - start_time).total_seconds()
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        logger.info(f"📤 {request.method} {request.url.path} -> {response.status_code} ({process_time:.3f}s)")
        
        return response
    except Exception as e:
        process_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"💥 {request.method} {request.url.path} -> ERROR ({process_time:.3f}s): {str(e)}")
        raise

logger.info("✅ FastAPI application initialized with enhanced middleware")

# ─────────────────────────────────────────────────────────────────────────────
# 5) Enhanced database setup with connection pooling and error handling
# ─────────────────────────────────────────────────────────────────────────────
def get_db_connection():
    """Get database connection with enhanced error handling and logging"""
    try:
        if not DB_PATH.exists():
            if os.getenv("VERCEL"):
                logger.error(f"❌ Database file {DB_PATH} does not exist in serverless environment")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database file not found in deployment"
                )
            logger.warning(f"⚠️  Database file {DB_PATH} does not exist - will be created on first use")
          # In serverless environments, use read-only mode with additional constraints
        if os.getenv("VERCEL"):
            # Try multiple connection approaches for Vercel
            connection_attempts = [
                # Attempt 1: Read-only with immutable flag
                f"file:{DB_PATH}?mode=ro&immutable=1",
                # Attempt 2: Read-only without immutable
                f"file:{DB_PATH}?mode=ro",
                # Attempt 3: Direct file path
                str(DB_PATH)
            ]
            
            conn = None
            last_error = None
            
            for i, conn_str in enumerate(connection_attempts):
                try:
                    logger.info(f"🔗 Attempting database connection {i+1}: {conn_str}")
                    if i < 2:  # URI connections
                        conn = sqlite3.connect(conn_str, uri=True, timeout=30.0)
                    else:  # Direct path
                        conn = sqlite3.connect(conn_str, timeout=30.0)
                    
                    # Test the connection with a simple query
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    logger.info(f"✅ Database connection {i+1} successful")
                    break
                    
                except sqlite3.Error as e:
                    logger.warning(f"⚠️  Connection attempt {i+1} failed: {e}")
                    last_error = e
                    if conn:
                        conn.close()
                    conn = None
                    continue
            
            if conn is None:
                raise sqlite3.Error(f"All connection attempts failed. Last error: {last_error}")
                
        else:
            conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
            # Enable WAL mode for better concurrent access (not available in read-only mode)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
        
        conn.row_factory = sqlite3.Row
        
        return conn
    except sqlite3.Error as e:
        logger.error(f"❌ Database connection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Database connection error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Unexpected database error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

def initialize_database():
    """Initialize database with proper schema and indexing"""
    logger.info("🗄️  Initializing database...")
    
    try:
        # Add debugging information for Vercel
        if os.getenv("VERCEL"):
            logger.info(f"📁 Project root: {PROJECT_ROOT}")
            logger.info(f"📊 Database path: {DB_PATH}")
            logger.info(f"📝 Database exists: {DB_PATH.exists() if DB_PATH else False}")
              # List files in data directory for debugging
            data_dir = PROJECT_ROOT / "data"
            if data_dir.exists():
                logger.info(f"📂 Data directory: {data_dir}")
                try:
                    files = list(data_dir.iterdir())
                    logger.info(f"📁 Files in data directory: {[f.name for f in files]}")
                except Exception as e:
                    logger.error(f"❌ Error listing data directory: {e}")
            else:
                logger.warning(f"⚠️  Data directory does not exist: {data_dir}")
                
            # Check if any of the possible database paths exist (for debugging)
            search_paths = [
                "/var/task/api/knowledge_base_compressed.db",
                "/var/task/data/knowledge_base_compressed.db",
                "./api/knowledge_base_compressed.db",
                "./data/knowledge_base_compressed.db",
                "api/knowledge_base_compressed.db",
                "data/knowledge_base_compressed.db"
            ]
            
            found_db = False
            for search_path in search_paths:
                if Path(search_path).exists():
                    logger.info(f"✅ Found database at: {search_path}")
                    found_db = True
                    break
            
            if not found_db:
                logger.warning("⚠️  Database file not found in any expected location")
        
        # Check if database file exists
        db_exists = DB_PATH and DB_PATH.exists()
        
        if not db_exists:
            if os.getenv("VERCEL"):
                logger.error("❌ Database file missing in serverless environment")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database file not found in deployment"
                )
            logger.info(f"📁 Creating new database file: {DB_PATH}")
        else:
            logger.info(f"📂 Using existing database: {DB_PATH}")
          # In serverless, we only need to verify the database exists and is readable
        if os.getenv("VERCEL"):
            # Use the same connection logic as get_db_connection()
            connection_attempts = [
                f"file:{DB_PATH}?mode=ro&immutable=1",
                f"file:{DB_PATH}?mode=ro",
                str(DB_PATH)
            ]
            
            conn = None
            last_error = None
            
            for i, conn_str in enumerate(connection_attempts):
                try:
                    logger.info(f"🔗 Database init attempt {i+1}: {conn_str}")
                    if i < 2:  # URI connections
                        conn = sqlite3.connect(conn_str, uri=True, timeout=30.0)
                    else:  # Direct path
                        conn = sqlite3.connect(conn_str, timeout=30.0)
                    
                    # Test with a simple query first
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    logger.info(f"✅ Database init connection {i+1} successful")
                    break
                    
                except sqlite3.Error as e:
                    logger.warning(f"⚠️  Init connection attempt {i+1} failed: {e}")
                    last_error = e
                    if conn:
                        conn.close()
                    conn = None
                    continue
            
            if conn is None:
                logger.error(f"❌ All database connection attempts failed: {last_error}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database connection error: {str(last_error)}"
                )
            
            cursor = conn.cursor()
            
            # Just verify we can read from key tables
            try:
                cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
                discourse_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM markdown_chunks") 
                markdown_count = cursor.fetchone()[0]
                
                conn.close()
                
                logger.info(f"✅ Database verification successful:")
                logger.info(f"   - Discourse chunks: {discourse_count}")
                logger.info(f"   - Markdown chunks: {markdown_count}")
                
                return {
                    "discourse_chunks": discourse_count,
                    "markdown_chunks": markdown_count,
                    "discourse_embeddings": 0,  # Not counted in serverless
                    "markdown_embeddings": 0    # Not counted in serverless
                }
                
            except sqlite3.OperationalError as e:
                conn.close()
                logger.error(f"❌ Database verification failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database verification error: {str(e)}"
                )
        
        # For local development, continue with full initialization
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS discourse_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            topic_id INTEGER,
            topic_title TEXT,
            post_number INTEGER,
            author TEXT,
            created_at TEXT,
            likes INTEGER,
            chunk_index INTEGER,
            content TEXT,
            url TEXT,
            embedding BLOB,
            reply_to_post_number INTEGER DEFAULT 0,
            created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_title TEXT,
            original_url TEXT,
            downloaded_at TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB,
            created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Add missing columns if they don't exist
        try:
            cursor.execute("ALTER TABLE discourse_chunks ADD COLUMN reply_to_post_number INTEGER DEFAULT 0")
            logger.info("✅ Added reply_to_post_number column to discourse_chunks")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE discourse_chunks ADD COLUMN created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
            cursor.execute("ALTER TABLE discourse_chunks ADD COLUMN updated_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
            logger.info("✅ Added timestamp columns to discourse_chunks")
        except sqlite3.OperationalError:
            pass  # Columns already exist
        
        try:
            cursor.execute("ALTER TABLE markdown_chunks ADD COLUMN created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
            cursor.execute("ALTER TABLE markdown_chunks ADD COLUMN updated_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
            logger.info("✅ Added timestamp columns to markdown_chunks")
        except sqlite3.OperationalError:
            pass  # Columns already exist
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_discourse_post_id ON discourse_chunks(post_id)",
            "CREATE INDEX IF NOT EXISTS idx_discourse_topic_id ON discourse_chunks(topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_discourse_reply_to ON discourse_chunks(reply_to_post_number)",
            "CREATE INDEX IF NOT EXISTS idx_discourse_embedding ON discourse_chunks(embedding) WHERE embedding IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_markdown_title ON markdown_chunks(doc_title)",
            "CREATE INDEX IF NOT EXISTS idx_markdown_embedding ON markdown_chunks(embedding) WHERE embedding IS NOT NULL"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.Error as e:
                logger.warning(f"⚠️  Could not create index: {e}")
        
        conn.commit()
        
        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embeddings = cursor.fetchone()[0]
        
        conn.close()
        
        logger.info(f"📊 Database statistics:")
        logger.info(f"   - Discourse chunks: {discourse_count} ({discourse_embeddings} with embeddings)")
        logger.info(f"   - Markdown chunks: {markdown_count} ({markdown_embeddings} with embeddings)")
        logger.info("✅ Database initialization completed")
        
        return {
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}", exc_info=True)
        raise

# Initialize database stats variable
db_stats = None

def ensure_database_initialized():
    """Ensure database is initialized before use"""
    global db_stats
    if db_stats is None:
        db_stats = initialize_database()
    return db_stats

# ─────────────────────────────────────────────────────────────────────────────
# 6) Enhanced cosine similarity with error handling
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors with enhanced error handling"""
    try:
        if not vec1 or not vec2:
            logger.warning("⚠️  Empty vectors provided for cosine similarity")
            return 0.0
        
        if len(vec1) != len(vec2):
            logger.warning(f"⚠️  Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
            return 0.0
        
        v1, v2 = np.array(vec1, dtype=np.float32), np.array(vec2, dtype=np.float32)
        
        # Check for zero vectors
        if np.allclose(v1, 0) or np.allclose(v2, 0):
            return 0.0
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        similarity = dot_product / (norm_v1 * norm_v2)
        
        # Ensure result is in valid range [-1, 1]
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"❌ Error calculating cosine similarity: {e}", exc_info=True)
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 7) Enhanced AIPipe connection testing and embedding function
# ─────────────────────────────────────────────────────────────────────────────
async def test_aipipe_connection() -> Dict[str, str]:
    """Test AIPipe connection and return status information"""
    try:
        logger.info("🔗 Testing AIPipe connection...")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            # Test basic connectivity with a simple embedding request
            test_url = f"{AIPIPE_OPENAI_URL}/embeddings"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": EMBEDDING_MODEL,
                "input": "test connection"
            }
            
            async with session.post(test_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'data' in result and len(result['data']) > 0:
                        logger.info("✅ AIPipe connection successful")
                        return {"status": "connected", "message": "AIPipe API is accessible"}
                    else:
                        logger.warning("⚠️  AIPipe responded but with unexpected format")
                        return {"status": "warning", "message": "Unexpected response format"}
                elif response.status == 401:
                    error_text = await response.text()
                    logger.error(f"❌ AIPipe authentication failed: {error_text}")
                    return {"status": "auth_failed", "message": "Invalid API key"}
                elif response.status == 429:
                    logger.warning("⚠️  AIPipe rate limit reached")
                    return {"status": "rate_limited", "message": "Rate limit exceeded"}
                else:
                    error_text = await response.text()
                    logger.error(f"❌ AIPipe connection failed with status {response.status}: {error_text}")
                    return {"status": "error", "message": f"HTTP {response.status}: {error_text}"}
                    
    except asyncio.TimeoutError:
        logger.error("❌ AIPipe connection timeout")
        return {"status": "timeout", "message": "Connection timeout"}
    except Exception as e:
        logger.error(f"❌ AIPipe connection error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

async def get_embedding(text: str, max_retries: int = None) -> List[float]:
    """Get embedding from AIPipe with enhanced error handling and retry logic"""
    if max_retries is None:
        max_retries = MAX_RETRIES
    
    if not API_KEY:
        logger.error("❌ API_KEY not set for embedding request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY environment variable not set"
        )
    
    if not text.strip():
        logger.warning("⚠️  Empty text provided for embedding")
        raise ValueError("Text cannot be empty")
    
    # Truncate text if too long (OpenAI has token limits)
    if len(text) > 8000:  # Conservative limit
        text = text[:8000]
        logger.warning(f"⚠️  Text truncated to 8000 characters for embedding")
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"🔄 Getting embedding (attempt {attempt + 1}/{max_retries}) for text length: {len(text)}")
            
            url = f"{AIPIPE_OPENAI_URL}/embeddings"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": EMBEDDING_MODEL,
                "input": text
            }
            
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'data' not in result or not result['data']:
                            raise ValueError("Invalid response format from AIPipe")
                        
                        embedding = result["data"][0]["embedding"]
                        
                        if not embedding or not isinstance(embedding, list):
                            raise ValueError("Invalid embedding format received")
                        
                        logger.debug(f"✅ Successfully received embedding with {len(embedding)} dimensions")
                        return embedding
                        
                    elif response.status == 429:
                        error_text = await response.text()
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"⚠️  Rate limited (attempt {attempt + 1}). Waiting {wait_time}s. Error: {error_text}")
                        await asyncio.sleep(wait_time)
                        
                    elif response.status == 401:
                        error_text = await response.text()
                        logger.error(f"❌ Authentication failed: {error_text}")
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid API key for AIPipe"
                        )
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ AIPipe API error (status {response.status}, attempt {attempt + 1}): {error_text}")
                        
                        if attempt + 1 >= max_retries:
                            raise HTTPException(
                                status_code=response.status,
                                detail=f"AIPipe API error after {max_retries} attempts: {error_text}"
                            )
                        
                        await asyncio.sleep(1 * (attempt + 1))  # Linear backoff for other errors
                        
        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout on embedding request (attempt {attempt + 1})")
            if attempt + 1 >= max_retries:
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail=f"Embedding request timeout after {max_retries} attempts"
                )
            await asyncio.sleep(2 * (attempt + 1))
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in get_embedding (attempt {attempt + 1}): {e}", exc_info=True)
            if attempt + 1 >= max_retries:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get embedding after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(1 * (attempt + 1))
    
    # Should not reach here, but as a safety net
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to get embedding after all retry attempts"
    )

# Test AIPipe connection on startup
async def startup_aipipe_test():
    """Test AIPipe connection during startup"""
    connection_status = await test_aipipe_connection()
    logger.info(f"🔌 AIPipe connection status: {connection_status['status']} - {connection_status['message']}")
    return connection_status

# We'll call this during startup


# ─────────────────────────────────────────────────────────────────────────────
# 8) Enhanced similar content search with better error handling and logging
# ─────────────────────────────────────────────────────────────────────────────
async def find_similar_content(query_emb: List[float], conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Find similar content in database with enhanced error handling and progress tracking"""
    try:
        logger.info(f"🔍 Searching for similar content (threshold: {SIMILARITY_THRESHOLD})")
        cursor = conn.cursor()
        results = []

        # --- Discourse chunks ---
        logger.debug("📊 Querying discourse_chunks table...")
        cursor.execute('''
        SELECT id, post_id, topic_id, topic_title, post_number, reply_to_post_number, 
               author, created_at, likes, chunk_index, content, url, embedding
        FROM discourse_chunks
        WHERE embedding IS NOT NULL
        ''')
        dc_rows = cursor.fetchall()
        logger.info(f"📚 Processing {len(dc_rows)} discourse chunks...")

        discourse_processed = 0
        discourse_matches = 0
        
        for i, chunk in enumerate(dc_rows):
            try:
                embedding_data = chunk["embedding"]
                if not embedding_data:
                    continue
                    
                embedding = json.loads(embedding_data)
                if not embedding or not isinstance(embedding, list):
                    logger.warning(f"⚠️  Invalid embedding format for discourse chunk {chunk['id']}")
                    continue
                
                similarity = cosine_similarity(query_emb, embedding)
                discourse_processed += 1
                
                if similarity >= SIMILARITY_THRESHOLD:
                    discourse_matches += 1
                    url = chunk["url"]
                    if not url.startswith("http"):
                        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
                    
                    results.append({
                        "source": "discourse",
                        "id": chunk["id"],
                        "post_id": chunk["post_id"],
                        "topic_id": chunk["topic_id"],
                        "post_number": chunk["post_number"],
                        "reply_to_post_number": chunk["reply_to_post_number"],
                        "title": chunk["topic_title"],
                        "url": url,
                        "content": chunk["content"],
                        "author": chunk["author"],
                        "created_at": chunk["created_at"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                
                # Progress logging for large datasets
                if (i + 1) % 1000 == 0:
                    logger.debug(f"🔄 Processed {i+1}/{len(dc_rows)} discourse rows ({discourse_matches} matches so far)")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  Invalid JSON in embedding for discourse chunk {chunk['id']}: {e}")
                continue
            except Exception as e:
                logger.error(f"❌ Error processing discourse chunk {chunk['id']}: {e}")
                continue

        logger.info(f"✅ Discourse processing complete: {discourse_processed} processed, {discourse_matches} matches")

        # --- Markdown chunks ---
        logger.debug("📊 Querying markdown_chunks table...")
        cursor.execute('''
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding
        FROM markdown_chunks
        WHERE embedding IS NOT NULL
        ''')
        md_rows = cursor.fetchall()
        logger.info(f"📄 Processing {len(md_rows)} markdown chunks...")

        markdown_processed = 0
        markdown_matches = 0
        
        for i, chunk in enumerate(md_rows):
            try:
                embedding_data = chunk["embedding"]
                if not embedding_data:
                    continue
                    
                embedding = json.loads(embedding_data)
                if not embedding or not isinstance(embedding, list):
                    logger.warning(f"⚠️  Invalid embedding format for markdown chunk {chunk['id']}")
                    continue
                
                similarity = cosine_similarity(query_emb, embedding)
                markdown_processed += 1
                
                if similarity >= SIMILARITY_THRESHOLD:
                    markdown_matches += 1
                    url = chunk["original_url"]
                    if not url or not url.startswith("http"):
                        url = f"https://docs.onlinedegree.iitm.ac.in/{chunk['doc_title']}"
                    
                    results.append({
                        "source": "markdown",
                        "id": chunk["id"],
                        "title": chunk["doc_title"],
                        "url": url,
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                
                # Progress logging for large datasets
                if (i + 1) % 1000 == 0:
                    logger.debug(f"🔄 Processed {i+1}/{len(md_rows)} markdown rows ({markdown_matches} matches so far)")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  Invalid JSON in embedding for markdown chunk {chunk['id']}: {e}")
                continue
            except Exception as e:
                logger.error(f"❌ Error processing markdown chunk {chunk['id']}: {e}")
                continue

        logger.info(f"✅ Markdown processing complete: {markdown_processed} processed, {markdown_matches} matches")

        # Sort and group results
        total_matches = len(results)
        logger.info(f"📊 Total matches found: {total_matches}")
        
        if not results:
            logger.warning("⚠️  No content found above similarity threshold")
            return []

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.debug(f"🏆 Best match: {results[0]['similarity']:.4f} from {results[0]['source']}")

        # Group by document/post and limit chunks per group
        grouped_results: Dict[str, List[Dict[str, Any]]] = {}
        for r_item in results:
            key = f"{r_item['source']}_{r_item.get('post_id', r_item.get('title'))}"
            grouped_results.setdefault(key, []).append(r_item)

        final_results = []
        for key, chunks_in_group in grouped_results.items():
            # Take top chunks from each group
            final_results.extend(chunks_in_group[:MAX_CONTEXT_CHUNKS])

        # Sort final combined list and take top N
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        final_count = min(len(final_results), MAX_RESULTS)
        
        logger.info(f"📋 Returning {final_count} results after grouping and filtering")
        if final_results:
            logger.debug(f"📊 Similarity range: {final_results[-1]['similarity']:.4f} to {final_results[0]['similarity']:.4f}")
        
        return final_results[:MAX_RESULTS]

    except sqlite3.Error as e:
        logger.error(f"❌ Database error in find_similar_content: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error during content search: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Unexpected error in find_similar_content: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content search error: {str(e)}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 9) Enhanced reply fetching with better error handling
# ─────────────────────────────────────────────────────────────────────────────
def fetch_replies_for_post(conn: sqlite3.Connection, topic_id: int, post_num: int) -> List[Dict[str, Any]]:
    """Fetch replies for a given post with enhanced error handling and logging"""
    try:
        logger.debug(f"🔍 Fetching replies for topic_id={topic_id}, post_number={post_num}")
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT post_id FROM discourse_chunks
            WHERE topic_id = ? AND reply_to_post_number = ?
        ''', (topic_id, post_num))
        
        reply_post_ids = [row["post_id"] for row in cursor.fetchall()]
        
        if not reply_post_ids:
            logger.debug(f"📭 No replies found for post_number={post_num}")
            return []
        
        logger.debug(f"📬 Found {len(reply_post_ids)} reply posts for post_number={post_num}")

        replies = []
        for r_post_id in reply_post_ids:
            try:
                cursor.execute('''
                    SELECT chunk_index, author, content, url FROM discourse_chunks
                    WHERE post_id = ? ORDER BY chunk_index ASC
                ''', (r_post_id,))
                
                chunk_rows = cursor.fetchall()
                if not chunk_rows:
                    logger.warning(f"⚠️  No chunks found for reply post_id={r_post_id}")
                    continue

                # Combine all chunks for this reply
                full_content = "".join(cr["content"] + "\n" for cr in chunk_rows)
                reply_author = chunk_rows[0]["author"]
                reply_url = chunk_rows[0]["url"]

                replies.append({
                    "post_id": r_post_id,
                    "author": reply_author,
                    "content": full_content.strip(),
                    "url": reply_url,
                    "chunk_count": len(chunk_rows)
                })
                
                logger.debug(f"✅ Built reply for post_id={r_post_id}, author={reply_author}, chunks={len(chunk_rows)}")
                
            except Exception as e:
                logger.error(f"❌ Error processing reply post_id={r_post_id}: {e}")
                continue
        
        logger.debug(f"📋 Returning {len(replies)} complete replies")
        return replies
        
    except sqlite3.Error as e:
        logger.error(f"❌ Database error in fetch_replies_for_post: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"❌ Unexpected error in fetch_replies_for_post: {e}", exc_info=True)
        return []

# ─────────────────────────────────────────────────────────────────────────────
# 10) Enhanced content enrichment with adjacent chunks and replies
# ─────────────────────────────────────────────────────────────────────────────
async def enrich_with_adjacent_chunks(conn: sqlite3.Connection, sim_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrich content with adjacent chunks and replies with enhanced error handling"""
    try:
        logger.info(f"🔧 Enriching {len(sim_chunks)} results with adjacent content...")
        cursor = conn.cursor()
        rich_chunks = []

        for idx, result_chunk in enumerate(sim_chunks):
            try:
                enriched_chunk = result_chunk.copy()
                add_content = ""
                source = result_chunk["source"]
                
                logger.debug(f"🔄 Processing {idx+1}/{len(sim_chunks)}: {source} chunk")

                if source == "discourse":
                    post_id = result_chunk["post_id"]
                    curr_chunk_idx = result_chunk["chunk_index"]
                    topic_id = result_chunk["topic_id"]
                    post_num = result_chunk["post_number"]

                    # Get adjacent chunks (previous and next)
                    adjacent_content = []
                    for offset in [-1, 1]:
                        adj_chunk_idx = curr_chunk_idx + offset
                        if adj_chunk_idx < 0 and offset == -1:
                            continue  # Skip previous if current is 0

                        try:
                            cursor.execute(
                                "SELECT content FROM discourse_chunks WHERE post_id = ? AND chunk_index = ?",
                                (post_id, adj_chunk_idx)
                            )
                            adj_chunk = cursor.fetchone()
                            
                            if adj_chunk:
                                adjacent_content.append(adj_chunk["content"])
                                logger.debug(f"  ✅ Found {'previous' if offset == -1 else 'next'} chunk at index {adj_chunk_idx}")
                                
                        except sqlite3.Error as e:
                            logger.warning(f"⚠️  Database error getting adjacent chunk: {e}")
                            continue
                    
                    if adjacent_content:
                        add_content += "\n".join(adjacent_content) + "\n"

                    # Fetch replies for this post
                    try:
                        replies = fetch_replies_for_post(conn, topic_id, post_num)
                        
                        if replies:
                            logger.debug(f"  📬 Adding {len(replies)} replies to content")
                            add_content += "\n\n---\nReplies:\n"
                            
                            for reply in replies:
                                add_content += f"\n[Reply by {reply['author']}]:\n{reply['content']}\n"
                                if reply['url']:
                                    add_content += f"Source URL: {reply['url']}\n"
                                    
                    except Exception as e:
                        logger.warning(f"⚠️  Error fetching replies for post {post_num}: {e}")

                elif source == "markdown":
                    title = result_chunk["title"]
                    curr_chunk_idx = result_chunk["chunk_index"]

                    # Get adjacent markdown chunks
                    adjacent_content = []
                    for offset in [-1, 1]:
                        adj_chunk_idx = curr_chunk_idx + offset
                        if adj_chunk_idx < 0 and offset == -1:
                            continue

                        try:
                            cursor.execute(
                                "SELECT content FROM markdown_chunks WHERE doc_title = ? AND chunk_index = ?",
                                (title, adj_chunk_idx)
                            )
                            adj_chunk = cursor.fetchone()
                            
                            if adj_chunk:
                                adjacent_content.append(adj_chunk["content"])
                                logger.debug(f"  ✅ Found {'previous' if offset == -1 else 'next'} markdown chunk at index {adj_chunk_idx}")
                                
                        except sqlite3.Error as e:
                            logger.warning(f"⚠️  Database error getting adjacent markdown chunk: {e}")
                            continue
                    
                    if adjacent_content:
                        add_content += "\n".join(adjacent_content) + "\n"

                # Add enriched content if any was found
                if add_content.strip():
                    enriched_chunk["content"] = f"{result_chunk['content']}\n\n{add_content.strip()}"
                    enriched_chunk["enriched"] = True
                    logger.debug(f"  📝 Content enriched (+{len(add_content)} chars)")
                else:
                    enriched_chunk["enriched"] = False
                    logger.debug(f"  📄 No additional content found")

                rich_chunks.append(enriched_chunk)
                
            except Exception as e:
                logger.error(f"❌ Error enriching chunk {idx+1}: {e}")
                # Still add the original chunk even if enrichment fails
                result_chunk["enriched"] = False
                rich_chunks.append(result_chunk)

        logger.info(f"✅ Content enrichment completed: {len(rich_chunks)} chunks processed")
        enriched_count = sum(1 for chunk in rich_chunks if chunk.get("enriched", False))
        logger.info(f"📊 Enrichment stats: {enriched_count}/{len(rich_chunks)} chunks were enriched")
        
        return rich_chunks
        
    except Exception as e:
        logger.error(f"❌ Critical error in enrich_with_adjacent_chunks: {e}", exc_info=True)
        # Return original chunks if enrichment completely fails
        logger.warning("⚠️  Returning original chunks due to enrichment failure")
        return sim_chunks

# ─────────────────────────────────────────────────────────────────────────────
# 11) Enhanced answer generation with AIPipe integration
# ─────────────────────────────────────────────────────────────────────────────
async def generate_answer(question: str, rich_chunks: List[Dict[str, Any]], max_retries: int = None) -> str:
    """Generate answer using AIPipe with enhanced error handling and retry logic"""
    if max_retries is None:
        max_retries = MAX_RETRIES
    
    if not API_KEY:
        logger.error("❌ API_KEY not set for answer generation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY environment variable not set"
        )

    for attempt in range(max_retries):
        try:
            logger.info(f"🤖 Generating answer (attempt {attempt + 1}/{max_retries}) for: '{question[:50]}...'")
            logger.debug(f"📚 Using {len(rich_chunks)} context chunks")
            
            # Build context from chunks
            context = ""
            context_stats = {"discourse": 0, "markdown": 0, "total_chars": 0}
            
            for r_chunk in rich_chunks:
                source_type = "Discourse post" if r_chunk["source"] == "discourse" else "Documentation"
                snippet = r_chunk["content"][:2000]  # Reasonable chunk size
                
                if len(r_chunk["content"]) > 2000:
                    snippet += "... [content truncated]"
                
                context += f"\n\n=== {source_type} ===\nURL: {r_chunk['url']}\nContent: {snippet}\n"
                
                context_stats[r_chunk["source"]] += 1
                context_stats["total_chars"] += len(snippet)
            
            logger.debug(f"📊 Context stats: {context_stats}")

            # Create the prompt
            prompt = f'''Answer the following question based ONLY on the provided context.
If you cannot answer the question based on the context, say "I don't have enough information to answer this question based on the provided sources."

Context:
{context}

Question: {question}

Instructions:
1. Provide a comprehensive yet concise answer based only on the context
2. Include a "Sources:" section listing the URLs and relevant text snippets you used
3. Make sure URLs are copied exactly from the context without modifications

Format your response as:
[Your answer here]

Sources:
1. URL: [exact_url_1], Text: [brief relevant quote or description]
2. URL: [exact_url_2], Text: [brief relevant quote or description]
[etc.]

Important: Only reference sources that were actually used in your answer.'''

            logger.debug(f"📝 Prompt length: {len(prompt)} characters")

            # Make API request to AIPipe
            url = f"{AIPIPE_OPENAI_URL}/chat/completions"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": CHAT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate answers based only on the given context. Always include exact source URLs and be precise about what information comes from which source."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1500
            }

            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if 'choices' not in result or not result['choices']:
                            raise ValueError("Invalid response format from AIPipe chat API")
                        
                        answer = result["choices"][0]["message"]["content"]
                        
                        if not answer:
                            raise ValueError("Empty answer received from AIPipe")
                        
                        logger.info(f"✅ Successfully generated answer ({len(answer)} characters)")
                        logger.debug(f"🧠 Answer preview: {answer[:100]}...")
                        
                        return answer
                        
                    elif response.status == 429:
                        error_text = await response.text()
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"⚠️  Rate limited (attempt {attempt + 1}). Waiting {wait_time}s. Error: {error_text}")
                        await asyncio.sleep(wait_time)
                        
                    elif response.status == 401:
                        error_text = await response.text()
                        logger.error(f"❌ Authentication failed for chat API: {error_text}")
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid API key for AIPipe chat API"
                        )
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ AIPipe chat API error (status {response.status}, attempt {attempt + 1}): {error_text}")
                        
                        if attempt + 1 >= max_retries:
                            raise HTTPException(
                                status_code=response.status,
                                detail=f"Chat API error after {max_retries} attempts: {error_text}"
                            )
                        
                        await asyncio.sleep(2 * (attempt + 1))  # Linear backoff for other errors
                        
        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout on chat request (attempt {attempt + 1})")
            if attempt + 1 >= max_retries:
                raise HTTPException(
                    status_code=status.HTTP_408_REQUEST_TIMEOUT,
                    detail=f"Chat request timeout after {max_retries} attempts"
                )
            await asyncio.sleep(3 * (attempt + 1))
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in generate_answer (attempt {attempt + 1}): {e}", exc_info=True)
            if attempt + 1 >= max_retries:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to generate answer after {max_retries} attempts: {str(e)}"
                )
            await asyncio.sleep(2 * (attempt + 1))
    
    # Should not reach here, but as a safety net
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to generate answer after all retry attempts"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 12) Enhanced multimodal query processing with vision capabilities
# ─────────────────────────────────────────────────────────────────────────────
async def process_multimodal_query(question: str, img_b64: Optional[str] = None) -> List[float]:
    """Process multimodal query with enhanced image handling and error recovery"""
    if not API_KEY:
        logger.error("❌ API_KEY not set for multimodal query processing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY environment variable not set"
        )

    try:
        logger.info(f"🖼️  Processing multimodal query: '{question[:50]}...', has_image: {img_b64 is not None}")
        
        if not img_b64:
            logger.debug("📝 No image provided, processing as text-only query")
            return await get_embedding(question)

        logger.info("🔍 Image provided, using vision model for description")
        
        # Prepare image data URL
        # Check if it already has a data URL prefix
        if not img_b64.startswith("data:"):
            img_data_url = f"data:image/jpeg;base64,{img_b64}"
        else:
            img_data_url = img_b64
        
        url = f"{AIPIPE_OPENAI_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",  # Vision-capable model
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Analyze this image in the context of the following question: {question}\n\nProvide a detailed description of what you see that might be relevant to answering the question. Focus on text, diagrams, charts, or any visual elements that could provide context."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": img_data_url}
                    }
                ]
            }],
            "max_tokens": 500,
            "temperature": 0.3
        }

        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT + 10)  # Extra time for vision processing
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if 'choices' not in result or not result['choices']:
                        logger.warning("⚠️  Invalid vision response format, falling back to text-only")
                        return await get_embedding(question)
                    
                    img_desc = result["choices"][0]["message"]["content"]
                    
                    if not img_desc:
                        logger.warning("⚠️  Empty image description, falling back to text-only")
                        return await get_embedding(question)
                    
                    logger.info(f"✅ Image analyzed successfully ({len(img_desc)} chars)")
                    logger.debug(f"🖼️  Image description preview: {img_desc[:200]}...")
                    
                    # Combine question with image description
                    combo_query = f"{question}\n\nImage context: {img_desc}"
                    logger.debug(f"📝 Combined query length: {len(combo_query)} characters")
                    
                    return await get_embedding(combo_query)
                    
                elif response.status == 401:
                    error_text = await response.text()
                    logger.error(f"❌ Authentication failed for vision API: {error_text}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid API key for vision processing"
                    )
                    
                elif response.status == 429:
                    error_text = await response.text()
                    logger.warning(f"⚠️  Rate limited on vision API: {error_text}")
                    logger.info("🔄 Falling back to text-only query due to rate limit")
                    return await get_embedding(question)
                    
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Vision API error (status {response.status}): {error_text}")
                    logger.info("🔄 Falling back to text-only query due to API error")
                    return await get_embedding(question)
                    
    except asyncio.TimeoutError:
        logger.error("⏰ Vision processing timeout, falling back to text-only")
        return await get_embedding(question)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in vision processing: {e}", exc_info=True)
        logger.info("🔄 Falling back to text-only query due to unexpected error")
        return await get_embedding(question)

# ─────────────────────────────────────────────────────────────────────────────
# 13) Enhanced LLM response parsing with better source extraction
# ─────────────────────────────────────────────────────────────────────────────
def parse_llm_response(llm_raw_resp: str) -> Dict[str, Any]:
    """Parse LLM response to extract answer and sources with enhanced error handling"""
    try:
        logger.debug("🔍 Parsing LLM response for answer and sources")
        
        # Try to split by various source headings
        parts = []
        source_headings = [
            "Sources:", "Source:", "References:", "Reference:", 
            "Bibliography:", "Links:", "URLs:"
        ]
        
        for heading in source_headings:
            if heading in llm_raw_resp:
                parts = llm_raw_resp.split(heading, 1)
                logger.debug(f"📋 Found '{heading}' section in response")
                break
        
        if not parts:
            logger.debug("📄 No sources section found, treating entire response as answer")
            parts = [llm_raw_resp]

        answer = parts[0].strip()
        links: List[LinkInfo] = []

        if len(parts) > 1:
            src_text = parts[1].strip()
            logger.debug(f"🔗 Processing sources section ({len(src_text)} chars)")
            
            # Enhanced regex pattern for flexible URL and text extraction
            patterns = [
                # Pattern 1: URL: [url], Text: [text]
                r"(?:URL:|url:)\s*(?:\[(.*?)\]|(\S+))(?:\s*,\s*|\s+)(?:Text:|text:)\s*(?:\[(.*?)\]|\"(.*?)\"|'(.*?)'|(.*?)(?=\n\d+\.|\nURL:|url:|$))",
                # Pattern 2: Just URLs with optional descriptions
                r"(?:https?://\S+)",
                # Pattern 3: Numbered lists with URLs
                r"\d+\.\s*(.*?)(?=\n\d+\.|\n\n|\Z)"
            ]
            
            found_links = False
            
            # Try the comprehensive pattern first
            pattern = re.compile(patterns[0], re.IGNORECASE | re.DOTALL)
            
            for match in pattern.finditer(src_text):
                try:
                    # Extract URL (from either group 1 or 2)
                    url = next((g for g in match.groups()[:2] if g), "").strip()
                    # Extract text (from groups 3-6)
                    text_content = next((g for g in match.groups()[2:] if g), "").strip()
                    
                    if not text_content:
                        text_content = "Source reference"
                    
                    # Clean up the text
                    text_content = re.sub(r'^["\'\[\]]+|["\'\[\]]+$', '', text_content)
                    
                    if url and url.startswith("http"):
                        links.append(LinkInfo(url=url, text=text_content))
                        found_links = True
                        logger.debug(f"✅ Extracted source: {url[:50]}...")
                        
                except Exception as e:
                    logger.warning(f"⚠️  Error processing regex match: {e}")
                    continue
            
            # Fallback: Simple URL extraction if regex didn't work
            if not found_links:
                logger.debug("🔄 Regex parsing failed, trying simple URL extraction")
                
                # Split by lines and look for URLs
                src_lines = src_text.split("\n")
                for line_num, line in enumerate(src_lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Remove list markers
                    line = re.sub(r'^\d+\.\s*|^-\s*|^\*\s*', '', line)
                    
                    # Find URLs in the line
                    url_matches = re.findall(r'(https?://\S+)', line)
                    
                    for url in url_matches:
                        try:
                            # Clean trailing punctuation
                            url = url.rstrip('.,;:)]')
                            
                            # Extract descriptive text (everything except the URL)
                            text_part = re.sub(r'https?://\S+', '', line).strip()
                            
                            # Clean up text
                            text_part = re.sub(r'^["\'\[\],:\s]+|["\'\[\],:\s]+$', '', text_part)
                            text_part = re.sub(r'(?:URL:|url:|Text:|text:)', '', text_part, flags=re.IGNORECASE)
                            text_part = text_part.strip()
                            
                            if not text_part:
                                text_part = f"Source {len(links) + 1}"
                            
                            links.append(LinkInfo(url=url, text=text_part))
                            found_links = True
                            logger.debug(f"✅ Simple extraction: {url[:50]}...")
                            
                        except Exception as e:
                            logger.warning(f"⚠️  Error in simple URL extraction: {e}")
                            continue

        # Log results
        logger.info(f"📊 Response parsing complete:")
        logger.info(f"   - Answer length: {len(answer)} characters")
        logger.info(f"   - Sources found: {len(links)}")
        
        if links:
            logger.debug("🔗 Sources extracted:")
            for i, link in enumerate(links[:3]):  # Log first 3 sources
                logger.debug(f"   {i+1}. {link.url[:60]}...")

        return {
            "answer": answer,
            "links": [link.model_dump() for link in links]
        }

    except Exception as e:
        logger.error(f"❌ Error parsing LLM response: {e}", exc_info=True)
        logger.warning("⚠️  Returning raw response due to parsing error")
        # Return the raw response as answer if parsing fails
        return {
            "answer": llm_raw_resp,
            "links": []
        }

# ─────────────────────────────────────────────────────────────────────────────
# 14) Enhanced main query endpoint with comprehensive error handling
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(req: QueryRequest):
    """Enhanced query endpoint with detailed logging and error handling"""
    request_start_time = datetime.now()
    
    try:
        logger.info(f"🔍 New query request: '{req.question[:50]}...' (has_image: {req.image is not None})")
        
        if not API_KEY:
            logger.error("❌ API_KEY not configured for query request")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Server configuration error",
                    "message": "API_KEY environment variable not set"
                }
            )

        # Validate question
        if not req.question.strip():
            logger.warning("⚠️  Empty question provided")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,                content={
                    "error": "Invalid input",
                    "message": "Question cannot be empty"
                }
            )

        conn = None
        try:
            # Ensure database is initialized
            logger.debug("🗄️  Ensuring database is initialized...")
            ensure_database_initialized()
            
            # Get database connection
            logger.debug("🗄️  Establishing database connection...")
            conn = get_db_connection()

            # Process multimodal query and get embedding
            logger.info("🔄 Processing query embedding...")
            embedding_start = datetime.now()
            query_emb = await process_multimodal_query(req.question, req.image)
            embedding_time = (datetime.now() - embedding_start).total_seconds()
            logger.info(f"✅ Query embedding completed ({embedding_time:.2f}s)")

            # Find similar content
            logger.info("🔍 Searching for similar content...")
            search_start = datetime.now()
            sim_chunks = await find_similar_content(query_emb, conn)
            search_time = (datetime.now() - search_start).total_seconds()
            logger.info(f"📊 Content search completed ({search_time:.2f}s) - found {len(sim_chunks)} matches")

            if not sim_chunks:
                processing_time = (datetime.now() - request_start_time).total_seconds()
                logger.warning(f"⚠️  No relevant content found ({processing_time:.2f}s total)")
                
                return QueryResponse(
                    answer="I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing your question or asking about a different topic.",
                    links=[],
                    metadata={
                        "processing_time_seconds": processing_time,
                        "chunks_found": 0,
                        "embedding_time": embedding_time,
                        "search_time": search_time
                    }
                )

            # Enrich content with adjacent chunks and replies
            logger.info("🔧 Enriching content with additional context...")
            enrich_start = datetime.now()
            rich_chunks = await enrich_with_adjacent_chunks(conn, sim_chunks)
            enrich_time = (datetime.now() - enrich_start).total_seconds()
            logger.info(f"✅ Content enrichment completed ({enrich_time:.2f}s)")

            # Generate answer using LLM
            logger.info("🤖 Generating AI answer...")
            generation_start = datetime.now()
            llm_raw_resp = await generate_answer(req.question, rich_chunks)
            generation_time = (datetime.now() - generation_start).total_seconds()
            logger.info(f"✅ Answer generation completed ({generation_time:.2f}s)")

            # Parse LLM response
            logger.debug("📝 Parsing LLM response...")
            parsed_resp = parse_llm_response(llm_raw_resp)

            # Create fallback links if LLM didn't provide sources
            if not parsed_resp["links"] and sim_chunks:
                logger.info("🔗 Creating fallback source links...")
                fallback_links: List[LinkInfo] = []
                unique_urls = set()
                
                for res_chunk in sim_chunks[:5]:  # Use top 5 chunks for fallback
                    url = res_chunk["url"]
                    if url not in unique_urls:
                        unique_urls.add(url)
                        
                        # Create meaningful snippet
                        content = res_chunk["content"]
                        if len(content) > 150:
                            snippet = content[:150] + "..."
                        else:
                            snippet = content
                        
                        # Clean snippet for display
                        snippet = re.sub(r'\s+', ' ', snippet).strip()
                        
                        fallback_links.append(LinkInfo(url=url, text=snippet))
                
                parsed_resp["links"] = [link.model_dump() for link in fallback_links]
                logger.info(f"📎 Added {len(fallback_links)} fallback source links")

            # Calculate total processing time and prepare metadata
            total_processing_time = (datetime.now() - request_start_time).total_seconds()
            
            metadata = {
                "processing_time_seconds": total_processing_time,
                "chunks_found": len(sim_chunks),
                "chunks_enriched": len(rich_chunks),
                "sources_provided": len(parsed_resp["links"]),
                "embedding_time": embedding_time,
                "search_time": search_time,
                "enrichment_time": enrich_time,
                "generation_time": generation_time,
                "has_image": req.image is not None
            }

            logger.info(f"✅ Query completed successfully ({total_processing_time:.2f}s total)")
            logger.info(f"📊 Final stats: answer={len(parsed_resp['answer'])} chars, sources={len(parsed_resp['links'])}")

            return QueryResponse(
                answer=parsed_resp["answer"],
                links=parsed_resp["links"],
                metadata=metadata
            )

        except HTTPException as e:
            # Re-raise HTTP exceptions with additional context
            processing_time = (datetime.now() - request_start_time).total_seconds()
            logger.error(f"🚨 HTTP error during query processing ({processing_time:.2f}s): {e.detail}")
            raise
            
        except Exception as e:
            processing_time = (datetime.now() - request_start_time).total_seconds()
            logger.error(f"❌ Unexpected error during query processing ({processing_time:.2f}s): {e}", exc_info=True)
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Query processing failed",
                    "message": f"An unexpected error occurred: {str(e)}",
                    "processing_time_seconds": processing_time
                }
            )
            
        finally:
            if conn:
                try:
                    conn.close()
                    logger.debug("🗄️  Database connection closed")
                except Exception as e:
                    logger.warning(f"⚠️  Error closing database connection: {e}")

    except Exception as e:
        # Catch-all for any errors in the route setup itself
        processing_time = (datetime.now() - request_start_time).total_seconds()
        logger.error(f"💥 Critical error in query endpoint ({processing_time:.2f}s): {e}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Critical server error",
                "message": "The server encountered an unexpected error",
                "processing_time_seconds": processing_time
            }
        )

# ─────────────────────────────────────────────────────────────────────────────
# 15) Enhanced health check endpoint with comprehensive diagnostics
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with AIPipe connectivity, database stats, and system info"""
    check_start_time = datetime.now()
    
    try:
        logger.info("🔍 Performing health check...")
        
        # Initialize response data
        health_data = {
            "status": "healthy",
            "database": "disconnected",
            "api_key_set": bool(API_KEY),
            "aipipe_connection": "unknown",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - app_start_time).total_seconds()
        }
        
        # Test AIPipe connection
        try:
            logger.debug("🔗 Testing AIPipe connection...")
            aipipe_status = await test_aipipe_connection()
            health_data["aipipe_connection"] = aipipe_status["status"]
            
            if aipipe_status["status"] not in ["connected", "warning"]:
                health_data["status"] = "degraded"
                logger.warning(f"⚠️  AIPipe connection issue: {aipipe_status['message']}")
            else:
                logger.debug("✅ AIPipe connection OK")
                
        except Exception as e:
            logger.error(f"❌ AIPipe health check failed: {e}")
            health_data["aipipe_connection"] = "error"
            health_data["status"] = "degraded"        # Test database connection and get statistics
        try:
            logger.debug("🗄️  Testing database connection...")
            # Ensure database is initialized
            stats = ensure_database_initialized()
            
            # Use a simple connection test
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Test basic connectivity
            cursor.execute("SELECT 1")
            cursor.fetchone()
              # Use cached stats from initialization
            health_data["discourse_chunks"] = stats.get("discourse_chunks", 0)
            health_data["markdown_chunks"] = stats.get("markdown_chunks", 0)
            health_data["discourse_embeddings"] = stats.get("discourse_embeddings", 0)
            health_data["markdown_embeddings"] = stats.get("markdown_embeddings", 0)
            
            health_data["database"] = "connected"
            logger.debug("✅ Database connection and stats OK")
            
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"❌ Database health check failed: {e}")
            health_data["database"] = "error"
            health_data["status"] = "unhealthy"
        except Exception as e:
            logger.error(f"❌ Unexpected database error: {e}")
            health_data["database"] = "error"
            health_data["status"] = "unhealthy"

        # Determine final status
        if not API_KEY:
            health_data["status"] = "unhealthy"
            logger.error("❌ API_KEY not configured")
        
        check_time = (datetime.now() - check_start_time).total_seconds()
        
        # Log health check results
        logger.info(f"🏥 Health check completed ({check_time:.3f}s):")
        logger.info(f"   - Status: {health_data['status']}")
        logger.info(f"   - Database: {health_data['database']}")
        logger.info(f"   - AIPipe: {health_data['aipipe_connection']}")
        logger.info(f"   - Uptime: {health_data['uptime_seconds']:.1f}s")
        
        if health_data["status"] == "healthy":
            return HealthResponse(**health_data)
        else:
            # Return appropriate HTTP status for unhealthy state
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE if health_data["status"] == "unhealthy" else status.HTTP_200_OK
            return JSONResponse(
                status_code=status_code,
                content=health_data
            )
            
    except Exception as e:
        check_time = (datetime.now() - check_start_time).total_seconds()
        logger.error(f"💥 Health check failed ({check_time:.3f}s): {e}", exc_info=True)
        
        error_response = {
            "status": "error",
            "database": "unknown",
            "api_key_set": bool(API_KEY),
            "aipipe_connection": "unknown",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - app_start_time).total_seconds(),
            "error": str(e)
        }
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )

# ─────────────────────────────────────────────────────────────────────────────
# 16) Application lifecycle events with enhanced startup/shutdown handling
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Enhanced startup event with comprehensive system initialization"""
    try:
        logger.info("🚀 Application startup initiated...")
        
        # Test AIPipe connection during startup
        logger.info("🔗 Testing AIPipe connectivity during startup...")
        connection_status = await startup_aipipe_test()
        
        if connection_status["status"] != "connected":
            logger.warning(f"⚠️  AIPipe connection issue detected: {connection_status['message']}")
            logger.warning("⚠️  Application will start but functionality may be limited")
        
        # Log final startup statistics
        logger.info("📊 Application startup completed:")
        logger.info(f"   - Database chunks: discourse={db_stats.get('discourse_chunks', 0)}, markdown={db_stats.get('markdown_chunks', 0)}")
        logger.info(f"   - Embeddings: discourse={db_stats.get('discourse_embeddings', 0)}, markdown={db_stats.get('markdown_embeddings', 0)}")
        logger.info(f"   - AIPipe status: {connection_status['status']}")
        logger.info(f"   - Configuration: threshold={SIMILARITY_THRESHOLD}, max_results={MAX_RESULTS}")
        
        logger.info("✅ RAG Knowledge Base API is ready to serve requests!")
        
    except Exception as e:
        logger.error(f"❌ Startup event failed: {e}", exc_info=True)
        # Don't prevent startup, but log the error
        logger.warning("⚠️  Application started with initialization errors")

@app.on_event("shutdown")
async def shutdown_event():
    """Enhanced shutdown event with cleanup"""
    try:
        logger.info("🛑 Application shutdown initiated...")
        
        # Calculate uptime
        uptime = (datetime.now() - app_start_time).total_seconds()
        logger.info(f"⏱️  Total uptime: {uptime:.1f} seconds ({uptime/3600:.2f} hours)")
        
        # Any cleanup tasks can be added here
        logger.info("🧹 Performing cleanup tasks...")
        
        logger.info("✅ Application shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ Shutdown event failed: {e}", exc_info=True)

# Signal handlers for graceful shutdown (only in non-serverless environments)
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    signal_name = signal.Signals(signum).name
    logger.info(f"📡 Received {signal_name} signal, initiating graceful shutdown...")
    
    # Let FastAPI handle the actual shutdown
    import sys
    sys.exit(0)

# Register signal handlers only in non-serverless environments
if not os.getenv("VERCEL") and not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
    try:
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        logger.info("✅ Signal handlers registered for graceful shutdown")
    except Exception as e:
        logger.warning(f"⚠️ Could not register signal handlers: {e}")
else:
    logger.info("🔧 Skipping signal handler registration in serverless environment")

# ─────────────────────────────────────────────────────────────────────────────
# 17) Enhanced server configuration and startup
# ─────────────────────────────────────────────────────────────────────────────
def create_app():
    """Factory function to create the FastAPI app"""
    return app

if __name__ == "__main__":
    try:
        # Enhanced server configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        reload = os.getenv("RELOAD", "True").lower() == "true"
        workers = int(os.getenv("WORKERS", "1"))
        log_level = os.getenv("LOG_LEVEL", "info").lower()
        
        logger.info("🖥️  Server configuration:")
        logger.info(f"   - Host: {host}")
        logger.info(f"   - Port: {port}")
        logger.info(f"   - Reload: {reload}")
        logger.info(f"   - Workers: {workers}")
        logger.info(f"   - Log level: {log_level}")
        
        # Additional uvicorn configuration
        uvicorn_config = {
            "app": "app:app",
            "host": host,
            "port": port,
            "reload": reload,
            "log_level": log_level,
            "access_log": True,
            "use_colors": True,
            "loop": "asyncio"
        }
        
        # Add workers only in production (not with reload)
        if not reload and workers > 1:
            uvicorn_config["workers"] = workers
            logger.info(f"   - Using {workers} workers (production mode)")
        elif reload:
            logger.info("   - Development mode with auto-reload")
        
        logger.info("🌟 Starting Enhanced RAG Knowledge Base API server...")
        logger.info(f"📚 Database: {DB_PATH}")
        logger.info(f"🔗 AIPipe endpoint: {AIPIPE_BASE_URL}")
        logger.info(f"🤖 Models: embedding={EMBEDDING_MODEL}, chat={CHAT_MODEL}")
        
        # Start the server
        uvicorn.run(**uvicorn_config)
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("👋 Server shutdown complete")

# Export the app for external runners (like gunicorn)
__all__ = ["app", "create_app"]
