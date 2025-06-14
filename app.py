import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
import logging
import traceback
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 1) Configure logging (DEBUG level for visibility)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Constants and environment loading
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.68  # Consider making this configurable if needed
MAX_RESULTS = 15
MAX_CONTEXT_CHUNKS = 4 # Max chunks from the same document/post to provide as context
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    logger.error("API_KEY env var not set. App may not function.")

# ─────────────────────────────────────────────────────────────────────────────
# 3) Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# ─────────────────────────────────────────────────────────────────────────────
# 4) Initialize FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Query API", description="API for RAG knowledge base queries")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Database setup / migrations
# ─────────────────────────────────────────────────────────────────────────────
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"DB connection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

# If DB file does not exist, create tables. Otherwise, ensure the column exists.
if not os.path.exists(DB_PATH):
    logger.debug("DB file not found. Creating new SQLite DB.")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
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
        reply_to_post_number INTEGER DEFAULT 0
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS markdown_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_title TEXT,
        original_url TEXT,
        downloaded_at TEXT,
        chunk_index INTEGER,
        content TEXT,
        embedding BLOB
    )
    ''')
    conn.commit()
    conn.close()
else:
    logger.debug("DB file exists. Checking columns.")
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("ALTER TABLE discourse_chunks ADD COLUMN reply_to_post_number INTEGER DEFAULT 0")
        logger.info("Added reply_to_post_number to discourse_chunks.")
    except sqlite3.OperationalError:
        logger.debug("reply_to_post_number col already exists.")
    conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6) Cosine similarity helper
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity(vec1, vec2):
    try:
        v1, v2 = np.array(vec1), np.array(vec2)
        if np.all(v1 == 0) or np.all(v2 == 0): return 0.0
        dot_product = np.dot(v1, v2)
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        return dot_product / (norm_v1 * norm_v2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}", exc_info=True)
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 7) Function to get embedding from AIPipe proxy (with retries)
# ─────────────────────────────────────────────────────────────────────────────
async def get_embedding(text: str, max_retries=3):
    if not API_KEY:
        logger.error("API_KEY env var not set for get_embedding")
        raise HTTPException(status_code=500, detail="API_KEY environment variable not set")

    for attempt in range(max_retries):
        try:
            logger.debug(f"Embedding text (len: {len(text)}, attempt {attempt+1}/{max_retries}).")
            url = "https://aipipe.org/openai/v1/embeddings"
            headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
            payload = {"model": "text-embedding-3-small", "input": text}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug("Successfully received embedding.")
                        return result["data"][0]["embedding"]
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"Rate limited (attempt {attempt+1}). Retrying. Details: {error_text}")
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        error_text = await response.text()
                        logger.error(f"Embedding API error (status {response.status}, attempt {attempt+1}): {error_text}")
                        # For non-429 errors, maybe don't retry or have specific handling
                        if attempt + 1 >= max_retries: # Raise if last attempt
                             raise HTTPException(status_code=response.status, detail=f"Error getting embedding: {error_text}")
                        await asyncio.sleep(3 * (attempt + 1)) # Wait before retry for other errors

        except Exception as e:
            logger.error(f"Exception in get_embedding (attempt {attempt+1}): {e}", exc_info=True)
            if attempt + 1 >= max_retries:
                raise HTTPException(status_code=500, detail=f"Failed to get embedding after {max_retries} attempts: {e}")
            await asyncio.sleep(3 * (attempt + 1))
    # Should not be reached if max_retries > 0, but as a fallback:
    raise HTTPException(status_code=500, detail="Failed to get embedding after multiple retries.")


# ─────────────────────────────────────────────────────────────────────────────
# 8) Find similar content in both tables (discourse + markdown)
# ─────────────────────────────────────────────────────────────────────────────
async def find_similar_content(query_emb: List[float], conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    try:
        logger.debug("Finding similar content in DB.")
        cursor = conn.cursor()
        results = []

        # --- Discourse chunks ---
        logger.debug("Querying discourse_chunks.")
        cursor.execute('''
        SELECT id, post_id, topic_id, topic_title, post_number, reply_to_post_number, author, created_at,
               likes, chunk_index, content, url, embedding
        FROM discourse_chunks
        WHERE embedding IS NOT NULL
        ''')
        dc_rows = cursor.fetchall()
        logger.debug(f"Fetched {len(dc_rows)} discourse rows.")

        for i, chunk in enumerate(dc_rows):
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_emb, embedding)
                if similarity >= SIMILARITY_THRESHOLD:
                    url = chunk["url"]
                    if not url.startswith("http"):
                        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
                    results.append({
                        "source": "discourse", "id": chunk["id"], "post_id": chunk["post_id"],
                        "topic_id": chunk["topic_id"], "post_number": chunk["post_number"],
                        "reply_to_post_number": chunk["reply_to_post_number"], "title": chunk["topic_title"],
                        "url": url, "content": chunk["content"], "author": chunk["author"],
                        "created_at": chunk["created_at"], "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                if (i + 1) % 1000 == 0: logger.debug(f"Processed {i+1}/{len(dc_rows)} discourse rows.")
            except Exception as e:
                logger.error(f"Error processing discourse chunk ID {chunk['id']}: {e}")

        # --- Markdown chunks ---
        logger.debug("Querying markdown_chunks.")
        cursor.execute('''
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding
        FROM markdown_chunks
        WHERE embedding IS NOT NULL
        ''')
        md_rows = cursor.fetchall()
        logger.debug(f"Fetched {len(md_rows)} markdown rows.")

        for i, chunk in enumerate(md_rows):
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_emb, embedding)
                if similarity >= SIMILARITY_THRESHOLD:
                    url = chunk["original_url"]
                    if not url or not url.startswith("http"): # Simplified condition
                        url = f"https://docs.onlinedegree.iitm.ac.in/{chunk['doc_title']}"
                    results.append({
                        "source": "markdown", "id": chunk["id"], "title": chunk["doc_title"],
                        "url": url, "content": chunk["content"], "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
                if (i + 1) % 1000 == 0: logger.debug(f"Processed {i+1}/{len(md_rows)} markdown rows.")
            except Exception as e:
                logger.error(f"Error processing markdown chunk ID {chunk['id']}: {e}")

        logger.debug(f"Total matching chunks before grouping: {len(results)}.")
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Group by document/post and limit chunks per group
        grouped_results: Dict[str, List[Dict[str, Any]]] = {}
        for r_item in results:
            key = f"{r_item['source']}_{r_item.get('post_id', r_item.get('title'))}"
            grouped_results.setdefault(key, []).append(r_item)

        final_results = []
        for key, chunks_in_group in grouped_results.items():
            # Already sorted by similarity due to overall sort, but re-sorting per group if MAX_CONTEXT_CHUNKS is small
            # chunks_in_group.sort(key=lambda x: x["similarity"], reverse=True) # This might be redundant if results are already sorted
            final_results.extend(chunks_in_group[:MAX_CONTEXT_CHUNKS])

        # Sort final combined list and take top N
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.debug(f"Returning {min(len(final_results), MAX_RESULTS)} results after grouping/truncation.")
        return final_results[:MAX_RESULTS]

    except Exception as e:
        logger.error(f"Error in find_similar_content: {e}", exc_info=True)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# 9) New: fetch replies for a given post_number (all chunks of each reply)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_replies_for_post(conn: sqlite3.Connection, topic_id: int, post_num: int) -> List[Dict[str, Any]]:
    logger.debug(f"Fetching replies for topic_id={topic_id}, post_number={post_num}")
    cursor = conn.cursor()

    cursor.execute('''
        SELECT DISTINCT post_id FROM discourse_chunks
        WHERE topic_id = ? AND reply_to_post_number = ?
    ''', (topic_id, post_num))
    reply_post_ids = [row["post_id"] for row in cursor.fetchall()]
    logger.debug(f"Found {len(reply_post_ids)} distinct reply post_ids for post_number={post_num}")

    replies = []
    for r_post_id in reply_post_ids:
        cursor.execute('''
            SELECT chunk_index, author, content, url FROM discourse_chunks
            WHERE post_id = ? ORDER BY chunk_index ASC
        ''', (r_post_id,))
        chunk_rows = cursor.fetchall()
        if not chunk_rows: continue

        full_content = "".join(cr["content"] + "\n" for cr in chunk_rows)
        # Assuming author and url are same for all chunks of a post
        reply_author = chunk_rows[0]["author"]
        reply_url = chunk_rows[0]["url"]

        replies.append({
            "post_id": r_post_id, "author": reply_author,
            "content": full_content.strip(), "url": reply_url
        })
        logger.debug(f"  Built reply for post_id={r_post_id}, author={reply_author}, chunks={len(chunk_rows)}")
    return replies

# ─────────────────────────────────────────────────────────────────────────────
# 10) Enrich content with adjacent chunks and replies (uses new fetch_replies_for_post)
# ─────────────────────────────────────────────────────────────────────────────
async def enrich_with_adjacent_chunks(conn: sqlite3.Connection, sim_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        logger.debug(f"Enriching {len(sim_chunks)} result(s).")
        cursor = conn.cursor()
        rich_chunks = []

        for result_chunk in sim_chunks:
            enriched_chunk = result_chunk.copy()
            add_content = ""
            # logger.debug(f"Processing: src={result_chunk['source']}, id={result_chunk.get('post_id')}, chunk_idx={result_chunk.get('chunk_index')}")

            if result_chunk["source"] == "discourse":
                post_id = result_chunk["post_id"]
                curr_chunk_idx = result_chunk["chunk_index"]
                topic_id = result_chunk["topic_id"]
                post_num = result_chunk["post_number"]
                # logger.debug(f"  Discourse: topic={topic_id}, post_num={post_num}, chunk_idx={curr_chunk_idx}")

                # Adjacent chunks (previous and next)
                for offset in [-1, 1]:
                    adj_chunk_idx = curr_chunk_idx + offset
                    if adj_chunk_idx < 0 and offset == -1: continue # Skip prev if current is 0

                    cursor.execute("SELECT content FROM discourse_chunks WHERE post_id = ? AND chunk_index = ?",
                                   (post_id, adj_chunk_idx))
                    adj_chunk = cursor.fetchone()
                    if adj_chunk:
                        # logger.debug(f"    Found {'prev' if offset == -1 else 'next'} chunk at index {adj_chunk_idx}")
                        add_content += adj_chunk["content"] + "\n"
                    # else:
                        # logger.debug(f"    No {'prev' if offset == -1 else 'next'} chunk at index {adj_chunk_idx}")
                
                # Fetch ALL replies for this post_number
                replies = fetch_replies_for_post(conn, topic_id, post_num)
                # logger.debug(f"    fetch_replies_for_post returned {len(replies)} replies")
                if replies:
                    add_content += "\n\n---\nReplies:\n"
                    for reply in replies:
                        # logger.debug(f"      Appending reply from post_id={reply['post_id']}, author={reply['author']}")
                        add_content += f"\n[Reply by {reply['author']}]:\n{reply['content']}\nSource URL: {reply['url']}\n"

            elif result_chunk["source"] == "markdown":
                title = result_chunk["title"]
                curr_chunk_idx = result_chunk["chunk_index"]
                # logger.debug(f"  Markdown: title='{title}', chunk_idx={curr_chunk_idx}")

                for offset in [-1, 1]:
                    adj_chunk_idx = curr_chunk_idx + offset
                    if adj_chunk_idx < 0 and offset == -1: continue

                    cursor.execute("SELECT content FROM markdown_chunks WHERE doc_title = ? AND chunk_index = ?",
                                   (title, adj_chunk_idx))
                    adj_chunk = cursor.fetchone()
                    if adj_chunk:
                        # logger.debug(f"    Found {'prev' if offset == -1 else 'next'} md chunk at index {adj_chunk_idx}")
                        add_content += adj_chunk["content"] + "\n"
                    # else:
                        # logger.debug(f"    No {'prev' if offset == -1 else 'next'} md chunk at index {adj_chunk_idx}")
            
            if add_content:
                # logger.debug("    Appending additional content.")
                enriched_chunk["content"] = f"{result_chunk['content']}\n\n{add_content.strip()}"
            # else:
                # logger.debug("    No additional content found.")
            rich_chunks.append(enriched_chunk)

        logger.debug(f"Finished enriching. Total: {len(rich_chunks)}.")
        return rich_chunks
    except Exception as e:
        logger.error(f"Error in enrich_with_adjacent_chunks: {e}", exc_info=True)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# 11) Generate an answer via LLM (with sources)
# ─────────────────────────────────────────────────────────────────────────────
async def generate_answer(question: str, rich_chunks: List[Dict[str, Any]], max_retries=2) -> str:
    if not API_KEY:
        logger.error("API_KEY env var not set for generate_answer")
        raise HTTPException(status_code=500, detail="API_KEY environment variable not set")

    for attempt in range(max_retries):
        try:
            logger.debug(f"Generating answer for: '{question[:50]}…', Chunks: {len(rich_chunks)}")
            context = ""
            for r_chunk in rich_chunks:
                src_type = "Discourse post" if r_chunk["source"] == "discourse" else "Documentation"
                snippet = r_chunk["content"][:1500] # Keep snippet length reasonable
                context += f"\n\n{src_type} (URL: {r_chunk['url']}):\n{snippet}"
            # logger.debug(f"Combined context (first 200 chars): {context[:200].replace(chr(10), ' ')}")

            prompt = f'''Answer the following question based ONLY on the provided context.
If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Return your response in this exact format:
1. A comprehensive yet concise answer
2. A "Sources:" section that lists the URLs and relevant text snippets you used to answer

Sources must be in this exact format:
Sources:
1. URL: [exact_url_1], Text: [brief quote or description]
2. URL: [exact_url_2], Text: [brief quote or description]

Make sure the URLs are copied exactly from the context without any changes.
'''
            # logger.debug("Sending payload to LLM API.")
            url = "https://aipipe.org/openai/v1/chat/completions"
            headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant providing answers based only on context. Always include sources with exact URLs."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.debug("Received answer from LLM.")
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"LLM rate limit (attempt {attempt+1}). Retrying. Details: {error_text}")
                        await asyncio.sleep(3 * (attempt + 1))
                    else:
                        error_text = await response.text()
                        logger.error(f"LLM API error (status {response.status}, attempt {attempt+1}): {error_text}")
                        if attempt + 1 >= max_retries:
                            raise HTTPException(status_code=response.status, detail=f"Error generating answer: {error_text}")
                        await asyncio.sleep(2 * (attempt+1)) # Wait before retry
        except Exception as e:
            logger.error(f"Exception in generate_answer (attempt {attempt+1}): {e}", exc_info=True)
            if attempt + 1 >= max_retries:
                raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")
            await asyncio.sleep(2 * (attempt+1)) # Wait before retry
    raise HTTPException(status_code=500, detail="Failed to generate answer after multiple retries.")


# ─────────────────────────────────────────────────────────────────────────────
# 12) Process multimodal query (text + optional image)
# ─────────────────────────────────────────────────────────────────────────────
async def process_multimodal_query(question: str, img_b64: Optional[str]) -> List[float]:
    if not API_KEY:
        logger.error("API_KEY env var not set for process_multimodal_query")
        raise HTTPException(status_code=500, detail="API_KEY environment variable not set")

    try:
        logger.debug(f"Processing multimodal query: '{question[:50]}…', image: {img_b64 is not None}")
        if not img_b64:
            logger.debug("No image, getting text-only embedding.")
            return await get_embedding(question)

        logger.debug("Image provided. Calling Vision LLM for description.")
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
        img_data_url = f"data:image/jpeg;base64,{img_b64}" # Assuming JPEG, make configurable if other types
        payload = {
            "model": "gpt-4o-mini", # Vision capable model
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe this image in the context of the question: {question}"},
                    {"type": "image_url", "image_url": {"url": img_data_url}}
                ]
            }]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    img_desc = result["choices"][0]["message"]["content"]
                    logger.debug(f"Image description (first 100 chars): {img_desc[:100].replace(chr(10), ' ')}")
                    combo_query = f"{question}\nImage context: {img_desc}"
                    return await get_embedding(combo_query)
                else:
                    error_text = await response.text()
                    logger.error(f"Error processing image (status {response.status}): {error_text}. Falling back to text-only.")
                    return await get_embedding(question) # Fallback
    except Exception as e:
        logger.error(f"Exception in process_multimodal_query: {e}", exc_info=True)
        logger.debug("Falling back to text-only embedding due to exception.")
        return await get_embedding(question) # Fallback

# ─────────────────────────────────────────────────────────────────────────────
# 13) Parse LLM response (extract answer + sources)
# ─────────────────────────────────────────────────────────────────────────────
def parse_llm_response(llm_raw_resp: str) -> Dict[str, Any]:
    try:
        logger.debug("Parsing LLM response.")
        # Try to split by "Sources:" first, then other common headings
        parts = []
        for heading in ["Sources:", "Source:", "References:", "Reference:"]:
            if heading in llm_raw_resp:
                parts = llm_raw_resp.split(heading, 1)
                break
        if not parts: # If no heading found, assume whole response is answer
             parts = [llm_raw_resp]

        answer = parts[0].strip()
        links: List[LinkInfo] = [] # Use Pydantic model for type hint

        if len(parts) > 1:
            src_text = parts[1].strip()
            # Regex to find "URL: [url]" or "URL: url" and "Text: [text]" or "Text: "text""
            # This regex is a bit more flexible for URL and Text extraction
            pattern = re.compile(
                r"(?:URL:|url:)\s*(?:\[(.*?)\]|(\S+))" +  # URL part
                r"(?:\s*,\s*|\s+)" +                      # Separator
                r"(?:Text:|text:)\s*(?:\[(.*?)\]|\"(.*?)\"|'(.*?)'|(.*?)(?=\n\d+\.|\nURL:|$))", # Text part
                re.IGNORECASE
            )
            
            for match in pattern.finditer(src_text):
                url = next((g for g in match.groups()[:2] if g), "").strip() # First two groups for URL
                text_content = next((g for g in match.groups()[2:] if g), "Source reference").strip() # Remaining for text

                if url.startswith("http"): # Basic URL validation
                    links.append(LinkInfo(url=url, text=text_content))
            
            # Fallback for simpler list items if regex fails or for lines not matching pattern
            if not links:
                src_lines = src_text.split("\n")
                for line in src_lines:
                    line = line.strip()
                    if not line: continue
                    line = re.sub(r'^\d+\.\s*|^-\s*', '', line) # Remove list markers

                    # Try to extract URL and Text more simply if complex regex failed
                    url_match = re.search(r'(https?://\S+)', line, re.IGNORECASE)
                    if url_match:
                        url = url_match.group(1).strip().rstrip('.,;:)]') # Clean trailing chars
                        
                        # Try to get text after URL or use a default
                        text_part = line.replace(url, "").strip()
                        text_match = re.search(r'(?:Text:|text:)\s*(.*)', text_part, re.IGNORECASE)
                        text = text_match.group(1).strip() if text_match else "Relevant passage"
                        if text.startswith("[") and text.endswith("]"): text = text[1:-1]
                        if text.startswith("\"") and text.endswith("\""): text = text[1:-1]
                        
                        if url: links.append(LinkInfo(url=url, text=text if text else "Source link"))


        logger.debug(f"Parsed: answer len={len(answer)}, sources={len(links)}.")
        return {"answer": answer, "links": [link.model_dump() for link in links]} # Convert LinkInfo to dict for response

    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}", exc_info=True)
        # Return the raw response as answer if parsing fails badly
        return {"answer": llm_raw_resp, "links": []}

# ─────────────────────────────────────────────────────────────────────────────
# 14) API route: /query
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse) # Use response_model for validation
async def query_knowledge_base(req: QueryRequest): # Renamed request to req
    try:
        logger.debug(f"/query: q='{req.question[:50]}…', img: {req.image is not None}")
        if not API_KEY:
            logger.error("API_KEY not set for /query")
            return JSONResponse(status_code=500, content={"error": "API_KEY environment variable not set"})

        conn = get_db_connection()
        try:
            logger.debug("Getting query embedding.")
            query_emb = await process_multimodal_query(req.question, req.image)

            logger.debug("Finding similar content.")
            sim_chunks = await find_similar_content(query_emb, conn)
            logger.debug(f"Found {len(sim_chunks)} similar chunk(s).")

            if not sim_chunks:
                logger.debug("No relevant results found.")
                return QueryResponse(answer="I couldn't find relevant information.", links=[])

            logger.debug("Enriching content.")
            rich_chunks = await enrich_with_adjacent_chunks(conn, sim_chunks)
            logger.debug(f"Enriched to {len(rich_chunks)} chunk(s).")

            logger.debug("Generating answer via LLM.")
            llm_raw_resp = await generate_answer(req.question, rich_chunks)

            logger.debug("Parsing LLM response.")
            parsed_resp = parse_llm_response(llm_raw_resp)

            # Fallback links if LLM didn't provide any
            if not parsed_resp["links"] and sim_chunks: # Use original sim_chunks for fallback
                logger.debug("No sources from LLM. Building fallback links from top sim_chunks.")
                fallback_links: List[LinkInfo] = []
                unique_urls = set()
                for res_chunk in sim_chunks[:5]: # Iterate through original similar chunks
                    url = res_chunk["url"]
                    if url not in unique_urls:
                        unique_urls.add(url)
                        snippet = (res_chunk["content"][:100] + "...") if len(res_chunk["content"]) > 100 else res_chunk["content"]
                        fallback_links.append(LinkInfo(url=url, text=snippet))
                parsed_resp["links"] = [link.model_dump() for link in fallback_links]

            logger.debug(f"Returning: ans_len={len(parsed_resp['answer'])}, links={len(parsed_resp['links'])}.")
            return QueryResponse(answer=parsed_resp["answer"], links=parsed_resp["links"])

        except HTTPException: # Re-raise HTTPExceptions directly
            raise
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": f"Error processing query: {e}"})
        finally:
            if conn: conn.close()

    except Exception as e: # Catch-all for unexpected errors in the route itself
        logger.error(f"Unhandled exception in /query: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Unhandled exception: {e}"})

# ─────────────────────────────────────────────────────────────────────────────
# 15) Health check endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    db_status = "disconnected"
    counts = {
        "discourse_chunks": 0, "markdown_chunks": 0,
        "discourse_embeddings": 0, "markdown_embeddings": 0
    }
    try:
        conn = sqlite3.connect(DB_PATH) # Use context manager if preferred
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        counts["discourse_chunks"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        counts["markdown_chunks"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        counts["discourse_embeddings"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        counts["markdown_embeddings"] = cursor.fetchone()[0]
        
        conn.close()
        db_status = "connected"
        
        return {
            "status": "healthy", "database": db_status,
            "api_key_set": bool(API_KEY), **counts
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY), "database": db_status, **counts}
        )

# ─────────────────────────────────────────────────────────────────────────────
# 16) Run the server
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
