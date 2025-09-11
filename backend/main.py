import os
import io
import json
import time
import hashlib
import sqlite3
import tempfile
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx2txt
import requests

# Load local .env if present (for local dev)
load_dotenv()

# ---------- CONFIG ----------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "8"))

# ---------- FastAPI app ----------
app = FastAPI(title="RAG Backend with Cerebras")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Embeddings (SentenceTransformers) ----------
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    embs = model.encode(texts, show_progress_bar=False)
    # normalize to unit vectors
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs.astype(np.float32)

# ---------- Vector store (file-based) ----------
def _session_path(session_id: str):
    p = os.path.join(DATA_DIR, session_id)
    os.makedirs(p, exist_ok=True)
    return p

def _index_files(session_id: str):
    base = _session_path(session_id)
    return {
        "emb": os.path.join(base, "embeddings.npy"),
        "meta": os.path.join(base, "meta.json"),
    }

def _load_meta(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"chunks": []}

def _save_meta(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def upsert_chunks(session_id: str, embeddings: np.ndarray, metadatas: List[Dict]):
    files = _index_files(session_id)
    meta = _load_meta(files["meta"])

    if os.path.exists(files["emb"]):
        emb_all = np.load(files["emb"])
        emb_all = np.vstack([emb_all, embeddings])
        meta["chunks"].extend(metadatas)
    else:
        emb_all = embeddings
        meta["chunks"] = metadatas

    np.save(files["emb"], emb_all)
    _save_meta(files["meta"], meta)
    return len(metadatas)

def search(session_id: str, query_embedding: np.ndarray, top_k: int = 4):
    files = _index_files(session_id)
    meta = _load_meta(files["meta"])
    if not os.path.exists(files["emb"]) or len(meta["chunks"]) == 0:
        return []

    emb_all = np.load(files["emb"])  # shape (N, dim)
    q = query_embedding.reshape(-1)
    sims = emb_all @ q  # cosine similarity (since normalized)
    idxs = np.argsort(sims)[-top_k:][::-1]
    results = []
    for rank, i in enumerate(idxs, start=1):
        if i < 0 or i >= len(meta["chunks"]):
            continue
        m = meta["chunks"][int(i)].copy()
        m["score"] = float(sims[int(i)])
        m["rank"] = int(rank)
        results.append(m)
    return results

# ---------- Text extraction & chunking ----------
def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode(errors="ignore")

def read_docx(file_bytes: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    tmp.write(file_bytes)
    tmp.close()
    try:
        return docx2txt.process(tmp.name) or ""
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

def read_any(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return read_pdf(file_bytes)
    if name.endswith(".docx"):
        return read_docx(file_bytes)
    return read_txt(file_bytes)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return [c.strip() for c in chunks if c.strip()]

# ---------- Memory (SQLite) ----------
DB_PATH = os.path.join(DATA_DIR, "memory.db")
def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS messages (session_id TEXT, role TEXT, content TEXT, ts REAL)")
    conn.commit()
    return conn

CONN = _get_conn()

def add_message(session_id: str, role: str, content: str):
    CONN.execute("INSERT INTO messages VALUES (?,?,?,?)", (session_id, role, content, time.time()))
    CONN.commit()

def get_recent_messages(session_id: str, limit: int = 8):
    cur = CONN.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY ts DESC LIMIT ?", (session_id, limit))
    rows = cur.fetchall()
    rows.reverse()
    return [{"role": r, "content": c} for (r, c) in rows]

def reset_session(session_id: str):
    CONN.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    CONN.commit()

# ---------- LLM via Cerebras ----------
def cerebras_generate(prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
    if not CEREBRAS_API_KEY:
        return "LLM not configured. Set CEREBRAS_API_KEY in backend environment."

    headers = {
        "Authorization": f"Bearer {CEREBRAS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama3.1-8b",  # Free tier model
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    try:
        r = requests.post(CEREBRAS_API_URL, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        out = r.json()
        if "choices" in out and len(out["choices"]) > 0:
            return out["choices"][0]["message"]["content"]
        return str(out)[:2000]
    except Exception as e:
        return f"LLM call failed: {e}"

def build_prompt(system: str, history: List[Dict], context_texts: List[str], user_message: str) -> str:
    ctx = "\n\n".join([f"[{i+1}] " + c for i, c in enumerate(context_texts)])
    hist = ""
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        hist += f"{role.upper()}: {content}\n"
    prompt = f"""System: {system}

Context (top matches):
{ctx}

Conversation so far:
{hist}

USER: {user_message}

Assistant, using ONLY the context above when relevant, answer the user clearly. If the context is insufficient,
say so and suggest what to upload or ask next. Cite sources like [1], [2] inline where appropriate.
"""
    return prompt

# ---------- API models ----------
class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: Optional[int] = 4

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"message": "Backend running"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...), session_id: str = Form(...)):
    data = await file.read()
    text = read_any(data, file.filename)
    if not text.strip():
        return {"ok": False, "message": "No text extracted from file."}

    chunks = chunk_text(text)
    embs = embed_texts(chunks)
    metadatas = []
    file_id = hashlib.md5((file.filename + session_id).encode("utf-8")).hexdigest()[:10]
    for i, ch in enumerate(chunks):
        metadatas.append({"file_id": file_id, "source": file.filename, "chunk_id": i, "text": ch})

    count = upsert_chunks(session_id, embs, metadatas)
    return {"ok": True, "filename": file.filename, "chunks": count}

@app.post("/chat")
def chat(req: ChatRequest):
    history = get_recent_messages(req.session_id, limit=MEMORY_TURNS)
    q_emb = embed_texts([req.message])[0]
    retrieved = search(req.session_id, q_emb, top_k=req.top_k or 4)
    context_texts = [r["text"] for r in retrieved]
    system = "You are a helpful RAG assistant."
    prompt = build_prompt(system, history, context_texts, req.message)
    answer = cerebras_generate(prompt)

    add_message(req.session_id, "user", req.message)
    add_message(req.session_id, "assistant", answer)

    cites = [{"source": r["source"], "chunk_id": r["chunk_id"], "score": r["score"], "rank": r["rank"]} for r in retrieved]
    return {"answer": answer, "citations": cites}

@app.get("/history")
def history(session_id: str, limit: int = 20):
    return {"messages": get_recent_messages(session_id, limit=limit)}

@app.post("/reset")
def reset(session_id: str):
    reset_session(session_id)
    return {"ok": True}
