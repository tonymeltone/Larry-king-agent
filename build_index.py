#!/usr/bin/env python3
"""
build_index.py
--------------
Reads all .txt transcript files from your lkl_transcripts folder,
splits them into chunks, and stores them in a local ChromaDB vector index.
 
Run this ONCE (or whenever you add new transcripts) before launching the app.
 
Usage:
    pip install chromadb anthropic
    python build_index.py
"""
 
import os
import re
import chromadb
from chromadb.utils import embedding_functions
 
# ── Configuration ─────────────────────────────────────────────────────────────
 
TRANSCRIPTS_DIR = r"E:\LarryKing\Transcripts\lkl_transcripts"   # your transcripts folder
CHROMA_DIR      = r"E:\LarryKing\Transcripts\chroma_db"          # where the index is saved
CHUNK_SIZE      = 1000   # characters per chunk
CHUNK_OVERLAP   = 150    # overlap between chunks so context isn't lost
 
# We use ChromaDB's built-in sentence transformer (free, runs locally, no API key)
EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks
 
 
def parse_metadata(text: str, filename: str) -> dict:
    """Extract title, date, segment from the file header we wrote."""
    title   = re.search(r"TITLE:\s*(.+)",   text)
    date    = re.search(r"DATE:\s*(.+)",    text)
    segment = re.search(r"SEGMENT:\s*(.+)", text)
    url     = re.search(r"URL:\s*(.+)",     text)
    return {
        "title":    title.group(1).strip()   if title   else filename,
        "date":     date.group(1).strip()    if date    else "unknown",
        "segment":  segment.group(1).strip() if segment else "1",
        "url":      url.group(1).strip()     if url     else "",
        "filename": filename,
    }
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    print(f"Reading transcripts from: {TRANSCRIPTS_DIR}")
    files = [f for f in os.listdir(TRANSCRIPTS_DIR)
             if f.endswith(".txt") and f != "index.txt"]
    print(f"Found {len(files)} transcript files.")
 
    # Set up ChromaDB
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name="larry_king_transcripts",
        embedding_function=EMBEDDING_FN,
    )
 
    already_indexed = set(
        m["filename"] for m in collection.get(include=["metadatas"])["metadatas"]
    )
    print(f"Already indexed: {len(already_indexed)} files. Skipping those.")
 
    added = 0
    for i, filename in enumerate(files, 1):
        if filename in already_indexed:
            continue
 
        filepath = os.path.join(TRANSCRIPTS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
 
        meta   = parse_metadata(text, filename)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
 
        ids       = [f"{filename}::chunk{j}" for j in range(len(chunks))]
        metadatas = [meta] * len(chunks)
 
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        added += 1
 
        if i % 50 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] Indexed: {filename}")
 
    print(f"\nDone! {added} new files indexed. Total chunks in DB: {collection.count()}")
    print(f"Index saved to: {CHROMA_DIR}")
 
 
if __name__ == "__main__":
    main()
 