import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import chromadb
from datetime import datetime, timedelta
from typing import Optional
import config
from embedder import embed_texts, embed_single, EMBEDDING_DIMENSIONS
from chunker import Chunk

# ─── ChromaDB Client ─────────────────────────────────────────
_chroma_client = None

def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(config.CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return _chroma_client


# ─── Session Management ──────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    safe_id = "".join(c if c.isalnum() else "_" for c in session_id)
    return f"session_{safe_id[:50]}"


def get_or_create_collection(session_id: str) -> chromadb.Collection:
    client = _get_client()
    name = get_collection_name(session_id)
    collection = client.get_or_create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "created_at": str(datetime.now().timestamp()),
            "session_id": session_id
        }
    )
    return collection


def cleanup_old_sessions():
    client = _get_client()
    cutoff = datetime.now() - timedelta(hours=config.SESSION_EXPIRY_HOURS)
    cutoff_timestamp = cutoff.timestamp()
    deleted = 0
    try:
        collections = client.list_collections()
        for col in collections:
            if not col.name.startswith("session_"):
                continue
            created_at = float(col.metadata.get("created_at", 0)) if col.metadata else 0
            if created_at < cutoff_timestamp:
                client.delete_collection(col.name)
                deleted += 1
    except Exception as e:
        print(f"   Cleanup warning: {e}")
    if deleted > 0:
        print(f"   Cleaned up {deleted} expired session(s)")


def delete_session(session_id: str) -> bool:
    client = _get_client()
    name = get_collection_name(session_id)
    try:
        client.delete_collection(name)
        return True
    except Exception:
        return False


# ─── Document Storage ────────────────────────────────────────

def store_chunks(chunks: list[Chunk], session_id: str) -> dict:
    if not chunks:
        return {"stored": 0, "error": "No chunks to store"}

    collection = get_or_create_collection(session_id)

    existing_ids = set()
    try:
        existing = collection.get()
        existing_ids = set(existing["ids"])
    except Exception:
        pass

    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

    if not new_chunks:
        return {"stored": 0, "skipped": len(chunks), "reason": "already exists"}

    texts = [c.text for c in new_chunks]
    print(f"   Generating embeddings for {len(new_chunks)} chunks...")
    embeddings = embed_texts(texts)

    # Verify counts match before storing
    if len(embeddings) != len(new_chunks):
        raise ValueError(
            f"Embedding count mismatch: got {len(embeddings)} embeddings "
            f"for {len(new_chunks)} chunks. Possible rate limit issue."
        )

    ids = [c.chunk_id for c in new_chunks]
    metadatas = [
        {
            "page": c.page,
            "filename": c.filename,
            "word_count": c.word_count,
            "chunk_index": c.chunk_index,
            "total_chunks": c.total_chunks
        }
        for c in new_chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    return {
        "stored": len(new_chunks),
        "skipped": len(chunks) - len(new_chunks),
        "total_in_collection": collection.count()
    }


# ─── Semantic Search ─────────────────────────────────────────

def search(
    query: str,
    session_id: str,
    top_k: int = None,
    filename_filter: Optional[str] = None
) -> list[dict]:
    if top_k is None:
        top_k = config.TOP_K_CHUNKS

    collection = get_or_create_collection(session_id)

    if collection.count() == 0:
        return []

    query_embedding = embed_single(query)
    where_filter = {"filename": filename_filter} if filename_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    formatted = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance

        if similarity < config.RELEVANCE_THRESHOLD:
            continue

        formatted.append({
            "chunk_id":   results["ids"][0][i],
            "text":       results["documents"][0][i],
            "page":       results["metadatas"][0][i]["page"],
            "filename":   results["metadatas"][0][i]["filename"],
            "word_count": results["metadatas"][0][i]["word_count"],
            "score":      round(similarity, 4)
        })

    formatted.sort(key=lambda x: x["score"], reverse=True)
    return formatted


# ─── Document Listing ────────────────────────────────────────

def list_documents(session_id: str) -> list[dict]:
    collection = get_or_create_collection(session_id)

    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    docs = {}

    for meta in results["metadatas"]:
        filename = meta["filename"]
        if filename not in docs:
            docs[filename] = {
                "filename": filename,
                "total_chunks": meta["total_chunks"],
                "pages": set()
            }
        docs[filename]["pages"].add(meta["page"])

    return [
        {
            "filename": d["filename"],
            "total_chunks": d["total_chunks"],
            "pages_indexed": len(d["pages"])
        }
        for d in docs.values()
    ]


def delete_document(filename: str, session_id: str) -> dict:
    collection = get_or_create_collection(session_id)

    results = collection.get(
        where={"filename": filename},
        include=["metadatas"]
    )

    if not results["ids"]:
        return {"deleted": 0, "error": "Document not found"}

    collection.delete(ids=results["ids"])
    return {"deleted": len(results["ids"]), "filename": filename}


# ─── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pdf_processor import extract_pdf
    from chunker import chunk_document

    print(f"\n{'='*50}")
    print(f"  {config.APP_NAME} — Vector Store Test")
    print(f"{'='*50}\n")

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        print("Usage: python vector_store.py path/to/your.pdf")
        sys.exit(0)

    TEST_SESSION = "test_session_001"
    print(f"Step 1: Extracting PDF...")
    doc = extract_pdf(test_path)
    if doc.error:
        print(f"❌ {doc.error}")
        sys.exit(1)
    print(f"✅ {doc.total_pages} pages, {doc.total_words} words\n")

    print(f"Step 2: Chunking...")
    chunks = chunk_document(doc)
    print(f"✅ {len(chunks)} chunks\n")

    print(f"Step 3: Storing...")
    result = store_chunks(chunks, TEST_SESSION)
    print(f"✅ Stored: {result['stored']}\n")

    print(f"Step 4: Searching...")
    results = search("What is the main contribution?", TEST_SESSION)
    if results:
        print(f"✅ Top result: {results[0]['filename']} p{results[0]['page']} ({results[0]['score']})")

    delete_session(TEST_SESSION)
    print(f"\n✅ Done")
