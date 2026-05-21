import chromadb
from chromadb.utils import embedding_functions
import hashlib
import os
import re
from typing import Dict, List, Any

CHROMA_PATH = "./data/chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Module-level singletons — embedding model and client are shared across users.
# Per-user collections are cached in _collections keyed by username.
_ef = None
_chroma_client = None
_collections: Dict[str, chromadb.Collection] = {}


def _collection_name(username: str) -> str:
    """Derive a safe ChromaDB collection name from a username."""
    safe = re.sub(r"[^a-z0-9_-]", "_", username.lower().strip())[:50]
    return f"papers_{safe or 'default'}"


def get_collection(username: str = "default") -> chromadb.Collection:
    """Return the per-user ChromaDB collection, creating it if needed."""
    global _ef, _chroma_client, _collections
    if username in _collections:
        return _collections[username]

    os.makedirs(CHROMA_PATH, exist_ok=True)
    if _ef is None:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = _chroma_client.get_or_create_collection(
        name=_collection_name(username),
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )
    _collections[username] = collection
    return collection


def add_paper_chunks(
    collection: chromadb.Collection,
    chunks: List[str],
    paper_id: str,
    title: str,
    authors: str,
    year: str,
    summary: str = "",
    batch_size: int = 50,
) -> None:
    """Embed and store paper chunks in the vector store."""
    ids, documents, metadatas = [], [], []

    for i, chunk in enumerate(chunks):
        uid = hashlib.md5(f"{paper_id}-{i}".encode()).hexdigest()
        ids.append(uid)
        documents.append(chunk)
        metadatas.append(
            {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "year": year,
                "chunk_index": i,
                "summary": summary,
            }
        )

    # Insert in batches to avoid OOM on large papers
    for start in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[start : start + batch_size],
            documents=documents[start : start + batch_size],
            metadatas=metadatas[start : start + batch_size],
        )


def query_papers(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 6,
) -> List[Dict[str, Any]]:
    """Retrieve the most relevant chunks for a user query."""
    results = collection.query(query_texts=[query], n_results=n_results)
    chunks = []
    if results and results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append(
                {
                    "text": doc,
                    "title": meta.get("title", "Unknown"),
                    "authors": meta.get("authors", "Unknown"),
                    "year": meta.get("year", "Unknown"),
                    "paper_id": meta.get("paper_id", ""),
                }
            )
    return chunks


def get_ingested_papers(collection: chromadb.Collection) -> List[Dict[str, str]]:
    """Return a deduplicated list of all papers in the knowledge base."""
    try:
        results = collection.get(include=["metadatas"])
        seen: Dict[str, Dict] = {}
        for meta in results["metadatas"]:
            pid = meta.get("paper_id", "")
            if pid and pid not in seen:
                seen[pid] = {
                    "paper_id": pid,
                    "title": meta.get("title", "Unknown"),
                    "authors": meta.get("authors", "Unknown"),
                    "year": meta.get("year", "Unknown"),
                    "summary": meta.get("summary", ""),
                }
        return list(seen.values())
    except Exception:
        return []


def get_paper_summary(collection: chromadb.Collection, paper_id: str) -> str:
    """Return the stored summary for a paper, or empty string if not found."""
    try:
        res = collection.get(where={"paper_id": paper_id}, limit=1, include=["metadatas"])
        if res["metadatas"]:
            return res["metadatas"][0].get("summary", "")
    except Exception:
        pass
    return ""


def paper_already_ingested(collection: chromadb.Collection, paper_id: str) -> bool:
    """Check whether a paper's chunks already exist in the store."""
    try:
        res = collection.get(where={"paper_id": paper_id}, limit=1)
        return len(res["ids"]) > 0
    except Exception:
        return False


def delete_paper(collection: chromadb.Collection, paper_id: str) -> int:
    """Remove all chunks belonging to a paper. Returns number of chunks deleted."""
    try:
        res = collection.get(where={"paper_id": paper_id})
        ids = res["ids"]
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0
