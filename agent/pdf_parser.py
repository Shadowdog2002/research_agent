import fitz  # PyMuPDF
import requests
import tempfile
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List


def _make_retry_session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=2.0,
        status_forcelist=(503, 429, 502, 504),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_and_extract(pdf_url: str, timeout: int = 30) -> str:
    """Download a PDF from a URL and extract its full text."""
    headers = {"User-Agent": "ResearchAgent/1.0 (academic research tool)"}
    session = _make_retry_session()
    response = session.get(pdf_url, headers=headers, timeout=timeout)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(response.content)
        tmp_path = f.name

    try:
        text = _extract_from_path(tmp_path)
    finally:
        os.unlink(tmp_path)

    return text


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text from raw PDF bytes (e.g. from a Streamlit file uploader)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return _extract_from_doc(doc)


def _extract_from_path(path: str) -> str:
    doc = fitz.open(path)
    return _extract_from_doc(doc)


def _extract_from_doc(doc: fitz.Document) -> str:
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 60,
) -> List[str]:
    """
    Split text into overlapping word-level chunks.
    chunk_size ~ 400 words ≈ ~500 tokens; safe for embedding models.
    """
    words = text.split()
    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
