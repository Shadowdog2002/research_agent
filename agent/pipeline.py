import time
from typing import Callable, Dict, List, Optional

from agent.arxiv_search import Paper, search_and_rerank
from agent.llm import (
    critique_literature_review,
    find_research_gaps,
    generate_literature_review,
    get_client,
    revise_literature_review,
    summarize_paper,
)
from agent.pdf_parser import (
    chunk_text,
    download_and_extract,
    extract_text_from_bytes,
)
from agent.rag import (
    add_paper_chunks,
    get_collection,
    get_paper_summary,
    paper_already_ingested,
)


def run_research_pipeline(
    topic: str,
    progress_cb: Optional[Callable[[float], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
    fetch_n: int = 20,
    min_score: float = 0.0,
) -> Dict:
    """
    Full agent loop:
      1. Search arXiv -> rerank -> top 5
      2. Download & parse each PDF
      3. Chunk -> embed -> store in ChromaDB (skip if already present)
      4. Summarise each paper with Claude
      5. Generate a unified literature review with Claude

    Returns a dict with keys: topic, papers (list), literature_review (str).
    """

    def _progress(pct: float):
        if progress_cb:
            progress_cb(pct)

    def _status(msg: str):
        if status_cb:
            status_cb(msg)

    llm = get_client()
    collection = get_collection()

    # -- 1. Search & rerank -------------------------------------------------------
    _status(f"Searching arXiv (fetching {fetch_n}, reranking to top 5)...")
    _progress(0.03)

    papers: List[Paper] = search_and_rerank(topic, fetch_n=fetch_n, top_k=5, min_score=min_score)
    if not papers:
        return {"error": "No papers found for this topic. Try a different query."}

    _status(f"Selected {len(papers)} papers after reranking")
    _progress(0.12)

    summaries: List[Dict] = []
    n = len(papers)

    for idx, paper in enumerate(papers):
        base = 0.12 + idx / n * 0.65          # progress occupies 12% -> 77%
        label = f"[{idx+1}/{n}] {paper.title[:55]}..."

        # -- 2. Check KB -> download only if needed --------------------------------
        if paper_already_ingested(collection, paper.paper_id):
            _status(f"Already in knowledge base: {label}")
            _progress(base)
            text = paper.abstract
            # Load stored summary; regenerate only if missing (legacy papers)
            summary = get_paper_summary(collection, paper.paper_id)
            if not summary:
                _status(f"Summarising {label}")
                summary = summarize_paper(llm, paper.title, paper.abstract, text)
        else:
            _status(f"Downloading {label}")
            _progress(base)

            try:
                text = download_and_extract(paper.pdf_url)
                time.sleep(3)
            except Exception:
                text = paper.abstract
                _status(f"Warning: PDF unavailable for paper {idx+1}, using abstract only")

            # -- 3. Summarise before embedding so summary is stored in KB ----------
            _status(f"Summarising {label}")
            _progress(base + 0.05)
            summary = summarize_paper(llm, paper.title, paper.abstract, text)

            # -- 4. Chunk -> embed -> store (with summary in metadata) -------------
            _status(f"Embedding & storing {label}")
            _progress(base + 0.10)
            chunks = chunk_text(text)
            add_paper_chunks(
                collection=collection,
                chunks=chunks,
                paper_id=paper.paper_id,
                title=paper.title,
                authors=", ".join(paper.authors[:3]),
                year=paper.published,
                summary=summary,
            )
        summaries.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": ", ".join(paper.authors[:3]),
                "year": paper.published,
                "abstract": paper.abstract,
                "summary": summary,
                "relevance_score": paper.relevance_score,
                "pdf_url": paper.pdf_url,
            }
        )

    # -- 5. Literature review (draft) ---------------------------------------------
    _status("Generating literature review (draft)...")
    _progress(0.78)

    lit_review_draft = generate_literature_review(llm, topic, summaries)

    # -- 6. CriticAgent: critique the draft ---------------------------------------
    _status("CriticAgent reviewing literature review...")
    _progress(0.83)

    critique = critique_literature_review(llm, topic, lit_review_draft, summaries)

    # -- 7. SynthesisAgent: revise based on critique ------------------------------
    _status("SynthesisAgent revising literature review...")
    _progress(0.88)

    lit_review = revise_literature_review(llm, topic, lit_review_draft, critique, summaries)

    # -- 8. Research gap analysis --------------------------------------------------
    _status("Identifying research gaps and future directions...")
    _progress(0.93)

    gaps = find_research_gaps(llm, topic, lit_review, summaries)

    _status("Done")
    _progress(1.0)

    return {
        "topic": topic,
        "papers": summaries,
        "literature_review": lit_review,
        "critique": critique,
        "research_gaps": gaps,
    }


def ingest_uploaded_pdf(
    pdf_bytes: bytes,
    filename: str,
    title: str,
    authors: str,
    year: str,
) -> str:
    """
    Ingest a user-uploaded PDF into the persistent RAG store.
    Returns a status message prefixed with OK:, INFO:, or Error:.
    """
    collection = get_collection()
    paper_id = f"upload_{filename}"

    if paper_already_ingested(collection, paper_id):
        return f"INFO: '{title}' is already in the knowledge base."

    try:
        text = extract_text_from_bytes(pdf_bytes)
    except Exception as e:
        return f"Error: Failed to parse PDF: {e}"

    chunks = chunk_text(text)
    if not chunks:
        return "Error: Could not extract any text from this PDF."

    try:
        summary = summarize_paper(get_client(), title, "", text)
    except Exception:
        summary = ""

    add_paper_chunks(
        collection=collection,
        chunks=chunks,
        paper_id=paper_id,
        title=title,
        authors=authors or "Unknown",
        year=year or "Unknown",
        summary=summary,
    )
    return f"OK: Ingested '{title}' — {len(chunks)} chunks stored."


def ingest_arxiv_paper(paper: Paper) -> str:
    """
    Download an arXiv paper by URL, chunk, embed, and store in the RAG store.
    Returns a status message prefixed with OK:, INFO:, or Error:.
    """
    collection = get_collection()

    if paper_already_ingested(collection, paper.paper_id):
        return f"INFO: '{paper.title}' is already in the knowledge base."

    try:
        text = download_and_extract(paper.pdf_url)
    except Exception:
        text = paper.abstract

    chunks = chunk_text(text)
    if not chunks:
        return "Error: Could not extract any text for this paper."

    try:
        summary = summarize_paper(get_client(), paper.title, paper.abstract, text)
    except Exception:
        summary = ""

    add_paper_chunks(
        collection=collection,
        chunks=chunks,
        paper_id=paper.paper_id,
        title=paper.title,
        authors=", ".join(paper.authors[:3]),
        year=paper.published,
        summary=summary,
    )
    return f"OK: Ingested '{paper.title}' — {len(chunks)} chunks stored."


def ingest_arxiv_papers_batch(papers: List[Paper]) -> List[str]:
    """
    Ingest multiple arXiv papers sequentially with a 3-second gap between
    papers that require a PDF download, to respect arXiv's rate limit.
    Returns a list of status strings in the same order as `papers`.
    """
    collection = get_collection()
    messages = []
    need_sleep = False
    for paper in papers:
        if need_sleep:
            time.sleep(3)
        # Only count this paper against the rate limit if it will actually
        # hit the network (i.e. it is not already in the knowledge base).
        will_download = not paper_already_ingested(collection, paper.paper_id)
        msg = ingest_arxiv_paper(paper)
        messages.append(msg)
        need_sleep = will_download
    return messages
