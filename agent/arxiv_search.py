import time
import urllib.error
import arxiv
from sentence_transformers import CrossEncoder
from dataclasses import dataclass, field
from typing import List

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_ARXIV_CLIENT = arxiv.Client(
    page_size=50,
    delay_seconds=3.0,
    num_retries=5,
)

_RERANKER = CrossEncoder(RERANKER_MODEL)


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    pdf_url: str
    published: str
    relevance_score: float = 0.0


def search_and_rerank(
    query: str,
    fetch_n: int = 20,
    top_k: int = 5,
    min_score: float = 0.0,
) -> List[Paper]:
    """
    Search arXiv for `fetch_n` papers, rerank with a cross-encoder,
    and return up to `top_k` results whose score exceeds `min_score`.
    """
    client = _ARXIV_CLIENT
    search = arxiv.Search(
        query=query,
        max_results=fetch_n,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    time.sleep(3)
    results = None
    for attempt in range(3):
        try:
            results = list(client.results(search))
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                time.sleep(10)
            else:
                raise
    if results is None:
        results = []

    if not results:
        return []

    # Score each (query, title + abstract snippet) pair
    pairs = [(query, f"{r.title}. {r.summary[:600]}") for r in results]
    scores = _RERANKER.predict(pairs)

    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)

    papers = []
    for score, r in ranked[:top_k]:
        if score < min_score:
            break  # ranked descending, so no later entry will pass either
        papers.append(
            Paper(
                paper_id=r.get_short_id(),
                title=r.title,
                authors=[a.name for a in r.authors],
                abstract=r.summary,
                pdf_url=r.pdf_url,
                published=str(r.published.year) if r.published else "Unknown",
                relevance_score=float(score),
            )
        )

    return papers
