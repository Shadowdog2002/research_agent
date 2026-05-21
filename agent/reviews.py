import json
import os
import re
import uuid
from datetime import datetime
from typing import Dict, List


def _reviews_path(username: str = "default") -> str:
    safe = re.sub(r"[^a-z0-9_-]", "_", username.lower().strip())[:50]
    return f"./data/reviews_{safe or 'default'}.json"


def save_review(
    topic: str,
    literature_review: str,
    papers: List[Dict],
    research_gaps: Dict = None,
    critique: str = "",
    username: str = "default",
) -> str:
    """Save a literature review (and optional gap analysis) to disk and return its ID."""
    path = _reviews_path(username)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    reviews = load_reviews(username)
    review_id = uuid.uuid4().hex[:8]
    reviews.append({
        "id": review_id,
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "literature_review": literature_review,
        "critique": critique,
        "research_gaps": research_gaps or {},
        "papers": [
            {
                "paper_id": p.get("paper_id", ""),
                "title": p.get("title", ""),
                "authors": p.get("authors", ""),
                "year": p.get("year", ""),
                "pdf_url": p.get("pdf_url", ""),
            }
            for p in papers
        ],
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)
    return review_id


def load_reviews(username: str = "default") -> List[Dict]:
    """Load all saved literature reviews for a user from disk."""
    try:
        with open(_reviews_path(username), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def delete_review(review_id: str, username: str = "default") -> bool:
    """Delete a review by ID. Returns True if found and deleted."""
    reviews = load_reviews(username)
    filtered = [r for r in reviews if r["id"] != review_id]
    if len(filtered) == len(reviews):
        return False
    with open(_reviews_path(username), "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    return True
