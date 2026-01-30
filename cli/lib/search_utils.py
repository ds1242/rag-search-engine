import json
import os
from typing import Any

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_ALPHA = 0.5

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_ROOT = os.path.join(PROJECT_ROOT, "cache")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0

DEFAULT_SEMANTIC_CHUNK = 4

SCORE_PRECISION = 3

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def format_search_result(doc_id: str, title: str, document: str, score: float, **metadata: Any) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
