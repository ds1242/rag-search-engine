import json
import os

DEFAULT_SEARCH_LIMIT = 5

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_ROOT = os.path.join(PROJECT_ROOT, "cache")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]
