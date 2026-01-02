import os

from .search_utils import PROJECT_ROOT

WORD_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

def load_stopwords() -> list[str]:
    with open(WORD_PATH, 'r') as f:
        stopwords = f.read().splitlines()

    return stopwords
