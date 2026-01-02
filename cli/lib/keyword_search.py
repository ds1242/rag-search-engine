import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    for movie in movies:
        preprocessed_query = preprocess_text(query)
        preprocessed_title = preprocess_text(movie['title'])

        
        if len(results) >= limit:
            break

        matched = False
        for query_word in preprocessed_query:
            if matched:
                break

            for title_word in preprocessed_title:
                if query_word in title_word:
                    results.append(movie)
                    matched = True
                    break

    return results


def preprocess_text(text: str) -> list[str]:
    text = text.lower()

    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    words = text.split()

    return words
