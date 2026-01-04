import string

from .word_utils import load_stopwords
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
from nltk.stem import PorterStemmer 

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    for movie in movies:
        query_tokens = remove_stopwords(query)
        title_tokens = remove_stopwords(movie['title'])

        if has_matching_tokens(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results

def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def preprocess_text(text: str) -> str:
    text = text.lower()

    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    return text 


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    return valid_tokens


def remove_stopwords(text: str) -> list[str]:
    stopwords = load_stopwords()

    valid_tokens = tokenize_text(text)

    for token in list(valid_tokens):
        if token in stopwords:
            valid_tokens.remove(token)

    stemmer = PorterStemmer()
    stemmed_words = []

    for token in valid_tokens:
        stemmed_word = stemmer.stem(token)
        stemmed_words.append(stemmed_word)

    return stemmed_words

            
