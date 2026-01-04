import string
from collections import defaultdict
from .word_utils import load_stopwords
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
from nltk.stem import PorterStemmer 

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie['title'])

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

    stopwords = load_stopwords()
    for token in list(valid_tokens):
        if token in stopwords:
            valid_tokens.remove(token)

    stemmer = PorterStemmer()
    stemmed_words = []

    for token in valid_tokens:
        stemmed_word = stemmer.stem(token)
        stemmed_words.append(stemmed_word)

    return stemmed_words

 
class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, dict]

    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}

    def __add_document(self, doc_id, text):
        text_tokens = tokenize_text(text)
        unique_tokens = set(text_tokens)
        for token in unique_tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term):
        documents = self.index[term]
        print(documents)




