from os.path import isfile
import string
import os
from collections import Counter, defaultdict
from .word_utils import load_stopwords
from .search_utils import PROJECT_ROOT, DEFAULT_SEARCH_LIMIT, load_movies
from nltk.stem import PorterStemmer 
import pickle

CACHE_ROOT = os.path.dirname(__file__)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
    
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for token in query_tokens:
        matching_ids = index.get_documents(token)
        for id in matching_ids:
            if id in seen:
                continue
            seen.add(id)
            results.append(index.docmap[id])
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

def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

def tf_command(doc_id, term) -> int:
    index = InvertedIndex()
    index.load()
    count = index.get_tf(doc_id, term)
    return count


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.index_path = os.path.join(os.path.join(PROJECT_ROOT, "cache"), "index.pkl")
        self.docmap_path = os.path.join(os.path.join(PROJECT_ROOT,"cache"), "docmap.pkl")
        self.tf_path = os.path.join(os.path.join(PROJECT_ROOT, "cache"), "term_frequencies.pkl")

    def __add_document(self, doc_id, text) -> None:
        text_tokens = tokenize_text(text)
        unique_tokens = set(text_tokens)
        for token in unique_tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(text_tokens)            

    def get_documents(self, term) -> list[int]:
        documents = self.index.get(term, set())
        return sorted(documents)

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")
            self.docmap[doc_id] = movie

    def save(self) -> None:
        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        with open(self.index_path, 'wb') as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        if os.path.isfile(self.index_path):
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
        else:
            raise Exception("file does not exist")

        if os.path.isfile(self.docmap_path):
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
        else:
            raise Exception("file does not exist")

        if os.path.isfile(self.tf_path):
            with open(self.tf_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        else:
            raise Exception("file does not exist")

    def get_tf(self, doc_id, term) -> int:
        token_term = tokenize_text(term)
        if len(token_term) != 1:
            raise ValueError('too many terms')
        
        token = token_term[0]
        return self.term_frequencies[doc_id].get(token, 0)


