from os.path import isfile
import string
import os
import pickle
import math
from collections import Counter, defaultdict

from .word_utils import load_stopwords
from .search_utils import BM25_B, PROJECT_ROOT, DEFAULT_SEARCH_LIMIT, load_movies, BM25_K1
from nltk.stem import PorterStemmer 

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

def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    count = index.get_tf(doc_id, term)
    return count

def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_idf(term)

def tfidf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_tfidf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float:
    idx = InvertedIndex()
    idx.load()

    bm25_tf = idx.get_bm25_tf(doc_id, term)

    return bm25_tf

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.index_path = os.path.join(os.path.join(PROJECT_ROOT, "cache"), "index.pkl")
        self.docmap_path = os.path.join(os.path.join(PROJECT_ROOT,"cache"), "docmap.pkl")
        self.tf_path = os.path.join(os.path.join(PROJECT_ROOT, "cache"), "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(os.path.join(PROJECT_ROOT, "cache"), "doc_lengths.pkl")

    def __add_document(self, doc_id, text) -> None:
        text_tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(text_tokens)
        unique_tokens = set(text_tokens)
        for token in unique_tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(text_tokens)            

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0

        doc_total = 0
        for _, value in self.doc_lengths.items():
            doc_total += value 

        return doc_total / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        documents = self.index.get(term, set())
        return sorted(list(documents))

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
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)

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

        if os.path.isfile(self.doc_lengths_path):
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        else:
            raise Exception("file does not exist")

    def get_tf(self, doc_id: int, term: str) -> int:
        token_term = tokenize_text(term)
        if len(token_term) != 1:
            raise ValueError('too many terms')
        
        token = token_term[0]
        return self.term_frequencies[doc_id].get(token, 0)

    def get_idf(self, term: str) -> float:
        token_term = tokenize_text(term)
        if len(token_term) != 1:
            raise ValueError("term must be a single token")

        token = token_term[0]
        matched_count = len(self.index[token]) 
        total_doc_count = len(self.docmap)
        return math.log((total_doc_count + 1) / (matched_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        idf_value = self.get_idf(term)
        tf_value = self.get_tf(doc_id, term)

        return tf_value * idf_value

    def get_bm25_idf(self, term: str) -> float:
        token_term = tokenize_text(term)
        if len(token_term) != 1:
            raise ValueError("to many terms")

        token = token_term[0]
        total_docs_count = len(self.docmap)
        matched_doc_count = len(self.index[token])

        return math.log((total_docs_count - matched_doc_count + 0.5) / (matched_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        avg_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_length)

        bm25_sat_val = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_sat_val

