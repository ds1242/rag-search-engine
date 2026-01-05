from os.path import isfile
import string
import os
from collections import defaultdict
from .word_utils import load_stopwords
from .search_utils import PROJECT_ROOT, DEFAULT_SEARCH_LIMIT, load_movies
from nltk.stem import PorterStemmer 
import pickle

CACHE_ROOT = os.path.dirname(__file__)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    index = InvertedIndex()
    query_tokens = tokenize_text(query)
    results = []

    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
    
    for token in query_tokens:
        docs = index.get_documents(token)
        if docs is not None:
            for doc in docs:
                if len(results) >= DEFAULT_SEARCH_LIMIT:
                    break
                results.append(index.docmap[doc])

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

def build_command():
    index = InvertedIndex()
    index.build()
    index.save()


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
        documents = self.index.get(term, set())
        return sorted(documents)

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")
            self.docmap[doc_id] = movie

    def save(self):
        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")

        with open(index_path, 'wb') as file:
            pickle.dump(self.index, file)


        with open(docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)

    def load(self):
        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        
        if os.path.isfile(index_path):
            with open(index_path, "rb") as f:
                self.index = pickle.load(f)
        else:
            raise Exception("file does not exist")

        if os.path.isfile(docmap_path):
            with open(docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
        else:
            raise Exception("file does not exist")






