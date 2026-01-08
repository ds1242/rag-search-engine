from os.path import isfile
from lib.search_utils import CACHE_ROOT, load_movies
import numpy as np
import os
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("text string is empty")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents

        movie_info_strings = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_string = f"{doc['title']}: {doc['description']}"
            movie_info_strings.append(movie_string)

        self.embeddings = self.model.encode(movie_info_strings, show_progress_bar=True)
        embeddings_path = os.path.join(CACHE_ROOT, "movie_embeddings.npy")
        np.save(embeddings_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if os.path.isfile(os.path.join(CACHE_ROOT, "movie_embeddings.npy")):
            self.embeddings = np.load(os.path.join(CACHE_ROOT, "movie_embeddings.npy"))

        if self.embeddings is not None and len(self.embeddings) == len(self.documents):
            return self.embeddings

        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if not self.embeddings or self.embeddings.size == 0:
            raise ValueError("No embeddings lodaded. Call `load_or_create_embeddings` first.")

        query_embedds = self.generate_embedding(query)


def verify_embeddings() -> None:
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def verify_model() -> None:
    search_instance = SemanticSearch()
    print(f"Model Loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def embed_text(text) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query) -> None:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape[0]}")

def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
