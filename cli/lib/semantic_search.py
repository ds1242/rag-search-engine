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


def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def verify_model():
    model = SemanticSearch()
    print(f"Model Loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

def embed_text(text):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
