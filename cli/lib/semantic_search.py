from lib.search_utils import CACHE_ROOT, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_SEMANTIC_CHUNK, load_movies
import numpy as np
import os
import re
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
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings lodaded. Call `load_or_create_embeddings` first.")

        query_embed = self.generate_embedding(query)
        results = []
        for i, doc_embed in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embed, doc_embed)
            results.append((similarity_score, self.documents[i]))

        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        final_results = []

        for result in sorted_results[:limit]:
            score = {
                    "score": result[0],
                    "title": result[1]['title'],
                    "description": result[1]['description']
            }
            final_results.append(score)

        return final_results


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

def search(query: str, limit: int):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)
    results = search_instance.search(query, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (score: {res['score']:.4f})\n{res['description'][:100] + "..."}")
    

def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
    chunks = fixed_size_chunking(text, chunk_size,overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def semantic_chunk_processing(text: str, chunk_size: int = DEFAULT_SEMANTIC_CHUNK, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text) 
    output = []
    i = 0
    n_chunks = len(chunks)

    while i < n_chunks:
        sentence_chunk = chunks[i : i + chunk_size]
        if output and len(sentence_chunk) <= overlap:
            break
        output.append(" ".join(sentence_chunk))
        i += chunk_size - overlap

    return output


def semantic_chunking(text: str, chunk_size: int = DEFAULT_SEMANTIC_CHUNK, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
    chunks = semantic_chunk_processing(text, chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")
