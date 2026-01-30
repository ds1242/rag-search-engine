import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.index = InvertedIndex()
        if not os.path.exists(self.index.index_path):
            self.index.build()
            self.index.save()

        def _bm25_search(self, query, limit):
            self.index.load()
            return self.index.bm25_search(query, limit)

        def weighted_search(self, query, alpha, limit=5):
            raise NotImplementedError("Weighted hybrid search is not implemented yet")

        def rrf_search(self, query, k, limit=10):
            raise NotImplementedError("RRF hybrid search is not implmented yet.")

