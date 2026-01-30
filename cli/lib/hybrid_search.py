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




def normalize_score(scores: list[float]) -> list[float]:
    if not scores:
        return []

    res = []
    max_score = max(scores)
    min_score = min(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    for score in scores:
        res.append((score - min_score) / (max_score - min_score))

    return res

def hybrid_score(bm25score, semantic_score, alpha=0.5):
    return alpha * bm25score + (1 - alpha) * semantic_score

    
    
