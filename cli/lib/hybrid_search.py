import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from lib.search_utils import load_movies

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.index = InvertedIndex()
        if not os.path.exists(self.index.index_path):
            self.index.build()
            self.index.save()

        self.doc_map = {}
        for doc in self.documents:
            self.doc_map[doc['id']] = doc
            

    def _bm25_search(self, query, limit):
        self.index.load()
        return self.index.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        normalized_bm25 = normalize_search_results(bm25_results)
        normalized_semantic = normalize_search_results(semantic_results)

        scores = {}
        for result in normalized_bm25:
            scores[result['id']] = {
                'title': self.doc_map[result['id']]['title'],
                'document': self.doc_map[result['id']]['document'],
                'bm25_score': result['normalized_score'],
                'semantic_score': 0.0,
            }

        for result in normalized_semantic:
            if result['id'] in scores:
                scores[result['id']]['semantic_score'] = result['normalized_score']
            else:
                scores[result['id']] = {
                    'title': self.doc_map[result['id']]['title'],
                    'document': self.doc_map[result['id']]['document'],
                    'bm25_score': 0.0,
                    'semantic_score': result['normalized_score']
                }

        for score in scores:
            score['hybrid_score'] = hybrid_score(score['bm25_score'], score['semantic_score'], alpha)

        scores = sorted(scores, key=lambda item: item['hybrid_score'], reverse=True)

        return scores


        def rrf_search(self, query, k, limit=10):
            raise NotImplementedError("RRF hybrid search is not implmented yet.")


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result['score'])

    normalized: list[float] = normalize_score(scores)
    for i, result in enumerate(results):
        result['normalized_score'] = normalized[i]

    return results


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

    
def weighted_search(query: str, alpha: float, limit: int):
    movies = load_movies()
    hybrid = HybridSearch(movies)
    scores = hybrid.weighted_search(query, alpha, limit)

    for i in range(0, limit):
        print(f"{i}. {scores[i]['title']}\n")
        print(f"     Hybrid Score: {scores[i]['hybrid_score']:.3f}")
        print(f"     BM25: {scores[i]['bm25_score']:.3f}, Semantic: {scores[i]['semantic_score']}")
        print(f"     {scores[i]['document']}")
