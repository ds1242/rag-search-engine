import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from lib.search_utils import format_search_result, load_movies

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

        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]


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

def hybrid_score(bm25score, semantic_score, alpha=0.5) -> float:
    return alpha * bm25score + (1 - alpha) * semantic_score

def combine_search_results(bm25_results: list[dict], semantic_results: list[dict], alpha: float = 0.5) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}
    for result in bm25_normalized:
        doc_id = result['id']
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                'title': result['title'],
                'document': result['document'],
                'bm25_score': 0.0,
                'semantic_score': 0.0,
            } 
        if result['normalized_score'] > combined_scores[doc_id]['bm25_score']:
            combined_scores[doc_id]['bm25_score'] = result['normalized_score']

    for result in semantic_normalized:
        doc_id = result['id']
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result['title'],
                "document": result['document'],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result['normalized_score'] > combined_scores[doc_id]['semantic_score']:
            combined_scores[doc_id]["semantic_score"] = result['normalized_score']

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data['bm25_score'], data['semantic_score'], alpha)
        result = format_search_result(
            doc_id = doc_id,
            title = data['title'],
            document = data['document'],
            score = score_value,
            bm25_score = data['bm25_score'],
            semantic_score = data['semantic_score'],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x['score'], reverse=True)

    
def weighted_search(query: str, alpha: float, limit: int):
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }
