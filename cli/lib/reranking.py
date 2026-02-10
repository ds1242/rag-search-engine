import os
import json
from time import sleep
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"


def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "0").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]

def llm_rerank_batch(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    if not documents:
        return []

    doc_map = {}
    doc_list = []
    for doc in documents:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )
    
    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    response = client.models.generate_content(model=model, contents=prompt)
    raw = (response.text or "").strip()
    start = raw.find("[")
    end = raw.rfind("]")
    json_str = raw[start : end + 1]
    ids_in_order = json.loads(json_str)

    reranked = []
    for i, doc_id in enumerate(ids_in_order, 1):
        if doc_id in doc_map:
            doc = doc_map[doc_id]
            reranked.append({**doc, "batch_rank": i})

    return reranked[:limit]

def cross_encode_rank(query: str, docs: list[dict], limit: int):
    pairs = []
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    for doc in docs:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    scores = cross_encoder.predict(pairs)

    for i, doc in enumerate(docs):
        doc['cross_encoder_score'] = scores[i]

    docs.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    return docs[:limit]

def rerank(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    if method == "batch":
        return llm_rerank_batch(query, documents, limit)
    if method == "cross_encoder":
        return cross_encode_rank(query, documents, limit)
    else:
        return documents[:limit]
