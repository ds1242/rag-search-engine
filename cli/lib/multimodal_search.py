from PIL import Image
from lib.search_utils import load_movies
from sentence_transformers import SentenceTransformer

from lib.semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, path):
        image = Image.open(path)
        return self.model.encode([image])[0]

    def search_with_image(self, path):
        curr_img_embed = self.embed_image(path)
        scores_with_indices = []
        for i, text_embed in enumerate(self.text_embeddings):
            scores_with_indices.append((i, cosine_similarity(curr_img_embed, text_embed)))


        sorted_list = sorted(scores_with_indices, key=lambda x: x[1], reverse=True)

        res = []
        for idx, score in sorted_list[:5]:
            curr = {
                "id": self.documents[idx]['id'],
                "title": self.documents[idx]['title'],
                "description": self.documents[idx]['description'],
                "score": score
            }
            res.append(curr)

        return res


def verify_image_embedding(path: str):
    movies = load_movies()
    searcher = MultimodalSearch(movies)

    embedding = searcher.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(path: str):
    movies = load_movies()
    searcher = MultimodalSearch(movies)

    results = searcher.search_with_image(path)

    return results
