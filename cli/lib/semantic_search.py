from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def generate_embedding(self, text):
        if text == "" or text = ' ':
            raise ValueError("text string is empty")

        embedding = self.model.encode(list(text))
        return embedding[0]




def verify_model():
    model = SemanticSearch()
    print(f"Model Loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")

def embed_text(text):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"Frist 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
