from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')




def verify_model():
    model = SemanticSearch()
    print(f"Model Loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")
