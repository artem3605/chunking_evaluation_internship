from sentence_transformers import SentenceTransformer


def generate_embeddings(texts: list[str]):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings
