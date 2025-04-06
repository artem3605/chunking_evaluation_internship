import torch
import torch.nn.functional as F

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, top_k):
        """
        Retrieve the top_k most similar vectors from the vector store. The function returns its indices.
        """
        similarities = [
            F.cosine_similarity(torch.tensor(query), torch.tensor(vec), dim=0).item() for vec in self.vector_store
        ]
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
        return top_indices