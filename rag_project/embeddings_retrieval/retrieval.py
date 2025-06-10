from rank_bm25 import BM25Okapi
from .embed_model import load_model
import numpy as np

class Retriever:
    def __init__(self, chunks):
        self.model = load_model()
        self.chunks = chunks
        self.bm25 = BM25Okapi([chunk.split() for chunk in chunks])
        self.chunk_embeddings = self.model.encode(chunks, convert_to_tensor=False)

    def dense_search(self, query, top_k=5):
        q_embed = self.model.encode([query])[0]
        scores = np.dot(self.chunk_embeddings, q_embed)
        top_k_idx = np.argsort(scores)[-top_k:][::-1]
        return [(self.chunks[i], scores[i]) for i in top_k_idx]

    def bm25_search(self, query, top_k=5):
        scores = self.bm25.get_scores(query.split())
        top_k_idx = np.argsort(scores)[-top_k:][::-1]
        return [(self.chunks[i], scores[i]) for i in top_k_idx]

    def hybrid_search(self, query, top_k=5):
        bm25_results = dict(self.bm25_search(query, top_k))
        dense_results = dict(self.dense_search(query, top_k))
        all_keys = set(bm25_results.keys()) | set(dense_results.keys())
        combined = [(chunk, bm25_results.get(chunk, 0) + dense_results.get(chunk, 0)) for chunk in all_keys]
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]
