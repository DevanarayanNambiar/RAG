import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = {}

    def add(self, embeddings, meta_list):
        embeddings = np.array(embeddings, dtype='float32')
        self.index.add(embeddings)
        for i, meta in enumerate(meta_list):
            self.metadata[len(self.metadata)] = meta

    def search(self, query_vector, k=5):
        query_vector = np.array([query_vector], dtype='float32')
        D, I = self.index.search(query_vector, k)
        results = [(self.metadata.get(i, None), D[0][j]) for j, i in enumerate(I[0])]
        return results
