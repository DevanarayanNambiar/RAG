from embeddings_retrieval.vector_store import VectorStore
import numpy as np

vs = VectorStore()
embeddings = np.array([[0.1]*384, [0.2]*384], dtype='float32')
vs.add(embeddings, ["meta1", "meta2"])

query = np.array([0.1]*384, dtype='float32')
print(vs.search(query))
