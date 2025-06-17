import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load any Sentence Transformer model (or change later)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Store texts and embeddings
stored_chunks = []
stored_embeddings = []

def embed_and_store(chunks):
    global stored_chunks, stored_embeddings
    embeddings = embedding_model.encode(chunks)
    
    # Convert to numpy float32
    embeddings = np.array(embeddings).astype('float32')

    # Store chunks
    stored_chunks = chunks
    stored_embeddings = embeddings

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_index(index, query, top_k=3):
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return [(stored_chunks[i], distances[0][idx]) for idx, i in enumerate(indices[0])]