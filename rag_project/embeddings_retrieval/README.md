# Member 2 Module - Embeddings and Retrieval

This module handles the core logic for:
- Embedding text chunks using SentenceTransformers
- Storing and searching embeddings using FAISS
- Performing hybrid search (dense + sparse retrieval) with BM25 and embeddings

## Structure
- `embed_model.py`: Loads embedding model and encodes text
- `vector_store.py`: FAISS-based vector DB for fast similarity search
- `retrieval.py`: Combines BM25 + dense search for improved relevance
- `test_scripts/`: Simple examples to test each component
