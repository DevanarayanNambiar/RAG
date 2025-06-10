from embeddings_retrieval.retrieval import Retriever

chunks = [
    "Python is a programming language.",
    "RAG stands for Retrieval-Augmented Generation.",
    "This system uses hybrid search combining dense and sparse methods."
]

retriever = Retriever(chunks)
query = "What is RAG?"
print(retriever.hybrid_search(query))
