from sentence_transformers import SentenceTransformer
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
def embed_chunks(chunks):
    model = load_model()
    return model.encode(chunks, convert_to_tensor=False)
