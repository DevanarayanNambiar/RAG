from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
import os
import fitz  # PyMuPDF
import re
from werkzeug.utils import secure_filename
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Flask setup
app = Flask(__name__)
CORS(app)

# Embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
stored_chunks = []
stored_embeddings = []
faiss_index = None

# PDF Reader
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Chunking
def chunk_text(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks

# Embed + Store
def embed_and_store(chunks):
    global stored_chunks, stored_embeddings, faiss_index
    embeddings = embedding_model.encode(chunks).astype('float32')
    stored_chunks = chunks
    stored_embeddings = embeddings
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

# Search
def retrieve_top_chunks(query, top_k=3):
    if faiss_index is None:
        return []
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [stored_chunks[i] for i in indices[0]]

# Routes
@app.route('/')
def index():
    return jsonify({'message': '✅ RAG backend is running'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(temp_path)
    text = extract_text_from_pdf(temp_path)
    chunks = chunk_text(text)
    embed_and_store(chunks)
    return jsonify({'message': f'{filename} uploaded and processed ✅', 'chunks': len(chunks)})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '').strip().lower()

        # ✅ Handle simple greetings
        if question in ["hi", "hello", "hlo", "hey", "good morning", "good evening"]:
            return jsonify({'answer': "Hello! How can I assist you today?", 'sources': []})

        # 🔍 Retrieve relevant context
        retrieved = retrieve_top_chunks(question)
        context = "\n\n".join(retrieved)

        # 🧠 LLM Prompt
        prompt = f"""You are an assistant answering based on the following content:\n\n{context}\n\nQuestion: {question}\nAnswer:"""

        # 🔗 Call LLM
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({'answer': answer, 'sources': retrieved})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
