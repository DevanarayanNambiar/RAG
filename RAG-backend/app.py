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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
bm25_index = None
bm25_vectorizer = None
uploaded_text = ""  # ✅ Store PDF content globally

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
    global stored_chunks, stored_embeddings, faiss_index, bm25_index, bm25_vectorizer
    stored_chunks = chunks
    embeddings = embedding_model.encode(chunks).astype('float32')
    stored_embeddings = embeddings
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    # BM25 (TF-IDF)
    bm25_vectorizer = TfidfVectorizer().fit(chunks)
    bm25_index = bm25_vectorizer.transform(chunks)

# Dense Search (FAISS)
def dense_retrieve(query, top_k=3):
    if faiss_index is None:
        return []
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [stored_chunks[i] for i in indices[0]]

# Sparse Search (BM25/TF-IDF)
def sparse_retrieve(query, top_k=3):
    if bm25_vectorizer is None:
        return []
    query_vec = bm25_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, bm25_index).flatten()
    top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
    return [stored_chunks[i] for i in top_indices]

# Hybrid Retrieval
def hybrid_retrieve(query, top_k=3):
    dense = dense_retrieve(query, top_k)
    sparse = sparse_retrieve(query, top_k)
    return list(dict.fromkeys(dense + sparse))[:top_k]

@app.route('/')
def index():
    return jsonify({'message': '✅ Hybrid RAG backend is running'})

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_text
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    chunk_size = int(request.form.get("chunkSize", 500))
    filename = secure_filename(file.filename)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(temp_path)
    text = extract_text_from_pdf(temp_path)
    uploaded_text = text
    chunks = chunk_text(text, chunk_size)
    embed_and_store(chunks)
    return jsonify({'message': f'{filename} uploaded and processed ✅', 'chunks': len(chunks)})

@app.route('/summary', methods=['GET'])
def get_summary():
    if not uploaded_text:
        return jsonify({'summary': 'No file uploaded yet.'})

    prompt = f"Summarize the following text briefly:\n\n{uploaded_text[:3000]}"
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    summary = response.choices[0].message.content.strip()
    return jsonify({'summary': summary})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '').strip().lower()
        use_hybrid = data.get("useHybrid", False)

        greetings = ["hi", "hello", "hlo", "hey", "good morning", "good evening"]
        if question in greetings:
            return jsonify({'answer': "Hello! How can I assist you today?", 'sources': []})

        file_questions = ["what is in there", "what this file contains", "what is the paper name", "what does the file contain", "what is the pdf about"]
        if any(q in question for q in file_questions):
            if not uploaded_text:
                return jsonify({'answer': "No file has been uploaded yet. Please upload a PDF first."})
            prompt = f"Based on the following PDF content, summarize what this file is about:\n\n{uploaded_text[:3000]}"
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content.strip()
            return jsonify({'answer': answer, 'sources': []})

        # Hybrid or Dense retrieval
        retrieved = hybrid_retrieve(question) if use_hybrid else dense_retrieve(question)
        context = "\n\n".join(retrieved)

        prompt = f"""You are an assistant answering based on the following content:\n\n{context}\n\nQuestion: {question}\nAnswer:"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({'answer': answer, 'sources': retrieved})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    questions = data.get("questions", [])
    results = []

    for q in questions:
        # Retrieve relevant context using hybrid RAG
        retrieved = hybrid_retrieve(q)
        context = "\n\n".join(retrieved)

        # Generate answer
        answer_prompt = f"""Use this content to answer:\n\n{context}\n\nQuestion: {q}\nAnswer:"""
        llm_response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": answer_prompt}]
        )
        answer = llm_response.choices[0].message.content.strip()

        # Ask LLM to evaluate
        eval_prompt = f"""
        Evaluate the following answer to a question using the provided context. Give scores from 0 to 100 for each metric.

        Metrics:
        1. Faithfulness: Is the answer factually correct based only on the context?
        2. Relevance: Is the answer directly related to the question?
        3. Completeness: Does the answer address all aspects of the question?
        4. Groundedness: Is the answer supported only by the given context?

        Question: {q}
        Context: {context}
        Answer: {answer}

        Reply in JSON like this:
        {{
        "faithfulness": <score>,
        "relevance": <score>,
        "completeness": <score>,
        "groundedness": <score>
        }}
        """

        eval_response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": eval_prompt}]
        )
        try:
            eval_json = eval_response.choices[0].message.content.strip()
            metrics = json.loads(eval_json)
        except Exception as e:
            # Fallback in case parsing fails
            metrics = {"faithfulness": 0, "relevance": 0, "completeness": 0, "groundedness": 0}

        results.append({
            "question": q,
            "answer": answer,
            "sources": retrieved,
            "metrics": metrics
        })

    return jsonify({"evaluations": results})


if __name__ == '__main__':
    app.run(debug=True)