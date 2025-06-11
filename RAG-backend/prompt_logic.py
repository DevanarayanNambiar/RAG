import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from dotenv import load_dotenv
from EmbeddingsRetrieval.embeddings_retrieval.retrieval import Retriever
from groq import Groq  # ← USES GROQ INSTEAD OF OPENAI

# Load environment variables
load_dotenv()

# Initialize Groq client with API key from .env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def construct_prompt(question, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Chunk {i+1}] {chunk[0]}\n\n"

    prompt = f"Answer the question based on the following context:\n\n{context}\nQuestion: {question}"
    return prompt

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",  # You can change to llama3-70b-8192 if needed
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Load dummy chunks for testing
    with open("EmbeddingsRetrieval/data/dummy_chunks.json", "r") as f:
        chunks = json.load(f)

    retriever = Retriever(chunks)

    # Ask a test question
    question = "What is the motive of G10x?"

    # Get top-k relevant chunks
    top_chunks = retriever.hybrid_search(question, top_k=2)

    for chunk, score in top_chunks:
        print(f"Score: {score:.4f} | Chunk: {chunk}")

    # Build prompt and ask LLM
    prompt = construct_prompt(question, top_chunks)
    print("-------- Prompt Sent to LLM --------")
    print(prompt)
    print("------------------------------------\n")

    answer = ask_llm(prompt)
    print("-------- LLM Answer --------")
    print(answer)
