import os
import json
from dotenv import load_dotenv

# These are your magic tools 🪄
from EmbeddingsRetrieval.embeddings_retrieval.retrieval import Retriever
from prompt_logic import construct_prompt, ask_llm

# 🍼 Load secret keys
load_dotenv()

# 🧱 Read the dummy chunks your friend made
with open("EmbeddingsRetrieval/data/dummy_chunks.json", "r") as f:
    chunks = json.load(f)

# 🛠️ Use your smart Retriever
retriever = Retriever(chunks)

# 🧠 Ask your question!
question = input("Ask me anything: ")

# 🔍 Find top 2 chunks related to the question
top_chunks = retriever.hybrid_search(question, top_k=2)

# 👀 Print which chunks it found
for chunk, score in top_chunks:
    print(f"Score: {score:.4f} | Chunk: {chunk}")

# 💌 Create the message for Groq
prompt = construct_prompt(question, top_chunks)
print("\n-------- Prompt Sent to LLM --------")
print(prompt)
print("------------------------------------\n")

# 🤖 Ask Groq and get the answer
answer = ask_llm(prompt)
print("-------- LLM Answer --------")
print(answer)
