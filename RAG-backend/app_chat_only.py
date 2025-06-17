from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
import os

# Load environment variables
load_dotenv()

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("🔑 GROQ_API_KEY:", GROQ_API_KEY)  # confirm loading

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file!")

# Initialize Groq client
# Directly set the key for testing (bypass .env)
client = Groq(api_key="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")

# Set up Flask
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({'message': '✅ Welcome to the RAG backend API!'})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '')
        print("📩 Question received:", question)

        # Call Groq API
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": question}]
        )

        answer = response.choices[0].message.content.strip()
        print("🤖 Answer:", answer)
        return jsonify({'answer': answer})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
