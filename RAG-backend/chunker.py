import re

def clean_text(text):
    # Remove extra newlines & spaces
    return re.sub(r'\n+', '\n', text.strip())

def chunk_text(text, chunk_size=300, overlap=50):
    sentences = re.split(r'(?<=[.!?]) +', clean_text(text))
    chunks = []
    chunk = []

    total_len = 0
    for sentence in sentences:
        sentence_len = len(sentence.split())
        if total_len + sentence_len > chunk_size:
            chunks.append(" ".join(chunk))
            # Start new chunk with overlap
            chunk = chunk[-overlap:] if overlap > 0 else []
            total_len = sum(len(c.split()) for c in chunk)
        chunk.append(sentence)
        total_len += sentence_len

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks