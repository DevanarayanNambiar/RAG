# ingestion/chunking_pipeline.py

import os
import fitz  # PyMuPDF
import docx
import pandas as pd
import json
import nltk
from typing import List, Dict

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


# ---------- Document Parsers ----------
def parse_pdf(file_path: str) -> List[Dict]:
    doc = fitz.open(file_path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            chunks.append({"text": text.strip(), "metadata": {"page": page_num}})
    return chunks

def parse_docx(file_path: str) -> List[Dict]:
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return [{"text": text.strip(), "metadata": {}}]

def parse_txt(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return [{"text": text.strip(), "metadata": {}}]

def parse_csv(file_path: str) -> List[Dict]:
    df = pd.read_csv(file_path)
    return [{"text": row.to_json(), "metadata": {"row_index": i}} for i, row in df.iterrows()]

def parse_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return [{"text": json.dumps(entry), "metadata": {"entry_index": i}} for i, entry in enumerate(data)]
    else:
        return [{"text": json.dumps(data), "metadata": {}}]


def get_parser_by_extension(extension: str):
    return {
        ".pdf": parse_pdf,
        ".docx": parse_docx,
        ".txt": parse_txt,
        ".csv": parse_csv,
        ".json": parse_json,
    }.get(extension.lower())


# ---------- Chunking ----------
def fixed_chunking(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def sentence_chunking(text: str, max_tokens: int = 512) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count <= max_tokens:
            current_chunk.append(sentence)
            current_len += word_count
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def chunk_document(parsed_chunks: List[Dict], method="fixed", **kwargs) -> List[Dict]:
    chunked_output = []
    for doc in parsed_chunks:
        if method == "sentence":
            chunks = sentence_chunking(doc["text"], **kwargs)
        else:
            chunks = fixed_chunking(doc["text"], **kwargs)
        for i, chunk in enumerate(chunks):
            chunked_output.append({
                "text": chunk,
                "metadata": {**doc["metadata"], "chunk_id": i}
            })
    return chunked_output


# ---------- Example Pipeline Runner ----------
def process_document(file_path: str, method="fixed", **kwargs):
    ext = os.path.splitext(file_path)[-1]
    parser = get_parser_by_extension(ext)
    if not parser:
        raise ValueError(f"Unsupported file type: {ext}")
    parsed = parser(file_path)
    chunked = chunk_document(parsed, method=method, **kwargs)
    return chunked


if __name__ == "__main__":
    chunks = process_document(r"C:\Users\91701\Documents\G10X\sample.txt")
    print(chunks)
