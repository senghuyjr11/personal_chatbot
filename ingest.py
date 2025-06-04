import os
import shutil

import torch
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# === Clear existing DB ===
db_path = "rag_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)

# === Load and split in one pass ===
with open("data/kyojuro.txt", "r", encoding="utf-8") as f:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,  # Increased chunk size
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? "]  # Better for natural text
    )
    chunks = splitter.create_documents([f.read()])

# === Faster embeddings with GPU/quantization ===
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # 30% faster than MiniLM
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 128  # Process in batches
    }
)

# === Faster DB creation ===
vectordb = Chroma.from_documents(
    chunks,
    embedding=embedding_model,
    persist_directory=db_path,
    client_settings=chromadb.config.Settings(anonymized_telemetry=False)
)
print("Vector DB created at:", db_path)
