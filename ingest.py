import os
import shutil
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# === Clear existing DB ===
db_path = "rag_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)

# === Load full personal file ===
with open("data/kyojuro.txt", "r", encoding="utf-8") as f:
    text = f.read()
    documents = [Document(page_content=text)]

# === Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# === Embed and store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory=db_path)

print("Vector DB created and saved at:", db_path)
