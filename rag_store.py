# rag_store.py
from langchain.vectorstores import FAISS

vector_store = None

def set_vector_store(store: FAISS):
    global vector_store
    vector_store = store

def get_vector_store() -> FAISS:
    return vector_store
