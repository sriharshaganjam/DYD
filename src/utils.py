import os
import numpy as np
# use langchain_community or langchain_huggingface depending on your install
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    from langchain_huggingface import HuggingFaceEmbeddings

MODEL = os.getenv("HUGGINGFACE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def get_hf_embedder(model_name=None):
    model_name = model_name or MODEL
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device":"cpu"})

def cosine_sim(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
