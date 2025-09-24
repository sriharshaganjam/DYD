
import os,  pathlib
from dotenv import load_dotenv
from typing import List, Dict, Any
from neo4j import GraphDatabase, basic_auth
import streamlit as st

# Import sentence transformers for query embedding
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# try import mistralai
try:
    from mistralai import Mistral
except Exception:
    Mistral = None

# ---------- Load .env ----------
project_root = pathlib.Path(__file__).resolve().parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    load_dotenv()

# ---------- Config ----------
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "open-mixtral-8x7b")

# Embedding model config
EMBED_MODEL_NAME = os.getenv("HUGGINGFACE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "384"))




def run_read_cypher(drv, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    params = params or {}
    with drv.session(database=NEO4J_DB) as s:
        res = s.run(query, params)
        return [r.data() for r in res]

# ---------- Mistral API Wrapper ----------
@st.cache_resource
def get_mistral_client(api_key: str):
    if Mistral is None or not api_key:
        return None
    return Mistral(api_key=api_key)

def mistral_request(client, model, messages, max_tokens=512, temperature=0.2):
    if client is None:
        raise RuntimeError("Mistral client not available")

    resp = None
    errors = []

    # Try different API patterns
    try:
        if hasattr(client, "chat") and hasattr(client.chat, "complete"):
            resp = client.chat.complete(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature
            )
    except Exception as e:
        errors.append(("chat.complete", str(e)))

    if resp is None:
        try:
            if hasattr(client, "chat_completion"):
                resp = client.chat_completion(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature
                )
        except Exception as e:
            errors.append(("chat_completion", str(e)))

    if resp is None:
        try:
            if hasattr(client, "chat") and callable(client.chat):
                resp = client.chat(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature
                )
        except Exception as e:
            errors.append(("chat", str(e)))

    if resp is None:
        raise RuntimeError(f"Unable to call Mistral API. Tried: {errors}")

    # Extract response
    try:
        return resp.choices[0].message.content
    except:
        pass
    try:
        return resp.choices[0].text
    except:
        pass
    try:
        return resp.content
    except:
        pass
    try:
        return resp.message.content
    except:
        pass
    return str(resp)



#The general

# ---------- Neo4j Connection ----------
@st.cache_resource
def create_driver(uri: str, user: str, password: str, database: str = "neo4j"):
    try:
        driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        with driver.session(database=database) as s:
            s.run("RETURN 1").single()
        return driver, None
    except Exception as e:
        return None, str(e)

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(EMBED_MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

embedding_model = load_embedding_model()

driver = None
neo4j_error = None
if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
    driver, neo4j_error = create_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, database=NEO4J_DB)
else:
    neo4j_error = "NEO4J credentials missing"

if neo4j_error:
    st.sidebar.error(" Neo4j: " + str(neo4j_error))
    st.error("Cannot connect to database. Please check configuration.")
    st.stop()







