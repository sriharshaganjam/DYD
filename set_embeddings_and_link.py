import os
import time
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Aura connection from .env ---
URI = os.getenv("NEO4J_URI")            # e.g. neo4j+s://xxxx.databases.neo4j.io
USER = os.getenv("NEO4J_USERNAME")      # usually "neo4j"
PWD  = os.getenv("NEO4J_PASSWORD")
DB   = os.getenv("NEO4J_DATABASE", "neo4j")

# --- Embedding model ---
MODEL_NAME = os.getenv("HUGGINGFACE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM  = int(os.getenv("EMBEDDING_DIM", "384"))

TOP_K = 5   # number of job matches per course/subject area

# --- Connect to Aura ---
print(f"Connecting to {URI} ...")
driver = GraphDatabase.driver(URI, auth=(USER, PWD))
model = SentenceTransformer(MODEL_NAME)


# -------------------------
# Helper DB functions
# -------------------------
def fetch_courses(tx):
    q = """
    MATCH (c:Course)
    RETURN c.course_code AS code, c.course_title AS title, c.subject_area AS subject
    """
    return [r.data() for r in tx.run(q)]

def fetch_subjectareas(tx):
    q = """
    MATCH (s:SubjectArea)<-[:BELONGS_TO]-(c:Course)
    RETURN s.name AS name, collect(c.course_code) AS course_codes
    """
    return [r.data() for r in tx.run(q)]

def fetch_jobs(tx):
    q = """
    MATCH (j:Job)
    RETURN j.job_id AS job_id, j.job_title AS job_title, j.ss_jd AS ss_jd
    """
    return [r.data() for r in tx.run(q)]

def set_embedding(tx, label, key, key_val, emb):
    q = f"MATCH (n:{label} {{{key}: $val}}) SET n.embedding = $emb"
    tx.run(q, val=key_val, emb=list(map(float, emb)))

def create_vector_index(tx, index_name, label):
    q = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (n:{label})
    ON (n.embedding)
    OPTIONS {{
      indexConfig: {{ `vector.dimensions`: {EMBED_DIM}, `vector.similarity_function`: 'cosine' }}
    }}
    """
    tx.run(q)

def match_jobs_for_node(tx, label, key, key_val, k):
    q = f"""
    MATCH (n:{label} {{{key}: $val}})
    CALL db.index.vector.queryNodes('job_embedding_index', $k, n.embedding) 
      YIELD node, score
    MATCH (j:Job) WHERE id(j) = id(node)
    MERGE (n)-[r:MATCHES_JOB]->(j)
    SET r.score = score
    """
    tx.run(q, val=key_val, k=k)


# -------------------------
# Main logic
# -------------------------
with driver.session(database=DB) as session:
    courses = session.read_transaction(fetch_courses)
    jobs = session.read_transaction(fetch_jobs)
    subjectareas = session.read_transaction(fetch_subjectareas)

print(f"Loaded {len(courses)} courses, {len(jobs)} jobs, {len(subjectareas)} subject areas")

# --- Prepare text for embeddings ---
course_texts = [(c["code"], f"{c.get('title','')} | {c.get('subject','')}".strip()) for c in courses]
job_texts    = [(j["job_id"], j.get("ss_jd") or j.get("job_title") or "") for j in jobs]

# --- Embed in batches ---
def embed_batch(texts, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        part = [t for _,t in texts[i:i+batch]]
        vecs = model.encode(part, convert_to_numpy=True, show_progress_bar=True)
        out.extend(vecs)
    return out

print("Embedding courses ...")
course_vecs = embed_batch(course_texts)
print("Embedding jobs ...")
job_vecs = embed_batch(job_texts)

# --- Store embeddings back ---
with driver.session(database=DB) as session:
    for (code,_), vec in zip(course_texts, course_vecs):
        session.write_transaction(set_embedding, "Course", "course_code", code, vec)
    for (jid,_), vec in zip(job_texts, job_vecs):
        session.write_transaction(set_embedding, "Job", "job_id", jid, vec)

# --- SubjectArea embeddings = mean of member course embeddings ---
course_map = {code: vec for (code,_), vec in zip(course_texts, course_vecs)}
subject_vecs = {}
for s in subjectareas:
    name = s["name"]
    members = [course_map[c] for c in s["course_codes"] if c in course_map]
    if members:
        subject_vecs[name] = np.mean(np.vstack(members), axis=0)
    else:
        subject_vecs[name] = model.encode(name, convert_to_numpy=True)

with driver.session(database=DB) as session:
    for name, vec in subject_vecs.items():
        session.write_transaction(set_embedding, "SubjectArea", "name", name, vec)

# --- Create vector indexes ---
with driver.session(database=DB) as session:
    session.write_transaction(create_vector_index, "course_embedding_index", "Course")
    session.write_transaction(create_vector_index, "job_embedding_index", "Job")
    session.write_transaction(create_vector_index, "subject_embedding_index", "SubjectArea")

print("Vector indexes created (may take a few seconds to build)")
time.sleep(5)

# --- Create MATCHES_JOB relationships ---
with driver.session(database=DB) as session:
    for code,_ in course_texts:
        session.write_transaction(match_jobs_for_node, "Course", "course_code", code, TOP_K)
    for name in subject_vecs.keys():
        session.write_transaction(match_jobs_for_node, "SubjectArea", "name", name, TOP_K)

print("✅ Done: embeddings stored, indexes created, MATCHES_JOB relationships built")
