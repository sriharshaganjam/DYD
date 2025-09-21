import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from src.neo4j_client import get_driver
from src.utils import get_hf_embedder, cosine_sim

load_dotenv()

COURSE_CSV = os.getenv("COURSE_CSV", "data/Playlist_Course_Data.csv")
JD_CSV = os.getenv("JD_CSV", "data/JD_Grid_with_new_data.csv")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

def main():
    from neo4j import GraphDatabase
    driver = get_driver()
    embedder = get_hf_embedder()

    courses = pd.read_csv(COURSE_CSV, dtype=str).fillna("")
    jds = pd.read_csv(JD_CSV, dtype=str).fillna("")

    def run_query(q, params=None):
        with driver.session() as session:
            session.run(q, params or {})

    # Create Course nodes
    for _, row in courses.iterrows():
        run_query("""
        MERGE (c:Course {course_code:$code})
        SET c.course_title=$title,
            c.subject_area=$subject_area,
            c.prereq_course_codes=$prereq
        """, {
            "code": row.get("course_code") or str(_),
            "title": row.get("course_title",""),
            "subject_area": row.get("Subject_area",""),
            "prereq": row.get("prereq_course_codes","")
        })

    # Create Job nodes
    for idx, row in jds.iterrows():
        job_id = row.get("job_id") or f"job_{idx}"
        run_query("""
        MERGE (j:Job {job_id:$job_id})
        SET j.job_title=$job_title,
            j.ss_jd=$ss_jd
        """, {
            "job_id": job_id,
            "job_title": row.get("job_title",""),
            "ss_jd": row.get("SS_JD", row.get("ss_jd",""))
        })

    # Embeddings
    course_texts = [(row.get("course_code") or str(_),
                    (row.get("course_title","")+" | "+row.get("Subject_area","")).strip())
                    for _, row in courses.iterrows()]
    job_texts = [(row.get("job_id") or f"job_{idx}",
                 row.get("SS_JD", row.get("ss_jd","")) or row.get("job_title",""))
                 for idx, row in jds.iterrows()]

    c_embs = embedder.embed_documents([t for _, t in course_texts])
    j_embs = embedder.embed_documents([t for _, t in job_texts])

    # Store embeddings
    for (code, _), emb in zip(course_texts, c_embs):
        run_query("MATCH (c:Course {course_code:$code}) SET c.embedding=$emb", {"code": code, "emb": emb})
    for (jid, _), emb in zip(job_texts, j_embs):
        run_query("MATCH (j:Job {job_id:$jid}) SET j.embedding=$emb", {"jid": jid, "emb": emb})

    # Vector indexes
    run_query(f"""
    CREATE VECTOR INDEX course_embedding_index IF NOT EXISTS
    FOR (c:Course) ON (c.embedding)
    OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBED_DIM}, `vector.similarity_function`: 'cosine'}}}}
    """)
    run_query(f"""
    CREATE VECTOR INDEX job_embedding_index IF NOT EXISTS
    FOR (j:Job) ON (j.embedding)
    OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBED_DIM}, `vector.similarity_function`: 'cosine'}}}}
    """)

    # Prereq relationships
    for _, row in courses.iterrows():
        code = row.get("course_code") or str(_)
        prereqs = [p.strip() for p in str(row.get("prereq_course_codes","")).split(",") if p.strip()]
        for prereq in prereqs:
            run_query("""
            MATCH (c:Course {course_code:$code})
            MERGE (p:Course {course_code:$prereq})
            MERGE (c)-[:REQUIRES]->(p)
            """, {"code": code, "prereq": prereq})

    # Match courses to jobs
    TOP_K = 5
    job_vecs = np.array(j_embs)
    for (c_code, _), c_emb in zip(course_texts, c_embs):
        sims = [cosine_sim(c_emb, j) for j in job_vecs]
        top_idx = np.argsort(sims)[::-1][:TOP_K]
        for idx in top_idx:
            run_query("""
            MATCH (c:Course {course_code:$c_code})
            MATCH (j:Job {job_id:$jid})
            MERGE (c)-[r:MATCHES_JOB]->(j)
            SET r.score=$score
            """, {"c_code": c_code, "jid": job_texts[idx][0], "score": float(sims[idx])})

    print("✅ Ingestion complete.")

if __name__ == "__main__":
    main()
