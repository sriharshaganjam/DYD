# test_neo4j_conn.py
import os, traceback
from neo4j import GraphDatabase, basic_auth

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")  # most Aura DBs use 'neo4j' by default

print("Using:")
print(" NEO4J_URI:", NEO4J_URI)
print(" NEO4J_USER:", NEO4J_USER)
print(" NEO4J_DATABASE:", NEO4J_DB)

if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
    print("Missing env variables. Please set NEO4J_URI, NEO4J_USER (or NEO4J_USERNAME) and NEO4J_PASSWORD")
    raise SystemExit(1)

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
    print("Driver created. Attempting a simple session and RETURN 1 ...")
    with driver.session(database=NEO4J_DB) as session:
        res = session.run("RETURN 1 as x")
        row = res.single()
        print("Query result:", row["x"])
    driver.close()
    print("Success: Able to connect and run a simple query.")
except Exception as e:
    print("ERROR: Exception while connecting to Neo4j:")
    traceback.print_exc()
    # print underlying .__cause__ if present
    cause = getattr(e, "__cause__", None)
    if cause:
        print("Underlying cause:", repr(cause))
    raise SystemExit(2)
