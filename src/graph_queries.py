# Central place to store reusable Cypher queries

GET_COURSE_INFO = """
MATCH (c:Course {course_code:$code})
OPTIONAL MATCH (c)-[:REQUIRES]->(p:Course)
RETURN c, collect(p) as prereqs
"""

GET_JOB_MATCHES = """
MATCH (c:Course {course_code:$code})-[r:MATCHES_JOB]->(j:Job)
RETURN j, r.score ORDER BY r.score DESC LIMIT 5
"""
