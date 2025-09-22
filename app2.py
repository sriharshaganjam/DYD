# app.py
"""
DYD — Jain University Course & Career Assistant
- Semantic search using sentence transformers embeddings
- Vector similarity matching for courses and jobs
- ASCII dependency tree visualization with robust fallback
- Comprehensive course and job opportunity analysis
- Conversational counselor persona
"""

import os, re, json, time, pathlib, traceback
from typing import List, Dict, Any, Optional
import streamlit as st
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
from rapidfuzz import fuzz, process
import numpy as np

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
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

# Embedding model config
EMBED_MODEL_NAME = os.getenv("HUGGINGFACE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# ---------- UI setup ----------
st.set_page_config(page_title="Jain University Course & Career Assistant", layout="wide")
st.title("🎓 Jain University — Course & Career Assistant")
st.markdown("*Discover courses, prerequisites, and career opportunities at Jain University*")

def mask(s: str) -> str:
    if not s:
        return "NOT SET"
    return s[:4] + "..." + s[-4:] if len(s) > 8 else s

# Enhanced sidebar
with st.sidebar:
    st.header("🔧 System Status")
    st.write("NEO4J_URI:", mask(NEO4J_URI))
    if os.getenv("NEO4J_USER"):
        st.write("NEO4J_USER: ✅ Connected")
    elif os.getenv("NEO4J_USERNAME"):
        st.write("NEO4J_USERNAME: ✅ Connected")
    else:
        st.write("NEO4J_USER: ❌ NOT SET")
    st.write("Mistral API:", "✅ Available" if (Mistral is not None and MISTRAL_API_KEY) else "❌ Not Available")
    st.write("Embeddings:", "✅ Available" if SentenceTransformer is not None else "❌ Install sentence-transformers")
    
    st.header("💡 Ask me about:")
    st.markdown("""
    **🎯 Courses at Jain University:**
    - "What courses are available in Computer Science?"
    - "Tell me about Data Science prerequisites"
    - "Which courses lead to AI jobs?"
    
    **💼 Career Opportunities:**
    - "What job opportunities do Jain students get?"
    - "Jobs for software engineering skills"
    - "Career paths in data analytics"
    
    **📊 Course Dependencies:**
    - "Prerequisites for advanced programming"
    - "Show me the dependency tree for ML courses"
    - "What comes after basic statistics?"
    """)

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
    st.sidebar.error("❌ Neo4j: " + str(neo4j_error))
else:
    st.sidebar.success("✅ Connected to Jain University Database")

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

# ---------- Semantic Search Functions ----------
def semantic_search_courses(_drv, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Semantic search for courses using vector similarity"""
    if not embedding_model:
        # Fallback: simple text search
        q = """
        MATCH (c:Course)
        WHERE toLower(c.course_title) CONTAINS toLower($q) OR toLower(c.course_code) CONTAINS toLower($q)
        RETURN c.course_code AS course_code, 
               c.course_title AS title, 
               c.subject_area AS subject_area,
               c.prereq_course_codes AS prereq_codes,
               0.5 as score
        LIMIT $top_k
        """
        return run_read_cypher(_drv, q, {"q": query_text, "top_k": top_k})
    
    # Embed the query
    query_vector = embedding_model.encode(query_text, convert_to_numpy=True)
    
    cypher = """
    CALL db.index.vector.queryNodes('course_embedding_index', $top_k, $query_vector) 
    YIELD node, score
    MATCH (c:Course) WHERE id(c) = id(node)
    OPTIONAL MATCH (c)-[:REQUIRES]->(pre:Course)
    OPTIONAL MATCH (post:Course)-[:REQUIRES]->(c)
    OPTIONAL MATCH (c)-[:BELONGS_TO]->(s:SubjectArea)
    OPTIONAL MATCH (c)-[:MATCHES_JOB]->(j:Job)
    RETURN c.course_code AS course_code,
           c.course_title AS title,
           c.subject_area AS subject_area,
           c.prereq_course_codes AS prereq_codes,
           score,
           collect(DISTINCT pre.course_code) AS direct_prerequisites,
           collect(DISTINCT post.course_code) AS postrequisites,
           collect(DISTINCT s.name) AS subject_areas,
           collect(DISTINCT j.job_title) AS job_matches
    ORDER BY score DESC
    """
    
    try:
        return run_read_cypher(_drv, cypher, {
            'query_vector': query_vector.tolist(),
            'top_k': top_k
        })
    except Exception as e:
        st.sidebar.warning(f"Semantic search error: {e}")
        return []

def semantic_search_jobs(_drv, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Semantic search for jobs using vector similarity"""
    if not embedding_model:
        return []
    
    query_vector = embedding_model.encode(query_text, convert_to_numpy=True)
    
    cypher = """
    CALL db.index.vector.queryNodes('job_embedding_index', $top_k, $query_vector) 
    YIELD node, score
    MATCH (j:Job) WHERE id(j) = id(node)
    OPTIONAL MATCH (c:Course)-[:MATCHES_JOB]->(j)
    OPTIONAL MATCH (s:SubjectArea)-[:MATCHES_JOB]->(j)
    RETURN j.job_id AS job_id,
           j.job_title AS job_title,
           j.ss_jd AS skills_description,
           score,
           collect(DISTINCT c.course_code) AS related_courses,
           collect(DISTINCT c.course_title) AS course_titles,
           collect(DISTINCT s.name) AS related_subjects
    ORDER BY score DESC
    """
    
    try:
        return run_read_cypher(_drv, cypher, {
            'query_vector': query_vector.tolist(),
            'top_k': top_k
        })
    except Exception as e:
        st.sidebar.warning(f"Job search error: {e}")
        return []

# ---------- Robust ASCII Tree Builder (from app.py) ----------
def get_course_dependencies(_drv, course_code: str, direction: str = "prerequisites") -> Dict[str, Any]:
    """Get detailed course dependency information with robust fallback"""
    
    if direction == "prerequisites":
        cypher = """
        MATCH (c:Course {course_code: $course_code})
        OPTIONAL MATCH prereq_path = (c)-[:REQUIRES*1..5]->(pre:Course)
        WITH c, collect(DISTINCT prereq_path) as paths
        RETURN c.course_code AS course_code,
               c.course_title AS title,
               c.prereq_course_codes AS prereq_codes,
               [path in paths WHERE path IS NOT NULL | [node in nodes(path) | {code: node.course_code, title: node.course_title}]] AS prerequisite_paths
        """
    else:
        cypher = """
        MATCH (c:Course {course_code: $course_code})
        OPTIONAL MATCH postreq_path = (c)<-[:REQUIRES*1..5]-(post:Course)
        WITH c, collect(DISTINCT postreq_path) as paths
        RETURN c.course_code AS course_code,
               c.course_title AS title,
               c.prereq_course_codes AS prereq_codes,
               [path in paths WHERE path IS NOT NULL | [node in nodes(path) | {code: node.course_code, title: node.course_title}]] AS postrequisite_paths
        """
    
    result = run_read_cypher(_drv, cypher, {'course_code': course_code})
    return result[0] if result else {}

def build_dependency_tree(_drv, course_code: str, direction: str = "prerequisites") -> Optional[str]:
    """Build ASCII tree visualization with robust fallback to prereq_course_codes property"""
    
    # Get dependency data from database
    deps = get_course_dependencies(_drv, course_code, direction)
    if not deps:
        return None

    # 1) Try to use relationship path data first
    tree_paths = deps.get(f"{direction}_paths", []) or []
    valid_paths = [path for path in tree_paths if path and len(path) > 1]

    # 2) If no relationship paths and looking for postrequisites, search for courses that have this course as prerequisite
    if not valid_paths and direction == "postrequisites":
        # Find courses that list this course in their prereq_course_codes
        search_query = """
        MATCH (c:Course)
        WHERE c.prereq_course_codes CONTAINS $course_code
        RETURN c.course_code AS course_code, c.course_title AS title
        LIMIT 20
        """
        
        dependent_courses = run_read_cypher(_drv, search_query, {"course_code": course_code})
        
        if dependent_courses:
            course_title = deps.get("title", "")
            header = f"{course_code}" + (f" - {course_title}" if course_title else "")
            lines = [f"📚 {header}", f"{'─' * (len(header) + 4)}", f"Courses This Unlocks:"]
            
            for i, dep_course in enumerate(dependent_courses):
                dep_code = dep_course.get("course_code", "")
                dep_title = dep_course.get("title", "")
                display = f"{dep_code}" + (f" - {dep_title[:40]}..." if dep_title and len(dep_title) > 40 else (f" - {dep_title}" if dep_title else ""))
                connector = "└── " if i == len(dependent_courses) - 1 else "├── "
                lines.append(f"{connector}{display}")
            
            return "\n".join(lines)

    # 3) If looking for prerequisites and no relationship paths, fall back to prereq_course_codes CSV property
    if not valid_paths and direction == "prerequisites":
        raw_prereqs = deps.get("prereq_codes") or ""
        # Parse CSV (accept comma, semicolon or newline separated values)
        codes = [c.strip() for c in re.split(r'[,;\n]+', raw_prereqs) if c.strip()]
        if not codes:
            return None

        # Fetch titles for the prerequisite codes
        cypher_titles = """
        UNWIND $codes AS code
        OPTIONAL MATCH (x:Course {course_code: code})
        RETURN code AS code, x.course_title AS title
        """
        titles_map = {}
        try:
            rows = run_read_cypher(_drv, cypher_titles, {"codes": codes})
            for r in rows:
                titles_map[r.get("code")] = r.get("title") or ""
        except Exception:
            titles_map = {}

        # Build simple two-level tree from CSV data
        course_title = deps.get("title", "")
        header = f"{course_code}" + (f" - {course_title}" if course_title else "")
        lines = [f"📚 {header}", f"{'─' * (len(header) + 4)}", f"Prerequisites:"]

        for i, code in enumerate(codes):
            title = titles_map.get(code, "")
            display = f"{code}" + (f" - {title[:40]}..." if title and len(title) > 40 else (f" - {title}" if title else ""))
            connector = "└── " if i == len(codes) - 1 else "├── "
            lines.append(f"{connector}{display}")

        return "\n".join(lines)

    if not valid_paths:
        return None

    # 3) Build tree from relationship path data (multi-hop)
    tree = {}
    for path in valid_paths:
        current = tree
        # Skip first node (target course) and build tree from dependencies
        if direction == "prerequisites":
            for node in path[1:]:
                code = node.get("code")
                title = node.get("title", "")
                display = f"{code}" + (f" - {title[:20]}..." if title and len(title) > 20 else (f" - {title}" if title else ""))
                current = current.setdefault(display, {})
        else:
            # For postrequisites, show what this course unlocks
            for node in path[1:]:
                code = node.get("code")
                title = node.get("title", "")
                display = f"{code}" + (f" - {title[:20]}..." if title and len(title) > 20 else (f" - {title}" if title else ""))
                current = current.setdefault(display, {})

    if not tree:
        return None

    # Render multi-level ASCII tree
    direction_label = "Prerequisites" if direction == "prerequisites" else "Courses This Unlocks"
    course_title = deps.get("title", "")
    header = f"{course_code}" + (f" - {course_title}" if course_title else "")
    lines = [f"📚 {header}", f"{'─' * (len(header) + 4)}", f"{direction_label}:"]

    def render_tree(subtree, prefix=""):
        items = list(subtree.items())
        for i, (display_name, children) in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{display_name}")
            if children:
                extension = "    " if is_last else "│   "
                render_tree(children, prefix + extension)

    render_tree(tree)
    return "\n".join(lines) if len(lines) > 3 else None

# ---------- Conversation Memory Helper ----------
def get_conversation_context() -> str:
    """Extract context from previous conversation messages"""
    if not st.session_state.messages:
        return ""
    
    context_parts = []
    # Look at last 6 messages to get recent context
    recent_messages = st.session_state.messages[-6:] if len(st.session_state.messages) > 6 else st.session_state.messages
    
    for msg in recent_messages:
        if msg["role"] == "user":
            context_parts.append(f"Student asked: {msg['content']}")
        elif msg["role"] == "assistant" and not msg.get("is_code", False):
            # Extract key information from assistant responses
            content = msg["content"]
            if "courses at Jain University" in content:
                # Extract course mentions
                course_matches = re.findall(r'([A-Z]{2,4}-\d{3})', content)
                if course_matches:
                    context_parts.append(f"Previously discussed courses: {', '.join(course_matches[:3])}")
            
            if "career opportunities" in content or "job" in content.lower():
                context_parts.append("Previously discussed career opportunities")
    
    return " | ".join(context_parts) if context_parts else ""

# ---------- Enhanced Query Processing with Memory ----------
def process_user_query(_drv, user_input: str) -> Dict[str, Any]:
    """Process user query with conversation memory context"""
    
    input_lower = user_input.lower()
    conversation_context = get_conversation_context()
    
    results = {
        'courses': [],
        'jobs': [],
        'ascii_tree': None,
        'search_type': 'general',
        'context': 'jain_university',
        'specific_course': None,
        'conversation_context': conversation_context
    }
    
    # Enhanced query processing that considers conversation context
    follow_up_patterns = [
        'tell me more', 'more information', 'learn more', 'details about',
        'what about', 'how about', 'dependencies', 'prerequisites', 'career prospects'
    ]
    
    is_follow_up = any(pattern in input_lower for pattern in follow_up_patterns)
    
    # Check for course code references from context or direct mention
    course_codes = re.findall(r'([A-Za-z]{2,}[-\s]?\d{2,3})', user_input, re.IGNORECASE)
    
    # If it's a follow-up and no specific course mentioned, try to extract from context
    if is_follow_up and not course_codes and conversation_context:
        context_courses = re.findall(r'([A-Z]{2,4}-\d{3})', conversation_context)
        if context_courses:
            course_codes = context_courses[:1]  # Use the most recent course
    
    # Enhanced search query that includes context
    search_query = user_input
    if conversation_context and is_follow_up:
        search_query = f"{user_input} {conversation_context}"
    
    # Check if asking about a specific course by name
    course_keywords = ['programming in python', 'python programming', 'web development', 'data science', 
                      'artificial intelligence', 'machine learning', 'algorithms', 'computer networks',
                      'history', 'mathematics', 'statistics', 'business', 'english']
    
    specific_course_query = None
    for keyword in course_keywords:
        if keyword in input_lower:
            specific_course_query = keyword
            break
    
    # Determine query intent
    is_job_query = any(term in input_lower for term in ['job', 'career', 'work', 'employment', 'opportunity', 'hire'])
    is_prereq_query = any(term in input_lower for term in ['prerequisite', 'prereq', 'pre-req', 'dependency', 'require', 'before', 'what do i need', 'study before'])
    is_postreq_query = any(term in input_lower for term in ['postrequisite', 'postreq', 'post-req', 'leads to', 'next', 'after', 'study after', 'what can i study'])
    is_pathway_query = any(term in input_lower for term in ['pathway', 'learning path', 'study after', 'what next', 'progression'])
    
    # Do semantic search for courses
    course_results = semantic_search_courses(_drv, search_query, top_k=10)
    results['courses'] = course_results
    
    # Process based on intent and content
    if course_codes and (is_prereq_query or is_postreq_query or is_pathway_query):
        # Specific course dependency query
        main_course = course_codes[0].upper().replace(' ', '-')
        direction = "prerequisites" if is_prereq_query else "postrequisites"
        results['ascii_tree'] = build_dependency_tree(_drv, main_course, direction)
        results['search_type'] = f'dependency_{direction}'
        results['specific_course'] = main_course
        
    elif specific_course_query or is_pathway_query:
        # Course pathway/progression query
        results['search_type'] = 'course_pathway'
        
        # If we found courses and user is asking about what comes after, build postrequisite tree
        if course_results and (is_postreq_query or is_pathway_query):
            main_course = course_results[0]['course_code']
            results['ascii_tree'] = build_dependency_tree(_drv, main_course, "postrequisites")
            results['specific_course'] = main_course
        elif course_results and is_prereq_query:
            main_course = course_results[0]['course_code']
            results['ascii_tree'] = build_dependency_tree(_drv, main_course, "prerequisites")
            results['specific_course'] = main_course
        
    elif is_job_query:
        # Job-focused query
        job_results = semantic_search_jobs(_drv, search_query, top_k=8)
        results['jobs'] = job_results
        results['search_type'] = 'job_search'
        
    else:
        # General course search
        results['search_type'] = 'course_search'
        
        # If dependency terms mentioned, try to show tree for top result
        if (is_prereq_query or is_postreq_query) and course_results:
            main_course = course_results[0]['course_code']
            direction = "prerequisites" if is_prereq_query else "postrequisites"
            results['ascii_tree'] = build_dependency_tree(_drv, main_course, direction)
            results['specific_course'] = main_course
    
    return results

# ---------- Response Generation (Enhanced from app2.py) ----------
def generate_response(query_results: Dict[str, Any], user_input: str, client) -> str:
    """Generate counselor-style response that explains learning pathways"""
    
    courses_count = len(query_results['courses'])
    jobs_count = len(query_results['jobs'])
    
    if courses_count == 0 and jobs_count == 0:
        return "I couldn't find specific information about that in our Jain University database. Could you rephrase your question or try asking about a different course or career area?"
    
    response_parts = []
    
    # Counselor-style opening based on query type
    if query_results['search_type'] == 'course_pathway' and courses_count > 0:
        main_course = query_results['courses'][0]
        course_code = main_course.get('course_code', '')
        title = main_course.get('title') or main_course.get('course_title', '')
        subject_area = main_course.get('subject_area', '')
        
        # Opening explanation
        opening = f"Let me tell you about {title}"
        if course_code:
            opening = f"Let me tell you about {course_code} - {title}"
        
        if subject_area:
            opening += f", which is a foundational course in {subject_area} at Jain University."
        else:
            opening += " at Jain University."
        
        response_parts.append(opening)
        response_parts.append("")
        
        # Show courses that use this as prerequisite or related courses
        advanced_courses = []
        for course in query_results['courses'][1:]:
            course_prereqs = course.get('prereq_codes', '') or course.get('direct_prerequisites', [])
            if isinstance(course_prereqs, str):
                course_prereqs = [p.strip() for p in course_prereqs.split(',') if p.strip()]
            
            # Check if current main course is a prerequisite for others
            if course_code and (course_code in str(course_prereqs) or course_code in course_prereqs):
                advanced_courses.append(course)
        
        # Show what this course opens up
        if len(query_results['courses']) > 1:
            response_parts.append("This course serves as a stepping stone to several advanced areas of study:")
            response_parts.append("")
            
            # Show the advanced courses that were found
            courses_to_show = advanced_courses if advanced_courses else query_results['courses'][1:5]
            
            for adv_course in courses_to_show:
                adv_code = adv_course.get('course_code', '')
                adv_title = adv_course.get('title') or adv_course.get('course_title', '')
                adv_subject = adv_course.get('subject_area', '')
                adv_jobs = adv_course.get('job_matches', [])
                
                course_line = f"**{adv_code} - {adv_title}**"
                if adv_subject and adv_subject != subject_area:
                    course_line += f" (moves into {adv_subject})"
                
                response_parts.append(course_line)
                
                if adv_jobs:
                    response_parts.append(f"This can lead to careers like: {', '.join(adv_jobs[:3])}")
                
                response_parts.append("")
        
        # Career opportunities from main course
        career_opps = main_course.get('job_matches', [])
        if career_opps:
            response_parts.append(f"Even with just {title}, you can already explore career opportunities such as: {', '.join(career_opps[:4])}")
            response_parts.append("")
        
        # Learning pathway advice
        if len(query_results['courses']) > 1:
            response_parts.append("My advice would be to master the fundamentals in this course first, then choose your next step based on your interests:")
            response_parts.append("")
            
            courses_for_advice = advanced_courses if advanced_courses else query_results['courses'][1:4]
            for adv_course in courses_for_advice:
                area_focus = adv_course.get('subject_area', '')
                course_code_adv = adv_course.get('course_code', '')
                course_title_adv = adv_course.get('title') or adv_course.get('course_title', '')
                
                if area_focus == 'AI':
                    response_parts.append(f"- If you're interested in artificial intelligence and machine learning, consider {course_code_adv} next")
                elif 'web' in course_title_adv.lower():
                    response_parts.append(f"- If you want to build websites and web applications, {course_code_adv} would be perfect")
                elif area_focus == 'CS':
                    response_parts.append(f"- To deepen your computer science knowledge, {course_code_adv} is a natural progression")
                else:
                    response_parts.append(f"- For {area_focus} specialization, consider {course_code_adv}")
            
            response_parts.append("")
        
    elif query_results['search_type'] == 'job_search' and jobs_count > 0:
        response_parts.append("Let me guide you through the career opportunities available to Jain University students in this field:")
        response_parts.append("")
        
        # Group jobs by type/level
        entry_jobs = []
        advanced_jobs = []
        
        for job in query_results['jobs'][:6]:
            job_title = job.get('job_title', '').lower()
            if any(term in job_title for term in ['trainee', 'intern', 'graduate', 'junior', 'associate']):
                entry_jobs.append(job)
            else:
                advanced_jobs.append(job)
        
        if entry_jobs:
            response_parts.append("**Starting your career**, you could begin with:")
            response_parts.append("")
            for job in entry_jobs[:3]:
                title = job.get('job_title', '')
                related_courses = job.get('related_courses', [])
                
                job_line = f"- **{title}**"
                if related_courses:
                    job_line += f" (prepare with: {', '.join(related_courses[:2])})"
                response_parts.append(job_line)
            response_parts.append("")
        
        if advanced_jobs:
            response_parts.append("**As you gain experience**, you could progress to:")
            response_parts.append("")
            for job in advanced_jobs[:3]:
                title = job.get('job_title', '')
                job_line = f"- **{title}**"
                response_parts.append(job_line)
            response_parts.append("")
        
        # Course preparation advice
        if courses_count > 0:
            response_parts.append("To prepare for these opportunities, I recommend focusing on these courses at Jain University:")
            response_parts.append("")
            for course in query_results['courses'][:3]:
                course_code = course.get('course_code', '')
                title = course.get('title') or course.get('course_title', '')
                response_parts.append(f"- **{course_code} - {title}**")
            response_parts.append("")
    
    else:
        # General course search - counselor style
        if courses_count > 0:
            response_parts.append("Based on your interests, I found several courses at Jain University that could be great for you:")
            response_parts.append("")
            
            # Group by subject area
            subject_groups = {}
            for course in query_results['courses'][:8]:
                subject = course.get('subject_area', 'Other')
                if subject not in subject_groups:
                    subject_groups[subject] = []
                subject_groups[subject].append(course)
            
            for subject, courses in subject_groups.items():
                if subject != 'Other' and courses:
                    response_parts.append(f"**In {subject}:**")
                    response_parts.append("")
                    
                    for course in courses[:3]:
                        course_code = course.get('course_code', '')
                        title = course.get('title') or course.get('course_title', '')
                        jobs = course.get('job_matches', [])
                        
                        course_line = f"- **{course_code} - {title}**"
                        response_parts.append(course_line)
                        
                        if jobs:
                            response_parts.append(f"  Leads to: {', '.join(jobs[:2])}")
                        response_parts.append("")
    
    # Closing counselor advice
    if query_results['search_type'] == 'course_pathway':
        response_parts.append("Remember, every expert was once a beginner. Focus on building strong fundamentals, and the advanced concepts will become much easier to grasp. What aspect interests you most?")
    elif jobs_count > 0:
        response_parts.append("The key is to start with courses that build foundational skills, then specialize based on where you want your career to go. Each step opens new doors.")
    else:
        response_parts.append("These are excellent options to explore. Consider what type of problems you enjoy solving - that will help guide your choice.")
    
    return "\n".join(response_parts)

# ---------- Chat Interface ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("is_code"):
            st.code(msg["content"], language="text")
        else:
            st.write(msg["content"])

# Main chat interface
user_input = st.chat_input("Ask me about courses, prerequisites, career opportunities, or anything related to Jain University...")

if user_input and driver:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Process the query
    with st.spinner("🔍 Searching Jain University database..."):
        query_results = process_user_query(driver, user_input)
    
    # Debug information
    st.sidebar.write(f"Debug - Query type: {query_results['search_type']}")
    st.sidebar.write(f"Debug - Courses found: {len(query_results['courses'])}")
    st.sidebar.write(f"Debug - ASCII tree: {'Yes' if query_results['ascii_tree'] else 'No'}")
    
    # Get Mistral client
    client = None
    if MISTRAL_API_KEY and Mistral is not None:
        client = get_mistral_client(MISTRAL_API_KEY)
    
    # Generate AI response
    try:
        assistant_response = generate_response(query_results, user_input, client)
        
        # Display response with streaming effect
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            displayed_text = ""
            words = assistant_response.split()
            
            for i, word in enumerate(words):
                displayed_text += word + " "
                message_placeholder.write(displayed_text + "▌")
                time.sleep(0.04)
                
            message_placeholder.write(displayed_text.rstrip())
        
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an issue generating a response. However, I found {len(query_results['courses'])} courses and {len(query_results['jobs'])} job opportunities at Jain University related to your query."
        
        with st.chat_message("assistant"):
            st.write(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Display ASCII dependency tree if available
    if query_results['ascii_tree']:
        with st.chat_message("assistant"):
            st.code(query_results['ascii_tree'], language="text")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": query_results['ascii_tree'], 
            "is_code": True
        })
    
    # Display summary statistics
    courses_count = len(query_results['courses'])
    jobs_count = len(query_results['jobs'])
    
    if courses_count > 0 or jobs_count > 0:
        with st.chat_message("assistant"):
            summary = f"📊 **Search Results:** {courses_count} courses at Jain University"
            if jobs_count > 0:
                summary += f", {jobs_count} job opportunities for Jain students"
            summary += f" | Powered by semantic search with {EMBED_DIM}D embeddings"
            st.info(summary)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"Search completed: {courses_count} courses, {jobs_count} job opportunities found"
        })

elif user_input and not driver:
    with st.chat_message("assistant"):
        st.error("❌ I'm unable to connect to the Jain University database right now. Please check the connection settings and try again.")

# Footer
st.markdown("---")
st.markdown("*🎓 Jain University Course & Career Assistant - Helping students navigate their academic and career journey*")