
import re
from typing import List, Dict, Any, Optional
import streamlit as st

from utils_def import embedding_model, run_read_cypher, mistral_request, MISTRAL_MODEL







# Performs semantic search on Course nodes using vector similarity (if embedding_model is available).
# Falls back to a simple text-based search (title/code match) when no embeddings are set.
# Supports retrieving related prerequisites, postrequisites, subject areas, and matched jobs.
# Returns a list of dictionaries with course metadata and similarity score.

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


















# Performs semantic search on Job nodes using vector similarity (if embedding_model is available).
# Retrieves jobs most relevant to the query along with their related courses and subject areas.
# Returns a list of dictionaries containing job details, similarity scores, and linked course/subject metadata.



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













# Retrieves a course’s prerequisite or postrequisite dependency paths up to 5 levels deep.
# Uses Cypher path queries to collect chains of related courses, returning them as structured dictionaries.
# Direction can be "prerequisites" (courses required before) or "postrequisites" (courses that depend on this one).
# Returns a single dictionary with course metadata and the dependency paths (if any).

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















# Builds an ASCII dependency tree for a given course, showing either prerequisites or postrequisites.
# Priority order for building the tree:
#   1) Multi-hop relationship paths (up to 5 levels deep).
#   2) Direct lookup of dependent courses (for postrequisites) via prereq_course_codes.
#   3) Fallback to CSV-style prereq_course_codes property (for prerequisites).
# Returns a formatted ASCII tree with course codes and truncated titles, or None if no dependencies found.


def build_dependency_tree(_drv, course_code: str, direction: str = "prerequisites") -> Optional[str]:
    """Build ASCII tree visualization with robust fallback to prereq_course_codes property

    NOTE: This is a minimal, backward-compatible fix for the prerequisite lookup.
    It uses the correct keys returned by the Cypher query (prerequisite_paths / postrequisite_paths),
    and falls back to parsing CSV-style prereq fields (handles both prereq_codes and prereq_course_codes).
    """
    # Get dependency data from database
    deps = get_course_dependencies(_drv, course_code, direction)
    if not deps:
        return None

    # Determine correct key names returned by the Cypher in get_course_dependencies
    if direction == "prerequisites":
        paths_key = "prerequisite_paths"
    else:
        paths_key = "postrequisite_paths"

    # 1) Try to use relationship path data first
    tree_paths = deps.get(paths_key, []) or []
    # valid_paths are those with at least one dependency node beyond the root
    valid_paths = [path for path in tree_paths if path and len(path) > 1]

    # 2) If no relationship paths and looking for postrequisites, search for courses that have this course as prerequisite
    if not valid_paths and direction == "postrequisites":
        search_query = """
        MATCH (c:Course)
        WHERE c.prereq_course_codes CONTAINS $course_code
        RETURN c.course_code AS course_code, c.course_title AS title
        LIMIT 200
        """
        dependent_courses = []
        try:
            dependent_courses = run_read_cypher(_drv, search_query, {"course_code": course_code})
        except Exception:
            dependent_courses = []

        if dependent_courses:
            course_title = deps.get("title", "")
            header = f"{course_code}" + (f" - {course_title}" if course_title else "")
            lines = [f"📚 {header}", f"{'─' * (len(header) + 4)}", f"Courses This Unlocks:"]

            for i, dep_course in enumerate(dependent_courses):
                dep_code = dep_course.get("course_code", "")
                dep_title = dep_course.get("title", "")
                display = f"{dep_code}" + (f" - {dep_title[:40]}..." if dep_title and len(dep_title) > 40 else (
                    f" - {dep_title}" if dep_title else ""))
                connector = "└── " if i == len(dependent_courses) - 1 else "├── "
                lines.append(f"{connector}{display}")

            return "\n".join(lines)

    # 3) If looking for prerequisites and no relationship paths, fall back to prereq_course_codes CSV property
    if not valid_paths and direction == "prerequisites":
        # Try both common keys
        raw_prereqs = deps.get("prereq_codes") or deps.get("prereq_course_codes") or ""
        # Parse CSV-ish formats (comma, semicolon, newline)
        codes = [c.strip() for c in re.split(r'[,;\n]+', raw_prereqs) if c.strip()]
        if not codes:
            # no prereq codes found — return None so caller can proceed
            return None

        # Fetch titles for the prerequisite codes (best-effort)
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

        # Build simple two-level tree from CSV data (preserve original formatting)
        course_title = deps.get("title", "")
        header = f"{course_code}" + (f" - {course_title}" if course_title else "")
        lines = [f"📚 {header}", f"{'─' * (len(header) + 4)}", f"Prerequisites:"]

        for i, code in enumerate(codes):
            title = titles_map.get(code, "")
            display = f"{code}" + (
                f" - {title[:40]}..." if title and len(title) > 40 else (f" - {title}" if title else ""))
            connector = "└── " if i == len(codes) - 1 else "├── "
            lines.append(f"{connector}{display}")

        return "\n".join(lines)

    # 4) If we have relationship paths (multi-hop), build multi-level tree exactly as before
    if not valid_paths:
        return None

    tree = {}
    for path in valid_paths:
        current = tree
        # Skip first node (target course) and build tree from dependencies
        if direction == "prerequisites":
            for node in path[1:]:
                code = node.get("code")
                title = node.get("title", "")
                display = f"{code}" + (
                    f" - {title[:20]}." if title and len(title) > 20 else (f" - {title}" if title else ""))
                current = current.setdefault(display, {})
        else:
            # For postrequisites, show what this course unlocks
            for node in path[1:]:
                code = node.get("code")
                title = node.get("title", "")
                display = f"{code}" + (
                    f" - {title[:20]}." if title and len(title) > 20 else (f" - {title}" if title else ""))
                current = current.setdefault(display, {})

    if not tree:
        return None

    # Render multi-level ASCII tree (same rendering logic as original)
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



















# Detects whether a user input is casual conversation (greetings, short replies, bot questions, noise)
# versus a meaningful course/career-related query.
# Uses regex patterns for common casual phrases and also checks for very short inputs
# that don’t contain course/career keywords. Returns True if casual, False otherwise.


def detect_casual_conversation(user_input: str) -> bool:
    """Detect if user input is casual conversation vs course/career query"""
    input_lower = user_input.lower().strip()

    # Greetings and casual phrases
    casual_patterns = [
        # Greetings
        r'^(hi|hello|hey+|hii+|sup|what\'s up)$',
        r'^good (morning|afternoon|evening)$',

        # Short responses
        r'^(ok|okay|yes|no|yep|nope|sure|thanks|thank you)$',

        # Questions about the bot
        r'^(who are you|what are you|what can you do)$',

        # Very short inputs (less than 3 chars, mostly punctuation/repetition)
        r'^.{1,2}$',
        r'^(.)\1{2,}$',  # repeated characters like "aaaa" or "!!!!"
    ]

    # Check if it matches casual patterns
    for pattern in casual_patterns:
        if re.match(pattern, input_lower):
            return True

    # Check if it's too short and doesn't contain course/career keywords
    if len(input_lower) < 4:
        course_keywords = ['course', 'class', 'job', 'career', 'study', 'learn', 'degree']
        if not any(keyword in input_lower for keyword in course_keywords):
            return True

    return False













# Generates friendly, context-aware responses for casual user inputs (greetings, small talk, short replies).
# Uses regex to detect patterns like greetings, bot questions, acknowledgments, or thanks.
# Returns tailored responses that reintroduce the bot’s role (Jain University Course & Career Assistant)
# and guide the user back toward course or career-related queries.
# Falls back to a default helper message if no casual pattern is matched.


def get_conversational_response(user_input: str) -> str:
    """Generate appropriate conversational responses for casual inputs"""
    input_lower = user_input.lower().strip()

    # Greetings
    if re.match(r'^(hi|hello|hey+|hii+)$', input_lower):
        return """Hi there! 👋 

I'm your Jain University Course & Career Assistant. I'm here to help you explore:

🎓 **Courses** - Find courses that match your interests
📚 **Prerequisites** - Discover what you need to study before advanced courses  
💼 **Career Paths** - Explore job opportunities for Jain University graduates
🗺️ **Learning Pathways** - Plan your academic journey

What would you like to know about? You can ask me things like:
- "What courses are available in Computer Science?"
- "What are the prerequisites for Data Science?"
- "What jobs can I get with an AI degree?"
- "Show me the learning pathway for becoming a software engineer"

How can I help you today?"""

    elif re.match(r'^(what\'s up|sup)$', input_lower):
        return """Not much! Just here waiting to help Jain University students like you navigate their academic journey! 🎓

I can help you discover courses, understand prerequisites, explore career opportunities, and plan your learning pathway.

What's on your mind? Looking for a specific course or career advice?"""

    elif re.match(r'^good (morning|afternoon|evening)$', input_lower):
        time_of_day = re.search(r'(morning|afternoon|evening)', input_lower).group(1)
        return f"""Good {time_of_day}! ☀️

Ready to explore what Jain University has to offer? I can help you with:
- Finding the right courses for your goals
- Understanding course prerequisites and dependencies  
- Discovering career opportunities
- Planning your academic pathway

What would you like to learn about today?"""

    elif re.match(r'^(who are you|what are you)$', input_lower):
        return """I'm the Jain University Course & Career Assistant! 🎓

I'm an AI assistant specifically designed to help Jain University students navigate their academic and career journey. I have access to:

📚 **Course Database** - All available courses with their details and prerequisites
🔗 **Dependency Mapping** - How courses connect and build upon each other
💼 **Career Opportunities** - Job prospects for different academic paths
🤖 **Smart Search** - I can understand your interests and find relevant courses

I use semantic search and vector embeddings to understand what you're looking for, even if you don't know the exact course names!

What can I help you discover today?"""

    elif re.match(r'^what can you do$', input_lower):
        return """Great question! Here's what I can help you with at Jain University: 🌟

🔍 **Course Discovery**
- Find courses based on your interests (even vague descriptions!)
- Search by subject area, career goals, or skills you want to learn

📊 **Prerequisite Analysis** 
- Show you what courses you need to take before others
- Display dependency trees so you can plan your path

🚀 **Career Guidance**
- Explore job opportunities for different academic paths
- See which courses lead to specific careers
- Understand the job market for Jain graduates

🗺️ **Academic Planning**
- Plan your learning progression
- Find the best sequence of courses for your goals
- Discover advanced courses you can take after mastering fundamentals

Just ask me naturally! For example:
- "I want to learn about artificial intelligence"
- "What do I need to study before advanced programming?"
- "What jobs can I get with a data science background?"

What interests you most?"""

    elif re.match(r'^(ok|okay|yes|yep|sure)$', input_lower):
        return """Perfect! What would you like to explore next? 

I'm here to help with anything related to Jain University courses and career planning. Feel free to ask about any subject that interests you! 🎓"""

    elif re.match(r'^(thanks|thank you)$', input_lower):
        return """You're welcome! 😊

I'm always here whenever you need help navigating Jain University's courses and career opportunities. Feel free to ask me anything else!

Good luck with your studies! 🎓✨"""

    elif re.match(r'^(no|nope)$', input_lower):
        return """No worries! If you change your mind or have any questions about Jain University courses or career paths, I'm here to help.

Is there anything else I can assist you with? 🎓"""

    else:
        # Default response for other casual inputs
        return """I'm not sure what you meant by that, but I'm here to help! 😊

I specialize in helping Jain University students with:
- Course recommendations and information
- Understanding prerequisites and dependencies
- Career guidance and job opportunities  
- Academic pathway planning

Try asking me something like "What courses are good for AI?" or "Show me computer science prerequisites" and I'll give you detailed, helpful information!

What would you like to know about?"""


















# Main query router: interprets user input and decides how to process it.
# Steps:
#   1. Detect and handle casual conversation (returns canned responses).
#   2. Identify intent using regex/keywords (job query, prerequisite/postrequisite, pathway, or general).
#   3. Run semantic search on courses (and jobs if relevant).
#   4. Build dependency trees when prerequisites/postrequisites/pathways are requested.
#   5. Populate a results dict with courses, jobs, ascii_tree (if any), search_type, context, and specific_course.
# Returns a structured dictionary that downstream components can render consistently.




# ---------- Query Processing ----------
def process_user_query(_drv, user_input: str) -> Dict[str, Any]:
    """Process user query and determine appropriate search strategy"""
    """Process user query and determine appropriate search strategy"""

    # NEW: Check for casual conversation first
    if detect_casual_conversation(user_input):
        return {
            'courses': [],
            'jobs': [],
            'ascii_tree': None,
            'search_type': 'casual_conversation',
            'context': 'jain_university',
            'specific_course': None,
            'conversational_response': get_conversational_response(user_input)
        }

    # Rest of your existing logic remains the same...
    input_lower = user_input.lower()
    results = {
        'courses': [],
        'jobs': [],
        'ascii_tree': None,
        'search_type': 'general',
        'context': 'jain_university',
        'specific_course': None
    }

    # Check for specific course codes or course names
    course_codes = re.findall(r'([A-Za-z]{2,}[-\s]?\d{2,3})', user_input, re.IGNORECASE)

    # Check if asking about a specific course by name
    course_keywords = ['programming in python', 'python programming', 'web development', 'data science',
                       'artificial intelligence', 'machine learning', 'algorithms', 'computer networks']

    specific_course_query = None
    for keyword in course_keywords:
        if keyword in input_lower:
            specific_course_query = keyword
            break

    # Determine query intent
    is_job_query = any(term in input_lower for term in ['job', 'career', 'work', 'employment', 'opportunity', 'hire'])
    is_prereq_query = any(term in input_lower for term in
                          ['prerequisite', 'prereq', 'pre-req', 'dependency', 'require', 'before', 'what do i need',
                           'study before', "pre req"])
    is_postreq_query = any(term in input_lower for term in
                           ['postrequisite', 'postreq', 'post-req', 'leads to', 'next', 'after', 'study after',
                            'what can i study', "post req"])
    is_pathway_query = any(
        term in input_lower for term in ['pathway', 'learning path', 'study after', 'what next', 'progression'])

    # Do semantic search for courses
    course_results = semantic_search_courses(_drv, user_input, top_k=10)
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
        job_results = semantic_search_jobs(_drv, user_input, top_k=8)
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















# ---------- Response Generation ----------
# Orchestrates how final answers are generated for the student:
#   • Handles casual conversation directly with canned responses (no LLM call).
#   • For explicit prerequisite queries, returns ASCII dependency trees or a graceful fallback.
#   • For all other queries, builds a rich context (courses, jobs, prior conversation) and
#     sends it to the LLM with a counselor-style system prompt.
#   • Uses generate_fallback_response() if no LLM client or if errors occur.
#   • Ensures responses stay student-friendly, concise, and focused on Jain University’s
#     courses, prerequisites, pathways, and career opportunities.



# ---------- Response Generation (Enhanced from app2.py) ----------
# Enhanced generate_response function with conversation history
def generate_response(query_results: Dict[str, Any], user_input: str, client,
                      conversation_history: List[Dict] = None) -> str:
    """
    Generate response for user queries with conversation context.

    Args:
        query_results: Results from process_user_query
        user_input: Current user input
        client: Mistral client
        conversation_history: List of previous messages for context
    """

    # Handle casual conversation (no LLM needed)
    if query_results.get("search_type") == "casual_conversation":
        return query_results.get("conversational_response",
                                 "Hello! How can I help you with Jain University courses today?")

    # Handle explicit prerequisite queries (return tree only)
    if query_results.get("search_type", "").startswith("dependency_prerequisites"):
        tree = query_results.get("ascii_tree")
        if tree:
            return tree
        else:
            top = None
            if query_results.get("courses"):
                top = query_results["courses"][0]
            if top and top.get("course_code"):
                return f"I couldn't find explicit prerequisite relationships in the database for {top.get('course_code')}. However, the course most closely matching your query is {top.get('course_code')} - {top.get('title', '')}. It appears there are no recorded prerequisites for that course."
            return "I couldn't find prerequisites for that course in the database."

    # For all other queries that need LLM response, use conversation context
    if not client:
        # Fallback to rule-based response if no LLM available
        return generate_fallback_response(query_results, user_input)

    # Prepare conversation context (last 2-3 exchanges)
    context_messages = []
    if conversation_history:
        # Get last 6 messages (3 exchanges: user->assistant->user->assistant->user->assistant)
        recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history

        for msg in recent_messages:
            # Skip code/tree messages from context to avoid clutter
            if not msg.get("is_code", False):
                role = "user" if msg["role"] == "user" else "assistant"
                context_messages.append({"role": role, "content": msg["content"]})

    # Prepare current query context
    courses_info = ""
    jobs_info = ""

    if query_results.get('courses'):
        courses_info = "Available courses:\n"
        for course in query_results['courses'][:5]:  # Top 5 courses
            course_code = course.get('course_code', '')
            title = course.get('title') or course.get('course_title', '')
            subject = course.get('subject_area', '')
            courses_info += f"- {course_code}: {title} ({subject})\n"

    if query_results.get('jobs'):
        jobs_info = "Related job opportunities:\n"
        for job in query_results['jobs'][:3]:  # Top 3 jobs
            job_title = job.get('job_title', '')
            related_courses = job.get('related_courses', [])
            jobs_info += f"- {job_title}"
            if related_courses:
                jobs_info += f" (requires: {', '.join(related_courses[:2])})"
            jobs_info += "\n"

    # System prompt for the counselor persona
    system_prompt = """You are a friendly, knowledgeable academic counselor for Jain University. Your role is to help students navigate their course selections and career paths.

Key guidelines:
- Be conversational, encouraging, and supportive like a real counselor
- Reference the conversation history to provide contextual advice
- Focus on practical guidance for academic and career planning
- Explain course progressions and career pathways clearly
- Keep responses concise but informative (2-3 paragraphs max)
- Use the provided course and job data to give specific recommendations
- If discussing prerequisites or course sequences, explain the learning progression logic"""

    # Build messages for Mistral API
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation context
    messages.extend(context_messages)

    # Add current query with database results
    current_context = f"Student query: {user_input}\n\n"
    if courses_info:
        current_context += courses_info + "\n"
    if jobs_info:
        current_context += jobs_info + "\n"

    # Add search type context for better responses
    search_type = query_results.get('search_type', 'general')
    if search_type == 'job_search':
        current_context += "Focus: This is a career-focused inquiry. Emphasize job opportunities and required skills.\n"
    elif search_type == 'course_pathway':
        current_context += "Focus: This is about course progression. Explain learning pathways and next steps.\n"
    elif 'dependency' in search_type:
        current_context += "Focus: This is about course dependencies. Explain prerequisites and course sequences.\n"

    messages.append({"role": "user", "content": current_context})

    try:
        response = mistral_request(
            client=client,
            model=MISTRAL_MODEL,
            messages=messages,
            max_tokens=600,  # Slightly longer for counselor-style responses
            temperature=0.3  # Slightly more creative but still focused
        )
        return response.strip()

    except Exception as e:
        st.sidebar.error(f"LLM Error: {str(e)}")
        return generate_fallback_response(query_results, user_input)


def generate_fallback_response(query_results: Dict[str, Any], user_input: str) -> str:
    """Fallback response when LLM is not available"""
    courses_count = len(query_results.get('courses', []))
    jobs_count = len(query_results.get('jobs', []))

    if courses_count == 0 and jobs_count == 0:
        return "I couldn't find specific information about that in our Jain University database. Could you rephrase your question or try asking about a different course or career area?"

    response_parts = []

    if courses_count > 0:
        response_parts.append("I found these relevant courses at Jain University:")
        for course in query_results['courses'][:3]:
            course_code = course.get('course_code', '')
            title = course.get('title') or course.get('course_title', '')
            response_parts.append(f"• {course_code} - {title}")

    if jobs_count > 0:
        response_parts.append("\nRelated career opportunities:")
        for job in query_results['jobs'][:3]:
            job_title = job.get('job_title', '')
            response_parts.append(f"• {job_title}")

    return "\n".join(response_parts)



