import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import plotly.graph_objects as go
import uuid

# Import career assistant utilities
from utils_def_1 import NEO4J_URI, Mistral, MISTRAL_API_KEY, SentenceTransformer, driver, get_mistral_client, \
    EMBED_DIM, mistral_request, MISTRAL_MODEL
from util_func_1 import process_user_query, generate_response

load_dotenv()

# Page config
st.set_page_config(
    page_title="Jain Design Your Degree - RIASEC",
    page_icon="🎓",
    layout="wide"
)

# Neo4j Configuration (use imported driver if available, else create new)
if not driver:
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


    @st.cache_resource
    def get_driver():
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            return driver
        except Exception as e:
            st.error(f"Neo4j connection failed: {e}")
            return None


    driver = get_driver()
else:
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# RIASEC Questions
RIASEC_QUESTIONS = [
    ("I like to work on cars", "R"),
    ("I like to build things", "R"),
    ("I like putting things together or assembling things.", "R"),
    ("I like to take care of animals", "R"),
    ("I like to cook", "R"),
    ("I am a practical person", "R"),
    ("I like working outdoors", "R"),
    ("I like working with numbers or charts", "I"),
    ("I'm good at math", "I"),
    ("I enjoy trying to figure out how things work", "I"),
    ("I like to analyze things (problems/situations)", "I"),
    ("I like to do puzzles", "I"),
    ("I enjoy science", "I"),
    ("I like to do experiments", "I"),
    ("I like to teach or train people", "S"),
    ("I like to play instruments or sing", "A"),
    ("I like to read about art and music", "A"),
    ("I like to draw", "A"),
    ("I enjoy creative writing", "A"),
    ("I am a creative person", "A"),
    ("I like acting in plays", "A"),
    ("I like helping people", "S"),
    ("I like to get into discussions about issues", "S"),
    ("I enjoy learning about other cultures", "S"),
    ("I am interested in healing people", "S"),
    ("I like trying to help people solve their problems", "S"),
    ("I like to work in teams", "S"),
    ("I am an ambitious person, I set goals for myself", "E"),
    ("I would like to start my own business", "E"),
    ("I am quick to take on new responsibilities", "E"),
    ("I like selling things", "E"),
    ("I like to lead", "E"),
    ("I like to try to influence or persuade people", "E"),
    ("I like to give speeches", "E"),
    ("I like to organize things, (files, desks/offices)", "C"),
    ("I like to have clear instructions to follow", "C"),
    ("I wouldn't mind working 8 hours per day in an office", "C"),
    ("I pay attention to details", "C"),
    ("I like to do filing or typing", "C"),
    ("I am good at keeping records of my work", "C"),
    ("I would like to work in an office", "C"),
]


# ==================== CHAT HISTORY FUNCTIONS ====================

def save_chat_message(username, role, content, is_code=False):
    """Save a single chat message to Neo4j"""
    if not driver:
        return

    try:
        with driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MATCH (u:User {username: $username})
                CREATE (m:ChatMessage {
                    id: $msg_id,
                    role: $role,
                    content: $content,
                    is_code: $is_code,
                    timestamp: datetime()
                })
                CREATE (u)-[:HAS_MESSAGE]->(m)
            """,
                  username=username,
                  msg_id=str(uuid.uuid4()),
                  role=role,
                  content=content,
                  is_code=is_code)
    except Exception as e:
        st.sidebar.warning(f"Failed to save message: {e}")


def load_chat_history(username):
    """Load chat history from Neo4j for a specific user"""
    if not driver:
        return []

    try:
        with driver.session(database=NEO4J_DATABASE) as s:
            result = s.run("""
                MATCH (u:User {username: $username})-[:HAS_MESSAGE]->(m:ChatMessage)
                RETURN m.role AS role, m.content AS content, m.is_code AS is_code, m.timestamp AS timestamp
                ORDER BY m.timestamp ASC
            """, username=username)

            messages = []
            for record in result:
                msg = {
                    "role": record["role"],
                    "content": record["content"]
                }
                if record.get("is_code"):
                    msg["is_code"] = True
                messages.append(msg)

            return messages
    except Exception as e:
        st.sidebar.warning(f"Failed to load chat history: {e}")
        return []


def clear_chat_history(username):
    """Clear all chat messages for a specific user"""
    if not driver:
        return

    try:
        with driver.session(database=NEO4J_DATABASE) as s:
            s.run("""
                MATCH (u:User {username: $username})-[:HAS_MESSAGE]->(m:ChatMessage)
                DETACH DELETE m
            """, username=username)
    except Exception as e:
        st.sidebar.warning(f"Failed to clear chat history: {e}")


# ==================== HELPER FUNCTIONS ====================

def calculate_riasec_scores(answers):
    data = []
    for question, flag in RIASEC_QUESTIONS:
        response = answers.get(question, 0)
        data.append({'flag': flag, 'response': response})

    df = pd.DataFrame(data)
    trait_stats = df.groupby('flag')['response'].agg(['sum', 'count']).reset_index()
    trait_stats['rate'] = trait_stats['sum'] / trait_stats['count']

    denom = float(trait_stats['rate'].sum())
    if denom == 0:
        scores = {flag: float(1 / 6) for flag in ['R', 'I', 'A', 'S', 'E', 'C']}
    else:
        trait_stats['score'] = trait_stats['rate'] / denom
        scores = {}
        for _, row in trait_stats.iterrows():
            scores[str(row['flag'])] = float(row['score'])

    for flag in ['R', 'I', 'A', 'S', 'E', 'C']:
        if flag not in scores:
            scores[flag] = 0.0

    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        'scores': scores,
        'top3': [str(t[0]) for t in top3],
        'riasec_vector': [float(scores['R']), float(scores['I']), float(scores['A']),
                          float(scores['S']), float(scores['E']), float(scores['C'])]
    }


def get_riasec_trait_description(trait):
    descriptions = {
        'R': 'Realistic - Practical, hands-on, mechanical work',
        'I': 'Investigative - Analytical, scientific, problem-solving',
        'A': 'Artistic - Creative, expressive, innovative',
        'S': 'Social - Helping, teaching, counseling others',
        'E': 'Enterprising - Leading, persuading, managing',
        'C': 'Conventional - Organizing, data management, structured work'
    }
    return descriptions.get(trait, trait)


def get_trait_name(trait):
    names = {
        'R': 'Realistic',
        'I': 'Investigative',
        'A': 'Artistic',
        'S': 'Social',
        'E': 'Enterprising',
        'C': 'Conventional'
    }
    return names.get(trait, trait)


def mask(s: str) -> str:
    if not s:
        return "NOT SET"
    return s[:4] + "..." + s[-4:] if len(s) > 8 else s


def extract_and_save_name(username, message):
    """Extract name from user message and save to profile"""
    if not driver:
        return

    # Simple name extraction (you may want more sophisticated NLP)
    lower_msg = message.lower()
    name_triggers = ["my name is", "i'm", "i am", "call me"]

    for trigger in name_triggers:
        if trigger in lower_msg:
            # Extract potential name after trigger
            start_idx = lower_msg.find(trigger) + len(trigger)
            potential_name = message[start_idx:].strip().split()[0]

            # Clean and save
            if potential_name:
                try:
                    with driver.session(database=NEO4J_DATABASE) as s:
                        s.run("""
                            MATCH (u:User {username: $username})
                            SET u.display_name = $name
                        """, username=username, name=potential_name)
                    return potential_name
                except Exception as e:
                    st.sidebar.warning(f"Failed to save name: {e}")
    return None


# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'chat_loaded' not in st.session_state:
    st.session_state.chat_loaded = False
if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False

# Custom CSS (merged from first code)
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
    }
    .stRadio>div {
        display: flex;
        gap: 10px;
    }
    /* Main container styling */
    .login-container {
        max-width: 450px;
        margin: 0 auto;
        padding: 40px 30px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }

    /* Header styling */
    .login-header {
        text-align: center;
        margin-bottom: 30px;
    }

    .login-title {
        font-size: 2.5em;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        animation: fadeInDown 0.6s ease-out;
    }

    .login-subtitle {
        color: #666;
        font-size: 1.1em;
        font-weight: 500;
        animation: fadeInUp 0.6s ease-out;
    }

    /* Icon styling */
    .login-icon {
        font-size: 4em;
        margin-bottom: 20px;
        animation: bounceIn 0.8s ease-out;
    }

    /* Form styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 12px 15px;
        font-size: 16px;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* Info box styling */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
        animation: slideInLeft 0.5s ease-out;
    }

    /* Divider styling */
    .custom-divider {
        display: flex;
        align-items: center;
        text-align: center;
        margin: 25px 0;
    }

    .custom-divider::before,
    .custom-divider::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid #e0e0e0;
    }

    .custom-divider span {
        padding: 0 15px;
        color: #999;
        font-weight: 500;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }

    .feature-title {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 5px;
    }

    .feature-desc {
        color: #666;
        font-size: 0.9em;
    }

    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .login-container {
            padding: 30px 20px;
        }
        .login-title {
            font-size: 2em;
        }
    }
</style>
""", unsafe_allow_html=True)


# Login function (enhanced with CSS from first code)
def show_login():
    # Create centered layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Header with icon
        st.markdown("""
        <div class="login-header">
            <div class="login-icon">🎓</div>
            <h1 class="login-title">Welcome Back</h1>
            <p class="login-subtitle">Design Your Degree at Jain University</p>
        </div>
        """, unsafe_allow_html=True)

        # Test credentials info
        st.info("🔐 **Test Account**\n\nUsername: `test_101`  \nPassword: `test_101`")

        # Login form
        with st.form("login_form"):
            st.text_input("👤 Username", key="login_username", placeholder="Enter your username")
            st.text_input("🔒 Password", key="login_password", type="password", placeholder="Enter your password")

            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("🚀 Login to Your Account")

            if submit:
                username = st.session_state.login_username
                password = st.session_state.login_password

                if not driver:
                    st.error("❌ Database connection error!")
                    return

                try:
                    with driver.session(database=NEO4J_DATABASE) as s:
                        result = s.run("""
                            MATCH (u:User {username:$username}) 
                            RETURN u.password AS password, u.riasec_completed AS riasec_completed
                        """, username=username).single()

                    if result and check_password_hash(result["password"], password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.chat_loaded = False  # Reset chat loaded flag

                        if result.get("riasec_completed"):
                            st.session_state.page = 'dashboard'
                        else:
                            st.session_state.page = 'survey'
                        st.success("✅ Login successful! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password!")
                except Exception as e:
                    st.error(f"❌ Login error: {e}")

        # Divider
        st.markdown("""
        <div class="custom-divider">
            <span>OR</span>
        </div>
        """, unsafe_allow_html=True)

        # Register button
        if st.button("✨ Create New Account"):
            st.session_state.page = 'register'
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature highlights
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 RIASEC Assessment</div>
            <div class="feature-desc">Discover your personality type and career interests</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">💼 Career Guidance</div>
            <div class="feature-desc">AI-powered course and career recommendations</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">📚 Course Matching</div>
            <div class="feature-desc">Find courses that align with your interests</div>
        </div>
        """, unsafe_allow_html=True)


# Register function
def show_register():
    st.title("🎓 Create Account")
    st.subheader("Jain Design Your Degree")

    with st.form("register_form"):
        username = st.text_input("Username *")
        email = st.text_input("Email (optional)")
        password = st.text_input("Password *", type="password")
        submit = st.form_submit_button("Register")

        if submit:
            if not username or not password:
                st.error("Username and password required!")
                return

            if not driver:
                st.error("Database connection error!")
                return

            hashed_password = generate_password_hash(password)

            try:
                with driver.session(database=NEO4J_DATABASE) as s:
                    exists = s.run("MATCH (u:User {username:$username}) RETURN u",
                                   username=username).single()
                    if exists:
                        st.error("Username already exists!")
                        return

                    s.run("""
                        CREATE (u:User {
                            username: $username, 
                            password: $password, 
                            email: $email,
                            created_at: datetime(),
                            riasec_completed: false
                        })
                    """, username=username, password=hashed_password, email=email)

                st.success("Registered successfully! Please login.")
                st.session_state.page = 'login'
                st.rerun()
            except Exception as e:
                st.error(f"Registration error: {e}")

    if st.button("Already have an account? Login"):
        st.session_state.page = 'login'
        st.rerun()


# Survey function
def show_survey():
    st.title("🎯 RIASEC Career Assessment")
    st.write("Discover your personality type and career interests")
    st.info("Answer each statement with **Yes** or **No** based on your preferences. All questions are mandatory.")

    with st.form("riasec_form"):
        answers = {}
        col1, col2 = st.columns(2)

        for i, (question, flag) in enumerate(RIASEC_QUESTIONS):
            col = col1 if i % 2 == 0 else col2
            with col:
                st.markdown(f"**{i + 1}. {question}**")
                answer = st.radio(f"Question {i + 1}", ["Yes", "No"], key=f"q_{i}", label_visibility="collapsed",
                                  horizontal=True)
                answers[question] = 1 if answer == "Yes" else 0
                st.markdown("---")

        submitted = st.form_submit_button("Submit Survey", type="primary")

        if submitted:
            riasec_results = calculate_riasec_scores(answers)

            if driver:
                try:
                    with driver.session(database=NEO4J_DATABASE) as s:
                        s.run("""
                            MATCH (u:User {username: $username})
                            SET u.riasec_answers = $answers,
                                u.riasec_scores = $scores,
                                u.riasec_top3 = $top3,
                                u.riasec_vector = $vector,
                                u.riasec_completed = true,
                                u.riasec_timestamp = datetime()
                        """,
                              username=st.session_state.username,
                              answers=json.dumps(answers),
                              scores=json.dumps(riasec_results['scores']),
                              top3=riasec_results['top3'],
                              vector=riasec_results['riasec_vector']
                              )

                        # Cosine similarity computation

                        s.run("""
                            MATCH (u:User {username:$username}), (c:Course)
                            WITH u, c,
                                 [x IN u.riasec_vector | coalesce(x, 0.0)] AS uVec,
                                 [x IN c.course_riasec_vector | coalesce(x, 0.0)] AS cVec
                            WITH u, c, uVec, cVec,
                                 reduce(dot=0.0, i IN range(0,size(uVec)-1) | dot + uVec[i]*cVec[i]) AS dotProduct,
                                 reduce(sumA=0.0, i IN range(0,size(uVec)-1) | sumA + uVec[i]*uVec[i]) AS sumASq,
                                 reduce(sumB=0.0, i IN range(0,size(cVec)-1) | sumB + cVec[i]*cVec[i]) AS sumBSq
                            WITH u, c, dotProduct, sqrt(sumASq) AS magA, sqrt(sumBSq) AS magB
                            WITH u, c, dotProduct, magA, magB,
                                 CASE WHEN magA = 0 OR magB = 0 THEN 0 ELSE dotProduct / (magA * magB) END AS cosineSim
                            ORDER BY cosineSim DESC
                            WITH u, collect({course:c, score:cosineSim}) AS matches
                            WITH u, matches[0..3] AS topMatches
                            UNWIND topMatches AS m
                            WITH u, m.course AS course, m.score AS score
                            MERGE (u)-[r:MATCHES_COURSE]->(course)
                            SET r.match_score = score,
                                r.course_title = course.course_title
                        """, username=st.session_state.username)

                    st.success("✅ Survey completed successfully! Courses matched!")
                    st.session_state.page = 'dashboard'
                    st.rerun()

                except Exception as e:
                    st.error(f"Error saving results: {e}")


def show_dashboard():
    st.title(f"Welcome, {st.session_state.username}! 👋")
    st.subheader("Your RIASEC Career Assessment Results")

    if not driver:
        st.error("Database connection error!")
        return

    try:
        with driver.session(database=NEO4J_DATABASE) as s:
            result = s.run("""
                MATCH (u:User {username: $username})
                RETURN u.riasec_scores AS scores,
                       u.riasec_top3 AS top3,

                       toString(u.riasec_timestamp) AS timestamp,
                       u.riasec_completed AS completed
            """, username=st.session_state.username).single()

        if not result or not result.get("completed"):
            st.warning("Please complete the RIASEC survey first.")
            if st.button("Go to Survey"):
                st.session_state.page = 'survey'
                st.rerun()
            return

        scores = json.loads(result["scores"]) if result["scores"] else {}
        top3 = result["top3"] if result["top3"] else ['R', 'I', 'A']
        timestamp = result["timestamp"]

        if timestamp:
            st.caption(f"Completed on: {timestamp}")

        # Create radar chart
        st.subheader("📊 Your RIASEC Profile")

        labels = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
        values = [
            round(scores.get('R', 0) * 100, 1),
            round(scores.get('I', 0) * 100, 1),
            round(scores.get('A', 0) * 100, 1),
            round(scores.get('S', 0) * 100, 1),
            round(scores.get('E', 0) * 100, 1),
            round(scores.get('C', 0) * 100, 1)
        ]

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            marker=dict(color='#667eea'),
            line=dict(color='#764ba2', width=2)
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Top 3 Traits
        st.subheader("🏆 Your Top 3 Personality Traits")

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i, trait in enumerate(top3):
            with cols[i]:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white;'>
                    <h2>#{i + 1}</h2>
                    <h3>{get_trait_name(trait)}</h3>
                    <p>{get_riasec_trait_description(trait)}</p>
                </div>
                """, unsafe_allow_html=True)

        # Trait descriptions
        st.subheader("📖 Understanding RIASEC Personality Types")

        with st.expander("View All Trait Descriptions"):
            descriptions = {
                'R': ('Realistic (Doers)',
                      'Practical, hands-on work with tools, machines, animals. Careers: Engineer, Technician, Mechanic, Architect'),
                'I': ('Investigative (Thinkers)',
                      'Analytical, scientific problem-solving and research. Careers: Scientist, Researcher, Analyst, Doctor'),
                'A': ('Artistic (Creators)',
                      'Creative expression through art, music, writing. Careers: Designer, Artist, Writer, Musician'),
                'S': ('Social (Helpers)',
                      'Working with and helping people. Careers: Teacher, Counselor, Nurse, Social Worker'),
                'E': ('Enterprising (Persuaders)',
                      'Leading, managing, and persuading others. Careers: Manager, Entrepreneur, Sales, Marketing'),
                'C': ('Conventional (Organizers)',
                      'Organizing data, following procedures. Careers: Accountant, Administrator, Banker, Analyst')
            }

            for trait, (title, desc) in descriptions.items():
                st.markdown(f"**{trait} - {title}**")
                st.write(desc)
                st.markdown("---")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake Survey"):
                st.session_state.page = 'survey'
                st.rerun()
        with col2:
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.page = 'login'
                st.session_state.chat_messages = []
                st.session_state.chat_loaded = False
                st.rerun()

    except Exception as e:
        st.error(f"Error loading dashboard: {e}")


# Career Assistant function
def show_assistant():
    st.title("🎓 Jain University — Design Your Degree Course & Career Assistant")
    st.markdown(
        "*Discover courses, prerequisites, and career opportunities to build a customized degree at Jain University*")

    # Load chat history from database if not already loaded
    if not st.session_state.chat_loaded:
        st.session_state.chat_messages = load_chat_history(st.session_state.username)
        st.session_state.chat_loaded = True

    # Sidebar information
    with st.sidebar:
        st.header("🔧 System Status")
        st.write("NEO4J_URI:", mask(NEO4J_URI))
        if os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME"):
            st.write("NEO4J: ✅ Connected")
        else:
            st.write("NEO4J: ❌ NOT SET")
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

        if st.button("🧪 Test Mistral API"):
            if MISTRAL_API_KEY and Mistral is not None:
                try:
                    client = get_mistral_client(MISTRAL_API_KEY)
                    if client:
                        test_messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Say hello in one sentence."}
                        ]
                        response = mistral_request(client, MISTRAL_MODEL, test_messages, max_tokens=50)
                        st.success(f"✅ Mistral API works! Response: {response[:100]}...")
                    else:
                        st.error("❌ Failed to create Mistral client")
                except Exception as e:
                    st.error(f"❌ Mistral API test failed: {str(e)}")
            else:
                st.error("❌ Mistral API key or module missing")

        if st.button("🔄 Clear Chat History"):
            clear_chat_history(st.session_state.username)
            st.session_state.chat_messages = []
            st.session_state.chat_loaded = False
            st.rerun()

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_code"):
                st.code(msg["content"], language="text")
            else:
                st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Ask me about courses, prerequisites, career opportunities, or anything related to Jain University...")

    if user_input and driver:
        # Extract and save name if mentioned
        extract_and_save_name(st.session_state.username, user_input)

        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        save_chat_message(st.session_state.username, "user", user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        # Process query
        with st.spinner("🔍 Searching Jain University database..."):
            query_results = process_user_query(driver, user_input)

        # Get Mistral client
        client = None
        if MISTRAL_API_KEY and Mistral is not None:
            client = get_mistral_client(MISTRAL_API_KEY)

        # Generate response
        try:
            conversation_context = st.session_state.chat_messages[:-1]

            with st.spinner("🤖 AI is thinking..."):
                assistant_response = generate_response(
                    query_results=query_results,
                    user_input=user_input,
                    client=client,
                    conversation_history=conversation_context
                )

            # Display ASCII tree if available
            if query_results['ascii_tree']:
                with st.chat_message("assistant"):
                    st.code(query_results['ascii_tree'], language="text")
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": query_results['ascii_tree'],
                    "is_code": True
                })
                save_chat_message(st.session_state.username, "assistant", query_results['ascii_tree'], True)

            # Display response with streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                displayed_text = ""
                words = assistant_response.split()

                for i, word in enumerate(words):
                    displayed_text += word + " "
                    message_placeholder.markdown(displayed_text + "▌")
                    time.sleep(0.04)

                message_placeholder.markdown(displayed_text.rstrip())

            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})
            save_chat_message(st.session_state.username, "assistant", assistant_response)

        except Exception as e:
            error_msg = f"I apologize, but I encountered an issue. However, I found {len(query_results['courses'])} courses and {len(query_results['jobs'])} job opportunities related to your query."
            with st.chat_message("assistant"):
                st.write(error_msg)
            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
            save_chat_message(st.session_state.username, "assistant", error_msg)
            st.sidebar.error(f"Error: {str(e)}")

        # Display summary
        courses_count = len(query_results['courses'])
        jobs_count = len(query_results['jobs'])

        if courses_count > 0 or jobs_count > 0:
            with st.chat_message("assistant"):
                summary = f"📊 **Search Results:** {courses_count} courses"
                if jobs_count > 0:
                    summary += f", {jobs_count} job opportunities"
                summary += f" | Powered by {EMBED_DIM}D embeddings"
                st.info(summary)

    elif user_input and not driver:
        with st.chat_message("assistant"):
            st.error("❌ Unable to connect to database. Please check connection settings.")


def show_home():
    st.title("Welcome to Jain Design Your Degree")
    st.write("Navigate using the sidebar to access different features.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 RIASEC Assessment
        Discover your personality type and career interests through our comprehensive assessment.

        - Complete the 42-question survey
        - Get personalized trait analysis
        - View your RIASEC profile
        - Match with relevant courses
        """)

    with col2:
        st.markdown("""
        ### 💬 Career Assistant
        Get AI-powered guidance for your academic journey.

        - Explore Jain University courses
        - Discover career opportunities
        - View course prerequisites
        - Get personalized recommendations
        """)


def show_profile():
    st.title(f"Profile: {st.session_state.username}")

    if not driver:
        st.error("Database connection error!")
        return

    try:
        with driver.session(database=NEO4J_DATABASE) as s:
            result = s.run("""
                MATCH (u:User {username: $username})
                RETURN u.email AS email,
                       u.created_at AS created_at,
                       u.riasec_completed AS riasec_completed,
                       u.riasec_top3 AS top3
            """, username=st.session_state.username).single()

        if result:
            st.info(f"📧 Email: {result.get('email', 'Not provided')}")
            st.info(f"📅 Account created: {result.get('created_at', 'Unknown')}")
            st.info(f"✅ RIASEC completed: {'Yes' if result.get('riasec_completed') else 'No'}")

            if result.get('top3'):
                st.success(f"🏆 Your top traits: {', '.join(result['top3'])}")
    except Exception as e:
        st.error(f"Error loading profile: {e}")


# Admin functions
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")


def fetch_all_users():
    """Fetch all users from the database"""
    if not driver:
        return []

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("""
                MATCH (u:User)
                OPTIONAL MATCH (u)-[:HAS_MESSAGE]->(m:ChatMessage)
                WITH u, count(m) as msg_count
                RETURN u.username AS username, 
                       u.email AS email, 
                       u.riasec_vector AS vector,
                       u.riasec_completed AS completed,
                       toString(u.created_at) AS created_at,
                       msg_count
                ORDER BY u.created_at DESC
            """)
            return [dict(record) for record in result]
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return []


def admin_login():
    """Admin login page"""
    st.title("🔑 Admin Login")
    st.markdown("---")

    with st.form("admin_login_form"):
        username = st.text_input("Admin Username")
        password = st.text_input("Admin Password", type="password")
        submit = st.form_submit_button("Login as Admin")

        if submit:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.success("✅ Logged in successfully as Admin")
                st.rerun()
            else:
                st.error("❌ Invalid admin credentials")

    if st.button("← Back to User Page"):
        st.session_state.page = 'login'
        st.rerun()


def admin_dashboard():
    """Admin dashboard with user statistics"""
    st.title("📊 Admin Dashboard")
    st.markdown("---")

    # Fetch user data
    data = fetch_all_users()

    if data:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Users", len(data))
        with col2:
            completed = sum(1 for u in data if u.get('completed'))
            st.metric("RIASEC Completed", completed)
        with col3:
            total_messages = sum(u.get('msg_count', 0) for u in data)
            st.metric("Total Messages", total_messages)
        with col4:
            active_users = sum(1 for u in data if u.get('msg_count', 0) > 0)
            st.metric("Active Users", active_users)

        st.markdown("---")

        # User table
        st.subheader("👥 All Users")
        df = pd.DataFrame(data)

        # Format the dataframe
        if not df.empty:
            display_df = df[['username', 'email', 'completed', 'msg_count', 'created_at']].copy()
            display_df.columns = ['Username', 'Email', 'RIASEC Done', 'Messages', 'Created At']
            display_df['RIASEC Done'] = display_df['RIASEC Done'].apply(lambda x: '✅' if x else '❌')
            display_df['Email'] = display_df['Email'].fillna('Not provided')

            st.dataframe(display_df, use_container_width=True)

        # User details expander
        st.markdown("---")
        st.subheader("🔍 User Details")

        selected_user = st.selectbox("Select a user to view details:",
                                     [u['username'] for u in data])

        if selected_user:
            user_data = next((u for u in data if u['username'] == selected_user), None)

            if user_data:
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Username:** {user_data['username']}")
                    st.write(f"**Email:** {user_data.get('email', 'Not provided')}")
                    st.write(f"**Created:** {user_data.get('created_at', 'Unknown')}")

                with col2:
                    st.write(f"**RIASEC Completed:** {'Yes ✅' if user_data.get('completed') else 'No ❌'}")
                    st.write(f"**Chat Messages:** {user_data.get('msg_count', 0)}")

                # Show RIASEC vector if available
                if user_data.get('vector'):
                    st.write("**RIASEC Vector:**")
                    vector = user_data['vector']
                    traits = ['R', 'I', 'A', 'S', 'E', 'C']
                    vector_str = ", ".join([f"{t}: {round(v, 3)}" for t, v in zip(traits, vector)])
                    st.code(vector_str)
    else:
        st.warning("No users found in the database.")

    st.markdown("---")

    # Logout button
    if st.button("🚪 Logout from Admin"):
        st.session_state.admin_logged_in = False
        st.rerun()


# Main app logic
def main():
    # Check if accessing admin page
    menu = st.sidebar.radio("Navigation", ["User Page", "Admin Login"], key="main_nav")

    if menu == "Admin Login":
        if not st.session_state.admin_logged_in:
            admin_login()
        else:
            admin_dashboard()
        return

    # Regular user flow
    if not st.session_state.logged_in:
        if st.session_state.page == 'register':
            show_register()
        else:
            show_login()
    else:
        # Sidebar navigation for logged-in users
        with st.sidebar:
            st.title("🎓 Navigation")
            st.write(f"Logged in as: **{st.session_state.username}**")
            st.markdown("---")

            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = 'home'
                st.rerun()

            if st.button("📊 Dashboard", use_container_width=True):
                st.session_state.page = 'dashboard'
                st.rerun()

            if st.button("📝 RIASEC Survey", use_container_width=True):
                st.session_state.page = 'survey'
                st.rerun()

            if st.button("💬 Career Assistant", use_container_width=True):
                st.session_state.page = 'assistant'
                st.rerun()

            if st.button("👤 Profile", use_container_width=True):
                st.session_state.page = 'profile'
                st.rerun()

            st.markdown("---")

            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.page = 'login'
                st.session_state.chat_messages = []
                st.session_state.chat_loaded = False
                st.rerun()

        # Main content
        if st.session_state.page == 'home':
            show_home()
        elif st.session_state.page == 'survey':
            show_survey()
        elif st.session_state.page == 'dashboard':
            show_dashboard()
        elif st.session_state.page == 'assistant':
            show_assistant()
        elif st.session_state.page == 'profile':
            show_profile()
        else:
            st.session_state.page = 'home'
            st.rerun()


if __name__ == "__main__":
    main()