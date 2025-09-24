# app.py
"""
DYD — Jain University Course & Career Assistant
- Semantic search using sentence transformers embeddings
- Vector similarity matching for courses and jobs
- ASCII dependency tree visualization with robust fallback
- Comprehensive course and job opportunity analysis
- Conversational counselor persona
"""

import os,  time
import streamlit as st
from utils_def import NEO4J_URI, Mistral, MISTRAL_API_KEY, SentenceTransformer, driver, get_mistral_client, \
    EMBED_DIM, mistral_request, MISTRAL_MODEL
from util_func import  process_user_query, \
    generate_response


# ---------- UI setup ----------
st.set_page_config(page_title="Jain University Course & Career Assistant", layout="wide")
st.title("🎓 Jain University — Design Your Degree Course & Career Assistant")
st.markdown("*Discover courses, prerequisites, and career opportunities to build a customized degree that you can curate at Jain University*")

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


# Modified chat interface section (replace the existing one)
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
    st.sidebar.write(f"Debug - Jobs found: {len(query_results['jobs'])}")
    st.sidebar.write(f"Debug - ASCII tree: {'Yes' if query_results['ascii_tree'] else 'No'}")
    st.sidebar.write(f"Debug - Conversation length: {len(st.session_state.messages)}")

    # Get Mistral client
    client = None
    if MISTRAL_API_KEY and Mistral is not None:
        client = get_mistral_client(MISTRAL_API_KEY)

    # Generate AI response with conversation history
    try:
        # Pass conversation history (excluding current user message)
        conversation_context = st.session_state.messages[:-1]  # Exclude the just-added user message

        with st.spinner("🤖 AI is thinking..."):
            assistant_response = generate_response(
                query_results=query_results,
                user_input=user_input,
                client=client,
                conversation_history=conversation_context
            )


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
        st.sidebar.error(f"Response generation error: {str(e)}")

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
        st.error(
            "❌ I'm unable to connect to the Jain University database right now. Please check the connection settings and try again.")








# Add this to your sidebar for testing
with st.sidebar:
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


# Footer
st.markdown("---")
st.markdown("*🎓 Jain University Course & Career Assistant - Helping students navigate their academic and career journey*")