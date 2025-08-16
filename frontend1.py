from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import logging
import requests
import uuid
import hashlib

API_URL = "http://127.0.0.1:9999"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

USERS = {
    "admin": "admin123",
    "guest": "guest123"
}

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()

# -------------------
# Login
# -------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(page_title="Login - AI Chatbot", layout="centered")
    st.title("🔐 Login to AI Chatbot")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.session_id = hash_string(username + str(uuid.uuid4()))
            st.session_state.chat_history = []
            st.rerun()
        else:
            st.error("❌ Invalid username or password")
    st.stop()

# -------------------
# UI Setup
# -------------------
st.set_page_config(page_title="LangGraph Agent UI", layout="wide")
st.sidebar.title("⚙️ Settings")
st.sidebar.write(f"Logged in as **{st.session_state.username}**")

system_prompt = st.sidebar.text_area("System Prompt (optional):", height=70)
provider = st.sidebar.radio("Model Provider:", ("Groq", "OpenAI", "DeepSeek"))
model_options = {
    "Groq": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    "OpenAI": ["gpt-4o-mini"],
    "DeepSeek": ["deepseek-chat"]
}
selected_model = st.sidebar.selectbox("Select Model:", model_options[provider])
allow_web_search = st.sidebar.checkbox("Enable Web Search")
# enable_rag = st.sidebar.checkbox("Enable RAG (Document-Based Search)")

# -------------------
# File Upload + Document List (Only if RAG is enabled)
# -------------------
# if enable_rag:
#     st.subheader("📄 Upload & Manage Documents for RAG")
#     uploaded_file = st.file_uploader("Upload a `.txt` or `.pdf` file", type=["txt", "pdf"])
#     if uploaded_file and st.button("Upload"):
#         files = {"file": (uploaded_file.name, uploaded_file.read())}
#         try:
#             res = requests.post(f"{API_URL}/upload", files=files)
#             data = res.json()
#             if "error" in data:
#                 st.error(f"❌ {data['error']}")
#             else:
#                 st.success(f"✅ Uploaded {uploaded_file.name} - {data.get('chunks', 'no')} chunks processed.")
#         except Exception as e:
#             st.error(f"Error uploading document: {str(e)}")

#     st.markdown("#### 📚 Uploaded Documents")
#     try:
#         res = requests.get(f"{API_URL}/documents")
#         docs = res.json()
#         if isinstance(docs, list):
#             for doc in docs:
#                 col1, col2 = st.columns([5, 1])
#                 with col1:
#                     st.write(f"📄 {doc['name']}")
#                 with col2:
#                     if st.button("❌ Delete", key=f"delete_{doc['id']}"):
#                         requests.delete(f"{API_URL}/documents/{doc['id']}")
#                         st.success("Document deleted.")
#                         st.rerun()
#     except Exception as e:
#         st.error(f"Couldn't fetch documents: {str(e)}")

# -------------------
# Chat Interface
# -------------------
st.title("🧠 AI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Scrollable chat box using expander (instead of raw HTML)
with st.container():
    st.subheader("🗨️ Conversation History")
    chat_box = st.container()
    with chat_box:
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"<div style='background-color:#DCF8C6;padding:10px;border-radius:10px;margin-bottom:5px;'><strong>🧑 You:</strong><br>{msg}</div>", unsafe_allow_html=True)
            elif role == "ai":
                st.markdown(f"<div style='background-color:#F1F0F0;padding:10px;border-radius:10px;margin-bottom:5px;'><strong>🤖 Agent:</strong><br>{msg}</div>", unsafe_allow_html=True)
            elif role == "error":
                st.markdown(f"<div style='background-color:#FFD2D2;padding:10px;border-radius:10px;margin-bottom:5px;'><strong>⚠️ Error:</strong><br>{msg}</div>", unsafe_allow_html=True)

st.markdown("---")

# Static Input at Bottom
with st.container():
    user_query = st.text_input("Type your message here...", value=st.session_state.user_input, key="user_input_input")
    col1, col2 = st.columns([1, 4])
    with col1:
        send = st.button("📤 Send")
    with col2:
        reset = st.button("🔁 Reset Conversation")

# -------------------
# Message Handling
# -------------------
if send and user_query.strip():
    try:
        payload = {
            "session_id": st.session_state.session_id,
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        st.session_state.chat_history.append(("user", user_query))
        st.session_state.user_input = ""

        logger.info("Sending request: %s", payload)
        response = requests.post(f"{API_URL}/chat", json=payload)

        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                st.session_state.chat_history.append(("error", data["error"]))
                st.error(f"❌ {data['error']}")
            else:
                ai_reply = data["response"]
                st.session_state.chat_history.append(("ai", ai_reply))
        else:
            st.session_state.chat_history.append(("error", f"Server error {response.status_code}"))
    except Exception as e:
        st.session_state.chat_history.append(("error", str(e)))
    st.rerun()

if reset:
    st.session_state.chat_history = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.user_input = ""
    st.success("🔁 Session has been reset.")
    st.rerun()

# -------------------
# Logout
# -------------------
if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.session_id = ""
    st.session_state.chat_history = []
    st.rerun()
