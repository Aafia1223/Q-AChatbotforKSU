import streamlit as st
import requests
import time

st.set_page_config(page_title="KSU AI Assistant", layout="wide")

# ----------------- CSS for Styling -----------------
st.markdown("""
<style>
    body {
        background-color: #f3f4f6;
    }

    .chat-container {
        max-width: 800px;
        margin: auto;
    }

    .bot, .user {
        padding: 12px 20px;
        border-radius: 20px;
        margin: 10px 0;
        width: fit-content;
        max-width: 80%;
        line-height: 1.4;
        font-size: 16px;
    }

    .bot {
        background-color: #dbeafe;
        align-self: flex-start;
    }

    .user {
        background-color: #fcd34d;
        align-self: flex-end;
    }

    .chat-bubble {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    .chat-wrapper {
        display: flex;
        flex-direction: column;
        min-height: 80vh;
        overflow-y: auto;
        padding: 10px;
    }

    .input-box {
        position: sticky;
        bottom: 10px;
        background: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 0 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Chat State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- Show Welcome Message on Load -----------------
if not st.session_state.messages:
    try:
        res = requests.get("http://127.0.0.1:8000/welcome")
        welcome = res.json()["response"]
        st.session_state.messages.append({"role": "bot", "content": welcome})
    except Exception as e:
        st.session_state.messages.append({
            "role": "bot",
            "content": "⚠️ Unable to fetch welcome message from the server. Is FastAPI running?"
        })

# ----------------- Display Chat -----------------
st.markdown("<div class='chat-container chat-wrapper'>", unsafe_allow_html=True)
for msg in st.session_state.messages:
    role = msg["role"]
    bubble_class = "bot" if role == "bot" else "user"
    st.markdown(f"<div class='chat-bubble {bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------- User Input -----------------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask something about KSU:", placeholder="e.g., What are the library hours?")
    submit = st.form_submit_button("Send")

# ----------------- Handle Submission -----------------
if submit and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        res = requests.post("http://127.0.0.1:8000/chat", json={
            "user_id": "user_123",
            "message": user_input
        })
        reply = res.json().get("response", "Sorry, I didn’t understand that.")
    except Exception as e:
        reply = "⚠️ Failed to connect to backend. Make sure FastAPI is running."

    # Show bot response with a short delay for realism
    with st.spinner("KSU Assistant is typing..."):
        time.sleep(0.5)
    st.session_state.messages.append({"role": "bot", "content": reply})

