# frontend/app.py
import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("🧠 PDF Chatbot")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.subheader("Settings")
    backend_url = st.text_input("Backend URL", DEFAULT_BACKEND)
    top_k = st.slider("Top K chunks", 1, 8, 4)
    if st.button("Reset Memory"):
        try:
            r = requests.post(f"{backend_url}/reset", params={"session_id": st.session_state.session_id}, timeout=30)
            if r.ok:
                st.success("Memory cleared.")
            else:
                st.error(f"Reset failed: {r.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("Upload files")
    uploaded = st.file_uploader("Upload PDF/TXT/DOCX", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True)
    if uploaded and st.button("Process Uploads"):
        for f in uploaded:
            with st.spinner(f"Ingesting {f.name}..."):
                files = {"file": (f.name, f.getvalue(), f.type or "application/octet-stream")}
                data = {"session_id": st.session_state.session_id}
                try:
                    r = requests.post(f"{backend_url}/ingest", files=files, data=data, timeout=120)
                    if r.ok:
                        res = r.json()
                        if res.get("ok"):
                            st.success(f"{f.name} → {res.get('chunks',0)} chunks")
                        else:
                            st.warning(f"{f.name} → {res}")
                    else:
                        st.error(f"{f.name} → {r.text}")
                except Exception as e:
                    st.error(f"Ingest failed: {e}")

st.markdown("### Chat")
if "history" not in st.session_state:
    st.session_state.history = []

# render history
for role, message in st.session_state.history:
    if role == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Assistant:** {message}")

prompt = st.chat_input("Ask a question...")
if prompt:
    st.session_state.history.append(("user", prompt))
    backend_url = st.sidebar.text_input(
    "Backend URL",
    DEFAULT_BACKEND,
    key="backend_url_input"
    )  # read latest
    with st.spinner("Thinking..."):
        try:
            payload = {"session_id": st.session_state.session_id, "message": prompt, "top_k": top_k}
            r = requests.post(f"{backend_url}/chat", json=payload, timeout=180)
            if r.ok:
                out = r.json()
                ans = out.get("answer", "")
                st.session_state.history.append(("assistant", ans))
                st.markdown(ans)
                # show citations if any
                cits = out.get("citations", [])
                if cits:
                    st.markdown("**Citations:**")
                    for c in cits:
                        st.write(f"- {c.get('source')} (chunk #{c.get('chunk_id')}, score={c.get('score'):.3f})")
            else:
                st.error(f"Backend error: {r.text}")
        except Exception as e:
            st.error(f"Request error: {e}")
