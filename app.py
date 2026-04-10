import streamlit as st
import time
from rag_backend import load_and_chunk, build_retrievers, get_rag_response

st.set_page_config(page_title="Manddy AI", layout="wide")

# -------------------- SESSION --------------------
if "sessions" not in st.session_state:
    st.session_state.sessions = {"Chat 1": []}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"

if "ready" not in st.session_state:
    st.session_state.ready = False

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("💬 Chats")

    if st.button("➕ New Chat"):
        name = f"Chat {len(st.session_state.sessions)+1}"
        st.session_state.sessions[name] = []
        st.session_state.current_chat = name

    for chat_name in st.session_state.sessions:
        if st.button(chat_name):
            st.session_state.current_chat = chat_name


    # ✅ ADD HERE
    if st.button("🗑 Clear Current Chat"):
        st.session_state.sessions[st.session_state.current_chat] = []
        st.rerun()

    if st.button("♻️ Reset App"):
        st.session_state.sessions = {"Chat 1": []}
        st.session_state.current_chat = "Chat 1"
        st.session_state.ready = False
        st.rerun()

    st.markdown("---")

    st.title("📄 Upload PDF")
    file = st.file_uploader("Upload", type="pdf")

    if st.button("🚀 Process"):
        if file:
            chunks = load_and_chunk(file)
            build_retrievers(chunks)
            st.session_state.ready = True
            st.success("Ready!")
        else:
            st.warning("Upload a PDF")

# -------------------- HEADER --------------------
st.title(f"🚀 {st.session_state.current_chat}")

chat = st.session_state.sessions[st.session_state.current_chat]

# -------------------- DISPLAY --------------------
for i, msg in enumerate(chat):

    if msg["role"] == "user":
        st.chat_message("user").write(msg["message"])

    else:
        st.chat_message("assistant").write(msg["message"])

        if msg["sources"]:
            with st.expander(f"📌 Sources {i//2 + 1}"):
                for d in msg["sources"]:
                    st.write(d.page_content[:300])
                    st.markdown("---")

# -------------------- INPUT --------------------
query = st.chat_input("Ask something...")

if query and st.session_state.ready:

    # USER
    chat.append({
        "role": "user",
        "message": query,
        "sources": None
    })

    st.chat_message("user").write(query)

    # BOT
    msg_box = st.chat_message("assistant")
    placeholder = msg_box.empty()

    start = time.time()
    response, docs = get_rag_response(query, chat)
    end = round(time.time() - start, 2)

    full = ""
    for word in response.split():
        full += word + " "
        placeholder.markdown(full + "▌")
        time.sleep(0.01)

    placeholder.markdown(full)

    chat.append({
        "role": "assistant",
        "message": full,
        "sources": docs
    })

    st.caption(f"⏱ {end}s")

    with st.expander("📌 Sources"):
        for d in docs:
            st.write(d.page_content[:300])
            st.markdown("---")

elif query:
    st.warning("Upload and process PDF first")