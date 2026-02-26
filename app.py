"""
Advanced RAG Chatbot - Streamlit App
Built with Groq + Pinecone + HuggingFace
All files are in the SAME folder - no src/ needed
"""

import streamlit as st
import time
from rag_engine import RAGEngine
from document_processor import DocumentProcessor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.85rem;
        color: #444;
    }
    .status-ok  { color: #22c55e; font-weight: 600; }
    .status-err { color: #ef4444; font-weight: 600; }
    .chat-meta  { font-size: 0.75rem; color: #9ca3af; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages":        [],
        "rag_engine":      None,
        "doc_processed":   False,
        "total_queries":   0,
        "total_chunks":    0,
        "processing_time": 0.0,
        "llm_model":       "llama-3.3-70b-versatile",
        "index_name":      "advanced-rag-chatbot",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8rem;">🤖 Advanced RAG Chatbot</h1>
    <p style="margin:0.3rem 0 0;opacity:0.85;font-size:0.95rem;">
        Powered by Groq · Pinecone · HuggingFace · Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("🔑 API Keys", expanded=True):
        groq_api_key = st.text_input("Groq API Key", type="password",
                                     placeholder="gsk_...",
                                     help="Free at console.groq.com")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password",
                                          placeholder="pcsk_...",
                                          help="Free at pinecone.io")
        pinecone_index = st.text_input("Pinecone Index Name",
                                        value="advanced-rag-chatbot")

    with st.expander("🤖 Model Settings", expanded=False):
        llm_model = st.selectbox("Groq LLM Model", [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ])
        embedding_model = st.selectbox("Embedding Model", [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "BAAI/bge-small-en-v1.5",
        ])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_k       = st.slider("Top-K Chunks", 1, 10, 5)

    with st.expander("📄 Chunking Settings", expanded=False):
        chunk_size    = st.slider("Chunk Size",    200, 1500, 512, 50)
        chunk_overlap = st.slider("Chunk Overlap",   0,  300,  50, 10)
        retrieval_strategy = st.selectbox("Retrieval Strategy", [
            "Semantic Search",
            "MMR (Diversity)",
        ])

    st.divider()
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files and groq_api_key and pinecone_api_key:
        if st.button("⚡ Process & Index Documents", use_container_width=True, type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    t0 = time.time()
                    processor = DocumentProcessor(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap)
                    all_chunks = []
                    for f in uploaded_files:
                        all_chunks.extend(processor.process_file(f))

                    rag = RAGEngine(
                        groq_api_key=groq_api_key,
                        pinecone_api_key=pinecone_api_key,
                        pinecone_index=pinecone_index,
                        llm_model=llm_model,
                        embedding_model_name=embedding_model,
                        temperature=temperature,
                        top_k=top_k,
                        retrieval_strategy=retrieval_strategy,
                    )
                    rag.index_documents(all_chunks)

                    st.session_state.rag_engine      = rag
                    st.session_state.doc_processed   = True
                    st.session_state.total_chunks    = len(all_chunks)
                    st.session_state.processing_time = round(time.time() - t0, 2)
                    st.session_state.llm_model       = llm_model
                    st.session_state.index_name      = pinecone_index
                    st.success(f"✅ Indexed {len(all_chunks)} chunks in {st.session_state.processing_time}s!")
                except Exception as e:
                    st.error(f"❌ {e}")
                    st.exception(e)
    elif uploaded_files:
        st.warning("⚠️ Enter both API keys first.")

    st.divider()
    st.header("📊 Stats")
    c1, c2 = st.columns(2)
    c1.metric("Queries", st.session_state.total_queries)
    c2.metric("Chunks",  st.session_state.total_chunks)

    if st.session_state.doc_processed:
        st.markdown('<p class="status-ok">✅ Knowledge base ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-err">⭕ Upload docs to begin</p>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.total_queries = 0
        st.rerun()

    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem;color:#6b7280;text-align:center;">
        Built with ❤️ by <strong>Anuj Vishwakarma</strong>
    </div>""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
col_chat, col_info = st.columns([2, 1])

with col_chat:
    st.subheader("💬 Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f'<div class="source-card">{src}</div>',
                                    unsafe_allow_html=True)
            if msg.get("time_taken"):
                st.markdown(
                    f'<p class="chat-meta">⏱ {msg["time_taken"]}s · {msg.get("model","")}</p>',
                    unsafe_allow_html=True)

    if prompt := st.chat_input("Ask anything about your documents..."):
        if not st.session_state.doc_processed:
            st.warning("⚠️ Upload and process documents first!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Retrieving & generating..."):
                    t0 = time.time()
                    try:
                        result  = st.session_state.rag_engine.query(
                            query=prompt,
                            chat_history=st.session_state.messages[:-1]
                        )
                        elapsed = round(time.time() - t0, 2)
                        answer  = result["answer"]
                        sources = result.get("sources", [])

                        st.markdown(answer)
                        if sources:
                            with st.expander("📚 Sources", expanded=False):
                                for src in sources:
                                    st.markdown(
                                        f'<div class="source-card">{src}</div>',
                                        unsafe_allow_html=True)
                        st.markdown(
                            f'<p class="chat-meta">⏱ {elapsed}s · {st.session_state.llm_model}</p>',
                            unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role":       "assistant",
                            "content":    answer,
                            "sources":    sources,
                            "time_taken": elapsed,
                            "model":      st.session_state.llm_model,
                        })
                        st.session_state.total_queries += 1
                    except Exception as e:
                        st.error(f"❌ {e}")
                        st.exception(e)

with col_info:
    st.subheader("🔬 How It Works")
    st.markdown("""
**Advanced RAG Pipeline:**

1. **📄 Document Ingestion** — PDF, TXT, DOCX
2. **✂️ Smart Chunking** — Recursive split + overlap
3. **🔢 Embedding** — HuggingFace Sentence Transformers
4. **🗄️ Pinecone** — Cloud vector store
5. **🔍 Retrieval** — Semantic or MMR strategy
6. **🔄 Query Rewriting** — Uses chat history
7. **🤖 Groq LLM** — Ultra-fast generation
8. **💾 Memory** — Sliding window history
    """)

    st.divider()
    st.subheader("🚀 Tech Stack")
    for label, val in [
        ("🤖 LLM",        "Groq — Llama 3.3 70B"),
        ("🗄️ Vector DB",  "Pinecone Serverless"),
        ("🔢 Embeddings", "HuggingFace ST"),
        ("🌐 UI",         "Streamlit"),
    ]:
        a, b = st.columns([1, 1.5])
        a.markdown(f"**{label}**")
        b.markdown(val)

    if st.session_state.doc_processed:
        st.divider()
        st.subheader("📈 Index Info")
        st.info(f"""
**Chunks:** {st.session_state.total_chunks}
**Index:** `{st.session_state.index_name}`
**Time:** {st.session_state.processing_time}s
        """)