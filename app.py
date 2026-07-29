"""
Hospital Chatbot – Streamlit UI (Streaming Edition)
────────────────────────────────────────────────────
Premium dark-themed chat interface for MedAssist AI.
Uses LangChain agent with SSE-style streaming for
instant token delivery to the user.
"""

import streamlit as st
import time, pathlib, sys

# ensure project root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAssist AI – Hospital Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS – premium dark theme with glassmorphism
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #e2e8f0 !important;
}

/* ── Header area ── */
.main-header {
    text-align: center;
    padding: 1rem 0 2rem;
}
.main-header h1 {
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* ── Chat bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    padding: 1rem 1.4rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.8rem 0;
    max-width: 75%;
    margin-left: auto;
    font-size: 0.95rem;
    line-height: 1.55;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
    animation: slideInRight 0.3s ease-out;
}
.bot-bubble {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    color: #e2e8f0;
    padding: 1rem 1.4rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.8rem 0;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.6;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    animation: slideInLeft 0.3s ease-out;
}
.bot-bubble strong { color: #a78bfa; }
.bot-bubble code { color: #34d399; }

/* ── Bubble label ── */
.bubble-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.3rem;
}
.user-label { color: #c4b5fd; text-align: right; }
.bot-label  { color: #6ee7b7; }

/* ── Retrieved docs ── */
.source-chip {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    margin: 0.2rem 0.3rem 0.2rem 0;
    font-weight: 500;
}

/* ── Quick action buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.2)) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    color: #c4b5fd !important;
    border-radius: 12px !important;
    padding: 0.55rem 1rem !important;
    font-weight: 500 !important;
    transition: all 0.25s ease !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(99,102,241,0.4), rgba(168,85,247,0.4)) !important;
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Input ── */
.stChatInput > div {
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.04) !important;
}

/* ── Animations ── */
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.typing-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #a78bfa;
    margin: 0 3px;
    animation: pulse 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* ── Streaming indicator ── */
.streaming-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    margin-bottom: 0.5rem;
}
.streaming-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #6ee7b7;
    animation: pulse 1s infinite;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.status-online {
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.3);
    color: #4ade80;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedAssist AI")
    st.markdown(
        '<span class="status-badge status-online">● Online</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "**LangChain-powered hospital pharmacy agent.**  \n"
        "Ask about medicines, stock, dosage, and alternatives."
    )
    st.markdown("---")

    # Data Management
    st.markdown("### ⚙️ Data Management")
    st.markdown("<p style='font-size:0.85rem;color:#e2e8f0;margin-bottom:0.5rem;'>Upload a new dataset (CSV):</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    
    if st.button("🔄 Ingest Medicine Data", key="ingest_btn", use_container_width=True):
        with st.spinner("Ingesting CSV into ChromaDB Cloud …"):
            try:
                import pathlib
                from config import CSV_PATH
                
                if uploaded_file is not None:
                    csv_full_path = pathlib.Path(__file__).resolve().parent / CSV_PATH
                    with open(csv_full_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                from ingest import ingest
                count = ingest()
                st.success(f"✅ {count} medicines ingested!")
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")

    st.markdown("---")

    # Quick actions
    st.markdown("### ⚡ Quick Queries")
    quick_queries = [
        "💊 Check Paracetamol stock",
        "🤒 Medicine for fever?",
        "💉 Is Insulin available?",
        "🦠 Antibiotic for infection?",
        "🧪 Alternatives for Ibuprofen?",
    ]
    for qq in quick_queries:
        if st.button(qq, key=f"qq_{qq}", use_container_width=True):
            # strip emoji prefix for cleaner query
            clean = qq.split(" ", 1)[1] if " " in qq else qq
            st.session_state["pending_query"] = clean

    st.markdown("---")

    # Agent info
    st.markdown("### 🤖 Agent Tools")
    st.markdown(
        "- 📦 **Stock Check** — availability lookup\n"
        "- 🔄 **Alternatives** — substitute finder\n"
        "- 💬 **FAQ / RAG** — general Q&A"
    )
    st.markdown("---")

    st.markdown(
        "<p style='color:#64748b;font-size:0.75rem;text-align:center;'>"
        "Powered by LangChain · Groq · ChromaDB<br>"
        "© 2026 MedAssist AI</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────
# Main area – header
# ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    '<h1>🏥 MedAssist AI</h1>'
    '<p>Your intelligent hospital pharmacy agent — powered by LangChain with real-time streaming</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# Chat state
# ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────────────────
# Render chat history
# ─────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="bubble-label user-label">You</div>'
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bubble-label bot-label">MedAssist AI</div>'
            f'<div class="bot-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        # show source chips if available
        if msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">💊 {s}</span>' for s in msg["sources"]
            )
            st.markdown(
                f'<div style="margin-bottom:1rem;">{chips}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────
# Process query with streaming
# ─────────────────────────────────────────────────────────
def process_query(query: str):
    """Run agent pipeline with SSE streaming and update chat."""
    # add user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(
        f'<div class="bubble-label user-label">You</div>'
        f'<div class="user-bubble">{query}</div>',
        unsafe_allow_html=True,
    )

    # Processing indicator
    status_placeholder = st.empty()
    status_placeholder.markdown(
        '<div class="bubble-label bot-label">MedAssist AI</div>'
        '<div class="streaming-badge">'
        '<span class="streaming-dot"></span> Generating response…'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        from agent import run_agent

        full_answer, docs = run_agent(query)
        sources = [d["metadata"]["medicine_name"] for d in docs] if docs else []
        
        # Clear processing indicator
        status_placeholder.empty()
        
        # Render the fresh answer immediately in UI
        st.markdown(
            f'<div class="bubble-label bot-label">MedAssist AI</div>'
            f'<div class="bot-bubble">{full_answer}</div>',
            unsafe_allow_html=True,
        )
        if sources:
            chips = "".join(f'<span class="source-chip">💊 {s}</span>' for s in sources)
            st.markdown(f'<div style="margin-bottom:1rem;">{chips}</div>', unsafe_allow_html=True)

    except Exception as e:
        status_placeholder.empty()
        full_answer = f"⚠️ Sorry, I encountered an error: {str(e)}"
        sources = []
        st.error(full_answer)

    # Store the complete answer in chat history
    display_answer = full_answer.replace("\n", "<br>") if full_answer else ""
    st.session_state.messages.append({
        "role": "assistant",
        "content": display_answer,
        "sources": sources,
    })


# ─────────────────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about medicines, stock, dosage, alternatives …")

# handle pending quick-action query
if "pending_query" in st.session_state:
    pq = st.session_state.pop("pending_query")
    process_query(pq)
elif user_input:
    process_query(user_input)

# ─────────────────────────────────────────────────────────
# Empty state prompt
# ─────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        '<div class="glass-card" style="text-align:center; margin-top:2rem;">'
        '<p style="font-size:2.8rem; margin-bottom:0.5rem;">💬</p>'
        '<p style="color:#a5b4fc; font-size:1.1rem; font-weight:500;">Start a conversation</p>'
        '<p style="color:#64748b; font-size:0.9rem;">'
        'Type a question below or use the quick queries in the sidebar.<br>'
        'Try: <em>"Do you have Paracetamol 500mg?"</em>'
        '</p>'
        '<p style="color:#6ee7b7; font-size:0.8rem; margin-top:0.5rem;">'
        '⚡ Responses stream in real-time — no more waiting!'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sample query cards
    col1, col2, col3 = st.columns(3)
    sample_queries = [
        ("🤒", "What medicine is recommended for fever?"),
        ("💊", "Do you have Ibuprofen for pain relief?"),
        ("🔄", "Is Paracetamol available? Any alternative?"),
    ]
    for col, (emoji, sq) in zip([col1, col2, col3], sample_queries):
        with col:
            st.markdown(
                f'<div class="glass-card" style="text-align:center;cursor:pointer;min-height:120px;'
                f'display:flex;flex-direction:column;justify-content:center;">'
                f'<p style="font-size:2rem;margin:0;">{emoji}</p>'
                f'<p style="color:#c4b5fd;font-size:0.85rem;margin:0.5rem 0 0;">{sq}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Ask →", key=f"sample_{sq}", use_container_width=True):
                st.session_state["pending_query"] = sq
                st.rerun()
