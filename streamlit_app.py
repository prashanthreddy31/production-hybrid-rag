"""
Hybrid RAG — Streamlit Frontend  (fixed + redesigned)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bug-fixes applied
─────────────────
1. st.chat_input() is at top level — cannot live inside st.columns()
   or any container; that's why `question` was always None.
2. st.rerun() only fires after a real interaction, not every render.
3. Streaming cursor "▌" stripped from the final saved message.
4. chat_container reference kept stable; history rendered inside it fresh
   on every run (no stale-ref issues).
5. source-card score-badge float fixed via overflow:hidden on parent.
6. Unused col_send column removed.
7. [NEW FIX] score cast to float safely to avoid crash on non-float scores.
8. [NEW FIX] clear_session() now also resets api_healthy state.
9. [NEW FIX] Streaming path omits meta chips (was showing misleading
   "✓ Grounded" badge on every streamed answer).

Design
──────
Perplexity-inspired light theme: crisp white canvas, deep slate text,
teal/cyan accent, soft card surfaces, generous spacing.
Fonts: Instrument Serif (display) + Geist (body).
"""
from __future__ import annotations

import uuid
import re
from typing import Iterator

import httpx
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE        = "http://localhost:8000/api/v1"
API_KEY         = "hybrid-rag-secret"
REQUEST_TIMEOUT = 60.0

HEADERS         = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    }

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@300;400;500;600&display=swap');

/* ── tokens ── */
:root {
    --bg:          #f8f9fb;
    --surface:     #ffffff;
    --surface2:    #f1f3f7;
    --surface3:    #e8ecf2;
    --border:      #dde2ec;
    --border2:     #c8cfde;
    --accent:      #0ea5a0;
    --accent2:     #0d9490;
    --accent-soft: #e6f7f7;
    --accent-glow: rgba(14,165,160,0.12);
    --text:        #0f172a;
    --text2:       #334155;
    --muted:       #64748b;
    --muted2:      #94a3b8;
    --user-bg:     #0ea5a0;
    --user-fg:     #ffffff;
    --bot-bg:      #ffffff;
    --bot-fg:      #0f172a;
    --green:       #059669;
    --green-soft:  #d1fae5;
    --red:         #dc2626;
    --red-soft:    #fee2e2;
    --radius:      12px;
    --radius-lg:   18px;
    --shadow-sm:   0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
    --shadow-md:   0 4px 16px rgba(15,23,42,0.08), 0 2px 6px rgba(15,23,42,0.05);
    --shadow-lg:   0 12px 40px rgba(15,23,42,0.1), 0 4px 12px rgba(15,23,42,0.06);
    --font-head:   'Instrument Serif', Georgia, serif;
    --font-body:   'Geist', 'SF Pro Text', system-ui, sans-serif;
}

/* ── global reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background:  var(--bg) !important;
    color:       var(--text) !important;
}

.stApp { background: var(--bg) !important; }

#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background:   var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow:   var(--shadow-sm) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text2) !important;
}

.sidebar-logo {
    display:       flex;
    align-items:   center;
    gap:           10px;
    padding:       6px 0 4px;
}
.sidebar-logo-icon {
    width:           36px;
    height:          36px;
    background:      var(--accent);
    border-radius:   10px;
    display:         flex;
    align-items:     center;
    justify-content: center;
    font-size:       1.1rem;
    flex-shrink:     0;
    box-shadow:      0 2px 8px var(--accent-glow);
}
.sidebar-logo-text {
    font-family: var(--font-head) !important;
    font-size:   1.2rem !important;
    color:       var(--text) !important;
    font-weight: normal !important;
    line-height: 1.2 !important;
}
.sidebar-logo-text em {
    color:       var(--accent) !important;
    font-style:  italic !important;
}

.sidebar-label {
    font-family:    var(--font-body) !important;
    font-size:      0.65rem !important;
    font-weight:    600 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color:          var(--muted2) !important;
    margin:         20px 0 8px;
}

/* ── hero header ── */
.hero {
    padding:       32px 36px 28px;
    border-radius: var(--radius-lg);
    background:    var(--surface);
    border:        1px solid var(--border);
    margin-bottom: 24px;
    position:      relative;
    overflow:      hidden;
    box-shadow:    var(--shadow-sm);
}
.hero::before {
    content:        "";
    position:       absolute;
    top: -80px; right: -80px;
    width:          260px;
    height:         260px;
    background:     radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content:        "";
    position:       absolute;
    bottom: -40px; left: 60px;
    width:          160px;
    height:         160px;
    background:     radial-gradient(circle, rgba(14,165,160,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size:      0.72rem;
    font-weight:    600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color:          var(--accent) !important;
    margin:         0 0 10px;
}
.hero h1 {
    font-family: var(--font-head) !important;
    font-size:   2.2rem !important;
    font-weight: normal !important;
    color:       var(--text) !important;
    margin:      0 !important;
    line-height: 1.2 !important;
}
.hero h1 em {
    font-style: italic;
    color:      var(--accent) !important;
}
.hero p {
    margin:      10px 0 0 !important;
    color:       var(--muted) !important;
    font-size:   0.93rem !important;
    line-height: 1.6 !important;
    max-width:   520px;
}

/* ── empty state ── */
.empty-state {
    text-align: center;
    padding:    80px 0;
}
.empty-state .icon {
    font-size:   3rem;
    opacity:     0.4;
}
.empty-state .hint {
    margin-top:  14px;
    font-size:   1rem;
    color:       var(--muted);
    font-family: var(--font-body);
}
.empty-state .hint strong {
    color: var(--text2);
}

/* ── chat bubbles ── */
.bubble-wrap {
    margin:   10px 0;
    overflow: hidden;
}

.user-bubble {
    background:    var(--user-bg);
    color:         var(--user-fg) !important;
    border-radius: 18px 18px 4px 18px;
    padding:       12px 18px;
    max-width:     72%;
    float:         right;
    font-size:     0.92rem;
    line-height:   1.55;
    box-shadow:    0 2px 12px rgba(14,165,160,0.25);
    font-family:   var(--font-body);
}

.assistant-bubble {
    background:    var(--bot-bg);
    color:         var(--bot-fg) !important;
    border:        1px solid var(--border);
    border-radius: 4px 18px 18px 18px;
    padding:       14px 18px;
    max-width:     85%;
    float:         left;
    font-size:     0.92rem;
    line-height:   1.65;
    box-shadow:    var(--shadow-sm);
    font-family:   var(--font-body);
}

/* ── chips ── */
.chips {
    margin:   6px 0 10px 52px;
    overflow: hidden;
    clear:    left;
}
.chip {
    display:       inline-block;
    background:    var(--surface2);
    border:        1px solid var(--border);
    border-radius: 999px;
    padding:       3px 10px;
    font-size:     0.73rem;
    font-weight:   500;
    margin-right:  6px;
    margin-bottom: 4px;
    color:         var(--muted) !important;
}
.chip.green {
    background:    var(--green-soft);
    border-color:  #a7f3d0;
    color:         var(--green) !important;
}
.chip.red {
    background:    var(--red-soft);
    border-color:  #fecaca;
    color:         var(--red) !important;
}
.chip.accent {
    background:    var(--accent-soft);
    border-color:  #99e6e4;
    color:         var(--accent) !important;
}

/* ── source cards ── */
.source-card {
    background:    var(--surface2);
    border-radius: var(--radius);
    border-left:   3px solid var(--accent);
    padding:       12px 14px;
    margin:        8px 0;
    overflow:      hidden;
    transition:    background 0.15s ease;
}
.source-card:hover {
    background: var(--surface3);
}
.source-title {
    font-weight:   600;
    color:         var(--text) !important;
    font-size:     0.86rem;
    font-family:   var(--font-body);
}
.score-badge {
    float:         right;
    background:    var(--accent-soft);
    color:         var(--accent) !important;
    border-radius: 999px;
    padding:       2px 9px;
    font-size:     0.70rem;
    font-weight:   600;
    border:        1px solid #99e6e4;
}
.source-snippet {
    margin-top:  6px;
    color:       var(--muted) !important;
    font-size:   0.81rem;
    line-height: 1.5;
}

/* ── buttons ── */
.stButton > button {
    background:    var(--surface) !important;
    color:         var(--text2) !important;
    border:        1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family:   var(--font-body) !important;
    font-weight:   500 !important;
    font-size:     0.88rem !important;
    transition:    all 0.15s ease !important;
    box-shadow:    var(--shadow-sm) !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color:        var(--accent) !important;
    box-shadow:   0 2px 8px var(--accent-glow) !important;
    transform:    translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── danger button ── */
.stButton.danger > button {
    color:        var(--red) !important;
    border-color: #fecaca !important;
}
.stButton.danger > button:hover {
    background:    var(--red-soft) !important;
    border-color:  var(--red) !important;
    box-shadow:    0 2px 8px rgba(220,38,38,0.12) !important;
    color:         var(--red) !important;
}

/* ── chat input ── */
.stChatInputContainer, [data-testid="stChatInput"] {
    background:    var(--surface) !important;
    border:        1px solid var(--border2) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow:    var(--shadow-md) !important;
    transition:    border-color 0.15s, box-shadow 0.15s !important;
}
.stChatInputContainer:focus-within, [data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow:   0 0 0 3px var(--accent-glow), var(--shadow-md) !important;
}
.stChatInputContainer textarea {
    background:  transparent !important;
    color:       var(--text) !important;
    font-family: var(--font-body) !important;
    font-size:   0.94rem !important;
}
.stChatInputContainer textarea::placeholder {
    color: var(--muted2) !important;
}

/* ── expander ── */
.streamlit-expanderHeader {
    background:    var(--surface2) !important;
    border-radius: var(--radius) !important;
    color:         var(--muted) !important;
    font-family:   var(--font-body) !important;
    font-size:     0.83rem !important;
}
details {
    border:        1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow:    var(--shadow-sm) !important;
}
details[open] {
    border-color: var(--border2) !important;
}

/* ── toggles ── */
[data-testid="stToggle"] label {
    color:       var(--text2) !important;
    font-family: var(--font-body) !important;
    font-size:   0.88rem !important;
}

/* ── spinner ── */
[data-testid="stSpinner"] p {
    color:       var(--muted) !important;
    font-size:   0.85rem !important;
    font-family: var(--font-body) !important;
}

/* ── divider ── */
hr { border-color: var(--border) !important; }

/* ── captions / small text ── */
.stCaption, [data-testid="stCaptionContainer"] p {
    color:       var(--muted) !important;
    font-family: var(--font-body) !important;
    font-size:   0.78rem !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }

/* ── status dot ── */
.status-dot {
    display:       inline-block;
    width:         7px;
    height:        7px;
    border-radius: 50%;
    margin-right:  6px;
    vertical-align: middle;
}
.status-dot.green { background: var(--green); }
.status-dot.red   { background: var(--red); }
.status-dot.grey  { background: var(--muted2); }

/* ── session id ── */
.session-id {
    font-family:   'Courier New', monospace !important;
    font-size:     0.72rem !important;
    color:         var(--muted2) !important;
    background:    var(--surface2);
    border:        1px solid var(--border);
    border-radius: 6px;
    padding:       3px 8px;
    display:       inline-block;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = None

# ── API helpers ───────────────────────────────────────────────────────────────

def check_health() -> dict:
    try:
        r = httpx.get(f"{API_BASE.rsplit('/api', 1)[0]}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "unreachable", "checks": {}}


def send_query(question: str, stream: bool = False) -> dict | None:
    payload = {
        "question":   question,
        "session_id": st.session_state.session_id,
        "stream":    stream,
    }
    try:
        r = httpx.post(
            f"{API_BASE}/query",
            json=payload,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    except httpx.ConnectError:
        st.error("Cannot reach the API. Is the FastAPI server running on localhost:8000?")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


def stream_query(question: str) -> Iterator[tuple]:
    """Yields (event_type, data) tuples.
    event_type == "token" -> data is a text string to append.
    event_type == "meta"  -> data is a dict with grounding/latency info.
    event_type == "error" -> data is an error message string.
    """
    import json as _json
    payload = {"question": question, "session_id": st.session_state.session_id}
    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/query/stream",
            json=payload,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        ) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    event = _json.loads(raw)
                except _json.JSONDecodeError:
                    yield ("token", raw)
                    continue
                etype = event.get("type")
                if etype == "token":
                    yield ("token", event.get("content", ""))
                elif etype == "meta":
                    yield ("meta", event)
    except Exception as e:
        yield ("error", f"\n\n⚠️ Stream error: {e}")


def clear_session() -> None:
    """Reset conversation and clear server-side session."""
    try:
        httpx.delete(
            f"{API_BASE}/session/{st.session_state.session_id}",
            headers=HEADERS,
            timeout=5,
        )
    except Exception:
        pass
    st.session_state.session_id  = str(uuid.uuid4())
    st.session_state.messages    = []
    st.session_state.api_healthy = None

def strip_citations(text: str) -> str:
    """Remove inline citation tags like [doc-1], [doc-2], [1], [doc-1][doc-2] from LLM output."""
    return re.sub(r'(\[doc-\d+\]|\[\d+\])+', '', text).strip()

# ── Render helpers ────────────────────────────────────────────────────────────

def render_message(msg: dict) -> None:
    role    = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(
            f'<div class="bubble-wrap"><div class="user-bubble">{content}</div></div>',
            unsafe_allow_html=True,
        )
        return

    # assistant bubble
    st.markdown(
        f'<div class="bubble-wrap"><div class="assistant-bubble">{content}</div></div>',
        unsafe_allow_html=True,
    )

    # meta chips — only show when we have real meta (non-streaming)
    if meta := msg.get("meta"):
        if meta:
            grounded = meta.get("is_fully_grounded", True)
            g_class  = "green" if grounded else "red"
            g_label  = "✓ Grounded" if grounded else "⚠ Partial"
            latency  = meta.get("latency_ms", "—")
            model    = meta.get("model", "—")
            st.markdown(
                f'<div class="chips">'
                f'<span class="chip {g_class}">{g_label}</span>'
                f'<span class="chip">⏱ {latency} ms</span>'
                f'<span class="chip accent">⬡ {model}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # source cards
    if sources := msg.get("sources"):
        with st.expander(f"📄  {len(sources)} source{'s' if len(sources) > 1 else ''}", expanded=False):
            for src in sources:
                title   = src.get("title") or src.get("source") or f"doc-{src.get('doc_index', '?')}"
                raw_score = src.get("score")
                try:
                    score_val = float(raw_score) if raw_score is not None else None
                except (TypeError, ValueError):
                    score_val = None
                score_h = f'<span class="score-badge">{score_val:.3f}</span>' if score_val is not None else ""
                snippet = src.get("content", "")[:300]
                idx     = src.get("doc_index", "?")
                st.markdown(
                    f'<div class="source-card">'
                    f'{score_h}'
                    f'<div class="source-title">[{idx}] {title}</div>'
                    f'<div class="source-snippet">{snippet}…</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">📚</div>
        <div class="sidebar-logo-text">Hybrid <em>RAG</em></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # System status
    st.markdown('<div class="sidebar-label">System Status</div>', unsafe_allow_html=True)

    if st.button("↻  Check Health", use_container_width=True):
        st.session_state.api_healthy = check_health()

    if st.session_state.api_healthy is not None:
        health = st.session_state.api_healthy
        status = health.get("status", "unknown")
        dot_class = "green" if status == "healthy" else "red"
        st.markdown(
            f'<span class="chip {dot_class}">'
            f'<span class="status-dot {dot_class}"></span>'
            f'{status.upper()}'
            f'</span>',
            unsafe_allow_html=True,
        )
        for svc, state in health.get("checks", {}).items():
            icon  = "✓" if state == "ok" else "✗"
            color = "green" if state == "ok" else "red"
            st.markdown(
                f'<span style="color:var(--{color});font-size:0.8rem">{icon}</span>'
                f'<span style="color:var(--muted);font-size:0.78rem"> {svc}: {state}</span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Session info
    st.markdown('<div class="sidebar-label">Session</div>', unsafe_allow_html=True)
    short_id = st.session_state.session_id[:16]
    st.markdown(
        f'<div style="margin-bottom:6px"><span class="session-id">{short_id}…</span></div>',
        unsafe_allow_html=True,
    )
    msg_count = len(st.session_state.messages)
    st.markdown(
        f'<span class="chip">{msg_count} message{"s" if msg_count != 1 else ""}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.button("🗑  Clear Conversation", use_container_width=True):
        clear_session()
        st.rerun()

    st.markdown("---")

    # Settings
    st.markdown('<div class="sidebar-label">Settings</div>', unsafe_allow_html=True)
    use_streaming = st.toggle("Stream response",     value=False)
    show_thinking = st.toggle("Show retrieval info", value=True)

    st.markdown("---")
    st.caption("Built with LangChain · Pinecone · Cohere")
    st.caption(f"API: `{API_BASE}`")

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered · Document Intelligence</div>
    <h1>Hybrid <em>RAG</em> Assistant</h1>
    <p>Ask questions across your document knowledge base — retrieved with vector + keyword hybrid search and answered with a large language model.</p>
</div>
""", unsafe_allow_html=True)

# Render conversation history
if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">'
        '<div class="icon">📖</div>'
        '<div class="hint">Ask a question about your documents<br>'
        '<strong>Try: "Summarise the key points in the uploaded reports"</strong></div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        render_message(msg)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

question = st.chat_input("Ask a question about your documents…")

if question:
    # persist user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    render_message({"role": "user", "content": question})

    if use_streaming:
        placeholder = st.empty()
        full_answer = ""
        stream_meta = None
        with st.spinner("Retrieving and generating…"):
            for etype, data in stream_query(question):
                if etype == "token":
                    full_answer += strip_citations(data)
                    placeholder.markdown(
                        f'<div class="bubble-wrap">'
                        f'<div class="assistant-bubble">{full_answer}▌</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif etype == "meta":
                    stream_meta = data
                elif etype == "error":
                    full_answer += data
        placeholder.markdown(
            f'<div class="bubble-wrap">'
            f'<div class="assistant-bubble">{full_answer}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if stream_meta and show_thinking:
            grounded = stream_meta.get("is_fully_grounded", True)
            g_class  = "green" if grounded else "red"
            g_label  = "✓ Grounded" if grounded else "⚠ Partial"
            latency  = stream_meta.get("elapsed_ms", "—")
            st.markdown(
                f'<div class="chips">'
                f'<span class="chip {g_class}">{g_label}</span>'
                f'<span class="chip">⏱ {latency} ms</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        meta_to_save = {
            "is_fully_grounded": stream_meta.get("is_fully_grounded", True),
            "latency_ms":        stream_meta.get("elapsed_ms", 0),
            "model":             "—",
        } if stream_meta else None
        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_answer,
            "sources": [],
            "meta":    meta_to_save,
        })        

    # Non-streaming path
    else:
        with st.spinner("Searching documents and generating answer…"):
            response = send_query(question)

        if response:
            answer  = strip_citations(response.get("answer", "No answer returned."))
            sources = response.get("sources", []) 

            if not sources: 
                st.error("No relevant documents were retrieved from the knowledge base.")
                st.stop()
            
            if not answer or not answer.strip():
                st.error("LLM failed to generate an answer.")
                st.stop()
            
            if show_thinking is False:
                sources =[]

            meta    = {
                "is_fully_grounded": response.get("is_fully_grounded", True),
                "latency_ms":        response.get("latency_ms", 0),
                "model":             response.get("model", "—"),
            }
            msg = {"role": "assistant", "content": answer, "sources": sources, "meta": meta}
            st.session_state.messages.append(msg)
            render_message(msg)
            st.rerun()
