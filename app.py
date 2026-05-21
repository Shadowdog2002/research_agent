import threading
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Research Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ---- Global ---- */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f3e; }
h1, h2, h3, h4 { color: #e2e8f0 !important; }
p, li, label { color: #cbd5e0; }

/* ---- Sidebar brand ---- */
.brand {
    text-align: center;
    padding: 1.4rem 0 1rem 0;
    border-bottom: 1px solid #2a2f3e;
    margin-bottom: 1rem;
}
.brand h2 { font-size: 1.25rem; margin: 0; color: #7c8cf8 !important; letter-spacing: .02em; }
.brand p  { font-size: .75rem; color: #718096; margin: .2rem 0 0 0; }

/* ---- Paper cards ---- */
.paper-card {
    background: #1a2035;
    border: 1px solid #2a2f3e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: .85rem;
    transition: border-color .2s;
}
.paper-card:hover { border-color: #4c5bce; }
.paper-title { font-size: .95rem; font-weight: 600; color: #a5b4fc; margin-bottom: .25rem; }
.paper-meta  { font-size: .75rem; color: #718096; }
.score-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #63b3ed;
    font-size: .7rem;
    padding: .15rem .55rem;
    border-radius: 999px;
    font-weight: 600;
    margin-left: .5rem;
}

/* ---- Chat bubbles ---- */
.chat-user {
    background: #1e3a5f;
    border-radius: 12px 12px 2px 12px;
    padding: .7rem 1rem;
    margin: .5rem 0;
    max-width: 80%;
    margin-left: auto;
    color: #e2e8f0;
    font-size: .9rem;
}
.chat-assistant {
    background: #1a2035;
    border: 1px solid #2a2f3e;
    border-radius: 12px 12px 12px 2px;
    padding: .7rem 1rem;
    margin: .5rem 0;
    max-width: 90%;
    color: #cbd5e0;
    font-size: .9rem;
}

/* ---- Stat pills in sidebar ---- */
.stat-pill {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #1a2035;
    border-radius: 8px;
    padding: .5rem .8rem;
    margin-bottom: .5rem;
    font-size: .8rem;
}
.stat-pill span:first-child { color: #a0aec0; }
.stat-pill span:last-child  { color: #7c8cf8; font-weight: 700; }

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #4c5bce 0%, #6b46c1 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: .5rem 1.4rem;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }

/* ---- Lit review block ---- */
.review-box {
    background: #1a2035;
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    line-height: 1.75;
}

/* ---- Tabs ---- */
[data-testid="stTabs"] button { color: #a0aec0 !important; font-size: .88rem; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #a5b4fc !important; border-bottom-color: #a5b4fc !important; }

/* ---- Inputs ---- */
.stTextInput input, .stTextArea textarea {
    background: #1a2035 !important;
    border: 1px solid #2a2f3e !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ---- Login page ---- */
.login-box {
    max-width: 380px;
    margin: 6rem auto 0 auto;
    background: #1a2035;
    border: 1px solid #2a2f3e;
    border-radius: 14px;
    padding: 2.5rem 2rem 2rem 2rem;
    text-align: center;
}
.login-box h2 { color: #7c8cf8 !important; margin-bottom: .25rem; }
.login-box p  { color: #718096; font-size: .85rem; margin-bottom: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Lazy imports after page config ────────────────────────────────────────────
from datetime import datetime

from agent.pipeline import ingest_arxiv_paper, ingest_arxiv_papers_batch, ingest_uploaded_pdf, run_research_pipeline
from agent.rag import delete_paper, get_collection, get_ingested_papers
from agent.llm import find_research_gaps, generate_literature_review, get_client
from agent.arxiv_search import search_and_rerank
from agent.reviews import delete_review, load_reviews, save_review
from agent.orchestrator import ResearchOrchestrator

# Module-level orchestrator singleton — initialises SecurityGuardrail once
_orchestrator = ResearchOrchestrator()


# ── Background pipeline helpers ───────────────────────────────────────────────
class PipelineState:
    def __init__(self):
        self.progress: float = 0.0
        self.status: str = "Starting…"
        self.result = None
        self.error: str | None = None
        self.done: bool = False
        self._lock = threading.Lock()

    def update(self, progress=None, status=None):
        with self._lock:
            if progress is not None:
                self.progress = progress
            if status is not None:
                self.status = status

    def finish(self, result):
        with self._lock:
            self.result = result
            self.done = True

    def fail(self, msg: str):
        with self._lock:
            self.error = msg
            self.done = True


def _pipeline_thread_fn(state: PipelineState, topic: str, fetch_n: int, min_score: float, username: str = "default"):
    try:
        result = run_research_pipeline(
            topic,
            progress_cb=lambda pct: state.update(progress=pct),
            status_cb=lambda msg: state.update(status=msg),
            fetch_n=fetch_n,
            min_score=min_score,
            username=username,
        )
        if isinstance(result, dict) and "error" in result:
            state.fail(result["error"])
        else:
            state.finish(result)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        state.fail(str(exc))


# ── Session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "username": "",             # set on login; empty means not logged in
        "page": "search",
        "results": None,            # pipeline output
        "chat_history": [],         # [{role, content}]
        "status_log": [],           # pipeline status messages
        "pipeline_running": False,  # True while thread is alive
        "pipeline_status": "",      # "" | "running" | "done" | "error"
        "pipeline_state": None,     # PipelineState object
        "pipeline_thread": None,    # threading.Thread
        "qa_processing": False,     # True while Claude generates a Q&A answer
        "arxiv_candidates": None,   # List[Paper] pending confirmation, or None
        "arxiv_query": "",          # Preserves last arXiv search query across reruns
        "arxiv_ingest_msgs": [],    # Status strings to display after rerun
        "current_review_id": None,  # ID of the review saved from the last pipeline run
        "pending_nav": None,        # Navigation target applied before sidebar renders
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ── Login gate ────────────────────────────────────────────────────────────────
if not st.session_state["username"]:
    st.markdown(
        '<div class="login-box">'
        '<h2>Research Agent</h2>'
        '<p>Enter a username to access your personal knowledge base.<br>'
        'Your papers and reviews are saved under your name.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    # Centre the form inside the login box using columns
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        with st.form("login_form"):
            username_input = st.text_input(
                "Username",
                placeholder="e.g. alice",
                label_visibility="collapsed",
            )
            sign_in = st.form_submit_button("Sign In", use_container_width=True)
        if sign_in:
            if username_input.strip():
                st.session_state["username"] = username_input.strip()
                st.rerun()
            else:
                st.warning("Please enter a username.")
    st.stop()

# ── Thread-status sync (runs on every render) ─────────────────────────────────
_t = st.session_state["pipeline_thread"]
if _t is not None and not _t.is_alive():
    _s = st.session_state["pipeline_state"]
    if _s and _s.done:
        st.session_state["pipeline_running"] = False
        st.session_state["pipeline_thread"] = None
        if _s.error:
            st.session_state["pipeline_status"] = "error"
        else:
            result = _s.result
            st.session_state["results"] = result
            st.session_state["pipeline_status"] = "done"
            rev_id = save_review(
                topic=result["topic"],
                literature_review=result["literature_review"],
                papers=result["papers"],
                research_gaps=result.get("research_gaps"),
                critique=result.get("critique", ""),
                username=st.session_state["username"],
            )
            st.session_state["current_review_id"] = rev_id
        st.session_state["pipeline_state"] = None


# ── Apply any pending programmatic navigation before sidebar renders ──────────
if st.session_state["pending_nav"]:
    st.session_state["nav_radio"] = st.session_state["pending_nav"]
    st.session_state["pending_nav"] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand"><h2>Research Agent</h2>'
        '<p>AI-powered literature assistant</p>'
        '<p>Thank you to arXiv for use of its open access interoperability.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="text-align:center;font-size:.8rem;color:#a0aec0;margin-bottom:.5rem;">'
        f'Signed in as <strong style="color:#7c8cf8">{st.session_state["username"]}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button("Sign Out", use_container_width=True):
        st.session_state["username"] = ""
        st.session_state["results"] = None
        st.session_state["chat_history"] = []
        st.rerun()

    nav = st.radio(
        "Navigation",
        ["Search Papers", "Literature Review", "Ask Questions", "Knowledge Base"],
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state["page"] = nav

    st.markdown("---")

    fetch_n = st.slider(
        "Papers to fetch from arXiv",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Fetches this many papers, reranks with a cross-encoder, then keeps the top 5.",
    )

    min_score = st.slider(
        "Min relevance score",
        min_value=-2.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help=(
            "Cross-encoder score cutoff. Papers below this threshold are dropped even if "
            "they are in the top 5. Score > 0 means the model considers the paper relevant; "
            "raise this to be stricter, lower it if too few papers are returned."
        ),
    )

    st.markdown("---")

    # Knowledge-base stats
    try:
        collection = get_collection(st.session_state["username"])
        papers = get_ingested_papers(collection)
        total_chunks = collection.count()
    except Exception:
        papers, total_chunks = [], 0

    st.markdown("**Knowledge Base**")
    st.markdown(
        f'<div class="stat-pill"><span>Papers ingested</span><span>{len(papers)}</span></div>'
        f'<div class="stat-pill"><span>Total chunks</span><span>{total_chunks}</span></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("results"):
        topic_label = st.session_state["results"].get("topic", "")
        st.markdown(
            f"**Last search:** `{topic_label[:35]}…`"
            if len(topic_label) > 35
            else f"**Last search:** `{topic_label}`"
        )

    # Pipeline status badge
    _status = st.session_state["pipeline_status"]
    if _status == "running":
        st.markdown("**Pipeline running...**")
    elif _status == "done":
        st.markdown("**Pipeline complete.**")
    elif _status == "error":
        st.markdown("**Pipeline error** — see Search tab")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: Search Papers
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["page"] == "Search Papers":
    st.title("Search Research Papers")
    st.markdown(
        f"Enter a research topic to search arXiv. "
        f"The agent fetches **{fetch_n}** papers, reranks them with a cross-encoder, "
        f"and processes the top 5."
    )

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. retrieval-augmented generation for question answering",
        key="topic_input",
    )

    run_col, _ = st.columns([1, 4])
    with run_col:
        run_btn = st.button(
            "Run Pipeline",
            use_container_width=True,
            disabled=st.session_state["pipeline_running"],
        )

    if run_btn and topic.strip():
        _guard = _orchestrator.guardrail.validate_topic(topic.strip())
        if not _guard.allowed:
            st.error(f"Input rejected by security guardrail: {_guard.reason}")
        else:
            state = PipelineState()
            thread = threading.Thread(
                target=_pipeline_thread_fn,
                args=(state, topic.strip(), fetch_n, min_score, st.session_state["username"]),
                daemon=True,
            )
            st.session_state["pipeline_state"] = state
            st.session_state["pipeline_thread"] = thread
            st.session_state["pipeline_running"] = True
            st.session_state["pipeline_status"] = "running"
            thread.start()
            st.rerun()
    elif run_btn:
        st.warning("Please enter a research topic first.")

    # Live progress while pipeline is running
    _pipe_state = st.session_state.get("pipeline_state")
    if st.session_state["pipeline_running"] and _pipe_state:
        st.progress(_pipe_state.progress, text=f"{int(_pipe_state.progress * 100)}%")
        st.markdown(f"**{_pipe_state.status}**")
        time.sleep(1.5)
        st.rerun()
    elif st.session_state["pipeline_status"] == "error":
        _err_state = st.session_state.get("pipeline_state")
        err_msg = _err_state.error if _err_state else "Pipeline failed."
        st.error(err_msg)
    elif st.session_state["pipeline_status"] == "done":
        st.success("Pipeline complete! See the Literature Review tab for the full review.")

    # ── Show paper cards if results exist ────────────────────────────────────
    if st.session_state["results"] and "papers" in st.session_state["results"]:
        res = st.session_state["results"]
        st.markdown(f"### Papers for: *{res['topic']}*")
        for p in res["papers"]:
            score = p.get("relevance_score", 0)
            authors_short = p["authors"].split(",")[0] + " et al." if "," in p["authors"] else p["authors"]
            st.markdown(
                f'<div class="paper-card">'
                f'<div class="paper-title">{p["title"]}'
                f'<span class="score-badge">score {score:.2f}</span></div>'
                f'<div class="paper-meta">{authors_short} &nbsp;|&nbsp; {p["year"]} '
                f'&nbsp;|&nbsp; <a href="{p["pdf_url"]}" target="_blank" style="color:#63b3ed">PDF ↗</a></div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("View summary"):
                st.markdown(p["summary"])


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: Literature Review
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["page"] == "Literature Review":
    st.title("Literature Review")

    res = st.session_state.get("results")
    if not res or "literature_review" not in res:
        st.info("No literature review yet. Run a search first.")
    else:
        st.markdown(f"**Topic:** {res['topic']}")

        tab_review, tab_gaps = st.tabs(["Review", "Research Gaps"])

        with tab_review:
            dl_col, _ = st.columns([1, 5])
            with dl_col:
                st.download_button(
                    "Download (.md)",
                    data=res["literature_review"],
                    file_name="literature_review.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            st.markdown("**Papers included in this review:**")
            for p in res.get("papers", []):
                authors_short = p["authors"].split(",")[0] + " et al." if "," in p["authors"] else p["authors"]
                pdf_link = f" · [PDF ↗]({p['pdf_url']})" if p.get("pdf_url") else ""
                st.markdown(f"- **{p['title']}** ({p['year']}) — {authors_short}{pdf_link}")

            st.markdown(
                f'<div class="review-box">{res["literature_review"]}</div>',
                unsafe_allow_html=True,
            )

            if res.get("critique"):
                with st.expander("CriticAgent Notes", expanded=False):
                    st.markdown(res["critique"])

        with tab_gaps:
            gaps = res.get("research_gaps", {})
            if not gaps or not gaps.get("gaps_text"):
                st.info("No gap analysis available for this result. Re-run the pipeline to generate one.")
            else:
                st.markdown(gaps["gaps_text"])

                queries = gaps.get("suggested_queries", [])
                if queries:
                    st.markdown("---")
                    st.markdown("### Suggested Searches to Fill These Gaps")
                    st.caption("Click **Search →** to pre-fill the topic and jump to the Search page.")
                    for i, q in enumerate(queries):
                        q_col, btn_col = st.columns([6, 1])
                        with q_col:
                            st.code(q, language=None)
                        with btn_col:
                            if st.button("Search →", key=f"gap_q_{i}", use_container_width=True):
                                st.session_state["topic_input"] = q
                                st.session_state["pending_nav"] = "Search Papers"
                                st.rerun()

    # ── Past reviews ──────────────────────────────────────────────────────────
    past_reviews = load_reviews(st.session_state["username"])
    if past_reviews:
        st.markdown("---")
        st.markdown("### Previous Reviews")
        for rev in reversed(past_reviews):
            try:
                dt_str = datetime.fromisoformat(rev["timestamp"]).strftime("%b %d, %Y %H:%M")
            except Exception:
                dt_str = rev.get("timestamp", "")
            n_papers = len(rev.get("papers", []))
            label = f"**{rev['topic']}** — {dt_str} · {n_papers} paper{'s' if n_papers != 1 else ''}"
            with st.expander(label):
                st.markdown("**Papers included:**")
                for p in rev.get("papers", []):
                    pdf_link = f" · [PDF ↗]({p['pdf_url']})" if p.get("pdf_url") else ""
                    st.markdown(f"- **{p['title']}** ({p['year']}) — {p['authors']}{pdf_link}")
                st.markdown("---")
                st.markdown(
                    f'<div class="review-box">{rev["literature_review"]}</div>',
                    unsafe_allow_html=True,
                )
                if rev.get("critique"):
                    with st.expander("CriticAgent Notes", expanded=False):
                        st.markdown(rev["critique"])
                rev_gaps = rev.get("research_gaps", {})
                if rev_gaps and rev_gaps.get("gaps_text"):
                    st.markdown("---")
                    st.markdown(rev_gaps["gaps_text"])
                    rev_queries = rev_gaps.get("suggested_queries", [])
                    if rev_queries:
                        st.markdown("**Suggested searches:**")
                        for i, q in enumerate(rev_queries):
                            q_col, btn_col = st.columns([6, 1])
                            with q_col:
                                st.code(q, language=None)
                            with btn_col:
                                if st.button("Search →", key=f"past_{rev['id']}_{i}", use_container_width=True):
                                    st.session_state["topic_input"] = q
                                    st.session_state["pending_nav"] = "Search Papers"
                                    st.rerun()
                if st.button("Delete Review", key=f"del_rev_{rev['id']}"):
                    delete_review(rev["id"], st.session_state["username"])
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: Ask Questions
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["page"] == "Ask Questions":
    st.title("Ask the Papers")
    st.markdown(
        "Ask anything about the research in your knowledge base. "
        "The agent retrieves relevant chunks and answers using Claude."
    )

    collection = get_collection(st.session_state["username"])
    paper_count = len(get_ingested_papers(collection))

    if paper_count == 0:
        st.warning("Your knowledge base is empty. Run a search or upload PDFs first.")
    else:
        st.caption(f"Querying across {paper_count} paper(s)")

        # Display chat history
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                conf_level = msg.get("confidence_level", "")
                conf_reason = msg.get("confidence_reason", "")
                _colors = {"High": "#48bb78", "Medium": "#ed8936", "Low": "#fc8181"}
                _color = _colors.get(conf_level, "")
                conf_html = ""
                if _color:
                    conf_html = (
                        f'<div style="font-size:.78rem;color:#a0aec0;margin-top:.6rem;'
                        f'padding-top:.4rem;border-top:1px solid #2a2f3e;">'
                        f'<strong style="color:{_color}">Confidence: {conf_level}</strong>'
                        + (f" — {conf_reason}" if conf_reason else "")
                        + "</div>"
                    )
                st.markdown(
                    f'<div class="chat-assistant">{msg["content"]}{conf_html}</div>',
                    unsafe_allow_html=True,
                )

        # Input
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input(
                "Your question",
                placeholder="What are the main findings about…?",
                label_visibility="collapsed",
            )
            ask_col, clear_col = st.columns([3, 1])
            with ask_col:
                submitted = st.form_submit_button(
                    "Ask →",
                    use_container_width=True,
                    disabled=st.session_state["qa_processing"],
                )
            with clear_col:
                cleared = st.form_submit_button("Clear chat", use_container_width=True)

        if cleared:
            st.session_state["chat_history"] = []
            st.rerun()

        if submitted and question.strip():
            _guard = _orchestrator.guardrail.validate_question(question.strip())
            if not _guard.allowed:
                st.error(f"Question rejected by security guardrail: {_guard.reason}")
            else:
                st.session_state["qa_processing"] = True
                try:
                    with st.spinner("Retrieving context & generating answer…"):
                        compact_history = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state["chat_history"][-6:]
                        ]
                        try:
                            ans = _orchestrator.answer(question, collection, compact_history)
                            if "error" in ans:
                                ans = {
                                    "answer": f"Error: {ans['error']}",
                                    "confidence_level": "",
                                    "confidence_reason": "",
                                }
                        except Exception as exc:
                            ans = {
                                "answer": f"Error: {exc}",
                                "confidence_level": "",
                                "confidence_reason": "",
                            }
                finally:
                    st.session_state["qa_processing"] = False

                st.session_state["chat_history"].append({"role": "user", "content": question})
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": ans["answer"],
                    "confidence_level": ans.get("confidence_level", ""),
                    "confidence_reason": ans.get("confidence_reason", ""),
                })
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state["page"] == "Knowledge Base":
    st.title("Knowledge Base")

    tab_view, tab_upload, tab_arxiv = st.tabs(
        ["Ingested Papers", "Upload PDF", "Find on arXiv"]
    )

    # ── View & manage ─────────────────────────────────────────────────────────
    with tab_view:
        collection = get_collection(st.session_state["username"])
        papers = get_ingested_papers(collection)

        if not papers:
            st.info("No papers ingested yet.")
        else:
            # Compute selection from previous render before rendering any widgets
            selected = [p for p in papers if st.session_state.get(f"kb_sel_{p['paper_id']}", False)]
            n_sel = len(selected)

            # ── Header controls ───────────────────────────────────────────────
            st.markdown(f"**{len(papers)} paper(s)** · {n_sel} / 6 selected for review")
            if n_sel == 6:
                st.caption("Maximum of 6 papers reached. Deselect one to change your selection.")

            topic_input = st.text_input(
                "Review topic",
                placeholder="e.g. self-supervised learning for vision",
                help="Theme for the generated review. Leave blank to auto-derive from the selected paper titles.",
            )

            gen_col, _ = st.columns([2, 5])
            with gen_col:
                gen_btn = st.button(
                    f"Generate Lit Review{f' ({n_sel})' if n_sel else ''}",
                    use_container_width=True,
                    disabled=n_sel == 0 or st.session_state["pipeline_running"],
                )

            st.markdown("---")

            # ── Paper list with checkboxes ────────────────────────────────────
            for p in papers:
                is_checked = st.session_state.get(f"kb_sel_{p['paper_id']}", False)
                chk_col, info_col, del_col = st.columns([1, 10, 1])
                with chk_col:
                    st.markdown("<div style='padding-top:1.1rem'>", unsafe_allow_html=True)
                    st.checkbox(
                        "",
                        key=f"kb_sel_{p['paper_id']}",
                        label_visibility="collapsed",
                        disabled=n_sel >= 6 and not is_checked,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with info_col:
                    st.markdown(
                        f'<div class="paper-card">'
                        f'<div class="paper-title">{p["title"]}</div>'
                        f'<div class="paper-meta">{p["authors"]} &nbsp;|&nbsp; {p["year"]} '
                        f'&nbsp;|&nbsp; ID: <code>{p["paper_id"]}</code></div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with del_col:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Delete", key=f"del_{p['paper_id']}", help="Remove from knowledge base"):
                        n = delete_paper(collection, p["paper_id"])
                        st.success(f"Removed {n} chunks for '{p['title']}'")
                        st.rerun()
                if p.get("summary"):
                    with st.expander("View summary"):
                        st.markdown(p["summary"])

            # ── Generate review from selection ────────────────────────────────
            if gen_btn and selected:
                if topic_input.strip():
                    topic_label = topic_input.strip()
                else:
                    first_titles = [p["title"].split(":")[0][:45] for p in selected[:2]]
                    topic_label = " · ".join(first_titles)

                # Normalise to the shape the pipeline produces
                norm_papers = [
                    {
                        "paper_id": p.get("paper_id", ""),
                        "title": p.get("title", ""),
                        "authors": p.get("authors", ""),
                        "year": p.get("year", ""),
                        "abstract": "",
                        "summary": p.get("summary", ""),
                        "relevance_score": 0.0,
                        "pdf_url": "",
                    }
                    for p in selected
                ]

                with st.spinner(
                    f"Generating review for {n_sel} selected paper{'s' if n_sel != 1 else ''}…"
                ):
                    llm = get_client()
                    lit_review = generate_literature_review(llm, topic_label, norm_papers)
                    gaps = find_research_gaps(llm, topic_label, lit_review, norm_papers)

                result = {
                    "topic": topic_label,
                    "papers": norm_papers,
                    "literature_review": lit_review,
                    "research_gaps": gaps,
                }
                st.session_state["results"] = result
                rev_id = save_review(
                    topic=topic_label,
                    literature_review=lit_review,
                    papers=norm_papers,
                    research_gaps=gaps,
                    username=st.session_state["username"],
                )
                st.session_state["current_review_id"] = rev_id
                st.session_state["pending_nav"] = "Literature Review"
                st.rerun()

    # ── Upload PDF ────────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("Upload a PDF to add it to your knowledge base for Q&A.")
        uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded:
            with st.form("upload_form"):
                title = st.text_input("Paper title", value=uploaded.name.replace(".pdf", ""))
                authors = st.text_input("Authors (comma-separated)")
                year = st.text_input("Year", value="2024")
                submit_upload = st.form_submit_button("Ingest PDF", use_container_width=True)

            if submit_upload:
                with st.spinner("Parsing and embedding…"):
                    msg = ingest_uploaded_pdf(
                        pdf_bytes=uploaded.read(),
                        filename=uploaded.name,
                        title=title,
                        authors=authors,
                        year=year,
                    )
                if msg.startswith("OK:"):
                    st.success(msg[3:].strip())
                elif msg.startswith("INFO:"):
                    st.info(msg[5:].strip())
                else:
                    st.error(msg)
                st.rerun()

    # ── Find on arXiv ─────────────────────────────────────────────────────────
    with tab_arxiv:
        if st.session_state["arxiv_candidates"] is None:
            # Show results from the previous ingest, then clear them
            for _msg in st.session_state["arxiv_ingest_msgs"]:
                if _msg.startswith("OK:"):
                    st.success(_msg[3:].strip())
                elif _msg.startswith("INFO:"):
                    st.info(_msg[5:].strip())
                else:
                    st.error(_msg)
            st.session_state["arxiv_ingest_msgs"] = []

            st.markdown("Search arXiv for papers and add one or more to your knowledge base.")

            query = st.text_input(
                "Search query",
                value=st.session_state["arxiv_query"],
                placeholder="e.g. vision transformers image classification",
                key="arxiv_query_input",
            )

            find_col, _ = st.columns([1, 4])
            with find_col:
                find_btn = st.button(
                    "Find Papers",
                    use_container_width=True,
                    disabled=st.session_state["pipeline_running"] or st.session_state["qa_processing"],
                )
            if st.session_state["pipeline_running"]:
                st.caption("Disabled while pipeline is running — avoids arXiv rate limit conflicts.")

            if find_btn and query.strip():
                st.session_state["arxiv_query"] = query.strip()
                with st.spinner("Searching arXiv and reranking…"):
                    try:
                        results = search_and_rerank(query.strip(), fetch_n=10, top_k=5, min_score=min_score)
                    except Exception as exc:
                        st.error(f"Search failed: {exc}")
                        results = []

                if results:
                    st.session_state["arxiv_candidates"] = results
                    st.rerun()
                else:
                    st.warning("No papers found. Try a different query.")
            elif find_btn:
                st.warning("Please enter a search query first.")

        else:
            papers = st.session_state["arxiv_candidates"]
            st.markdown(f"**Found {len(papers)} papers — select which to add:**")

            for paper in papers:
                authors_display = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_display += " et al."
                abstract_snippet = paper.abstract[:250] + "…" if len(paper.abstract) > 250 else paper.abstract

                chk_col, card_col = st.columns([1, 11])
                with chk_col:
                    st.markdown("<div style='padding-top:1rem'>", unsafe_allow_html=True)
                    st.checkbox("", key=f"chk_{paper.paper_id}", label_visibility="collapsed")
                    st.markdown("</div>", unsafe_allow_html=True)
                with card_col:
                    st.markdown(
                        f'<div class="paper-card">'
                        f'<div class="paper-title">'
                        f'<a href="{paper.pdf_url}" target="_blank" style="color:#a5b4fc;text-decoration:none;">'
                        f"{paper.title}</a>"
                        f'<span class="score-badge">score {paper.relevance_score:.2f}</span>'
                        f"</div>"
                        f'<div class="paper-meta">'
                        f"{authors_display} &nbsp;|&nbsp; {paper.published} "
                        f"&nbsp;|&nbsp; ID: <code>{paper.paper_id}</code>"
                        f"</div>"
                        f'<p style="color:#a0aec0;font-size:.85rem;margin-top:.5rem;">{abstract_snippet}</p>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            selected = [p for p in papers if st.session_state.get(f"chk_{p.paper_id}", False)]
            n_sel = len(selected)

            ingest_label = (
                f"Ingest {n_sel} paper{'s' if n_sel != 1 else ''}"
                if n_sel else "Ingest selected"
            )
            ingest_col, again_col, _ = st.columns([2, 2, 3])
            with ingest_col:
                ingest_btn = st.button(
                    ingest_label,
                    use_container_width=True,
                    disabled=st.session_state["pipeline_running"] or n_sel == 0,
                )
            with again_col:
                again_btn = st.button(
                    "Search again",
                    use_container_width=True,
                )

            if again_btn:
                st.session_state["arxiv_candidates"] = None
                st.rerun()

            if ingest_btn and selected:
                with st.spinner(f"Ingesting {n_sel} paper{'s' if n_sel != 1 else ''}…"):
                    try:
                        msgs = ingest_arxiv_papers_batch(selected)
                    except Exception as exc:
                        msgs = [f"Error: Unexpected error: {exc}"]

                st.session_state["arxiv_ingest_msgs"] = msgs
                st.session_state["arxiv_candidates"] = None
                st.rerun()
