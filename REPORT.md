# Written Report: AI Research Agent — Multi-Agent System

## 1. Multi-Agent Architecture

The system is a research automation assistant built around two LLM-powered agents and two specialized processing modules, coordinated by a central orchestrator. The distinction is intentional: agents use language models to reason and generate; modules use deterministic ML pipelines to retrieve and transform data.

**LLM Agents:**

| Agent | File | Role |
|---|---|---|
| **SynthesisAgent** | `agent/llm.py` | Calls Claude (claude-sonnet-4-6) to produce per-paper summaries, a draft literature review, and a revised literature review incorporating the CriticAgent's feedback; also generates a research gap analysis with suggested follow-up queries |
| **CriticAgent** | `agent/llm.py` | Reviews the draft literature review and returns a structured numbered critique identifying 3–5 specific weaknesses (missing themes, unsupported claims, structural gaps, missing citations) |
| **QAAgent** | `agent/llm.py` + `agent/rag.py` | Retrieves the top-6 most semantically relevant chunks from ChromaDB and sends them to Claude to generate a grounded, cited answer with a confidence rating |

**Processing Modules:**

| Module | File | Role |
|---|---|---|
| **SearchModule** | `agent/arxiv_search.py` | Queries the arXiv API for candidate papers and reranks them using a cross-encoder model (`ms-marco-MiniLM-L-6-v2`) to surface the most semantically relevant results |
| **IngestionModule** | `agent/pdf_parser.py` + `agent/rag.py` | Downloads PDFs, extracts full text via PyMuPDF, splits it into overlapping 400-word chunks, embeds each chunk with `all-MiniLM-L6-v2`, and stores vectors + metadata in a persistent ChromaDB collection |

**Orchestration** (`agent/orchestrator.py`): The `ResearchOrchestrator` class coordinates the full pipeline — SearchModule → IngestionModule → SynthesisAgent for paper processing, and QAAgent independently for question answering. Components communicate through two channels: a shared ChromaDB collection (IngestionModule writes; QAAgent reads) and direct Python function calls through the orchestrator. The pipeline runs on a background thread (Python `threading.Thread`) so the UI remains navigable during long searches.

**Architecture diagram:**
```
User Input
    │
    ▼
SecurityGuardrail ── blocks invalid / malicious inputs
    │
    ▼
ResearchOrchestrator
    ├──► SearchModule    — arXiv fetch + cross-encoder rerank
    │         │
    ├──► IngestionModule — PDF → chunks → ChromaDB
    │         │
    └──► SynthesisAgent  — per-paper summaries (Claude)
              │
              ▼
         SynthesisAgent  — draft literature review (Claude)
              │
              ▼
         CriticAgent     — critique the draft: 3–5 weaknesses (Claude)
              │
              ▼
         SynthesisAgent  — revised literature review + gap analysis (Claude)

    └──► QAAgent (independent) — RAG retrieval → Claude answer
                                     └─ confidence scoring
```

---

## 2. Security, Safety, and Guardrails

**Input validation** (`agent/guardrails.py`): A `SecurityGuardrail` class sits in front of all agents and modules. Every user-supplied topic and question is validated for length (topics ≤ 300 characters, questions ≤ 1,000 characters) and scanned against 12 regex patterns covering common prompt injection techniques ("ignore previous instructions", "you are now", "act as", etc.) and code injection markers (`__import__`, `os.system`, `eval`). Blocked inputs surface a visible error in the UI — no agent or module is invoked.

**LLM role constraints**: Each Claude call uses a scoped system prompt that restricts Claude to a specific role (research summarizer, literature reviewer, Q&A assistant). Claude is explicitly instructed to cite sources, avoid speculation, and indicate when information is not in the provided context. This prevents open-ended generation and keeps outputs grounded in the retrieved documents.

**Confidence scoring as an output guardrail**: The QAAgent parses a `CONFIDENCE: High/Medium/Low` signal from every Claude response. This tells the user how well the knowledge base actually covers their question — a practical guardrail against over-trusting answers on topics not well-covered by the ingested papers.

**Secrets management**: The Anthropic API key is stored in a `.env` file loaded by `python-dotenv`. The `.env` file is excluded from version control via `.gitignore`. A `.env.example` template is provided so users can supply their own key without exposing secrets in the repository.

**No PII storage**: The system stores only paper metadata (title, authors, year, abstract chunks) in ChromaDB. No user queries or chat history are persisted to disk.

---

## 3. Implementation Approach

**Stack**: Python 3.9+, Streamlit (UI), Anthropic SDK (Claude API), ChromaDB (vector store), sentence-transformers (embeddings + reranking), PyMuPDF (PDF extraction), arxiv (arXiv API client). No external agent framework (LangChain, AutoGen, etc.) — the orchestration is a custom lightweight class, which keeps the system predictable and debuggable.

**Instantiation and coordination**: Processing modules are stateless (the cross-encoder and embedding model are loaded once at import time as module-level singletons). The `ResearchOrchestrator` holds a reference to the `SecurityGuardrail` and lazy-initialises the Claude client on first use. The background pipeline thread communicates progress and status back to the UI via injected callback functions (`progress_cb`, `status_cb`).

**Error handling**: PDF download failures fall back silently to the paper's abstract. arXiv rate limit errors (HTTP 429) are retried up to 3 times with exponential backoff. A 3-second delay is enforced between paper downloads to respect arXiv's rate limit. ChromaDB ingestion is idempotent — papers already in the store are skipped based on a deterministic MD5 chunk ID scheme.

**Testing approach**: Manual end-to-end testing against live arXiv queries. Guardrail logic is independently testable (pure Python regex, no external calls). The confidence scoring system provides an implicit quality signal on Q&A outputs.

---

## 4. Use of AI / LLMs and Collaboration

**Where Claude is used**: Claude (claude-sonnet-4-6) is invoked at five distinct points by the three LLM agents, each with a narrowly scoped prompt:
1. **Summarize** (SynthesisAgent) — extract Objective / Methods / Key Findings / Significance from a single paper
2. **Literature review draft** (SynthesisAgent) — synthesise findings across all papers into a 5-section academic review
3. **Critique** (CriticAgent) — review the draft and return a numbered list of 3–5 specific weaknesses
4. **Literature review revision** (SynthesisAgent) — regenerate an improved review that directly addresses the critique
5. **Gap analysis** (SynthesisAgent) — identify open research questions and suggest follow-up arXiv queries
6. **Q&A** (QAAgent) — answer a user question grounded in retrieved chunks, with citations and confidence

**Collaboration pattern**: The SynthesisAgent's summaries feed into the draft review; the CriticAgent's critique feeds into the revision; and the revised review feeds into the gap analysis — a sequential refinement chain. The QAAgent reads ChromaDB chunks written by the IngestionModule, making the agent and module loosely coupled through shared storage rather than direct calls. The processing modules handle all data retrieval and transformation so the LLM agents receive clean, structured inputs.

**Autonomy vs. control**: The system is deliberately low-autonomy. The pipeline is a fixed sequential workflow — agents do not make branching decisions or call each other recursively. Claude is given explicit output format constraints (structured markdown, CONFIDENCE: delimiter) and told to refuse rather than speculate. The user controls which papers are ingested and can delete them at any time. This design prioritises predictability and transparency over open-ended autonomy.
