from typing import Callable, Dict, List, Optional

from agent.guardrails import SecurityGuardrail
from agent.llm import answer_question, get_client
from agent.pipeline import run_research_pipeline
from agent.rag import query_papers


class ResearchOrchestrator:
    """
    Coordinates four specialized agents in a sequential research pipeline.

    Agents
    ------
    SearchAgent    (agent/arxiv_search.py)
        Queries arXiv for candidate papers and reranks them with a
        cross-encoder model to surface the most relevant results.

    IngestionAgent (agent/pdf_parser.py + agent/rag.py)
        Downloads PDFs, extracts full text via PyMuPDF, splits it into
        overlapping word-level chunks, embeds each chunk with
        all-MiniLM-L6-v2, and stores everything in a persistent
        ChromaDB vector store.

    SynthesisAgent (agent/llm.py)
        Calls Claude to produce per-paper summaries, a unified literature
        review, and a research gap analysis with suggested arXiv queries.

    QAAgent        (agent/llm.py + agent/rag.py)
        Embeds the user's question, retrieves the top-6 semantically
        similar chunks from ChromaDB, and sends them to Claude to
        generate a grounded, cited answer with a confidence rating.

    Communication
    -------------
    Agents communicate through two channels:
      - Shared ChromaDB collection: IngestionAgent writes chunks;
        QAAgent reads them at query time.
      - Direct Python calls through this orchestrator: each agent is
        a module-level function or singleton, invoked in sequence by
        run_pipeline() and answer().

    Security
    --------
    A SecurityGuardrail validates every user input before it reaches
    any agent. Blocked inputs are returned as error dicts so the UI
    can display a clear rejection message without invoking Claude.
    """

    def __init__(self):
        self.guardrail = SecurityGuardrail()
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_client()
        return self._llm

    def run_pipeline(
        self,
        topic: str,
        fetch_n: int = 20,
        min_score: float = 0.0,
        progress_cb: Optional[Callable[[float], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None,
        username: str = "default",
    ) -> Dict:
        """
        Orchestrate the full research pipeline.

        Flow: Guardrail → SearchAgent → IngestionAgent → SynthesisAgent
        Returns a result dict with keys: topic, papers, literature_review,
        research_gaps — or {"error": reason} if blocked or failed.
        """
        guard = self.guardrail.validate_topic(topic)
        if not guard.allowed:
            return {"error": f"Input blocked by security guardrail: {guard.reason}"}

        return run_research_pipeline(
            topic=guard.text,
            progress_cb=progress_cb,
            status_cb=status_cb,
            fetch_n=fetch_n,
            min_score=min_score,
            username=username,
        )

    def answer(
        self,
        question: str,
        collection,
        chat_history: List[Dict],
    ) -> Dict:
        """
        Route a question through the guardrail and QAAgent.

        Flow: Guardrail → RAG retrieval → QAAgent (Claude)
        Returns a dict with keys: answer, confidence_level, confidence_reason
        — or {"error": reason} if the input was blocked.
        """
        guard = self.guardrail.validate_question(question)
        if not guard.allowed:
            return {"error": guard.reason}

        chunks = query_papers(collection, guard.text, n_results=6)
        return answer_question(self.llm, guard.text, chunks, chat_history)
