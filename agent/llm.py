import anthropic
import os
import re
from typing import List, Dict

MODEL = "claude-sonnet-4-6"


def get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file."
        )
    return anthropic.Anthropic(api_key=api_key)


def summarize_paper(
    client: anthropic.Anthropic,
    title: str,
    abstract: str,
    full_text: str,
) -> str:
    """Generate a structured summary of one paper."""
    # Keep first ~6 000 words to stay well within context
    truncated = " ".join(full_text.split()[:6000])

    response = client.messages.create(
        model=MODEL,
        max_tokens=700,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Summarize this research paper concisely and accurately.\n\n"
                    f"**Title:** {title}\n\n"
                    f"**Abstract:** {abstract}\n\n"
                    f"**Full text (excerpt):** {truncated}\n\n"
                    "Return exactly this structure:\n"
                    "**Objective:** (1 sentence)\n"
                    "**Methods:** (1–2 sentences)\n"
                    "**Key Findings:** (3 bullet points)\n"
                    "**Significance:** (1 sentence)"
                ),
            }
        ],
    )
    return response.content[0].text


def _trim_incomplete(text: str) -> str:
    """Trim a truncated response to the last complete sentence.

    Removes a dangling section header at the tail, then trims to the
    rightmost sentence-ending punctuation if the text ends mid-sentence.
    Only called when stop_reason == 'max_tokens'.
    """
    text = re.sub(r'\n{1,2}#+\s+[^\n]+\s*$', '', text).rstrip()

    if re.search(r'[.!?][\'"]?\s*$', text):
        return text

    last_end = max(
        text.rfind('. '), text.rfind('! '), text.rfind('? '),
        text.rfind('.\n'), text.rfind('!\n'), text.rfind('?\n'),
        text.rfind('.'),
    )
    if last_end > 0:
        text = text[:last_end + 1].rstrip()

    return text


def generate_literature_review(
    client: anthropic.Anthropic,
    topic: str,
    paper_summaries: List[Dict],
) -> str:
    """Synthesise a full literature review from per-paper summaries."""
    blocks = "\n\n---\n\n".join(
        f"**{s['title']}** ({s['year']})\nAuthors: {s['authors']}\n\n{s['summary']}"
        for s in paper_summaries
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": (
                    f'Write a structured academic literature review on: "{topic}"\n\n'
                    f"Papers to synthesise:\n{blocks}\n\n"
                    "Structure the review as:\n"
                    "## Introduction\n"
                    "Contextualise the topic and explain why it matters.\n\n"
                    "## Key Themes\n"
                    "Group papers thematically and discuss each theme.\n\n"
                    "## Findings & Consensus\n"
                    "What do these papers collectively establish?\n\n"
                    "## Gaps & Contradictions\n"
                    "Where do papers disagree, or what remains unexplored?\n\n"
                    "## Conclusion\n"
                    "Current state of the field and promising future directions.\n\n"
                    "Use in-text citations as (Author et al., Year). "
                    "Write in formal academic style. "
                    "If you are running long, write shorter paragraphs for later sections "
                    "rather than leaving any sentence or section unfinished."
                ),
            }
        ],
    )
    text = response.content[0].text
    if response.stop_reason == "max_tokens":
        text = _trim_incomplete(text)
    return text


def critique_literature_review(
    client: anthropic.Anthropic,
    topic: str,
    literature_review: str,
    paper_summaries: List[Dict],
) -> str:
    """CriticAgent: review a draft literature review and return a numbered critique."""
    paper_list = "\n".join(
        f"- {s['title']} ({s['year']}) by {s['authors']}"
        for s in paper_summaries
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=(
            "You are a CriticAgent — a rigorous academic peer reviewer. "
            "Your role is to identify specific weaknesses in draft literature reviews "
            "so they can be improved. Be concrete and constructive, not generic."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f'Review this draft literature review on "{topic}" and identify its weaknesses.\n\n'
                    f"**Papers available:**\n{paper_list}\n\n"
                    f"**Draft review:**\n{literature_review}\n\n"
                    "Identify 3–5 specific weaknesses. For each one, note:\n"
                    "- What is missing or wrong\n"
                    "- Which section it affects\n"
                    "- How it should be improved\n\n"
                    "Format as a numbered list. Be specific — reference actual paper titles "
                    "or claims from the review where possible."
                ),
            }
        ],
    )
    return response.content[0].text


def revise_literature_review(
    client: anthropic.Anthropic,
    topic: str,
    original_review: str,
    critique: str,
    paper_summaries: List[Dict],
) -> str:
    """SynthesisAgent: regenerate an improved literature review addressing the critique."""
    blocks = "\n\n---\n\n".join(
        f"**{s['title']}** ({s['year']})\nAuthors: {s['authors']}\n\n{s['summary']}"
        for s in paper_summaries
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": (
                    f'Revise this literature review on "{topic}" to address the critique below.\n\n'
                    f"**Original review:**\n{original_review}\n\n"
                    f"**Critique to address:**\n{critique}\n\n"
                    f"**Papers for reference:**\n{blocks}\n\n"
                    "Produce an improved version with the same 5-section structure:\n"
                    "## Introduction\n"
                    "## Key Themes\n"
                    "## Findings & Consensus\n"
                    "## Gaps & Contradictions\n"
                    "## Conclusion\n\n"
                    "Directly address each point raised in the critique. "
                    "Use in-text citations as (Author et al., Year). "
                    "Write in formal academic style. "
                    "If you are running long, write shorter paragraphs for later sections "
                    "rather than leaving any sentence or section unfinished."
                ),
            }
        ],
    )
    text = response.content[0].text
    if response.stop_reason == "max_tokens":
        text = _trim_incomplete(text)
    return text


def find_research_gaps(
    client: anthropic.Anthropic,
    topic: str,
    literature_review: str,
    paper_summaries: List[Dict],
) -> Dict:
    """Identify research gaps and suggest follow-up arXiv queries.

    Returns a dict with keys: gaps_text (markdown), suggested_queries (list of strings).
    """
    paper_list = "\n".join(
        f"- {s['title']} ({s['year']}) by {s['authors']}"
        for s in paper_summaries
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": (
                    f'Based on this literature review on "{topic}", identify what is missing.\n\n'
                    f"**Papers reviewed:**\n{paper_list}\n\n"
                    f"**Literature review:**\n{literature_review}\n\n"
                    "Produce two things:\n\n"
                    "1. A structured '## Future Research Directions' markdown section covering:\n"
                    "   - 3–5 specific open research questions (with a short explanation of why each is open)\n"
                    "   - Key methodological limitations in current work\n"
                    "   - Underexplored sub-topics or application domains\n\n"
                    "2. After the section, on a new line write exactly:\n"
                    "SUGGESTED_QUERIES:\n"
                    "Then list 3–5 concrete arXiv search queries (one per line, prefixed with '- ') "
                    "that would help fill the identified gaps. Make queries specific and searchable."
                ),
            }
        ],
    )

    text = response.content[0].text

    if "SUGGESTED_QUERIES:" in text:
        gaps_text, queries_raw = text.split("SUGGESTED_QUERIES:", 1)
        gaps_text = gaps_text.strip()
        queries = []
        for line in queries_raw.strip().splitlines():
            line = line.strip().lstrip("-").lstrip("0123456789.").strip()
            if line:
                queries.append(line)
    else:
        gaps_text = text.strip()
        queries = []

    return {
        "gaps_text": gaps_text,
        "suggested_queries": queries,
    }


def answer_question(
    client: anthropic.Anthropic,
    question: str,
    context_chunks: List[Dict],
    chat_history: List[Dict],
) -> Dict:
    """Answer a user question grounded in retrieved paper chunks.

    Returns a dict with keys: answer, confidence_level (High/Medium/Low), confidence_reason.
    """
    context_text = "\n\n---\n\n".join(
        f"[{c['title']} ({c['year']})]\n{c['text']}"
        for c in context_chunks
    )

    system = (
        "You are a research assistant with access to a curated collection of "
        "academic papers. Answer questions accurately and specifically, always "
        "citing the relevant papers by title and year. "
        "If the answer cannot be found in the provided context, say so clearly "
        "rather than speculating.\n\n"
        "After your answer, on a new line, add a confidence rating in exactly this format:\n"
        "CONFIDENCE: High — <one sentence explaining why>\n"
        "Use 'High' if the papers directly and fully address the question, "
        "'Medium' if they partially cover it or require some inference, "
        "'Low' if the coverage is sparse, tangential, or the answer is mostly inferred. "
        "Keep your answer focused. If you must be concise, still finish your current "
        "sentence completely before writing the CONFIDENCE line."
    )

    messages = list(chat_history) + [
        {
            "role": "user",
            "content": (
                f"**Question:** {question}\n\n"
                f"**Relevant excerpts from the knowledge base:**\n{context_text}\n\n"
                "Provide a thorough, well-cited answer."
            ),
        }
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    text = response.content[0].text
    if response.stop_reason == "max_tokens":
        text = _trim_incomplete(text)

    # Parse the confidence line appended by Claude
    if "\nCONFIDENCE:" in text:
        answer_text, conf_part = text.rsplit("\nCONFIDENCE:", 1)
        answer_text = answer_text.strip()
        conf_line = conf_part.strip()
        for sep in (" — ", " – ", " - "):
            if sep in conf_line:
                level, reason = conf_line.split(sep, 1)
                level = level.strip()
                reason = reason.strip()
                break
        else:
            level = conf_line.strip()
            reason = ""
    else:
        answer_text = text.strip()
        level = "Unknown"
        reason = ""

    level = level.capitalize()
    if level not in ("High", "Medium", "Low"):
        level = "Unknown"

    return {
        "answer": answer_text,
        "confidence_level": level,
        "confidence_reason": reason,
    }
