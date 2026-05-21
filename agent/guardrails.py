import re
from dataclasses import dataclass, field
from typing import List

MAX_TOPIC_LEN = 300
MAX_QUESTION_LEN = 1000

# Patterns that indicate prompt injection or code injection attempts
_INJECTION_PATTERNS: List[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"you\s+are\s+now\b",
    r"forget\s+(everything|all|your)",
    r"\bsystem\s*prompt\b",
    r"\bjailbreak\b",
    r"act\s+as\s+(if\s+you\s+are|a\s+)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"override\s+(your\s+)?(instructions|guidelines|rules)",
    r"<\s*script",
    r"__import__",
    r"os\.system",
    r"eval\s*\(",
]


@dataclass
class GuardrailResult:
    allowed: bool
    text: str
    reason: str = ""


class SecurityGuardrail:
    """
    Input validation layer that sits in front of all agents.

    Checks:
      - Length limits (prevents context-stuffing attacks)
      - Prompt injection patterns (jailbreak attempts, role overrides)
      - Code injection markers (__import__, os.system, eval)
    """

    def validate_topic(self, text: str) -> GuardrailResult:
        """Validate a research topic before passing it to SearchAgent."""
        text = text.strip()
        if not text:
            return GuardrailResult(False, text, "Topic cannot be empty.")
        if len(text) > MAX_TOPIC_LEN:
            return GuardrailResult(
                False, text,
                f"Topic exceeds maximum length of {MAX_TOPIC_LEN} characters."
            )
        hit = self._check_injection(text)
        if hit:
            return GuardrailResult(
                False, text,
                f"Input contains a disallowed pattern and was blocked for safety."
            )
        return GuardrailResult(True, text)

    def validate_question(self, text: str) -> GuardrailResult:
        """Validate a Q&A question before passing it to QAAgent."""
        text = text.strip()
        if not text:
            return GuardrailResult(False, text, "Question cannot be empty.")
        if len(text) > MAX_QUESTION_LEN:
            return GuardrailResult(
                False, text,
                f"Question exceeds maximum length of {MAX_QUESTION_LEN} characters."
            )
        hit = self._check_injection(text)
        if hit:
            return GuardrailResult(
                False, text,
                "Input contains a disallowed pattern and was blocked for safety."
            )
        return GuardrailResult(True, text)

    def _check_injection(self, text: str) -> bool:
        """Return True if any injection pattern matches."""
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
