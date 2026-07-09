"""LLM access layer.

Uses Claude (via langchain-anthropic) when ANTHROPIC_API_KEY is set.
Falls back to a small deterministic stub when no key is present, so the
whole project can be cloned and demoed with zero API cost/setup. Swap
in a real key at any time -- no other code changes needed.
"""

import os
import re
from typing import Optional

_ANTHROPIC_MODEL = "claude-sonnet-4-6"


class StubLLM:
    """Deterministic, keyword-based fallback used when no API key is set.

    This is NOT meant to be smart -- it exists so `python demo/run_demo.py`
    works out of the box for anyone cloning the repo, and so unit tests
    don't need network access or a paid API key.
    """

    def invoke(self, prompt: str) -> str:
        p = prompt.lower()
        if "classify the following support ticket" in p:
            if any(w in p for w in ["crash", "freeze", "error", "doesn't work", "does not work",
                                     "nothing happens", "fails", "not working"]):
                category, priority = "bug", "high"
            elif any(w in p for w in ["feature request", "would be great", "can this be",
                                       "increase", "custom date"]):
                category, priority = "feature_request", "medium"
            elif any(w in p for w in ["how do i", "where is", "how to", "can't find"]):
                category, priority = "question", "low"
            else:
                category, priority = "other", "low"
            if any(w in p for w in ["blocking", "migration", "urgent", "crashes"]):
                priority = "critical" if category == "bug" else priority
            return f"category: {category}\npriority: {priority}"

        if "draft a helpful, concise reply" in p:
            if "known issue" in p or "root cause" in p:
                return (
                    "Hi there — thanks for the report! This is a known issue our team is "
                    "actively working on. We've included a workaround below based on our "
                    "internal documentation and will update this ticket once the fix ships."
                )
            if "roadmap" in p or "feature status" in p:
                return (
                    "Thanks for the suggestion! This isn't available yet, but it's on our "
                    "roadmap and we've logged your request so Product can track demand. "
                    "We'll post here if the timeline firms up."
                )
            if "how to invite" in p or "settings > members" in p:
                return (
                    "Happy to help! You can invite teammates from Settings > Members > "
                    "Invite Member — details are in the steps below."
                )
            return (
                "Hi there — thanks for reaching out. We've looked into this using our "
                "internal documentation and included the relevant guidance below."
            )

        return "Acknowledged."


class ClaudeLLM:
    """Thin wrapper around langchain_anthropic.ChatAnthropic."""

    def __init__(self, model: str = _ANTHROPIC_MODEL):
        from langchain_anthropic import ChatAnthropic

        self._client = ChatAnthropic(model=model, temperature=0, max_tokens=500)

    def invoke(self, prompt: str) -> str:
        result = self._client.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)


_llm_singleton: Optional[object] = None


def get_llm():
    """Returns a cached LLM instance: real Claude if ANTHROPIC_API_KEY is set,
    otherwise the offline stub."""
    global _llm_singleton
    if _llm_singleton is not None:
        return _llm_singleton

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            _llm_singleton = ClaudeLLM()
            return _llm_singleton
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[llm] Falling back to StubLLM, failed to init ChatAnthropic: {exc}")

    _llm_singleton = StubLLM()
    return _llm_singleton


def parse_key_value(text: str, key: str, default: str = "unknown") -> str:
    """Small helper to pull `key: value` lines out of an LLM response."""
    match = re.search(rf"{key}\s*:\s*([a-zA-Z_]+)", text, re.IGNORECASE)
    return match.group(1).strip().lower() if match else default
