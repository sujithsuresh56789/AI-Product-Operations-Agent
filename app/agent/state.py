"""Shared state object passed between LangGraph nodes.

Every node reads from and writes back into this TypedDict. Keeping the
schema in one place makes the graph easy to reason about and test.
"""

from typing import List, Optional, TypedDict


class RetrievedDoc(TypedDict):
    source: str
    text: str
    score: float


class AgentState(TypedDict, total=False):
    # --- input ---
    ticket_id: str
    ticket_title: str
    ticket_description: str

    # --- classify_ticket ---
    category: str          # bug | feature_request | question | other
    priority: str          # low | medium | high | critical

    # --- check_duplicates ---
    duplicate_of: Optional[str]

    # --- retrieve_docs ---
    retrieved_docs: List[RetrievedDoc]

    # --- decide_action ---
    action: str             # auto_respond | route_to_engineering | escalate | close_duplicate

    # --- generate_response ---
    response_draft: str

    # --- update_ticket ---
    jira_update: dict

    # --- trace (for demo/debugging, not required in production) ---
    trace: List[str]
