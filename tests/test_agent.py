"""Unit tests for individual nodes and the compiled graph.

Run with:  PYTHONPATH=. pytest -v
These use the offline StubLLM (no ANTHROPIC_API_KEY needed / no network calls).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.agent.graph import triage_ticket
from app.agent.nodes import check_duplicates, classify_ticket, decide_action
from app.services.jira_mock import MockJiraClient
from app.services.vector_store import search_docs


@pytest.fixture
def bug_state():
    return {
        "ticket_id": "TEST-1",
        "ticket_title": "App crashes on upload",
        "ticket_description": "The app crashes every time I upload a large file.",
        "trace": [],
    }


@pytest.fixture
def question_state():
    return {
        "ticket_id": "TEST-2",
        "ticket_title": "How do I invite a teammate?",
        "ticket_description": "Where is the option to invite a colleague to my workspace?",
        "trace": [],
    }


def test_classify_ticket_detects_bug(bug_state):
    result = classify_ticket(bug_state)
    assert result["category"] == "bug"
    assert result["priority"] in {"high", "critical"}


def test_classify_ticket_detects_question(question_state):
    result = classify_ticket(question_state)
    assert result["category"] == "question"


def test_decide_action_routes_high_priority_bug_to_engineering(bug_state):
    state = {**bug_state, "category": "bug", "priority": "high", "duplicate_of": None}
    result = decide_action(state)
    assert result["action"] == "route_to_engineering"


def test_decide_action_closes_duplicates(bug_state):
    state = {**bug_state, "category": "bug", "priority": "high", "duplicate_of": "OPS-1"}
    result = decide_action(state)
    assert result["action"] == "close_duplicate"


def test_mock_jira_client_add_comment_roundtrip():
    client = MockJiraClient()
    ticket = client.list_tickets()[0]
    client.add_comment(ticket["id"], "test comment")
    assert "test comment" in client.get_ticket(ticket["id"])["comments"]


def test_vector_search_returns_relevant_doc():
    results = search_docs("export to csv not downloading", top_k=1)
    assert results
    assert results[0]["source"] == "kb_export_csv"


def test_check_duplicates_finds_similar_earlier_ticket():
    state = {
        "ticket_id": "OPS-105",
        "ticket_title": "Export button not working (duplicate?)",
        "ticket_description": "Clicking export on the reports screen does not download anything.",
        "trace": [],
    }
    result = check_duplicates(state)
    assert result["duplicate_of"] == "OPS-101"


def test_full_graph_runs_end_to_end_for_known_bug():
    # Ticket must already exist in the (mock) Jira store, mirroring how the
    # real agent would be triggered by a Jira webhook for an existing issue.
    client = MockJiraClient()
    ticket = client.get_ticket("OPS-101")

    result = triage_ticket(ticket)
    assert result["category"] == "bug"
    assert result["action"] in {"route_to_engineering", "auto_respond"}
    assert result["jira_update"]["ticket_id"] == "OPS-101"
    assert len(result["trace"]) == 6
