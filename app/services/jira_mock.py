"""Mock Jira client for portfolio/demo purposes.

The method signatures deliberately mirror the subset of the real Jira
REST API (`jira` Python package / `atlassian-python-api`) that this
agent needs: get issue, add comment, transition status, add labels,
assign. Swapping this module for a real Jira client later means the
LangGraph nodes in agent/nodes.py do not need to change -- only the
import at the top of this file does.

Data is stored in-memory (seeded from data/tickets.json) for the
lifetime of the process.
"""

import json
from pathlib import Path
from typing import List, Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TICKETS_FILE = DATA_DIR / "tickets.json"


class MockJiraClient:
    def __init__(self):
        self._tickets = {t["id"]: t for t in json.loads(TICKETS_FILE.read_text())}

    def list_tickets(self) -> List[dict]:
        return list(self._tickets.values())

    def get_ticket(self, ticket_id: str) -> Optional[dict]:
        return self._tickets.get(ticket_id)

    def add_comment(self, ticket_id: str, comment: str) -> dict:
        ticket = self._require(ticket_id)
        ticket.setdefault("comments", []).append(comment)
        return {"ticket_id": ticket_id, "action": "comment_added", "comment": comment}

    def add_labels(self, ticket_id: str, labels: List[str]) -> dict:
        ticket = self._require(ticket_id)
        existing = set(ticket.get("labels", []))
        existing.update(labels)
        ticket["labels"] = sorted(existing)
        return {"ticket_id": ticket_id, "action": "labels_added", "labels": ticket["labels"]}

    def transition_status(self, ticket_id: str, status: str) -> dict:
        ticket = self._require(ticket_id)
        ticket["status"] = status
        return {"ticket_id": ticket_id, "action": "status_transitioned", "status": status}

    def assign(self, ticket_id: str, assignee: str) -> dict:
        ticket = self._require(ticket_id)
        ticket["assignee"] = assignee
        return {"ticket_id": ticket_id, "action": "assigned", "assignee": assignee}

    def link_duplicate(self, ticket_id: str, duplicate_of: str) -> dict:
        ticket = self._require(ticket_id)
        ticket["duplicate_of"] = duplicate_of
        return {"ticket_id": ticket_id, "action": "linked_duplicate", "duplicate_of": duplicate_of}

    def _require(self, ticket_id: str) -> dict:
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            raise KeyError(f"Unknown ticket: {ticket_id}")
        return ticket


_client_singleton: Optional[MockJiraClient] = None


def get_jira_client() -> MockJiraClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = MockJiraClient()
    return _client_singleton
