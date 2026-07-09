"""FastAPI surface for the AI product operations agent.

Run with:  uvicorn app.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent.graph import triage_ticket
from app.services.jira_mock import get_jira_client
from app.services.vector_store import search_docs

app = FastAPI(
    title="AI Product Operations Agent",
    description=(
        "LangGraph + ChromaDB agent that triages support tickets, retrieves "
        "relevant internal docs (RAG), drafts responses, and writes decisions "
        "back to Jira. Jira is mocked for this portfolio demo -- see "
        "app/services/jira_mock.py."
    ),
    version="0.1.0",
)


class DocSearchResult(BaseModel):
    source: str
    text: str
    score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tickets")
def list_tickets():
    return get_jira_client().list_tickets()


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    ticket = get_jira_client().get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticket: {ticket_id}")
    return ticket


@app.post("/tickets/{ticket_id}/triage")
def triage(ticket_id: str):
    """Runs the full LangGraph agent on a ticket and applies the resulting
    (mock) Jira updates. Returns the full state trace for transparency."""
    ticket = get_jira_client().get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticket: {ticket_id}")

    result = triage_ticket(ticket)
    return result


@app.get("/docs/search", response_model=list[DocSearchResult])
def docs_search(q: str, top_k: int = 2):
    """Direct access to the RAG retrieval step, useful for debugging /
    demonstrating the vector search independent of the full agent."""
    return search_docs(q, top_k=top_k)
