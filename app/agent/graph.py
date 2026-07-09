"""Wires the node functions in app/agent/nodes.py into a LangGraph graph.

Flow:

    classify_ticket -> check_duplicates -> retrieve_docs -> decide_action
        -> generate_response -> update_ticket -> END

retrieve_docs is skipped internally (returns empty) when the ticket was
already flagged as a duplicate, but stays in the graph so the trace is
consistent and the node is easy to unit test in isolation.
"""

from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    check_duplicates,
    classify_ticket,
    decide_action,
    generate_response,
    retrieve_docs,
    update_ticket,
)
from app.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_ticket", classify_ticket)
    graph.add_node("check_duplicates", check_duplicates)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("decide_action", decide_action)
    graph.add_node("generate_response", generate_response)
    graph.add_node("update_ticket", update_ticket)

    graph.set_entry_point("classify_ticket")
    graph.add_edge("classify_ticket", "check_duplicates")
    graph.add_edge("check_duplicates", "retrieve_docs")
    graph.add_edge("retrieve_docs", "decide_action")
    graph.add_edge("decide_action", "generate_response")
    graph.add_edge("generate_response", "update_ticket")
    graph.add_edge("update_ticket", END)

    return graph.compile()


_agent_singleton = None


def get_agent():
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = build_graph()
    return _agent_singleton


def triage_ticket(ticket: dict) -> dict:
    """Runs the compiled graph for a single ticket dict (id/title/description)."""
    agent = get_agent()
    initial_state = {
        "ticket_id": ticket["id"],
        "ticket_title": ticket["title"],
        "ticket_description": ticket["description"],
        "trace": [],
    }
    return agent.invoke(initial_state)
