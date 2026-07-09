"""Node functions for the product-ops LangGraph agent.

Each function takes the current AgentState and returns a partial dict
of updates (LangGraph merges these into the running state). Keeping
each node small and single-purpose makes the graph easy to test node
by node, and easy to re-order or extend later (e.g. add a
"sentiment_check" node without touching the others).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.agent.llm import get_llm, parse_key_value
from app.agent.state import AgentState
from app.services.jira_mock import get_jira_client
from app.services.vector_store import search_docs

# Tuned against the sample dataset: near-duplicate tickets worded quite
# differently (e.g. "Export to CSV does nothing" vs "Export button not
# working") land around 0.3 cosine similarity on short-text TF-IDF,
# while unrelated tickets sit well below 0.1.
DUPLICATE_SIMILARITY_THRESHOLD = 0.25


def _log(state: AgentState, message: str) -> list:
    trace = list(state.get("trace", []))
    trace.append(message)
    return trace


def classify_ticket(state: AgentState) -> dict:
    """Classifies category (bug/feature_request/question/other) and priority."""
    llm = get_llm()
    prompt = (
        "Classify the following support ticket.\n"
        f"Title: {state['ticket_title']}\n"
        f"Description: {state['ticket_description']}\n\n"
        "Respond with two lines:\n"
        "category: <bug|feature_request|question|other>\n"
        "priority: <low|medium|high|critical>"
    )
    response = llm.invoke(prompt)
    category = parse_key_value(response, "category", default="other")
    priority = parse_key_value(response, "priority", default="low")

    return {
        "category": category,
        "priority": priority,
        "trace": _log(state, f"classify_ticket -> category={category}, priority={priority}"),
    }


def check_duplicates(state: AgentState) -> dict:
    """TF-IDF cosine-similarity duplicate check against other open tickets.

    Only compares against tickets with an earlier `created_at`, so a
    later ticket gets linked to the original rather than the other way
    around. Interface (return a ticket id or None) stays stable even if
    this is swapped for embedding-based similarity in production.
    """
    jira = get_jira_client()
    this_ticket = jira.get_ticket(state["ticket_id"])
    this_title = this_ticket["title"] if this_ticket else state["ticket_title"]
    this_description = this_ticket["description"] if this_ticket else state["ticket_description"]
    this_created_at = this_ticket["created_at"] if this_ticket else None

    others = [
        t for t in jira.list_tickets()
        if t["id"] != state["ticket_id"]
        and (this_created_at is None or t["created_at"] < this_created_at)
    ]

    duplicate_of = None
    if others:
        texts = [f"{this_title} {this_description}"] + [
            f"{o['title']} {o['description']}" for o in others
        ]
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        similarities = cosine_similarity(matrix[0:1], matrix[1:])[0]

        best_idx = similarities.argmax()
        if similarities[best_idx] >= DUPLICATE_SIMILARITY_THRESHOLD:
            duplicate_of = others[best_idx]["id"]

    return {
        "duplicate_of": duplicate_of,
        "trace": _log(state, f"check_duplicates -> duplicate_of={duplicate_of}"),
    }


def retrieve_docs(state: AgentState) -> dict:
    """RAG step: pulls relevant internal documentation from ChromaDB."""
    if state.get("duplicate_of"):
        # No need to retrieve docs for a ticket we're about to close as a duplicate.
        return {"retrieved_docs": [], "trace": _log(state, "retrieve_docs -> skipped (duplicate)")}

    query = f"{state['ticket_title']} {state['ticket_description']}"
    docs = search_docs(query, top_k=2)
    sources = [d["source"] for d in docs]
    return {
        "retrieved_docs": docs,
        "trace": _log(state, f"retrieve_docs -> sources={sources}"),
    }


def decide_action(state: AgentState) -> dict:
    """Rule-based routing decision informed by the earlier steps."""
    if state.get("duplicate_of"):
        action = "close_duplicate"
    elif state["category"] == "bug" and state["priority"] in ("high", "critical"):
        action = "route_to_engineering"
    elif state["category"] == "feature_request":
        action = "route_to_engineering"
    else:
        action = "auto_respond"

    return {"action": action, "trace": _log(state, f"decide_action -> action={action}")}


def generate_response(state: AgentState) -> dict:
    """Drafts a customer-facing reply grounded in retrieved documentation."""
    if state["action"] == "close_duplicate":
        draft = (
            f"Thanks for the report! This looks like the same issue as "
            f"{state['duplicate_of']}, so we're linking and closing this one to keep "
            f"discussion in a single place. We'll post updates there."
        )
        return {"response_draft": draft, "trace": _log(state, "generate_response -> duplicate reply")}

    context = "\n\n".join(f"[{d['source']}]\n{d['text']}" for d in state.get("retrieved_docs", []))
    llm = get_llm()
    prompt = (
        "Draft a helpful, concise reply to a customer support ticket, "
        "grounded in the internal documentation context below. Do not "
        "invent information that isn't supported by the context.\n\n"
        f"Ticket: {state['ticket_title']} — {state['ticket_description']}\n\n"
        f"Internal documentation context:\n{context}"
    )
    draft = llm.invoke(prompt)
    return {"response_draft": draft, "trace": _log(state, "generate_response -> drafted reply")}


def update_ticket(state: AgentState) -> dict:
    """Writes the agent's decisions back to (mock) Jira."""
    jira = get_jira_client()
    ticket_id = state["ticket_id"]
    updates = []

    updates.append(jira.add_comment(ticket_id, state["response_draft"]))
    updates.append(jira.add_labels(ticket_id, [state["category"], f"priority:{state['priority']}"]))

    if state["action"] == "close_duplicate":
        updates.append(jira.link_duplicate(ticket_id, state["duplicate_of"]))
        updates.append(jira.transition_status(ticket_id, "closed"))
    elif state["action"] == "route_to_engineering":
        updates.append(jira.assign(ticket_id, "engineering-triage-queue"))
        updates.append(jira.transition_status(ticket_id, "in_progress"))
    else:  # auto_respond
        updates.append(jira.transition_status(ticket_id, "awaiting_customer"))

    return {
        "jira_update": {"ticket_id": ticket_id, "updates": updates},
        "trace": _log(state, f"update_ticket -> {len(updates)} write(s) applied"),
    }
