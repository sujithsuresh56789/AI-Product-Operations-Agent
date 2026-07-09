# AI Product Operations Agent

An agent that triages incoming support tickets the way a product-ops person
would: classify it, check whether it's a duplicate, pull the relevant
internal documentation, decide what to do, draft a reply, and write all of
that back to Jira — automatically.

Built with **LangGraph** (agent orchestration), **ChromaDB** (retrieval /
RAG over internal docs), **Claude** (classification + drafting), and
**FastAPI** (HTTP interface). Jira is mocked for this portfolio demo so
the whole thing runs with zero external accounts — see [Design notes](#design-notes)
for how that swaps out for the real Jira API.

```
Ticket in  →  classify_ticket  →  check_duplicates  →  retrieve_docs (RAG)
           →  decide_action    →  generate_response →  update_ticket  →  Jira
```

## Why this exists

Manually triaging support tickets — figuring out if something's a bug or a
feature request, checking whether it's already been reported, digging up
the relevant internal doc, and drafting a first response — is repetitive
and takes a support/product-ops engineer real time every day. This agent
automates that loop end-to-end while keeping every decision auditable
(each run returns a full trace of what the agent did and why).

## What it actually does

| Step | Node | What happens |
|---|---|---|
| 1 | `classify_ticket` | Claude classifies the ticket as `bug` / `feature_request` / `question` / `other`, plus a priority |
| 2 | `check_duplicates` | TF-IDF similarity search against other open tickets; links and short-circuits duplicates |
| 3 | `retrieve_docs` | Semantic search over an internal knowledge base in ChromaDB (RAG) |
| 4 | `decide_action` | Routes to `auto_respond`, `route_to_engineering`, or `close_duplicate` |
| 5 | `generate_response` | Claude drafts a reply grounded in the retrieved docs |
| 6 | `update_ticket` | Writes labels, comments, assignment, and status back to (mock) Jira |

## Example run

```
$ python demo/run_demo.py

OPS-101: Export to CSV button does nothing on Reports page
Category:     bug
Priority:     high
Duplicate of: None
Action:       route_to_engineering
Retrieved:    ['kb_export_csv', 'kb_custom_date_ranges']
Response:     Hi there — thanks for the report! This is a known issue our
              team is actively working on. We've included a workaround...

OPS-105: Export button not working (duplicate?)
Category:     bug
Priority:     high
Duplicate of: OPS-101          <-- correctly caught as a duplicate
Action:       close_duplicate
Retrieved:    []                <-- retrieval skipped, no point RAG-ing a closed ticket
Response:     Thanks for the report! This looks like the same issue as
              OPS-101, so we're linking and closing this one...
```

Full output for all 6 sample tickets is in [`demo/example_run.txt`](demo/example_run.txt).

## Getting started

```bash
git clone <your-repo-url>
cd ai-product-ops-agent
pip install -r requirements.txt

# optional — see below
cp .env.example .env

# run the agent over the sample tickets and print a full trace
python demo/run_demo.py

# or run the API
uvicorn app.main:app --reload
# → open http://127.0.0.1:8000/docs for interactive Swagger UI
```

### Running with real Claude responses

By default this project runs on a small offline stub LLM (`app/agent/llm.py`)
so anyone can clone it and see it work immediately, with no API key and no
network calls. To use real Claude for classification and drafting, set
`ANTHROPIC_API_KEY` in `.env` — no other code changes needed.

### Running the tests

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -v
```

8 tests cover individual nodes (classification, duplicate detection,
routing), the mock Jira client, vector search, and a full end-to-end graph
run.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/tickets` | List all mock tickets |
| `GET` | `/tickets/{id}` | Get a single ticket |
| `POST` | `/tickets/{id}/triage` | Run the full agent on a ticket, apply the resulting Jira updates, return the full trace |
| `GET` | `/docs/search?q=...` | Run the RAG retrieval step directly |

## Project structure

```
app/
  agent/
    state.py     # shared state schema passed between LangGraph nodes
    llm.py        # Claude wrapper + offline stub fallback
    nodes.py      # the 6 node functions (classify, dedupe, retrieve, decide, respond, update)
    graph.py       # wires the nodes into a LangGraph StateGraph
  services/
    vector_store.py  # ChromaDB + TF-IDF embeddings over data/docs
    jira_mock.py       # mock Jira client (same method shapes as a real client)
  main.py            # FastAPI app
data/
  tickets.json     # 6 sample support tickets
  docs/               # 6 internal knowledge-base articles used for RAG
demo/
  run_demo.py       # CLI script that triages every sample ticket
tests/
  test_agent.py    # unit + integration tests
```

## Design notes

- **Why LangGraph instead of a single prompt?** Ticket triage is naturally
  a multi-step decision process with branching (a duplicate doesn't need
  fresh doc retrieval; a question doesn't get routed to engineering).
  Modeling it as an explicit graph makes each step independently testable
  and the whole flow auditable via `state["trace"]`, instead of hoping one
  big prompt gets every step right.
- **Why a mock Jira client instead of the real API?** So the project is
  runnable by anyone with `pip install -r requirements.txt` — no Jira
  account, no API tokens. `app/services/jira_mock.py` mirrors the method
  shapes of a real Jira client (`add_comment`, `transition_status`,
  `assign`, etc.), so swapping in `atlassian-python-api` later is a
  one-file change; the LangGraph nodes don't need to know the difference.
- **Why TF-IDF instead of a hosted embedding model?** Same reasoning —
  zero external dependencies to run the demo. `TfidfEmbeddingFunction` in
  `vector_store.py` implements ChromaDB's `EmbeddingFunction` interface,
  so swapping in OpenAI/Voyage/Anthropic embeddings for production is a
  one-line change with no changes to the retrieval node.
- **Duplicate detection** uses TF-IDF cosine similarity over ticket
  title+description rather than exact/fuzzy string matching, since
  real-world duplicate reports are rarely worded the same way (e.g.
  "Export to CSV does nothing" vs. "Export button not working").

## Possible extensions

- Swap the mock Jira client for a real one via `atlassian-python-api`
- Swap TF-IDF for a hosted embedding model for better semantic recall
- Add a `sentiment_check` node to flag frustrated customers for human review
- Persist agent decisions to a database for analytics on triage accuracy over time
- Add human-in-the-loop approval before `update_ticket` writes for high-risk actions

## License

MIT
