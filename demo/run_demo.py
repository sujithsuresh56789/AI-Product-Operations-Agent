"""Runs the agent over every mock ticket and prints a readable trace.

Usage:
    python demo/run_demo.py

Works with zero setup (uses the offline StubLLM) or with a real Claude
model if ANTHROPIC_API_KEY is set in the environment / a .env file.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from app.agent.graph import triage_ticket  # noqa: E402

TICKETS_FILE = Path(__file__).resolve().parents[1] / "data" / "tickets.json"


def main():
    tickets = json.loads(TICKETS_FILE.read_text())
    print(f"Running agent over {len(tickets)} mock tickets...\n")

    for ticket in tickets:
        print("=" * 78)
        print(f"{ticket['id']}: {ticket['title']}")
        print("-" * 78)

        result = triage_ticket(ticket)

        print(f"Category:     {result['category']}")
        print(f"Priority:     {result['priority']}")
        print(f"Duplicate of: {result['duplicate_of']}")
        print(f"Action:       {result['action']}")
        print(f"Retrieved:    {[d['source'] for d in result['retrieved_docs']]}")
        print(f"Response:     {result['response_draft'][:160]}"
              f"{'...' if len(result['response_draft']) > 160 else ''}")
        print("\nTrace:")
        for step in result["trace"]:
            print(f"  - {step}")
        print()

    print("=" * 78)
    print("Done. This was run with the offline StubLLM unless ANTHROPIC_API_KEY "
          "was set -- see README for how to switch to real Claude responses.")


if __name__ == "__main__":
    main()
