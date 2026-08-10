"""
pipeline/graph.py — LangGraph orchestrator for the ClaimSight pipeline.

This module is a thin wiring layer. All logic lives in /agents/*.
Exposes `app` (compiled graph) at module level for:
  - LangGraph Studio discovery (langgraph.json)
  - FastAPI import (api/main.py)
  - CLI / eval usage

Pipeline topology (sequential):
    START → extract → policy_retrieval → verification → fraud_scoring → summary → END
"""

import os
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # In Docker, env vars are injected via docker-compose

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

from langgraph.graph import StateGraph, START, END

from agents.state import ClaimState, empty_state
from agents.extract_agent import node_extract
from agents.policy_retrieval_agent import node_policy_retrieval
from agents.verification_agent import node_verification
from agents.fraud_scoring_agent import node_fraud_scoring
from agents.summary_agent import node_summary


def build_graph() -> object:
    """Construct and compile the ClaimSight LangGraph state machine."""
    workflow = StateGraph(ClaimState)

    # Register nodes
    workflow.add_node("extract",           node_extract)
    workflow.add_node("policy_retrieval",  node_policy_retrieval)
    workflow.add_node("verification",      node_verification)
    workflow.add_node("fraud_scoring",     node_fraud_scoring)
    workflow.add_node("summary",           node_summary)

    # Linear pipeline edges
    workflow.add_edge(START,              "extract")
    workflow.add_edge("extract",          "policy_retrieval")
    workflow.add_edge("policy_retrieval", "verification")
    workflow.add_edge("verification",     "fraud_scoring")
    workflow.add_edge("fraud_scoring",    "summary")
    workflow.add_edge("summary",          END)

    return workflow.compile()


# Module-level compiled graph (required by LangGraph Studio)
app = build_graph()


# ─── Convenience run function ─────────────────────────────────── #

def run_pipeline(doc_id: str, image_path: str) -> dict:
    """
    Run the full pipeline for a single document.

    Args:
        doc_id:     Unique identifier for the document.
        image_path: Absolute or relative path to the claim image.

    Returns:
        Final ClaimState dict after all 5 nodes execute.
    """
    initial = empty_state(doc_id=doc_id, image_path=image_path)
    return app.invoke(initial)
