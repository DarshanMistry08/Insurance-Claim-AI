from __future__ import annotations
import os
import re
import time
import logging
import json
import requests

import psycopg2
from sentence_transformers import SentenceTransformer

from agents.state import ClaimState, PolicyClause, NodeLog, Flag
from agents.tracing import NodeTracer

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "claims_db"),
    "user":     os.getenv("DB_USER", "claims"),
    "password": os.getenv("DB_PASSWORD", ""),
}

TOP_K = int(os.getenv("POLICY_RETRIEVAL_TOP_K", "3"))
COVERAGE_LIMITS: dict[str, float] = {
    "water":  25_000,
    "fire":   500_000,
    "theft":  15_000,
    "storm":  75_000,
    "auto":   100_000,
}

_emb_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _emb_model


def _build_query(extracted_fields: dict) -> str:
    def val(field: str) -> str:
        f = extracted_fields.get(field, {})
        if isinstance(f, dict):
            return f.get("value", "unknown")
        return str(f) if f else "unknown"

    incident_type = val("incident_type")
    amount = val("amount_claimed")
    return f"coverage for {incident_type} damage claim amount {amount}"


def _retrieve_clauses(query: str) -> list[PolicyClause]:
    model = _get_model()
    query_emb = model.encode(query).tolist()

    clauses: list[PolicyClause] = []

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Join chunks with policy_clauses for clause_id and policy_doc
        cursor.execute(
            """
            SELECT
                COALESCE(pc.clause_id, 'CHUNK_' || c.chunk_id::text) AS clause_id,
                COALESCE(pc.policy_doc, d.file_path)                 AS policy_doc,
                c.chunk_text,
                1 - (c.embedding <=> %s::vector)                     AS similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.document_id
            LEFT JOIN policy_clauses pc ON pc.chunk_id = c.chunk_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_emb, query_emb, TOP_K),
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        for clause_id, policy_doc, chunk_text, similarity in rows:
            clauses.append(PolicyClause(
                clause_id=clause_id,
                policy_doc=policy_doc,
                chunk_text=chunk_text,
                similarity_score=round(float(similarity), 4),
            ))

    except Exception as e:
        logger.error("policy_retrieval: DB error %s", e)

    return clauses


def _check_coverage_limit(extracted_fields: dict, clauses: list[PolicyClause]) -> list[Flag]:
    flags: list[Flag] = []

    def val(field: str) -> str:
        f = extracted_fields.get(field, {})
        return (f.get("value", "N/A") if isinstance(f, dict) else str(f))

    incident_type = val("incident_type").lower()
    amount_str = val("amount_claimed")

    try:
        amount = float(re.sub(r"[^0-9.]", "", amount_str))
    except (ValueError, TypeError):
        return flags

    limit = COVERAGE_LIMITS.get(incident_type)
    if limit and amount > limit:
        clause_ref = ""
        for c in clauses:
            if incident_type in c.chunk_text.lower() or incident_type in c.clause_id.lower():
                clause_ref = c.clause_id
                break

        code_map = {
            "water": "AMOUNT_EXCEEDS_WATER_LIMIT",
            "fire":  "AMOUNT_EXCEEDS_FIRE_LIMIT" if amount <= 500_000 else "AMOUNT_EXCEEDS_DWELLING_LIMIT",
            "theft": "AMOUNT_EXCEEDS_THEFT_LIMIT",
            "storm": "AMOUNT_EXCEEDS_STORM_LIMIT",
            "auto":  "AMOUNT_EXCEEDS_AUTO_LIMIT",
        }
        code = code_map.get(incident_type, "AMOUNT_EXCEEDS_COVERAGE_LIMIT")

        flags.append(Flag(
            code=code,
            severity="HIGH",
            description=(
                f"Claimed amount {amount_str} exceeds the policy limit of "
                f"${limit:,.0f} for {incident_type} damage (clause {clause_ref})."
            ),
            field_ref="amount_claimed",
            clause_ref=clause_ref,
        ))

    return flags


def node_policy_retrieval(state: ClaimState) -> dict:
    """
    Retrieves top-k policy clauses via pgvector and raises coverage-limit flags.
    Skips retrieval for non-claim document types.
    """
    doc_id = state["doc_id"]
    doc_type = state.get("document_type", "claim_form")
    extracted_fields = state.get("extracted_fields", {})

    with NodeTracer("policy_retrieval_agent", trace_id=doc_id, inputs={"doc_id": doc_id}) as tracer:
        t0 = time.perf_counter()

        if doc_type != "claim_form":
            log = NodeLog(
                node_name="policy_retrieval_agent",
                latency_ms=0.5,
                langfuse_span_id=tracer.span_id,
                error="",
            )
            tracer.end(outputs={"clauses_retrieved": 0, "flags_raised": 0})
            return {
                "retrieved_clauses": [],
                "policy_context": f"Document classified as '{doc_type}'. Policy chunk retrieval is not required for non-claim documents.",
                "flags": [],
                "agent_trace": [log.model_dump()],
            }

        error_msg = ""
        clauses: list[PolicyClause] = []

        try:
            query = _build_query(extracted_fields)
            clauses = _retrieve_clauses(query)
            logger.info("policy_retrieval: retrieved %d clauses for doc_id=%s", len(clauses), doc_id)
        except Exception as e:
            error_msg = str(e)
            logger.error("policy_retrieval: failed doc_id=%s error=%s", doc_id, e)

        raw_policy_context = "\n\n".join(
            f"[{c.clause_id}] {c.chunk_text}" for c in clauses
        ) if clauses else "No policy context found."

        if clauses:
            prompt = f"""You are an Insurance Policy Analysis AI.

IMPORTANT RULES:

1. NEVER return the complete policy text.
2. NEVER return the same policy context for different claims.
3. ONLY return the policy clauses that are relevant to the uploaded claim.
4. Ignore unrelated policy information.
5. Every response MUST be different if:
    - claim type changes
    - damage type changes
    - amount changes
    - policy changes
    - coverage changes
6. DO NOT use predefined summaries.

ANALYSIS PROCESS

Step 1:
Analyze the extracted claim information.

- incident type
- amount claimed
- policy number
- date
- claimant details
- supporting documents
- damages

Step 2:
Analyze the policy chunks.

Step 3:
Retrieve ONLY those clauses which are directly related to:

- claim type
- damages
- exclusions
- limitations
- coverage amount
- deductibles
- waiting periods

OUTPUT RULES

If policy information is relevant:

Return:

------------------------------------------------

RELEVANT POLICY FINDINGS

Covered Items:
- mention only covered damages.

Coverage Limit:
- mention only the applicable limit.

Restrictions:
- mention only applicable restrictions.

Exclusions:
- mention only applicable exclusions.

Deductibles:
- mention only if available.

Policy Match Status:
- Covered
- Partially Covered
- Not Covered

------------------------------------------------

If no relevant policy information exists write:

"No relevant policy clauses were found for this claim."

------------------------------------------------

DO NOT:

- copy entire policy documents
- repeat previous responses
- hallucinate policy information
- generate generic coverage summaries

IMPORTANT:

The response MUST depend completely on:
- uploaded claim
- retrieved policy chunks

If two claims are different the retrieved policy context MUST also be different.

CLAIM INFORMATION:
{json.dumps(extracted_fields, indent=2)}

POLICY CHUNKS:
{raw_policy_context}
"""
            try:
                resp = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}},
                    timeout=90,
                )
                resp.raise_for_status()
                policy_context = resp.json().get("response", "").strip()
            except Exception as e:
                logger.error("policy_retrieval: LLM call failed: %s", e)
                policy_context = raw_policy_context
        else:
            policy_context = "No relevant policy clauses were found for this claim."

        new_flags = _check_coverage_limit(extracted_fields, clauses)

        latency_ms = (time.perf_counter() - t0) * 1000
        log = NodeLog(
            node_name="policy_retrieval_agent",
            latency_ms=round(latency_ms, 1),
            langfuse_span_id=tracer.span_id,
            error=error_msg,
        )

        tracer.end(outputs={
            "clauses_retrieved": len(clauses),
            "flags_raised": len(new_flags),
        })

    return {
        "retrieved_clauses": [c.model_dump() for c in clauses],
        "policy_context": policy_context,
        "flags": [f.model_dump() for f in new_flags],
        "agent_trace": [log.model_dump()],
    }
